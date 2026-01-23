"""
Candidates Discovery Stage

In this stage, candidates for actual dataset generation in next steps
are proposed through an agentic step.

1. A first AI Agent analyses a random subset of datasets
    drawned from the whole available collection;
        a. for each dataset, it inspects its available metadata and a sample
            of few rows;
        b. if the dataset is valid, the Agent proposes a set of data discovery
            tasks to perform on it against the BLEND index.
2. The proposed tasks are executed and each identified pair of candidate
    matching datasets is stored.
        a. Schema matching techniques are applied on the Join-discovered candidates,
            avoiding to consider datasets with nearly-identical or identical schema
            as joinable, while instead would be better to consider them only as unionable.
3. The identified matches are used to build a graph representing the relationships
    discovered among datasets: also, the graph is enriched with metadata about the
    overlap ratio between each pair of tables, and metrics from schema matching
    techniques.
4. The graph is navigated in a random-walk or metadata-driven fashion to generate
    paths of related involved, with 1...N datasets involved.
    If a boolean predicate is specified, the generation is conducted on the sub-graph
    induced by the predicate.
        a. In the random-walk case, the graph is just traversed focusing only on
            a specified type or relationships (e.g. Union-only), avoiding loops
            and repeteated nodes.
        b. In the metadata-driven case, the graph is again traversed in a random-walk
            style, but giving more weight to one of the available metrics (overlap_ratio, ...)
    In both case, each generated path is then used as final candidate.
"""

import json
import os
import random
import resource
import time
from functools import partial
from pathlib import Path

import dotenv
import networkx as nx
import polars as pl
from blend import BLEND
from tqdm import tqdm
from wrapt_timeout_decorator import timeout

from conf import OrQAConfig

from .agent import agent
from .graph import graph_builder, graph_explorer
from .utils import load_datasets_metadata, pl_read_dataset, remove_file_extension

# make the API key for LLM available
dotenv.load_dotenv(Path(__file__).parent.parent.joinpath(".env"))


# NOTE: this implementation of BLEND is currently based on DuckDB, and
# in some cases it needs to fetch a lot of data. We try to limit the memory
# usage in order to avoid a machine complete block (maybe works maybe not,
# no time for complete tests)
def memory_limit_half():
    """Limit max memory usage to half."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # Convert KiB to bytes, and divide in two to half
    resource.setrlimit(resource.RLIMIT_AS, (int(get_memory() * 1024 / 2), hard))


def get_memory():
    with open("/proc/meminfo", "r") as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ("MemFree:", "Buffers:", "Cached:"):
                free_memory += int(sline[1])
    return free_memory  # KiB


def _f_nop(x: str):
    return x


def _f_comma_dot_filter_tokens(x: str):
    x = x.strip().replace(",", "").lower()
    return x if x != "total" else None


_string_to_float_methods = [_f_nop, _f_comma_dot_filter_tokens]


def sample_seed_datasets(
    datasets_path: Path, n_datasets_to_sample: int, seed: int
) -> list[tuple[str, Path, str, str]]:
    datasets = os.listdir(datasets_path)

    assert len(datasets) >= n_datasets_to_sample, (
        f"Too many datasets to sample: {len(datasets)} < {n_datasets_to_sample}"
    )

    random.seed(seed)
    sample = random.sample(datasets, n_datasets_to_sample)

    # remove the filetype extension from each sample filename
    sample = [(remove_file_extension(f), datasets_path.joinpath(f)) for f in sample]

    sample = [
        (
            filename,
            filepath,
            *(filename.split("::") if "::" in filename else ("", filename)),
        )
        for filename, filepath in sample
    ]

    return sample  # ty: ignore


def generate_tasks(cfg: OrQAConfig, seed_datasets: list[tuple[str, Path, str]]):
    datasets_path = cfg.datasets_path

    # for each of these sample datasets, fetch its relative metadata
    print("Loading metadata for sampled datasets...")
    metadata = load_datasets_metadata(
        cfg.metadata_path.joinpath("metadata.json"),
        [s[3 if len(s) == 4 else 2] for s in seed_datasets],
    )

    litellm_config_path = cfg.llm_config_path.joinpath("litellm.yaml")

    # setup the Agent
    # In this case, we call this entity "agent", even if
    # it is just a wrapper of a LLM client, without any
    # needs of tool-calling or memory or other properties
    print("Loading LLM-agent")
    _agent = agent.CandidatesDiscoveryAgent(litellm_config_path)

    results = {}

    for dataset_filename, dataset_path, dataset_name, dataset_id in tqdm(
        seed_datasets, desc="Generating Tasks"
    ):
        try:
            _metadata = metadata[dataset_id]
            _format = dataset_path.suffix.replace(".", "")

            results[dataset_id] = _agent.propose_tasks(
                dataset_path,
                _format,
                _metadata,
                cfg.polars_opts.scan,
                cfg.candidates_discovery.min_dataset_height,
                cfg.candidates_discovery.limit_to_n_columns,
                cfg.candidates_discovery.sample_size,
                seed=cfg.seed,
            )
            time.sleep(30)
        except Exception as e:
            print(f"\n\n{e}\n\n")

    with open(
        cfg.candidates_discovery.proposed_tasks_path, "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    return results


@timeout(30)
def _execute_union_search(index: BLEND, table, k):
    return index.union_search(table, k)


@timeout(30)
def _execute_single_join_search(index: BLEND, column, k):
    return index.single_column_join_search(column, k)


@timeout(45)
def _execute_multi_join_search(index: BLEND, table, k):
    return index.multi_column_join_search(table, k)


@timeout(30)
def _execute_correlation_search(index: BLEND, keys, targets, k, hash_size):
    return index.correlation_search(keys, targets, k, hash_size)


def execute_tasks(cfg: OrQAConfig, tasks: dict) -> list[dict]:
    """
    Given a collection of data discovery tasks (Union, Join, Multi-column Join, Join-Correlation),
    executes them on the BLEND index.

    :param cfg: The OrQA configuration.
    :param tasks: A dictionary of task_id-task.
    :return: A list of dictionaries, one for each executed task.
    """
    datasets_path = cfg.datasets_path
    _format = cfg.datasets_format

    tasks = sorted(
        list(tasks.values()), key=lambda c: c.get("dataset_id", "") if c else ""
    )  # ty: ignore

    # instantiate the BLEND index
    index = BLEND(
        cfg.indexing.index_database_path,
        clean_args=cfg.indexing.clean_args,
        xash_size=cfg.indexing.xash_size,
    )

    top_k = cfg.candidates_discovery.top_k_results_per_task

    # a collection where we will store our effective candidates as dictionaries
    candidates = []

    # for each task, execute it over the index
    for task_set in tqdm(tasks, desc="Executing tasks: "):
        if not task_set:
            continue

        dataset_filename = task_set["dataset"]
        dataset_path = datasets_path.joinpath(f"{dataset_filename}.{_format}")

        _tasks = task_set["tasks"]
        if isinstance(_tasks, str):
            print(f"Strange! _tasks is str: {_tasks}")
            continue

        union_tasks = _tasks.get("union_tasks", [])
        join_tasks = _tasks.get("join_tasks", [])
        join_correlation = _tasks.get("join_correlation_tasks", [])

        df = pl_read_dataset(dataset_path, cfg.polars_opts.read)

        for task in tqdm(union_tasks, desc="Union tasks", position=1, leave=False):
            columns = task["columns"]
            table = df.select(columns).rows()

            try:
                top_res = _execute_union_search(index, table, top_k)
            except (TimeoutError, RuntimeError):
                print(f"Timeout on Union: {dataset_filename}")
                continue

            for candidate in top_res:
                cand_id = candidate[0]
                if cand_id == dataset_filename:
                    continue
                candidates.append(
                    {
                        "Q": dataset_filename,
                        "R": cand_id,
                        "task": "U",
                        "q_columns": columns,
                    }
                )

        for task in tqdm(join_tasks, desc="Join tasks", position=1, leave=False):
            columns = task["columns"]
            if len(columns) == 1:
                column = df.get_column(columns[0]).to_list()
                try:
                    top_res = _execute_single_join_search(index, column, top_k)
                except (TimeoutError, RuntimeError):
                    print(f"Timeout on Single-Join: {dataset_filename}")
                    continue
                for cand_id, r_join_key, _ in top_res:
                    if cand_id == dataset_filename:
                        continue
                    candidates.append(
                        {
                            "Q": dataset_filename,
                            "R": cand_id,
                            "task": "J",
                            "q_join_key": columns[0],
                            "r_join_key_pos": r_join_key,
                        }
                    )
            else:
                table = df.select(columns).unique().rows()
                try:
                    top_res = _execute_multi_join_search(index, table, top_k)
                except (TimeoutError, RuntimeError):
                    print(f"Timeout on Multi-Join: {dataset_filename}")
                    continue
                for cand_id, r_join_keys, _ in top_res:
                    if cand_id == dataset_filename:
                        continue
                    candidates.append(
                        {
                            "Q": dataset_filename,
                            "R": cand_id,
                            "task": "J",
                            "q_join_keys": columns,
                            "r_join_keys_pos": r_join_keys,
                        }
                    )

        for task in tqdm(
            join_correlation, desc="Join-Correlation tasks", position=1, leave=False
        ):
            # get the column names
            key_column = task["join_column"]
            target_column = task["correlation_column"]

            # get the actual column
            keys = df.get_column(key_column).to_list()
            targets = df.get_column(target_column)

            is_float = targets.dtype.is_numeric()
            if not is_float:
                for method in _string_to_float_methods:
                    try:
                        targets = (
                            df.get_column(target_column)
                            .map_elements(method, pl.String)
                            .cast(pl.Float32)
                        )
                        is_float = True
                        break
                    except pl.exceptions.InvalidOperationError as e:
                        print(f"Method: {method}")
                        print(str(e).replace("\n\n", "\n"))
                        continue

            if not is_float:
                continue

            targets = targets.to_list()
            hash_size = cfg.candidates_discovery.qcr_hash_size

            try:
                top_res = _execute_correlation_search(
                    index, keys, targets, top_k, hash_size
                )
            except (TimeoutError, RuntimeError):
                print(f"Timeout on Join-Correlation: {dataset_filename}")
                continue

            for cand_id, r_key_column, r_target_column, score in top_res:
                candidates.append(
                    {
                        "Q": dataset_filename,
                        "R": cand_id,
                        "task": "JC",
                        "q_key": key_column,
                        "q_target": target_column,
                        "r_key": r_key_column,
                        "r_target": r_target_column,
                    }
                )
        with open(cfg.candidates_discovery.tasks_results_path, "w") as file:
            json.dump(candidates, file, indent=4)
    return candidates


def build_matches_graph(cfg: OrQAConfig, matches: list[dict]) -> nx.MultiDiGraph:
    G = graph_builder.build_matches_graph(
        matches, cfg.datasets_path, cfg.polars_opts.read
    )

    nx.write_gml(G, cfg.candidates_discovery.matches_graph_path)
    return G


def overlap_ratio_only_predicate(edge_data: dict, overlap_threshold: float) -> bool:
    return edge_data["overlap_ratio"] >= overlap_threshold


def explore_matches_graph(
    cfg: OrQAConfig, G: nx.MultiDiGraph, seed_datasets: list[tuple[str, Path, str]]
):
    _overlap_ratio_only_predicate = partial(
        overlap_ratio_only_predicate,
        overlap_threshold=cfg.candidates_discovery.overlap_ratio_threshold,
    )

    random_walks = []

    for dataset_filename, dataset_path, dataset_name, dataset_id in tqdm(
        seed_datasets, desc="Exploring Graph"
    ):
        # FIX: Here we do not check whether a Join column is considered
        # also in any other Join/Join-Correlation candidate match during
        # search.
        for edge_labels in [["U"], ["J", "JC"]]:
            sub_graph = graph_explorer.fetch_matches(
                G,
                dataset_id,
                None,  # _overlap_ratio_only_predicate,
                edge_labels,  # ty: ignore
                cfg.candidates_discovery.max_path_length,
            )

            for random_walk in nx.generate_random_paths(
                sub_graph,
                cfg.candidates_discovery.n_paths_for_dataset,
                cfg.candidates_discovery.max_path_length,
                weight="overlap_ratio",
                seed=cfg.seed,
                source=dataset_id,
            ):
                random_walks.append(
                    {
                        "Q": dataset_id,
                        "operation_type": edge_labels,
                        "datasets": random_walk,  # this should be a list
                    }
                )

    with open(cfg.candidates_discovery.candidates_path, "w") as file:
        json.dump(random_walks, file)


def get_seed_datasets(cfg: OrQAConfig):
    # sample dataset seeds for the candidates discovery step
    if cfg.candidates_discovery.seeds_datasets_path.exists():
        with open(cfg.candidates_discovery.seeds_datasets_path) as file:
            sample = json.load(file)
    else:
        sample = sample_seed_datasets(
            cfg.datasets_path, cfg.candidates_discovery.n_random_dataset_seeds, cfg.seed
        )

    return sample


def candidates_discovery(cfg: OrQAConfig):
    GENERATE_TASKS = True
    EXECUTE_TASKS = False
    BUILD_GRAPH = False
    EXPLORE_GRAPH = False

    seed_datasets = get_seed_datasets(cfg)

    if GENERATE_TASKS:
        generate_tasks(cfg, seed_datasets)
    with open(cfg.candidates_discovery.proposed_tasks_path, "r") as file:
        generated_tasks = json.load(file)

    memory_limit_half()

    if EXECUTE_TASKS:
        execute_tasks(cfg, generated_tasks)
    with open(
        cfg.candidates_discovery.tasks_results_path,
        "r",
    ) as file:
        discovered_candidates = json.load(file)

    if BUILD_GRAPH:
        G = build_matches_graph(cfg, discovered_candidates)
    else:
        G = nx.read_gml(cfg.candidates_discovery.matches_graph_path)

    if EXPLORE_GRAPH:
        explore_matches_graph(cfg, G, seed_datasets)
