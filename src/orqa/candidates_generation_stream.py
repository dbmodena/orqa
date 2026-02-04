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

The workflow changes now:
    For each dataset seed:
        1. Propose tasks
        2. Execute tasks
        3. Integrate results into neighbors graph
        4. Go to 1
"""

from orqa.graph.matches_graph import DatasetMatchesGraph

from networkx.algorithms import union

import json
import os
import random
import resource
import time
from pathlib import Path

import dotenv
import networkx as nx
import polars as pl
from blend import BLEND
from tqdm import tqdm
from wrapt_timeout_decorator import timeout

from conf import OrQAConfig

from .agent.agent import CandidatesDiscoveryAgent

from .graph import matches_graph
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


SEP = "__"


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
            *(filename.split(SEP) if SEP in filename else ("", filename)),
        )
        for filename, filepath in sample
    ]

    return sample  # ty: ignore


def generate_tasks(
    cfg: OrQAConfig,
    dataset_path: Path,
    metadata: dict,
    agent: CandidatesDiscoveryAgent,
) -> dict | None:
    dataset_format = dataset_path.suffix.replace(".", "")

    return agent.propose_tasks(
        dataset_path,
        dataset_format,
        metadata,
        cfg.polars_opts.scan,
        cfg.candidates_discovery.min_dataset_height,
        cfg.candidates_discovery.limit_to_n_columns,
        cfg.candidates_discovery.sample_size,
        cfg.seed,
    )


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


def execute_tasks(
    index: BLEND,
    top_k: int,
    tasks: dict,
    query_id: str,
    query_dataset_path: Path,
    opts_read: dict,
    qcr_hash_size: int,
) -> list[dict]:
    df = pl_read_dataset(query_dataset_path, opts_read)

    # the list where execution tasks results will be stored
    candidates = []

    union_tasks = tasks.get("union_tasks", [])
    join_tasks = tasks.get("join_tasks", [])
    join_correlation_tasks = tasks.get("join_correlation_tasks", [])

    print(f"Union tasks: {len(union_tasks)}")
    print(f"Join tasks: {len(join_tasks)}")
    print(f"Join-Correlation tasks: {len(join_correlation_tasks)}")

    for task in tqdm(union_tasks, desc="Union tasks", position=1, leave=False):
        columns = task["columns"]
        table = df.select(columns).rows()

        try:
            top_res = _execute_union_search(index, table, top_k)
            for candidate in top_res:
                cand_id = candidate[0]
                if cand_id != query_id:
                    candidates.append(
                        {
                            "Q": query_id,
                            "R": cand_id,
                            "task": "U",
                            "q_columns": columns,
                        }
                    )
        except (TimeoutError, RuntimeError):
            print(f"Timeout on Union: {query_id}")

    for task in tqdm(join_tasks, desc="Join tasks", position=1, leave=False):
        columns = task["columns"]
        if len(columns) == 1:
            column = df.get_column(columns[0]).to_list()
            try:
                top_res = _execute_single_join_search(index, column, top_k)
                for cand_id, r_join_key, _ in top_res:
                    if cand_id != query_id:
                        candidates.append(
                            {
                                "Q": query_id,
                                "R": cand_id,
                                "task": "J",
                                "q_join_key": columns[0],
                                "r_join_key_pos": r_join_key,
                            }
                        )
            except (TimeoutError, RuntimeError):
                print(f"Timeout on Single-Join: {query_id}")
        else:
            table = df.select(columns).unique().rows()
            try:
                top_res = _execute_multi_join_search(index, table, top_k)
                for cand_id, r_join_keys, _ in top_res:
                    if cand_id != query_id:
                        candidates.append(
                            {
                                "Q": query_id,
                                "R": cand_id,
                                "task": "J",
                                "q_join_keys": columns,
                                "r_join_keys_pos": r_join_keys,
                            }
                        )
            except (TimeoutError, RuntimeError):
                print(f"Timeout on Multi-Join: {query_id}")

    for task in tqdm(
        join_correlation_tasks, desc="Join-Correlation tasks", position=1, leave=False
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

        try:
            top_res = _execute_correlation_search(
                index, keys, targets, top_k, qcr_hash_size
            )

            for cand_id, r_key_column, r_target_column, score in top_res:
                if cand_id != query_id:
                    candidates.append(
                        {
                            "Q": query_id,
                            "R": cand_id,
                            "task": "JC",
                            "q_key": key_column,
                            "q_target": target_column,
                            "r_key": r_key_column,
                            "r_target": r_target_column,
                        }
                    )
        except (TimeoutError, RuntimeError):
            print(f"Timeout on Join-Correlation: {query_id}")
    return candidates


def pipeline(cfg: OrQAConfig):
    seed_datasets = get_seed_datasets(cfg)

    # load metadata for seed datasets
    # FIX: could not be better to load all metadata in memory directly?
    # or store them in document-oriented-like db?
    datasets_metadata = load_datasets_metadata(
        cfg.metadata_path.joinpath("metadata.json"),
        [s[3 if len(s) == 4 else 2] for s in seed_datasets],
    )

    # setup the Agent
    # In this case, we call this entity "agent", even if
    # it is just a wrapper of a LLM client, without any
    # needs of tool-calling or memory or other properties
    print("Loading LLM-agent")
    litellm_config_path = cfg.llm_config_path.joinpath("litellm.yaml")
    agent = CandidatesDiscoveryAgent(litellm_config_path)

    tokens_budget = 1_000_000
    n_datasets_limit = 100

    q = seed_datasets

    _format = cfg.datasets_format

    # instantiate the BLEND index
    index = BLEND(
        cfg.indexing.index_database_path,
        clean_args=cfg.indexing.clean_args,
        xash_size=cfg.indexing.xash_size,
    )

    top_k = cfg.candidates_discovery.top_k_results_per_task

    # instantiate the matches graph
    G = matches_graph.DatasetMatchesGraph()

    while q:
        if tokens_budget <= 0 or n_datasets_limit <= 0:
            break
        # get the first item in the queue
        dataset_id, dataset_path, resource_name, resource_id = q.pop(0)

        # get its metadata
        metadata = datasets_metadata[resource_id]

        # Propose tasks for this dataset
        tasks_completion = generate_tasks(cfg, dataset_path, metadata, agent)
        if not tasks_completion:
            continue

        total_tokens = tasks_completion["token_usage"]["total_tokens"]
        tasks = tasks_completion["tasks"]
        tokens_budget -= total_tokens
        n_datasets_limit -= 1

        print("\n" + " BUDGET UPDATE ".center(100, "="))
        print(f" Remaining tokens: {tokens_budget} ".center(100, "-"))
        print(f" Remaining datasets: {n_datasets_limit} ".center(100, "-"))
        print("=" * 100 + "\n")

        # wait some time (for remote groq calls...)
        time.sleep(30)
        save_list_to_jsonlines(cfg.candidates_discovery.proposed_tasks_path, [tasks])

        # execute the tasks
        print("\n" + " EXECUTING TASKS ".center(100, "="))
        candidates = execute_tasks(
            index,
            top_k,
            tasks,
            dataset_id,
            dataset_path,
            cfg.polars_opts.read,
            cfg.candidates_discovery.qcr_hash_size,
        )
        print("\n" + " DONE ".center(100, "="))

        # TODO: tag or filter with metadata from Valentine Schema matching:
        #   - Joinable pairs with (almost-)identical schema should not be joined;
        #   - Unionable pairs should have a very similar schema instead;
        save_list_to_jsonlines(cfg.candidates_discovery.tasks_results_path, candidates)

        # Add the identified matches to the queue of datasets that have to
        # be analysed
        for candidate in candidates:
            q.append(
                (  # ty: ignore
                    candidate["R"],
                    cfg.datasets_path / f"{candidate['R']}.{cfg.datasets_format}",
                    *str(candidate["R"]).split(SEP),
                )
            )

        # Once we have executed the tasks, we can add the identified
        # matches to the graph
        print("\n" + " EXTENDING GRAPH ".center(100, "="))
        G.add(candidates, cfg.datasets_path, cfg.polars_opts.read, verbose=False)
        print("\n" + " DONE ".center(100, "="))
        G.save(cfg.candidates_discovery.matches_graph_path)

    G.save(cfg.candidates_discovery.matches_graph_path)


def save_list_to_jsonlines(path: Path, objects: list):
    with open(path, "a") as file:
        file.writelines([json.dumps(o) + "\n" for o in objects])


def get_seed_datasets(cfg: OrQAConfig) -> list[tuple[str, Path, str, str]]:
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
    pipeline(cfg)
    # with open(cfg.candidates_discovery.tasks_results_path, "r") as file:
    #     candidates = [json.loads(line) for line in file.readlines()]
    #
    # G = DatasetMatchesGraph()
    # G.add(candidates, cfg.datasets_path, cfg.polars_opts.read, verbose=False)
    # G.save(cfg.candidates_discovery.matches_graph_path)


def candidates_discovery_old(cfg: OrQAConfig):
    # TODO: export these as configuration options
    GENERATE_TASKS = False
    EXECUTE_TASKS = False
    BUILD_GRAPH = True
    EXPLORE_GRAPH = True

    # TODO: consider as seeds only valid datasets
    # (i.e. datasets with at least N rows in general)
    seed_datasets = get_seed_datasets(cfg)

    if GENERATE_TASKS:
        generate_tasks(cfg, seed_datasets)
    with open(cfg.candidates_discovery.proposed_tasks_path, "r") as file:
        generated_tasks = json.load(file)

    memory_limit_half()

    if EXECUTE_TASKS:
        discovered_candidates = execute_tasks(cfg, generated_tasks)
    else:
        try:
            with open(
                cfg.candidates_discovery.tasks_results_path,
                "r",
            ) as file:
                discovered_candidates = json.load(file)

        except FileNotFoundError:
            print(
                f"File {cfg.candidates_discovery.tasks_results_path} not found: have you executed tasks yet?"
            )
            return

    if BUILD_GRAPH:
        G = build_matches_graph(cfg, discovered_candidates)
    else:
        try:
            G = nx.read_gml(cfg.candidates_discovery.matches_graph_path)
        except FileNotFoundError:
            print(
                f"File {cfg.candidates_discovery.matches_graph_path} not found: have you generated the graph yet?"
            )
            return

    if EXPLORE_GRAPH:
        explore_matches_graph(cfg, G, seed_datasets)
