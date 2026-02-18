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

import json
import os
import random
import resource
import time
from functools import lru_cache
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
from .schema_matching.valentine_matcher import instantiate_matcher, schema_matching
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
COMPLETION_CALLS_TIMEOUT = 1


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
    x = x.strip().replace(",", "").replace("£", "").lower()
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


@timeout(90)
def _execute_multi_join_search(index: BLEND, table, k):
    return index.multi_column_join_search(table, k, verbose=True)


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
                                "q_columns": [columns[0]],
                                "r_columns": [r_join_key],
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
                                "q_columns": columns,
                                "r_columns": r_join_keys,
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
            casting_exprs = [
                pl.col(target_column),
                pl.col(target_column)
                .str.strip_chars()
                .str.replace_all(r"[£,]", "", literal=False),
            ]
            for expr in casting_exprs:
                try:
                    targets = (
                        df.lazy()  # ty: ignore
                        .with_columns(expr)
                        .collect()
                        .get_column(target_column)
                        .cast(pl.Float32)
                    )
                except pl.exceptions.InvalidOperationError as e:
                    print(str(e).replace("\n\n", "\n"))
                else:
                    is_float = True
                    break

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


@lru_cache(64)
def _load_dataframe(path: Path, opts: str, seed: int) -> pl.DataFrame:
    df = pl_read_dataset(path, json.loads(opts))
    return df.sample(min(30, df.height), seed=seed)


def load_dataframe(path: Path, opts: dict, seed: int) -> pl.DataFrame:
    return _load_dataframe(path, json.dumps(opts), seed)


def evaluate_matches(
    cfg: OrQAConfig,
    blend_matches: list[dict],
):
    matcher_name = "coma"

    matcher = instantiate_matcher(matcher_name, use_instances=False)

    for blend_match in tqdm(blend_matches, desc="Scanning BLEND matches"):
        Q_name = blend_match["Q"]
        R_name = blend_match["R"]
        task = blend_match["task"]

        Q = load_dataframe(
            cfg.datasets_path / f"{Q_name}.csv", cfg.polars_opts.read, cfg.seed
        )
        R = load_dataframe(
            cfg.datasets_path / f"{R_name}.csv", cfg.polars_opts.read, cfg.seed
        )

        q_columns = r_columns = None
        q_key = r_key = None
        q_target = r_target = None

        match task:
            case "U":
                q_columns = blend_match["q_columns"]
            case "J" | "MJ":
                q_columns = blend_match["q_join_keys"]

                r_columns_pos = blend_match["r_join_keys_pos"]
                r_columns = [R.columns[i] for i in r_columns_pos]
            case "JC":
                q_key = blend_match["q_key"]
                q_target = blend_match["q_target"]

                r_key_pos = blend_match["r_key"]
                r_target_pos = blend_match["r_target"]
                r_key = R.columns[r_key_pos]
                r_target = R.columns[r_target_pos]

        try:
            match_t = time.time()
            matches, global_avg, spec_avg = schema_matching(
                matcher,
                task,
                Q.to_pandas(),
                R.to_pandas(),
                q_columns,  # ty: ignore
                r_columns,
                q_key,
                r_key,
                q_target,
                r_target,
            )
            match_t = time.time() - match_t
        except Exception as exc:
            print(f"Error with Q={Q_name}, R={R_name}: {exc}")
            blend_match["sm_global_avg"] = None
            blend_match["sm_spec_avg"] = None
            blend_match["#matches"] = None
        else:
            blend_match["sm_global_avg"] = global_avg
            blend_match["sm_spec_avg"] = spec_avg
            blend_match["#matches"] = len(matches)
            blend_match["time(s)"] = match_t
        finally:
            blend_match["#Q_schema"] = len(Q.columns)
            blend_match["#R_schema"] = len(R.columns)
            blend_match["#Q_req_columns"] = len(q_columns) if q_columns else None
            blend_match["#R_req_columns"] = len(r_columns) if r_columns else None


def pipeline(cfg: OrQAConfig):
    seed_datasets = get_seed_datasets(cfg)

    # save seed datasets
    with open(cfg.candidates_discovery.seeds_datasets_path, "w") as file:
        json.dump([d[0] for d in seed_datasets], file)

    # load metadata for seed datasets
    # FIX: could not be better to load all metadata in memory directly?
    # or store them in document-oriented-like db?
    datasets_metadata = load_datasets_metadata(
        cfg.metadata_path.joinpath("metadata.json"),
        None,  # [s[3 if len(s) == 4 else 2] for s in seed_datasets],
        source=cfg.source,
    )

    # setup the Agent
    # In this case, we call this entity "agent", even if
    # it is just a wrapper of a LLM client, without any
    # needs of tool-calling or memory or other properties
    print("Loading LLM-agent")
    litellm_config_path = cfg.llm_config_path.joinpath("litellm.yaml")
    agent = CandidatesDiscoveryAgent(litellm_config_path)

    tokens_budget = 1_000_000
    n_datasets_limit = 500

    _format = cfg.datasets_format

    # instantiate the BLEND index
    index = BLEND(
        cfg.indexing.index_database_path,
        clean_args=cfg.blend_opts.clean_args,
        xash_size=cfg.blend_opts.xash_size,
        max_cell_length=cfg.blend_opts.max_cell_length,
    )

    top_k = cfg.candidates_discovery.top_k_results_per_task

    # instantiate the matches graph
    G = matches_graph.DatasetMatchesGraph()

    # keep track of already analyzed datasets
    visited = set()

    # Place the candidates inside q. Once we have analyzed all the
    # initial seeds, switch to the candidates.
    q = set()

    while seed_datasets or q:
        # first pop the seeds, then switch to the candidates
        if len(seed_datasets) > 0:
            dataset_id, dataset_path, resource_name, resource_id = seed_datasets.pop()
        else:
            dataset_id, dataset_path, resource_name, resource_id = q.pop()

        # avoid duplicated analyses
        if resource_id in visited:
            continue

        # if we run out of budget, stop generation
        if tokens_budget <= 0 or n_datasets_limit <= 0:
            break

        print("\n" + f" DATASET {dataset_id} ".center(100, "=") + "\n")

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

        if isinstance(tasks, str):
            continue

        print("\n" + " BUDGET UPDATE ".center(100, "="))
        print(f" Remaining tokens: {tokens_budget} ".center(100, "-"))
        print(f" Remaining datasets: {n_datasets_limit} ".center(100, "-"))
        print("=" * 100 + "\n")

        # wait some time (for remote groq calls...)
        time.sleep(COMPLETION_CALLS_TIMEOUT)
        save_list_to_jsonlines(
            cfg.candidates_discovery.proposed_tasks_path, [tasks_completion]
        )

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

        # Add the identified matches to the queue of datasets that have to be analysed
        for candidate in candidates:
            filename = str(candidate["R"])
            q.add(
                (
                    candidate["R"],
                    cfg.datasets_path / f"{candidate['R']}.{cfg.datasets_format}",
                    *(filename.split(SEP) if SEP in filename else ("", filename)),
                )
            )

        # Once we have executed the tasks, we can add the identified
        # matches to the graph
        print("\n" + " EXTENDING GRAPH ".center(100, "="))
        G.add(
            candidates,
            cfg.datasets_path,
            cfg.polars_opts.read,
            "coma",
            {"use_instances": False},
            verbose=False,
        )
        print("\n" + " DONE ".center(100, "="))
        G.save(cfg.candidates_discovery.matches_graph_path)

        save_list_to_jsonlines(cfg.candidates_discovery.tasks_results_path, candidates)

    G.save(cfg.candidates_discovery.matches_graph_path)


def save_list_to_jsonlines(path: Path, objects: list):
    with open(path, "a") as file:
        file.writelines([json.dumps(o) + "\n" for o in objects])


def get_seed_datasets(cfg: OrQAConfig) -> list[tuple[str, Path, str, str]]:
    # sample dataset seeds for the candidates discovery step
    if False and cfg.candidates_discovery.seeds_datasets_path.exists():
        with open(cfg.candidates_discovery.seeds_datasets_path) as file:
            sample = json.load(file)
    else:
        sample = sample_seed_datasets(
            cfg.datasets_path, cfg.candidates_discovery.n_random_dataset_seeds, cfg.seed
        )

    return sample


def generate_random_walks(cfg: OrQAConfig):
    graph_gml_path = cfg.candidates_discovery.matches_graph_path
    random_walks_path = cfg.candidates_discovery.candidates_path

    G = matches_graph.DatasetMatchesGraph()
    G.load(graph_gml_path)

    seed_datasets = get_seed_datasets(cfg)
    random_walks = {}

    for dataset_id, dataset_path, resource_id, resource_name in tqdm(
        seed_datasets, desc="Generating random walks"
    ):
        if dataset_id not in G._G:
            continue

        random_walks[dataset_id] = []

        for path_length in range(1, cfg.candidates_discovery.max_path_length + 1):
            random_walks[dataset_id].append(
                {
                    "path_length": path_length,
                    "paths": G.generate_random_walks(
                        dataset_id,
                        cfg.candidates_discovery.n_paths_for_dataset,
                        path_length,
                        None,
                        cfg.seed,
                    ),
                }
            )

    with open(random_walks_path, "w") as file:
        json.dump(random_walks, file, indent=4)


def candidates_discovery(cfg: OrQAConfig):
    pipeline(cfg)

    # generate_random_walks(cfg)

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
