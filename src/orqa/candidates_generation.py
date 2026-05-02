"""
Candidates Discovery Stage

In this stage, candidates for actual dataset generation in next steps
are proposed through an agentic step.

The workflow changes now:
    For each dataset seed:
        1. Propose tasks
        2. Execute tasks
        3. Integrate results into neighbors graph
        4. Go to 1
"""

import faulthandler
import json
import os
import random
import time
import ctypes
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None

import dotenv
import polars as pl
from blend import BLEND
from tqdm import tqdm
from wrapt_timeout_decorator import timeout

from conf import OrQAConfig

from .agent.agent import CandidatesDiscoveryAgent
from .graph import matches_graph
from .utils import (
    load_normalized_datasets_metadata,
    pl_read_dataset,
    remove_file_extension,
)

SEP = "__"
COMPLETION_CALLS_TIMEOUT = 1
PRINT_PAD = 120


## (Temporary solution) Added a check for illegal table names in duck db enviroment
def has_unsafe_sql_chars(dataset_id: str) -> bool:
    """Skip datasets whose IDs would break SQL identifiers or string literals."""
    import re
    # Reject names with quotes, semicolons, parentheses, or other SQL-unsafe chars
    if re.search(r"""['"`;()\[\]{}\\@#$%^&*!~<>|/]""", dataset_id):
        return True
    # Reject names that start with a digit (invalid SQL identifier)
    if dataset_id and dataset_id[0].isdigit():
        return True
    return False

# NOTE: this implementation of BLEND is currently based on DuckDB, and
# in some cases it needs to fetch a lot of data. We try to limit the memory
# usage in order to avoid a machine complete block (maybe works maybe not,
# no time for complete tests)
def memory_limit_half():
    """Limit max memory usage to half."""
    if resource is None:
        raise OSError("resource module is not available on this platform")

    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    # Convert KiB to bytes, and divide in two to half
    resource.setrlimit(resource.RLIMIT_AS, (int(get_memory() * 1024 / 2), hard))


def get_memory():
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("Could not read available physical memory on Windows")
        return status.ullAvailPhys // 1024

    if not os.path.exists("/proc/meminfo"):
        try:
            return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")) // 1024
        except (AttributeError, OSError, ValueError) as e:
            raise FileNotFoundError(
                "/proc/meminfo is not available on this platform"
            ) from e

    with open("/proc/meminfo", "r") as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ("MemFree:", "Buffers:", "Cached:"):
                free_memory += int(sline[1])
    return free_memory  # KiB


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
    sample = [(remove_file_extension(f), datasets_path / f) for f in sample]

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


@timeout(60)
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
        try:
            columns = task["columns"]
            top_res = _execute_union_search(index, df, top_k)
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
        except TimeoutError as e:
            print(f"Timeout on Union for {query_id}: {e}")
        except (RuntimeError, MemoryError) as e:
            print(f"RuntimeError or MemoryError on Union for {query_id}: {e}")

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
            except TimeoutError as e:
                print(f"Timeout on Join for {query_id}: {e}")
            except (RuntimeError, MemoryError) as e:
                print(f"RuntimeError or MemoryError on Join for {query_id}: {e}")
        else:
            try:
                top_res = _execute_multi_join_search(index, df.select(columns), top_k)
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
            except TimeoutError as e:
                print(f"Timeout on Multi-Join for {query_id}: {e}")
            except (RuntimeError, MemoryError) as e:
                print(f"RuntimeError or MemoryError on Multi-Join for {query_id}: {e}")

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
                pl.col(target_column).cast(pl.Float32),
                pl.col(target_column)
                .str.strip_chars()
                .str.replace_all(r"[£,*]", "", literal=False)
                .cast(pl.Float32),
                pl.col(target_column).cast(pl.Float32, strict=False),
            ]
            for i, expr in enumerate(casting_exprs):
                try:
                    targets = (
                        df.lazy()  # ty: ignore
                        .with_columns(expr)
                        .select(target_column)
                        .collect()
                        .get_column(target_column)
                    )
                except pl.exceptions.InvalidOperationError as e:
                    error_message = str(e).replace("\n\n", "\n")
                    print(
                        f"Cast failed for JC attempt {i + 1}/{len(casting_exprs)}: "
                        f"{error_message}"
                    )
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
        except TimeoutError as e:
            print(f"Timeout on Join-Correlation for {query_id}: {e}")
        except (RuntimeError, MemoryError) as e:
            print(f"RuntimeError or MemoryError on Join-Correlation for {query_id}: {e}")

    return candidates


def is_resumable(cfg: OrQAConfig) -> bool:
    """Check if there are intermediate result files to resume from."""
    p = cfg.candidates_discovery.tasks_results_path
    return p.exists() and p.stat().st_size > 0


def load_recovery_state(
    cfg: OrQAConfig,
) -> tuple[
    matches_graph.DatasetMatchesGraph,
    set,   # visited (resource_ids already fully processed)
    set,   # recovered_q (candidate tuples discovered but not yet processed)
    int,   # tokens_spent
    int,   # datasets_done (for n_datasets_limit accounting)
]:
    """
    Rebuilds pipeline state from the intermediate files:
      - tasks_results   → which Q datasets were executed, which R datasets were found
      - proposed_tasks  → how many tokens were spent (only for executed datasets)

    Datasets present in proposed_tasks but absent from tasks_results are considered
    NOT done: their cached proposal is ignored and the agent will re-propose them.
    """
    tasks_results_path  = cfg.candidates_discovery.tasks_results_path
    proposed_tasks_path = cfg.candidates_discovery.proposed_tasks_path

    # ── 1. Load task results ────────────────────────────────────────────────
    all_candidates: list[dict] = []
    executed_q_ids: set[str]   = set()

    if tasks_results_path.exists():
        with open(tasks_results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                all_candidates.append(c)
                executed_q_ids.add(c["Q"])

    # ── 2. Count tokens only for datasets that were actually executed ───────
    #    (proposed-but-not-executed entries are intentionally excluded so their
    #     budget is not pre-consumed; they will be re-proposed fresh)
    tokens_spent = 0
    if proposed_tasks_path.exists():
        with open(proposed_tasks_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry["dataset"] in executed_q_ids:
                    tokens_spent += entry["token_usage"]["total_tokens"]

    # ── 3. Rebuild graph from scratch (don't trust the possibly-corrupt file) 
    G = matches_graph.DatasetMatchesGraph()
    if all_candidates:
        print("Rebuilding graph from task results...")
        G.add(
            all_candidates,
            cfg.datasets_path,
            cfg.polars_opts.read,
            "coma",
            {"use_instances": False},
            verbose=False,
        )
        G.save(cfg.candidates_discovery.matches_graph_path)
        print(f"Graph rebuilt: {len(executed_q_ids)} Q-nodes, {len(all_candidates)} edges.")

    # ── 4. Rebuild the candidate queue ─────────────────────────────────────
    #    R datasets that were discovered but never used as a query yet
    all_r_ids     = {c["R"] for c in all_candidates}
    unvisited_r   = all_r_ids - executed_q_ids

    recovered_q: set[tuple] = set()
    for r_id in unvisited_r:
        filepath = cfg.datasets_path / f"{r_id}.{cfg.datasets_format}"
        parts    = r_id.split(SEP) if SEP in r_id else ("", r_id)
        recovered_q.add((r_id, filepath, *parts))

    return G, executed_q_ids, recovered_q, tokens_spent, len(executed_q_ids)


def pipeline(cfg: OrQAConfig):
    if resource is None or not os.path.exists("/proc/meminfo"):
        print(
            "Skipping memory limit setup because the required OS utilities are "
            "not available on this platform."
        )
    else:
        try:
            memory_limit_half()
        except (AttributeError, FileNotFoundError, OSError, ValueError) as e:
            print(f"Skipping memory limit setup: {e}")

    if hasattr(faulthandler, "enable"):
        try:
            faulthandler.enable()
        except (AttributeError, OSError, RuntimeError) as e:
            print(f"Could not enable faulthandler: {e}")

    time_stat_records = []
    seed_datasets = get_seed_datasets(cfg)

    # save seed datasets
    with open(cfg.candidates_discovery.seeds_datasets_path, "w") as file:
        json.dump([d[0] for d in seed_datasets], file)

    # load metadata for seed datasets
    # FIX: could not be better to load all metadata in memory directly?
    # or store them in document-oriented-like db?
    datasets_metadata = load_normalized_datasets_metadata(
        cfg.normalized_metadata_filepath
    )

    # setup the Agent
    # In this case, we call this entity "agent", even if
    # it is just a wrapper of a LLM client, without any
    # needs of tool-calling or memory or other properties
    print("Loading LLM-agent")
    litellm_config_path = cfg.llm_config_path / "litellm.yaml"
    agent = CandidatesDiscoveryAgent(litellm_config_path)

    tokens_budget = 1_000_000
    n_datasets_limit = 400#1000

    _format = cfg.datasets_format

    # instantiate the BLEND index
    index = BLEND(
        cfg.indexing.index_database_path,
        clean_args=cfg.blend_opts.clean_args,
        xash_size=cfg.blend_opts.xash_size,
        max_cell_length=cfg.blend_opts.max_cell_length,
    )

    top_k = cfg.candidates_discovery.top_k_results_per_task


    # ── State initialisation (fresh or resumed) ────────────────────────────

    G       = matches_graph.DatasetMatchesGraph()
    visited = set()
    q       = set()

    #if is_resumable(cfg):                                               # ← NEW
    #    print("Resuming from intermediate state...")                    # ← NEW
    #    G, visited, q, tokens_spent, datasets_done = load_recovery_state(cfg)   # ← NEW 
    #    tokens_budget    -= tokens_spent                                # ← NEW
    #    n_datasets_limit -= datasets_done                               # ← NEW
    #    print(                                                          # ← NEW
    #        f"  visited={len(visited)}, queued={len(q)}, "             # ← NEW
    #        f"budget left={tokens_budget}, datasets left={n_datasets_limit}" # ← NEW
    #    )                                                               # ← NEW
    # ───────────────────────────────────────────────────────────────────────

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

        # we skip (temporary)
        if has_unsafe_sql_chars(dataset_id):
            print(f"Skipping dataset with unsafe SQL chars in name: {dataset_id}")
            visited.add(resource_id)
            continue
        # if we run out of budget, stop generation
        if tokens_budget <= 0 or n_datasets_limit <= 0:
            break

        try:
            print("\n" + f" DATASET {dataset_id} ".center(PRINT_PAD, "=") + "\n")

            # get its metadata
            metadata = datasets_metadata.get(resource_id)
            if metadata is None:
                print(
                    f"Metadata not found for resource_id={resource_id} "
                    f"in {cfg.normalized_metadata_filepath}"
                )
                continue

            # Propose tasks for this dataset
            g_t = time.time()
            tasks_completion = generate_tasks(cfg, dataset_path, metadata, agent)
            if not tasks_completion:
                continue
            g_t = time.time() - g_t

            total_tokens = tasks_completion["token_usage"]["total_tokens"]
            tasks = tasks_completion["tasks"]
            tokens_budget -= total_tokens
            n_datasets_limit -= 1

            if isinstance(tasks, str):
                continue

            print("\n" + " BUDGET UPDATE ".center(PRINT_PAD, "="))
            print(f" Remaining tokens: {tokens_budget} ".center(PRINT_PAD, "-"))
            print(f" Remaining datasets: {n_datasets_limit} ".center(PRINT_PAD, "-"))
            print("=" * 100 + "\n")

            # wait some time (for remote groq calls...)
            time.sleep(COMPLETION_CALLS_TIMEOUT)
            save_list_to_jsonlines(
                cfg.candidates_discovery.proposed_tasks_path, [tasks_completion]
            )

            # execute the tasks
            print("\n" + " EXECUTING TASKS ".center(PRINT_PAD, "="))
            e_t = time.time()
            candidates = execute_tasks(
                index,
                top_k,
                tasks,
                dataset_id,
                dataset_path,
                cfg.polars_opts.read,
                cfg.candidates_discovery.qcr_hash_size,
            )
            e_t = time.time() - e_t
            print("\n" + " DONE ".center(PRINT_PAD, "="))

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
            print("\n" + " EXTENDING GRAPH ".center(PRINT_PAD, "="))
            a_t = time.time()
            G.add(
                candidates,
                cfg.datasets_path,
                cfg.polars_opts.read,
                "coma",
                {"use_instances": False},
                verbose=False,
            )
            print("\n" + " DONE ".center(PRINT_PAD, "="))
            a_t = time.time()
            G.save(cfg.candidates_discovery.matches_graph_path)

            save_list_to_jsonlines(
                cfg.candidates_discovery.tasks_results_path, candidates
            )

            time_stat_records.append(
                {
                    "Q": dataset_id,
                    "tasks_proposal_time": g_t,
                    "tasks_execution_time": e_t,
                    "graph_increment_time": a_t,
                }
            )
            save_time_statistics(
                time_stat_records, cfg.statistics_path / "generation_time_stats.csv"
            )
            visited.add(resource_id) 
        except OSError as e:
            print(f"SegFault (?): {e}")

    G.save(cfg.candidates_discovery.matches_graph_path)


def save_time_statistics(records: list, path: Path):
    if path.exists():
        with open(path, "a",encoding="utf-8") as file:
            pl.DataFrame(records).write_csv(
                file, include_header=False, float_precision=3
            )
    else:
        pl.DataFrame(records).write_csv(path, float_precision=3)


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
                        cfg.candidates_discovery.overlap_ratio_threshold,
                        cfg.candidates_discovery.sm_macro_avg_threshold,
                        cfg.candidates_discovery.sm_micro_avg_threshold,
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
