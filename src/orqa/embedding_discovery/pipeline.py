"""
Candidates Discovery Stage (embeddings + clustering + Valentine).

For each dataset in a BFS frontier seeded with a random sample:
    1. Nominate neighbor datasets via cluster membership over metadata
       embeddings (cosine-KMeans clusters with soft boundary overlap).
    2. Gate each (Q, R) pair with Valentine schema matching.
    3. Ask the LLM to select union/join/join-correlation tasks for the pair.
    4. Verify join-correlation tasks with an actual join + correlation.
    5. Emit candidate dicts (with metrics inline), extend the matches graph
       and enqueue R for its own discovery pass.
"""

import json
import logging
import random
import time
from pathlib import Path

import polars as pl

from conf import OrQAConfig

from ..agent.agent import PairTaskSelectionAgent
from ..graph import matches_graph
from ..utils import (
    load_dataset_info,
    load_normalized_datasets_metadata,
    pl_read_dataset,
    remove_file_extension,
    select_columns,
)
from ..utils.pipeline_logger import PipelineLogger
from ..agent.llm_client.EmbeddingClient import EmbeddingClient
from .clustering import ClusterNeighborIndex, compute_cluster_projection, save_cluster_projection
from .embeddings import (
    SEP,
    EmbeddingCache,
    build_embedding_text,
    dataset_id_to_resource_id,
    load_raw_normalized_metadata,
)
from .verification import (
    check_join_correlation,
    passes_schema_gate,
    verify_pair_schema,
)

PRINT_PAD = 120

# Row/column caps for the pandas frames handed to Valentine: coma is
# O(cols²) and schema-oriented, so a slim slice is enough evidence.
VALENTINE_MAX_ROWS = 200


def sample_seed_datasets(
    datasets_path: Path, n_datasets_to_sample: int, seed: int
) -> list[tuple[str, Path, str, str]]:
    datasets = [f for f in sorted(datasets_path.iterdir()) if f.is_file()]

    assert len(datasets) >= n_datasets_to_sample, (
        f"Too many datasets to sample: {len(datasets)} < {n_datasets_to_sample}"
    )

    random.seed(seed)
    sample = random.sample(datasets, n_datasets_to_sample)
    return [_queue_entry(remove_file_extension(f.name), f) for f in sample]


def _queue_entry(dataset_id: str, filepath: Path) -> tuple[str, Path, str, str]:
    parts = dataset_id.split(SEP) if SEP in dataset_id else ("", dataset_id)
    return (dataset_id, filepath, *parts)  # ty: ignore


def get_seed_datasets(cfg: OrQAConfig) -> list[tuple[str, Path, str, str]]:
    return sample_seed_datasets(
        cfg.datasets_path, cfg.candidates_discovery.n_random_dataset_seeds, cfg.seed
    )


def save_list_to_jsonlines(path: Path, objects: list):
    with open(path, "a") as file:
        file.writelines([json.dumps(o) + "\n" for o in objects])


def save_time_statistics(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path, "a", encoding="utf-8") as file:
            pl.DataFrame(records).write_csv(
                file, include_header=False, float_precision=3
            )
    else:
        pl.DataFrame(records).write_csv(path, float_precision=3)


def build_embedding_texts(cfg: OrQAConfig) -> dict[str, str]:
    """One embedding document per dataset on disk that has metadata."""
    raw_metadata = load_raw_normalized_metadata(cfg.normalized_metadata_filepath)
    scan_opts = cfg.polars_opts.scan

    texts: dict[str, str] = {}
    skipped = 0
    for filepath in sorted(cfg.datasets_path.iterdir()):
        if not filepath.is_file():
            continue
        dataset_id = remove_file_extension(filepath.name)
        record = raw_metadata.get(dataset_id_to_resource_id(dataset_id))
        if record is None:
            skipped += 1
            continue
        texts[dataset_id] = build_embedding_text(
            record,
            dataset_path=filepath,
            scan_opts=scan_opts,
            max_chars=cfg.candidates_discovery.embedding_text_max_chars,
        )

    if skipped:
        print(f"Skipped {skipped} datasets without metadata records.")
    return texts


def is_resumable(cfg: OrQAConfig) -> bool:
    """Check if there are intermediate result files to resume from."""
    p = cfg.candidates_discovery.tasks_results_path
    return p.exists() and p.stat().st_size > 0


def load_recovery_state(
    cfg: OrQAConfig,
) -> tuple[
    matches_graph.DatasetMatchesGraph,
    set,        # visited (resource_ids already fully processed)
    set,        # recovered queue entries (discovered but not yet processed)
    set,        # seen_pairs (unordered pairs already verified)
    int,        # tokens_spent
    int,        # datasets_done
]:
    """
    Rebuilds pipeline state from the intermediate files:
      - tasks_results   → which Q datasets were executed, which R were found
      - proposed_tasks  → how many tokens were spent (executed datasets only)

    Candidates now carry their metrics inline, so the graph rebuild is a
    cheap loop with no Valentine replay.
    """
    tasks_results_path = cfg.candidates_discovery.tasks_results_path
    proposed_tasks_path = cfg.candidates_discovery.proposed_tasks_path

    all_candidates: list[dict] = []
    executed_q_ids: set[str] = set()

    if tasks_results_path.exists():
        with open(tasks_results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                all_candidates.append(c)
                executed_q_ids.add(c["Q"])

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

    G = matches_graph.DatasetMatchesGraph()
    if all_candidates:
        print("Rebuilding graph from task results...")
        G.add_precomputed(all_candidates)
        G.save(cfg.candidates_discovery.matches_graph_path)
        print(
            f"Graph rebuilt: {len(executed_q_ids)} Q-nodes, "
            f"{len(all_candidates)} edges."
        )

    seen_pairs = {frozenset((c["Q"], c["R"])) for c in all_candidates}

    unvisited_r = {c["R"] for c in all_candidates} - executed_q_ids
    recovered_q = {
        _queue_entry(r_id, cfg.datasets_path / f"{r_id}.{cfg.datasets_format}")
        for r_id in unvisited_r
    }

    visited = {dataset_id_to_resource_id(q_id) for q_id in executed_q_ids}
    return G, visited, recovered_q, seen_pairs, tokens_spent, len(executed_q_ids)


def _materialize_candidates(
    q_id: str,
    r_id: str,
    tasks: dict,
    evidence: dict,
    cosine_sim: float,
    Q_df: pl.DataFrame,
    R_df: pl.DataFrame,
    cd_cfg,
    log: "PipelineLogger | None" = None,
) -> list[dict]:
    """Convert the LLM's task selection into verified candidate dicts."""
    log = log or PipelineLogger()
    base_metrics = {
        "sm_macro_avg": evidence["sm_macro_avg"],
        "sm_micro_avg": evidence["sm_micro_avg"],
        "sm_n_matches": evidence["sm_n_matches"],
        "sm_time": evidence["sm_time"],
        "cosine_sim": round(cosine_sim, 3),
    }
    matches = evidence["matches"]
    candidates = []

    for task in tasks.get("union_tasks", []):
        candidates.append(
            {
                "Q": q_id,
                "R": r_id,
                "task": "U",
                "q_columns": task["q_columns"],
                "r_columns": task["r_columns"],
                "matches": matches,
                "metrics": dict(base_metrics),
            }
        )

    for task in tasks.get("join_tasks", []):
        candidates.append(
            {
                "Q": q_id,
                "R": r_id,
                "task": "J",
                "q_columns": task["q_columns"],
                "r_columns": task["r_columns"],
                "matches": matches,
                "metrics": dict(base_metrics),
            }
        )

    for task in tasks.get("join_correlation_tasks", []):
        correlation = check_join_correlation(
            Q_df,
            R_df,
            task["q_key"],
            task["r_key"],
            task["q_target"],
            task["r_target"],
            threshold=cd_cfg.correlation_threshold,
            method=cd_cfg.correlation_method,
            min_joined_rows=cd_cfg.min_joined_rows_for_correlation,
        )
        if correlation is None:
            log.jc_dropped(q_id, r_id, task["q_target"], task["r_target"])
            continue
        candidates.append(
            {
                "Q": q_id,
                "R": r_id,
                "task": "JC",
                "q_key": task["q_key"],
                "q_target": task["q_target"],
                "r_key": task["r_key"],
                "r_target": task["r_target"],
                "matches": matches,
                "metrics": {
                    **base_metrics,
                    "correlation": round(correlation, 3),
                    "correlation_method": cd_cfg.correlation_method,
                },
            }
        )

    return candidates


def pipeline(cfg: OrQAConfig):
    cd = cfg.candidates_discovery

    # ── Embeddings + cluster neighbor index ─────────────────────────────────
    print(" BUILDING METADATA EMBEDDINGS ".center(PRINT_PAD, "="))
    texts = build_embedding_texts(cfg)
    client = EmbeddingClient(
        cfg.llm_config_path / "litellm.yaml", batch_size=cd.embedding_batch_size
    )
    cache = EmbeddingCache(cd.embeddings_cache_path)
    ids, vectors = cache.get_or_compute(texts, client)

    print(" CLUSTERING DATASETS ".center(PRINT_PAD, "="))
    index = ClusterNeighborIndex(
        ids,
        vectors,
        target_cluster_size=cd.target_cluster_size,
        overlap_margin=cd.cluster_overlap_margin,
        max_cluster_size=cd.max_cluster_size,
        seed=cfg.seed,
    )
    if cd.clusters_path is not None:
        # Persisted purely for the query browser's Stats page (cluster
        # map) — never read back into the discovery pipeline itself, so a
        # failure here must never abort a real discovery run.
        try:
            projection = compute_cluster_projection(
                ids,
                vectors,
                target_cluster_size=cd.target_cluster_size,
                overlap_margin=cd.cluster_overlap_margin,
                max_cluster_size=cd.max_cluster_size,
                seed=cfg.seed,
            )
            save_cluster_projection(cd.clusters_path, projection)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Could not persist cluster projection to %s: %s", cd.clusters_path, exc
            )
    print("=" * PRINT_PAD + "\n")

    # ── Seeds, metadata, agent ─────────────────────────────────────────────
    seed_datasets = get_seed_datasets(cfg)
    with open(cd.seeds_datasets_path, "w") as file:
        json.dump([d[0] for d in seed_datasets], file)

    prompt_metadata = load_normalized_datasets_metadata(
        cfg.normalized_metadata_filepath
    )

    print("Loading LLM-agent")
    agent = PairTaskSelectionAgent(cfg.llm_config_path / "litellm.yaml")

    tokens_budget = cd.tokens_budget
    n_datasets_limit = cd.max_datasets_to_process

    # ── State initialisation (fresh or resumed) ────────────────────────────
    G = matches_graph.DatasetMatchesGraph()
    visited: set[str] = set()
    q: set[tuple] = set()
    seen_pairs: set[frozenset] = set()

    if is_resumable(cfg):
        print("Resuming from intermediate state...")
        G, visited, q, seen_pairs, tokens_spent, datasets_done = (
            load_recovery_state(cfg)
        )
        tokens_budget -= tokens_spent
        n_datasets_limit -= datasets_done
        print(
            f"  visited={len(visited)}, queued={len(q)}, "
            f"budget left={tokens_budget}, datasets left={n_datasets_limit}"
        )

    # Per-dataset caches so a dataset serving many pairs is loaded once.
    info_cache: dict[str, tuple[dict, dict]] = {}

    def dataset_info(dataset_id: str, dataset_path: Path):
        if dataset_id not in info_cache:
            info_cache[dataset_id] = load_dataset_info(
                dataset_path,
                cfg.polars_opts.read,
                cd.limit_to_n_columns,
                cd.sample_size,
                cfg.seed,
            )
        return info_cache[dataset_id]

    time_stat_records = []
    log = PipelineLogger()

    # ── BFS over the discovery frontier ────────────────────────────────────
    while seed_datasets or q:
        if seed_datasets:
            dataset_id, dataset_path, resource_name, resource_id = seed_datasets.pop()
        else:
            dataset_id, dataset_path, resource_name, resource_id = q.pop()

        if resource_id in visited:
            continue
        if tokens_budget <= 0 or n_datasets_limit <= 0:
            break

        q_metadata = prompt_metadata.get(resource_id)
        if q_metadata is None:
            log.warning(f"Metadata not found for resource_id={resource_id}; skipping {dataset_id}.")
            visited.add(resource_id)
            continue

        try:
            q_info, q_typings = dataset_info(dataset_id, dataset_path)
        except Exception as exc:
            log.warning(f"Could not load dataset {dataset_id}: {exc}")
            visited.add(resource_id)
            continue

        if q_info["num_rows"] < cd.min_dataset_height or q_info["num_columns"] == 0:
            log.info(f"Dataset {dataset_id} too small; skipping.")
            visited.add(resource_id)
            continue

        neighbors = index.query_neighbors(
            dataset_id, cd.top_k_neighbors, cd.cosine_similarity_threshold
        )
        log.discovery_dataset_start(dataset_id, len(neighbors))

        n_datasets_limit -= 1
        visited.add(resource_id)
        candidates: list[dict] = []
        nomination_t = time.time()

        try:
            Q_df = pl_read_dataset(dataset_path, cfg.polars_opts.read)
        except Exception as exc:
            log.warning(f"Could not read dataset {dataset_id}: {exc}")
            continue
        Q_valentine = (
            Q_df.select(select_columns(Q_df.columns, cd.limit_to_n_columns))
            .head(VALENTINE_MAX_ROWS)
            .to_pandas()
        )

        for r_id, cosine_sim in neighbors:
            pair = frozenset((dataset_id, r_id))
            if len(pair) < 2 or pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            r_path = cfg.datasets_path / f"{r_id}.{cfg.datasets_format}"
            r_resource_id = dataset_id_to_resource_id(r_id)
            r_metadata = prompt_metadata.get(r_resource_id)
            if r_metadata is None:
                continue

            try:
                r_info, r_typings = dataset_info(r_id, r_path)
            except Exception as exc:
                log.warning(f"Could not load neighbor {r_id}: {exc}")
                continue
            if (
                r_info["num_rows"] < cd.min_dataset_height
                or r_info["num_columns"] == 0
            ):
                continue

            # ── Valentine gate (before any LLM tokens) ────────────────────
            try:
                R_df = pl_read_dataset(r_path, cfg.polars_opts.read)
                evidence = verify_pair_schema(
                    Q_valentine,
                    R_df.select(select_columns(R_df.columns, cd.limit_to_n_columns))
                    .head(VALENTINE_MAX_ROWS)
                    .to_pandas(),
                    cd.sm_matcher,
                    cd.sm_matcher_kwargs,
                )
            except Exception as exc:
                log.warning(f"Schema matching failed for ({dataset_id}, {r_id}): {exc}")
                continue

            if not passes_schema_gate(
                evidence, cd.sm_macro_avg_threshold, cd.sm_micro_avg_threshold
            ):
                log.pair_rejected(
                    dataset_id, r_id, cosine_sim,
                    evidence["sm_macro_avg"], evidence["sm_micro_avg"],
                )
                continue

            # ── LLM column selection ──────────────────────────────────────
            selection = agent.select_tasks(
                q_info,
                q_typings,
                q_metadata,
                r_info,
                r_typings,
                r_metadata,
                evidence["matches"],
                cosine_sim,
            )
            if not selection or not selection.get("tasks"):
                continue

            tokens_budget -= selection["token_usage"]["total_tokens"]
            save_list_to_jsonlines(
                cd.proposed_tasks_path,
                [{"dataset": dataset_id, **selection}],
            )

            pair_candidates = _materialize_candidates(
                dataset_id,
                r_id,
                selection["tasks"],
                evidence,
                cosine_sim,
                Q_df,
                R_df,
                cd,
                log,
            )
            candidates.extend(pair_candidates)
            for c in pair_candidates:
                log.pair_verified(
                    c["Q"], c["R"], c["task"], cosine_sim,
                    evidence["sm_macro_avg"], evidence["sm_micro_avg"],
                )

            if pair_candidates:
                q.add(_queue_entry(r_id, r_path))

            if tokens_budget <= 0:
                log.warning("Token budget exhausted.")
                break

        nomination_t = time.time() - nomination_t
        log.discovery_budget(tokens_budget, n_datasets_limit)

        if not candidates:
            continue

        graph_t = time.time()
        G.add_precomputed(candidates)
        G.save(cd.matches_graph_path)
        graph_t = time.time() - graph_t

        save_list_to_jsonlines(cd.tasks_results_path, candidates)

        time_stat_records.append(
            {
                "Q": dataset_id,
                "pair_processing_time": nomination_t,
                "graph_increment_time": graph_t,
            }
        )
        save_time_statistics(
            time_stat_records,
            cfg.write_statistics_path / "generation_time_stats_semantic.csv",
        )
        time_stat_records = []

    G.save(cd.matches_graph_path)
    print("\n" + " CANDIDATES DISCOVERY COMPLETED ".center(PRINT_PAD, "="))


def candidates_discovery(cfg: OrQAConfig):
    pipeline(cfg)
