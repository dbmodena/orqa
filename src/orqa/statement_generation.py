import asyncio
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from .agent.agent import TableAnalysisAgent
from .agent.agents.StatementAgent import StatementAgent
from .agent.agents.SingleStatementAgent import SingleStatementAgent
from .benchmark.index import load_index
from .benchmark.questions import get_entry, store_entry
from .utils import (
    dataset_index_shape,
    load_normalized_datasets_metadata,
    save_json,
    load_json,
)
from conf import OrQAConfig, JUDGE_MODE_COUNTS
from dataclasses import dataclass, field

_TIMEOUT_SECONDS_PER_5K_TOKENS: int = 15
_TOKENS_PER_BUCKET: int = 5_000


def _compute_timeout(max_tokens: int) -> float:
    """Return the wall-clock timeout (seconds) for a call capped at *max_tokens*.

    Uses ceiling division so that even a 1-token call gets the full first bucket.
    """
    buckets = math.ceil(max_tokens / _TOKENS_PER_BUCKET)
    return buckets * _TIMEOUT_SECONDS_PER_5K_TOKENS


# ── Match formatting: relationship spec -> QueryLink dict ─────────────────────

_LINK_TYPE_MAP = {"merge": "join", "merge_correlation": "join-correlation", "union": "union"}


def _relationship_to_link(spec: dict) -> dict:
    """
    Convert one query_candidates.json relationship spec into a QueryLink-shaped
    dict (``type``/``tables``/``key_columns``/``correlated_columns``/
    ``description`` — see ``structured_outputs.QueryLink``).

    Keeps each pairwise join/union/join-correlation as its OWN link, so
    ``QueryPlanner._build_constraint_links`` preserves them as separate
    building blocks instead of synthesising one description that reads like a
    prescribed chain (Table_1 joined to Table_0, then to Table_2 reads as a
    linear pipeline even though the two relationships are independent).

    For ``join``/``join-correlation`` links, ``key_columns`` is a list of
    ``{table_alias: column_name}`` PAIRS rather than a flat deduplicated list
    — BLEND matches join keys by VALUE overlap, not by name, so the two sides
    of a key are frequently named differently (e.g. ``Table_1.dbn`` <->
    ``Table_0.school_id``), and ``left_on``/``right_on`` are reliably
    same-length, position-aligned lists (verified single/multi-column key
    search). Flattening them into one list loses which column belongs to
    which table, and which two columns are actually the same key.

    For ``union`` links, ``left_cols``/``right_cols`` are the LLM-proposed
    query-side columns vs. whatever the schema matcher found ANYTHING for on
    the candidate side (see ``query_candidates._make_union_match``) — they're
    only a real correspondence when ``column_scores`` is populated (schema
    match cleared the confidence threshold), in which case they're already
    filtered, ranked, and positionally aligned by that function and are
    zipped into pairs here, WITH the confidence score attached (union
    evidence is a schema/name-similarity signal, not a value-overlap
    guarantee like a join key, so the score matters). When ``column_scores``
    is empty, no pair was confident enough — each column is listed
    separately, still tagged with its own table, without claiming a
    cross-table correspondence that isn't actually known.
    """
    left, right = spec.get("left", ""), spec.get("right", "")
    mtype = spec.get("type", "")
    correlated_columns: list[dict] = []
    if mtype == "union":
        left_cols, right_cols = spec.get("left_cols", []), spec.get("right_cols", [])
        scores = spec.get("column_scores") or []
        if scores and len(scores) == len(left_cols) == len(right_cols):
            columns = [
                {left: lc, right: rc, "score": f"{s:.2f}"}
                for lc, rc, s in zip(left_cols, right_cols, scores)
            ]
        else:
            columns = [{left: c} for c in left_cols] + [{right: c} for c in right_cols]
    else:
        left_on, right_on = spec.get("left_on", []), spec.get("right_on", [])
        columns = [{left: lk, right: rk} for lk, rk in zip(left_on, right_on)]
        if mtype == "merge_correlation":
            corr = spec.get("correlation_cols") or {}
            if corr:
                entry = dict(corr)
                # Only populated by the semantic pipeline, which actually
                # computes this at discovery time (query_candidates.py's
                # _make_join_correlation_match) — omitted (not fabricated)
                # for classical/BLEND JC tasks, which never measure it.
                corr_value = spec.get("correlation_value")
                if corr_value is not None:
                    entry["correlation"] = f"{corr_value:.2f}"
                    method = spec.get("correlation_method")
                    if method:
                        entry["correlation_method"] = method
                correlated_columns = [entry]
    return {
        "type": _LINK_TYPE_MAP.get(mtype, "other"),
        "tables": [t for t in (left, right) if t],
        "key_columns": columns,
        "correlated_columns": correlated_columns,
        "description": spec.get("description") or f"{mtype}: {left} - {right}",
    }


def _relationships_to_links(specs: list[dict]) -> list[dict]:
    return [_relationship_to_link(spec) for spec in specs]


# ── Statement generation ──────────────────────────────────────────────────────

def _build_match_inputs(
    match: dict, csv_folder: Path, datasets_metadata: dict, extension: str
) -> tuple[list[Path], dict, list[dict], dict]:
    """
    Unpack a match record into agent inputs.

    Returns
    -------
    dataset_paths, aliases, metadatas, involved_cols
    """
    dataset_paths, metadatas = [], []
    aliases       = {}
    involved_cols = {}
    for alias, dataset in match["aliases"].items():
        path = csv_folder / f"{dataset}.{extension}"
        dataset_paths.append(path)
        aliases[alias]        = dataset
        metadatas.append(datasets_metadata.get(dataset))
        involved_cols[alias]  = match["columns_by_table"].get(alias, [])
    return dataset_paths, aliases, metadatas, involved_cols


def _involved_columns_by_dataset(all_matches: list[dict]) -> dict[str, list[str]]:
    """Union of each dataset's join/union/correlation columns across every
    candidate it appears in.

    ``match["columns_by_table"]`` is keyed by alias (``Table_0`` ...); the
    same real dataset can wear different aliases (and expose different key
    columns) across candidates, so this resolves aliases back to dataset
    names and merges. Feeds the pre-generation table analysis so it forces
    the same involved columns into its budget that per-candidate generation
    does (see ``StatementOrchestrator._run`` / ``utils.select_columns``).
    """
    by_dataset: dict[str, set[str]] = {}
    for match in all_matches:
        columns_by_table = match.get("columns_by_table", {}) or {}
        for alias, dataset in (match.get("aliases", {}) or {}).items():
            by_dataset.setdefault(dataset, set()).update(
                columns_by_table.get(alias, []) or []
            )
    return {dataset: sorted(cols) for dataset, cols in by_dataset.items()}


def _is_single_table_candidate(match: dict) -> bool:
    """Check if a candidate has a single dataset and no cross-table relationships."""
    aliases = match.get("aliases", {})
    if len(aliases) != 1:
        return False
    has_relationships = bool(
        match.get("relationships")
        or match.get("match_specs")       # legacy chain-format files
        or match.get("SQL_matches")
        or match.get("PANDAS_matches")
    )
    return not has_relationships


def _sample_single_table_datasets(
    datasets_path: Path,
    count: int,
    extension: str = "csv",
    seed: int = 0,
    limit_to_n_columns: int | None = None,
    scan_opts: dict | None = None,
) -> list[Path]:
    """
    Return up to ``count`` randomly sampled dataset paths.

    The glob result is sorted and the sampler seeded with the workflow
    seed, so every run (and every sibling workflow sharing the yaml seed)
    selects the same datasets.

    When ``limit_to_n_columns`` is given, a candidate is skipped (and the
    next one drawn instead) when it has no rows, no columns, or MORE than
    ``limit_to_n_columns`` columns — the same up-front "only ever work with
    whole, integral tables" filter the discovery pipeline applies. Without
    this, single-table sampling reads straight off disk and a too-wide
    table would slip past every discovery-side gate into statement
    generation. An unreadable file is treated as unusable and skipped.
    """
    all_files = sorted(datasets_path.glob(f"*.{extension}"))

    if limit_to_n_columns is None:
        return random.Random(seed).sample(all_files, min(count, len(all_files)))

    scan_opts = scan_opts or {}
    shuffled = all_files[:]
    random.Random(seed).shuffle(shuffled)

    picked: list[Path] = []
    for filepath in shuffled:
        if len(picked) >= count:
            break
        try:
            n_columns, has_rows = dataset_index_shape(filepath, scan_opts)
        except Exception:
            continue
        if n_columns == 0 or not has_rows or n_columns > limit_to_n_columns:
            continue
        picked.append(filepath)
    return picked


def _cap_matches(all_matches: list, count: Optional[int], seed: int = 0) -> list:
    """Cap cross-table candidate matches to ``count``, seeded like
    ``_sample_single_table_datasets`` so runs (and sibling workflows sharing
    the yaml seed) pick the same subset. ``None`` (or a count at/above the
    current total, so there is nothing to actually cap) returns
    ``all_matches`` unchanged and in its original order — sampling would
    otherwise reshuffle it via ``random.sample``, needlessly scrambling the
    index-based resume/dedup keys the caller relies on.
    """
    if count is None or count >= len(all_matches):
        return all_matches
    return random.Random(seed).sample(all_matches, count)


def _get_formatted_match(match: dict, kind: str) -> str:
    """
    Legacy string-list fallback for the ``match`` constraint: the flat
    ``{kind}_matches`` prose block from pre-``relationships`` candidate
    files. Only reached when ``_get_match_for_planner`` returns ``None``
    (no structured ``relationships``/``match_specs`` on the match) — modern
    ``query_candidates.json`` always carries ``relationships``, so the
    structured path in ``_get_match_for_planner`` is what actually runs.
    """
    return "\n".join(match.get(f"{kind}_matches", []))


def _get_match_for_planner(match: dict) -> Any:
    """
    Return what's passed as the ``match`` constraint into ``generate_statements``
    (and from there into ``QueryPlanner._build_constraint_links``).

    Prefers the structured per-relationship link list — one ``QueryLink`` per
    verified join/union/join-correlation — over the flattened prose block
    ``_get_formatted_match`` builds for legacy string-only match formats:
    ``_build_constraint_links`` collapses anything that isn't already a
    list/tuple into a single synthesized link, which reads as one prescribed
    chain even when the underlying relationships are independent pairs.
    """
    specs = match.get("relationships") or match.get("match_specs")
    if specs:
        return _relationships_to_links(specs)
    return None


def _already_succeeded(results: dict, kind: str, idx: str) -> bool:
    """Return True if the queries file has a 'success' entry for this id and kind."""
    entry = get_entry(results, kind, idx)
    return bool(entry) and entry.get("_meta", {}).get("status") == "success"


def _already_processed(results: dict, kind: str, idx: str) -> bool:
    """Return True if the queries file already has an entry for this id and kind."""
    return get_entry(results, kind, idx) is not None


def _store_generation(
    results: dict,
    kind: str,
    entry_id: str,
    content: dict,
    aliases: dict,
    generation_time: float,
) -> str:
    """
    Store one candidate's generation output in the shared hierarchy
    (kind -> single_table|multi_table -> query id -> question number),
    with the run-wide fields under "_meta". Returns the entry status.
    """
    result = content["result"]
    queries = result.get("queries") or []
    meta = {
        "model": content["model"].split("/")[-1],
        # A structural pre-generation failure carries its own status (e.g.
        # "insufficient_rows", "unjustifiable_group") set by the agent;
        # otherwise the status is derived from whether queries survived.
        "status": result.get("status") or ("success" if queries else "failure"),
        "tokens": content["token_usage"],
        "tables": aliases,
        "errors": content["errors"],
        "generation_time": generation_time,
        "avg_cols": content["avg_cols"],
    }
    if content.get("proposed_columns") is not None:
        meta["proposed_columns"] = content["proposed_columns"]
    if content.get("query_plan"):
        meta["query_plan"] = content["query_plan"]
    extra = {k: v for k, v in result.items() if k not in ("queries", "status")}
    # "traceable" (the phase-grouped TraceableQuerySet dump, carrying every
    # approved query's execution.reliability) is a SIBLING of "result" on
    # `content` (see JudgementResponseAgent._assemble_result's return shape),
    # not nested inside it — so it never appears in `result.items()` above.
    # Merge it in explicitly, same as proposed_columns/query_plan above,
    # or it silently never reaches the saved JSON.
    if content.get("traceable") is not None:
        extra["traceable"] = content["traceable"]
    if extra:
        meta["result_extra"] = extra
    store_entry(results, kind, entry_id, queries, meta)
    return meta["status"]

def create_statements(
    config_path: Path,
    csv_folder: Path,
    candidates_file: Path,
    output_file: Path,
    kind: str,
    max_cols: int,
    extension: str,
    datasets_metadata: dict | None = None,
    bad_tokens: list | None = None,
    enable_single_table: bool = False,
    single_table_query_count: int = 0,
    languages: list= ["English"],
    seed: int = 0,
    search_index=None,
    keyword_search_top_k_coefficient: float = 5.0,
    multi_table_query_count: Optional[int] = None,
    gate_unretrievable_groups: bool = False,
    retrieval_gate_enabled: bool = True,
    plan_judge_count: Optional[int] = None,
    code_judge_count: Optional[int] = None,
    scan_opts: dict | None = None,
) -> list[dict]:
    bad_tokens = bad_tokens or []

    if datasets_metadata is None:
        datasets_metadata = {}

    all_matches: list = load_json(candidates_file)
    all_matches = _cap_matches(all_matches, multi_table_query_count, seed=seed)
    results = load_json(output_file) if output_file.exists() else {}

    # Table analyses (description + keywords) are cached per table id and
    # model in the candidates-discovery folder, so a table already seen by an
    # earlier candidate/run is fetched from disk instead of re-analysed.
    analysis_cache_path = candidates_file.parent / "table_analysis_cache.json"

    # ── Stage 0: pre-generation table analysis ────────────────────────────
    # Every table this run will touch (all cross-table candidate datasets plus
    # the deterministically sampled single tables — same seed as the loop
    # below, so the samples are identical) is analysed HERE, before any
    # statement generation. The generation agents then read every description/
    # keyword set straight from the cache, fully decoupling table analysis
    # from the statement-generation orchestrator.
    involved_paths = [
        csv_folder / f"{dataset}.{extension}"
        for match in all_matches
        for dataset in match.get("aliases", {}).values()
    ]
    # Per-DATASET union of the join/union/correlation columns across every
    # candidate that table appears in, so table analysis is built from the
    # same columns each candidate's generation forces into its per-table
    # column budget (utils.select_columns), not just the first max_cols.
    involved_columns_by_dataset = _involved_columns_by_dataset(all_matches)
    if enable_single_table and single_table_query_count > 0:
        involved_paths.extend(
            _sample_single_table_datasets(
                csv_folder, single_table_query_count, extension=extension,
                seed=seed, limit_to_n_columns=max_cols, scan_opts=scan_opts,
            )
        )
    TableAnalysisAgent(
        config_path,
        analysis_cache_path,
        languages=languages,
        bad_tokens=bad_tokens,
        max_cols=max_cols,
        seed=seed,
    ).analyze_tables(
        involved_paths, datasets_metadata, involved_columns_by_dataset
    )

    cross_agent = StatementAgent(
        config_path, kind, bad_tokens, languages=languages, seed=seed,
        analysis_cache_path=analysis_cache_path,
        search_index=search_index,
        keyword_search_top_k_coefficient=keyword_search_top_k_coefficient,
        gate_unretrievable_groups=gate_unretrievable_groups,
        retrieval_gate_enabled=retrieval_gate_enabled,
        plan_judge_count=plan_judge_count,
        code_judge_count=code_judge_count,
    )


# ── Single-table generation ───────────────────────────────────────────────
    if enable_single_table and single_table_query_count > 0:
        print(f"\nStarting single-table generation ({single_table_query_count} datasets)...")
        sampled = _sample_single_table_datasets(
            csv_folder, single_table_query_count, extension=extension,
            seed=seed, limit_to_n_columns=max_cols, scan_opts=scan_opts,
        )
        single_agent = SingleStatementAgent(
            config_path, kind, bad_tokens, languages=languages, seed=seed,
            analysis_cache_path=analysis_cache_path,
            search_index=search_index,
            keyword_search_top_k_coefficient=keyword_search_top_k_coefficient,
            gate_unretrievable_groups=gate_unretrievable_groups,
            retrieval_gate_enabled=retrieval_gate_enabled,
            plan_judge_count=plan_judge_count,
            code_judge_count=code_judge_count,
        )

        for st_idx, csv_path in enumerate(sampled):
            dataset_name = csv_path.stem
            aliases = {"Table_0": dataset_name}
            str_idx = f"st_{st_idx}"

            if _already_processed(results, kind, str_idx):
                print(f"[st_{st_idx}] Already processed — skipping.")
                continue

            start = __import__("time").perf_counter()

            content = single_agent.generate_statements(
                csv_path, aliases, kind,
                datasets_metadata.get(dataset_name),
                max_cols, sample_size=5,
            )

            generation_time = __import__("time").perf_counter() - start

            _store_generation(results, kind, str_idx, content, aliases, generation_time)
            save_json(results, output_file)
            sys.stdout.flush()

            actual_tokens = sum(content["token_usage"].values()) if isinstance(content["token_usage"], dict) else content["token_usage"]
            cooldown = _compute_timeout(actual_tokens)
            print(f"[st_{st_idx}] Consumed {actual_tokens} tokens — cooling down for {cooldown}s.")
            __import__("time").sleep(cooldown)
            
    # ── Cross-table generation ────────────────────────────────────────────────
    for idx, match in enumerate(all_matches):
        #if _is_single_table_candidate(match):
        #    continue
        #continue
        str_idx = str(idx)
        if _already_processed(results, kind, str_idx):
            print(f"[{idx}] Already processed — skipping.")
            continue

        dataset_paths, aliases, metadatas, involved_cols = _build_match_inputs(
            match, csv_folder, datasets_metadata, extension
        )

        match_for_planner = _get_match_for_planner(match)
        if match_for_planner is None:
            match_for_planner = _get_formatted_match(match, kind)

        start = __import__("time").perf_counter()

        content = cross_agent.generate_statements(
            dataset_paths, aliases, kind,
            match_for_planner,
            involved_cols, metadatas,
            max_cols, sample_size=5
        )

        generation_time = __import__("time").perf_counter() - start

        _store_generation(results, kind, str_idx, content, aliases, generation_time)
        save_json(results, output_file)
        sys.stdout.flush()

        actual_tokens = sum(content["token_usage"].values()) if isinstance(content["token_usage"], dict) else content["token_usage"]
        cooldown = _compute_timeout(actual_tokens)
        print(f"[{idx}] Consumed {actual_tokens} tokens — cooling down for {cooldown}s.")
        __import__("time").sleep(cooldown)

    

    print(f"\nResults saved to {output_file}")
    return results


# ── Async streaming generation (used by web_app.py) ──────────────────────────

async def stream_generate_statements(
    cfg: OrQAConfig,
    kind: str,
    resume_from: int = 0,
) -> AsyncGenerator[dict, None]:
    """
    Async generator that runs statement generation match-by-match and yields
    progress dicts after each one.

        { "type": "progress",  "idx", "total", "successes", "failures",
          "status", "aliases", "query_count" }
        { "type": "done",      "successes", "failures", "total" }
        { "type": "error",     "message" }

    Args:
        kind:        query language to generate ("PANDAS", "SQL", …)
        resume_from: skip any match whose index is already present in the
                     output file — only indices >= resume_from are candidates,
                     and among those only ones not already stored are run.
    """
    loop = asyncio.get_event_loop()

    try:
        metadata = await loop.run_in_executor(
            None,
            load_normalized_datasets_metadata, cfg.metadata_path.joinpath("metadata.json"),
            None,
            cfg.source,
        )
        all_matches: list = await loop.run_in_executor(
            None, load_json, cfg.statement_generation.query_candidates_path
        )
        all_matches = _cap_matches(
            all_matches, cfg.statement_generation.multi_table_query_count, seed=cfg.seed
        )
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return

    output_file: Path = cfg.statement_generation.queries_path
    results     = load_json(output_file) if output_file.exists() else {}
    total       = len(all_matches)
    successes = failures = 0

    enable_single_table      = cfg.statement_generation.enable_single_table
    single_table_query_count = cfg.statement_generation.single_table_query_count

    analysis_cache_path = (
        cfg.statement_generation.query_candidates_path.parent
        / "table_analysis_cache.json"
    )

    # ── Stage 0: pre-generation table analysis (see create_statements) ────
    # Analyse every involved table up front so the generation agents below
    # only ever READ descriptions/keywords from the cache.
    def _pre_analyze() -> None:
        involved = [
            cfg.datasets_path / f"{dataset}.{cfg.datasets_format}"
            for match in all_matches
            for dataset in match.get("aliases", {}).values()
        ]
        involved_columns_by_dataset = _involved_columns_by_dataset(all_matches)
        if enable_single_table and single_table_query_count:
            involved.extend(
                _sample_single_table_datasets(
                    cfg.datasets_path, single_table_query_count, cfg.datasets_format,
                    cfg.seed, cfg.candidates_discovery.limit_to_n_columns,
                    cfg.polars_opts.scan,
                )
            )
        TableAnalysisAgent(
            cfg.llm_config_path / "litellm.yaml",
            analysis_cache_path,
            bad_tokens=cfg.statement_generation.bad_tokens,
            max_cols=cfg.candidates_discovery.limit_to_n_columns,
            seed=cfg.seed,
        ).analyze_tables(involved, metadata, involved_columns_by_dataset)

    try:
        await loop.run_in_executor(None, _pre_analyze)
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return

    progress_idx = 0  # Track overall progress across single + cross table
    st_total = 0  # Will be set if single-table generation runs

    # ── Single-table generation FIRST (random CSV sampling) ───────────────────────────
    if enable_single_table and single_table_query_count:
        sampled = await loop.run_in_executor(
            None, _sample_single_table_datasets,
            cfg.datasets_path, single_table_query_count, cfg.datasets_format, cfg.seed,
            cfg.candidates_discovery.limit_to_n_columns, cfg.polars_opts.scan,
        )
        st_total = len(sampled)

        single_agent = SingleStatementAgent(
            cfg.llm_config_path / "litellm.yaml",
            kind,
            cfg.statement_generation.bad_tokens,
            seed=cfg.seed,
            analysis_cache_path=analysis_cache_path,
            gate_unretrievable_groups=cfg.mcp_search.gate_unretrievable_groups,
            retrieval_gate_enabled=cfg.mcp_search.retrieval_gate_enabled,
            plan_judge_count=JUDGE_MODE_COUNTS[cfg.judges.plan_mode],
            code_judge_count=JUDGE_MODE_COUNTS[cfg.judges.code_mode],
        )
        for st_idx, csv_path in enumerate(sampled):
            dataset_name = csv_path.stem
            aliases = {"Table_0": dataset_name}

            def _process_single(csv_path=csv_path, aliases=aliases):
                content = single_agent.generate_statements(
                    csv_path, aliases, kind,
                    metadata.get(csv_path.stem),
                    cfg.candidates_discovery.limit_to_n_columns, sample_size=5,
                )
                return content

            try:
                content = await loop.run_in_executor(None, _process_single)
            except Exception as exc:
                yield {"type": "error", "message": str(exc)}
                return

            result = content["result"]

            if result.get("queries"): successes += 1
            else:                     failures  += 1

            status = _store_generation(
                results, kind, f"st_{st_idx}", content, aliases, content["time_elapsed"]
            )

            await loop.run_in_executor(None, save_json, results, output_file)

            actual_tokens = sum(content["token_usage"].values()) if isinstance(content["token_usage"], dict) else content["token_usage"]
            cooldown = _compute_timeout(actual_tokens)
            print(f"[st_{st_idx}] Consumed {actual_tokens} tokens — cooling down for {cooldown}s.")
            await asyncio.sleep(cooldown)

            progress_idx += 1
            yield {
                "type":        "progress",
                "idx":         progress_idx,
                "total":       total + st_total,
                "successes":   successes,
                "failures":    failures,
                "status":      status,
                "aliases":     list(aliases.values()),
                "query_count": len(result.get("queries", [])),
            }

    # ── Cross-table generation THEN ────────────────────────────────────────────────
    cross_agent = StatementAgent(
        cfg.llm_config_path / "litellm.yaml",
        kind,
        cfg.statement_generation.bad_tokens,
        seed=cfg.seed,
        analysis_cache_path=analysis_cache_path,
        gate_unretrievable_groups=cfg.mcp_search.gate_unretrievable_groups,
        retrieval_gate_enabled=cfg.mcp_search.retrieval_gate_enabled,
        plan_judge_count=JUDGE_MODE_COUNTS[cfg.judges.plan_mode],
        code_judge_count=JUDGE_MODE_COUNTS[cfg.judges.code_mode],
    )
    for idx, match in enumerate(all_matches):
        #if idx < resume_from:
        continue

        def _process_cross(match=match):
            dataset_paths, aliases, metadatas, involved_cols = _build_match_inputs(
                match, cfg.datasets_path, metadata, cfg.datasets_format
            )
            match_for_planner = _get_match_for_planner(match)
            if match_for_planner is None:
                match_for_planner = _get_formatted_match(match, kind)
            content = cross_agent.generate_statements(
                dataset_paths, aliases, kind,
                match_for_planner,
                involved_cols, metadatas,
                cfg.candidates_discovery.limit_to_n_columns, sample_size=5,
            )
            return content, aliases

        try:
            content, aliases = await loop.run_in_executor(None, _process_cross)
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}
            return

        result = content["result"]

        if result.get("queries"): successes += 1
        else:                     failures  += 1

        status = _store_generation(
            results, kind, str(idx), content, aliases, content["time_elapsed"]
        )

        await loop.run_in_executor(None, save_json, results, output_file)

        actual_tokens = sum(content["token_usage"].values()) if isinstance(content["token_usage"], dict) else content["token_usage"]
        cooldown = _compute_timeout(actual_tokens)
        print(f"[{idx}] Consumed {actual_tokens} tokens — cooling down for {cooldown}s.")
        await asyncio.sleep(cooldown)

        progress_idx += 1
        yield {
            "type":        "progress",
            "idx":         progress_idx,
            "total":       total + st_total,
            "successes":   successes,
            "failures":    failures,
            "status":      status,
            "aliases":     list(aliases.values()),
            "query_count": len(result.get("queries", [])),
        }


# ── Entry point (CLI via main.py) ─────────────────────────────────────────────

def generate_statements(cfg: OrQAConfig) -> None:
    metadata = load_normalized_datasets_metadata(cfg.normalized_metadata_filepath)
    # Built once, shared by every plan judge panel this run spins up: the
    # SAME reverse index tasks.mcp_search points at, so the plan judge's
    # keyword-searchability check (see
    # orqa.agent.utility.keyword_searchability) verifies against the exact
    # index a downstream retrieval agent would actually search. `None` when
    # it can't be built (metadata not indexed yet, Elasticsearch down, ...)
    # — the check then no-ops rather than blocking generation.
    search_index = load_index(cfg)
    for lang in ["PANDAS","SQL"]:
        create_statements(
            cfg.llm_config_path.joinpath("litellm.yaml"),
            cfg.datasets_path,
            cfg.statement_generation.query_candidates_path,
            cfg.statement_generation.queries_path,
            lang,
            cfg.candidates_discovery.limit_to_n_columns,
            cfg.datasets_format,
            datasets_metadata=metadata,
            bad_tokens=cfg.statement_generation.bad_tokens,
            enable_single_table=cfg.statement_generation.enable_single_table,
            single_table_query_count=cfg.statement_generation.single_table_query_count,
            languages=cfg.statement_generation.detected_languages,
            seed=cfg.seed,
            search_index=search_index,
            keyword_search_top_k_coefficient=cfg.mcp_search.keyword_search_top_k_coefficient,
            multi_table_query_count=cfg.statement_generation.multi_table_query_count,
            gate_unretrievable_groups=cfg.mcp_search.gate_unretrievable_groups,
            retrieval_gate_enabled=cfg.mcp_search.retrieval_gate_enabled,
            plan_judge_count=JUDGE_MODE_COUNTS[cfg.judges.plan_mode],
            code_judge_count=JUDGE_MODE_COUNTS[cfg.judges.code_mode],
            scan_opts=cfg.polars_opts.scan,
        )