import asyncio
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Iterator, Optional

from .agent.agent import TableAnalysisAgent
from orqa.agent.agent import StatementAgent
from orqa.agent.agent import SingleStatementAgent
from .utils import memory_guard
from .utils.generation_trace import open_trace, trace_group
from .utils.questions import META_KEY, SECTION_SINGLE, get_entry, store_entry
from .retrieval.panel import RetrieverPanel, make_llm_keyword_extractor
from .utils import (
    dataset_id_to_resource_id,
    dataset_index_shape,
    load_normalized_datasets_metadata,
    save_json,
    load_json,
)
from conf import OrQAConfig, JUDGE_MODE_COUNTS
from solr.index import load_index
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)


def _build_gate_index(cfg: OrQAConfig, search_index: Any) -> Any:
    """The index the gate's KEYWORD search and the anchor search run on: the
    portal's Solr core, matching per ``retrieval_contract.keyword_match``
    (``all`` = ``q.op=AND``, what LakeGen sends), so the gate verifies against
    the retriever LakeGen queries, not an approximation of it.
    """
    if search_index is None:
        return None
    contract = cfg.search.retrieval_contract
    gate = search_index.with_match(contract.keyword_match)
    print(
        f"[gate] keyword search: Solr core {cfg.search.solr.core!r}, "
        f"q.op={'AND' if contract.keyword_match == 'all' else 'OR'}, "
        f"retrievers={list(contract.retrievers)}.", flush=True,
    )
    return gate


def _build_retrieval_panel(
    cfg: OrQAConfig,
    search_index: Any,
    keyword_index: Optional[Any] = None,
    with_keywords: bool = True,
) -> Optional[RetrieverPanel]:
    """The retriever panel behind the retrievability gate (see
    ``orqa.retrieval.gate``) — ``None`` when the gate
    is disabled (``retrieval_gate_enabled`` / ``retrieval_contract.enabled``)
    or no index is available, so callers only need to check
    for ``None`` rather than re-deriving these flags themselves.

    ``keyword_index`` is what ``llm_keywords`` and the lexical gate search —
    the gate index (``_build_gate_index``). The dense retriever is the Solr
    core's own KNN, and the core's document universe (``resource_ids``)
    restricts every ranking. ``with_keywords=False`` skips the LLM keyword
    extractor (and with it the ``llm_keywords`` and ``hybrid_rrf``
    retrievers).
    """
    if search_index is None:
        return None
    if (
        not cfg.search.retrieval_gate_enabled
        or not cfg.search.retrieval_contract.enabled
    ):
        return None
    keyword_extractor = None
    try:
        if with_keywords:
            keyword_extractor = make_llm_keyword_extractor(cfg.llm_config_path / "litellm.yaml")
    except Exception:
        logger.warning(
            "Could not build the llm_keywords retriever; the retrieval "
            "panel proceeds with lexical/dense retrievers only.",
            exc_info=True,
        )
    lexical = search_index
    with_match = getattr(search_index, "with_match", None)
    if callable(with_match):
        # Solr: ``lexical_question`` searches the raw question as free text,
        # which under q.op=AND would need every word of the question in one
        # document — OR ranks it. The panel takes its lexical retriever from
        # the dense index's ``lexical_index`` (a Solr index's is itself), so
        # the OR sibling serves both: its KNN ignores q.op, and siblings share
        # the client and the query-embedding cache.
        lexical = with_match("any")
    dense = lexical if callable(getattr(lexical, "semantic_search_many", None)) else None
    resource_ids = getattr(search_index, "resource_ids", None)
    contract = cfg.search.retrieval_contract
    return RetrieverPanel(
        lexical,
        dense_index=dense,
        keyword_extractor=keyword_extractor,
        retrievers=contract.retrievers,
        keyword_index=keyword_index,
        rrf_k=contract.hybrid_rrf_k,
        rrf_depth=contract.hybrid_rrf_depth,
        universe=resource_ids() if callable(resource_ids) else None,
    )

# Pacing between generated groups, per 5k tokens the group consumed. NOT a
# rate limit the provider enforces: OCI's on-demand ceiling is applied by
# dynamic throttling — undocumented, moving with overall demand — so no fixed
# pause can be calibrated to stay under it, and the client backs off on a real
# 429 instead (see LLMClient._completion_with_backoff).
#
# Measured on a 99-group UK run: 15s/5k spent 12.55h sleeping against 6.03h
# generating (68% of wall clock) and still did not prevent throttling — the
# one group that was throttled ran at 35.7k tokens/min, BELOW the 42.3k median,
# while groups at 60k and 81k went through untouched. The sleep also sits
# between groups, whereas the ~150k tokens of a single group are a burst of
# many calls inside it, so it never paced the thing that actually trips the
# limit. Kept small rather than zero purely to space consecutive bursts.
# Cut from 3 to 1 on 2026-11-07 (user's call): on the dedicated Qwen endpoint,
# which has no tenant throttle, the 3 s rate idled ~40-55 s after each ~3 min
# single-table group.
_TIMEOUT_SECONDS_PER_5K_TOKENS: int = 1
_TOKENS_PER_BUCKET: int = 5_000


def _compute_timeout(max_tokens: int) -> float:
    """Return the cooldown (seconds) to pace a call of *max_tokens* tokens.

    Callers pass the tokens a generation ACTUALLY consumed, not a cap.
    Uses ceiling division so that even a 1-token call gets the full first bucket.
    """
    buckets = math.ceil(max_tokens / _TOKENS_PER_BUCKET)
    return buckets * _TIMEOUT_SECONDS_PER_5K_TOKENS


def _tokens_used(content: dict) -> int:
    """Total tokens a generation output reports (its ``token_usage`` is either
    a usage dict or a bare number). A usage dict's ``total_tokens`` already
    counts the prompt and completion: summing its values counted every token
    twice (and doubled every cooldown)."""
    usage = content.get("token_usage")
    if not isinstance(usage, dict):
        return usage or 0
    if "total_tokens" in usage:
        return usage["total_tokens"] or 0
    return sum(v for v in usage.values() if isinstance(v, (int, float)))


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

    For ``union`` links, ``left_cols``/``right_cols`` are only a real
    correspondence when ``column_scores`` is populated (see
    ``query_candidates._make_union_match``): either the aligned pairs the LLM
    selected for the pair (semantic pipeline) or the schema matches that
    cleared the confidence threshold (classical pipeline). They are
    positionally aligned by that function and are zipped into pairs here,
    WITH the confidence score attached when the matcher scored the pair
    (union evidence is a schema/name-similarity signal, not a value-overlap
    guarantee like a join key, so the score matters). When ``column_scores``
    is empty there is no correspondence — each column is listed separately,
    still tagged with its own table, without claiming a cross-table
    correspondence that isn't actually known.
    """
    left, right = spec.get("left", ""), spec.get("right", "")
    mtype = spec.get("type", "")
    correlated_columns: list[dict] = []
    if mtype == "union":
        left_cols, right_cols = spec.get("left_cols", []), spec.get("right_cols", [])
        scores = spec.get("column_scores") or []
        if scores and len(scores) == len(left_cols) == len(right_cols):
            columns = [
                {left: lc, right: rc, **({"score": f"{s:.2f}"} if s is not None else {})}
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
        metadatas.append(datasets_metadata.get(dataset_id_to_resource_id(dataset)))
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


# A single-table slot whose table cannot be found within the contract's top K
# is dropped — nothing is written for it — and refilled from the next table of
# the same seeded order. At most this many replacements are tried per slot, so
# a portal where almost nothing is retrievable cannot loop over its whole
# corpus for one slot.
MAX_SINGLE_TABLE_REPLACEMENTS = 25

# The status a generation run reports when the table(s) it is fixed to cannot
# be found by the retrievers within the top K: the pre-planning keyword check,
# the question stage and the plan judge all end this way.
UNRETRIEVABLE_STATUS = "unretrievable_group"


def _iter_single_table_replacements(
    datasets_path: Path,
    exclude: set[Path],
    extension: str = "csv",
    seed: int = 0,
    limit_to_n_columns: int | None = None,
    scan_opts: dict | None = None,
) -> Iterator[Path]:
    """Lazily yield usable single-table candidates beyond the ones already
    sampled, in the same seeded order ``_sample_single_table_datasets`` draws
    from and under the same usability filter (rows, columns, width) — so a
    table that stands in for an unretrievable one is chosen exactly the way
    the original was, and the same on every run. ``exclude``: paths to skip
    (the sampled ones, and any table a stored entry already used)."""
    all_files = sorted(datasets_path.glob(f"*.{extension}"))
    ordered = all_files[:]
    random.Random(seed).shuffle(ordered)

    for filepath in ordered:
        if filepath in exclude:
            continue
        if limit_to_n_columns is not None:
            try:
                n_columns, has_rows = dataset_index_shape(filepath, scan_opts or {})
            except Exception:
                continue
            if n_columns == 0 or not has_rows or n_columns > limit_to_n_columns:
                continue
        yield filepath


def _stored_single_table_stems(results: dict, kind: str) -> set[str]:
    """The tables the stored single-table entries of ``kind`` were generated
    on — a resumed run must not hand one of them to another slot."""
    stems: set[str] = set()
    for entry in (results.get(kind, {}).get(SECTION_SINGLE) or {}).values():
        tables = (entry.get(META_KEY) or {}).get("tables") or {}
        stems.update(str(stem) for stem in tables.values())
    return stems


@dataclass
class SingleTableSlot:
    """Outcome of filling one single-table slot: the table that produced an
    entry (``None`` when every candidate was unretrievable), its generation
    output and duration, and the tables skipped on the way."""
    path: Optional[Path]
    content: Optional[dict]
    elapsed: float
    skipped: list[Path] = field(default_factory=list)


def _generate_single_table_slot(
    primary: Path,
    replacements: Iterator[Path],
    generate: Callable[[Path], dict],
    max_replacements: int = MAX_SINGLE_TABLE_REPLACEMENTS,
    on_skip: Optional[Callable[[Path, dict], None]] = None,
) -> SingleTableSlot:
    """Generate for ``primary``; while the run reports it unretrievable
    (``UNRETRIEVABLE_STATUS``) drop it and try the next of ``replacements``.

    Only that status skips a table — any other outcome (success, a failure,
    "no_valid_question", ...) is the slot's result, exactly as before. A
    skipped table leaves nothing behind: the caller writes no entry for it.
    """
    skipped: list[Path] = []
    candidate: Optional[Path] = primary
    while candidate is not None:
        start = __import__("time").perf_counter()
        content = generate(candidate)
        elapsed = __import__("time").perf_counter() - start
        if (content.get("result") or {}).get("status") != UNRETRIEVABLE_STATUS:
            return SingleTableSlot(candidate, content, elapsed, skipped)
        skipped.append(candidate)
        if on_skip is not None:
            on_skip(candidate, content)
        if len(skipped) > max_replacements:
            break
        candidate = next(replacements, None)
    return SingleTableSlot(None, None, 0.0, skipped)


def _cap_matches(
    all_matches: list,
    count: Optional[int],
    seed: int = 0,
) -> list:
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
    retrieval_panel: Optional[RetrieverPanel] = None,
    retrieval_contract_min_agreement: int = 2,
    retrieval_contract_top_k_per_table: int = 10,
    retrieval_contract_max_top_k: int = 20,
    retrieval_contract_single_table_top_k: int = 1,
    max_question_corrections: int = 5,
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
        retrieval_panel=retrieval_panel,
        retrieval_contract_min_agreement=retrieval_contract_min_agreement,
        retrieval_contract_top_k_per_table=retrieval_contract_top_k_per_table,
        retrieval_contract_max_top_k=retrieval_contract_max_top_k,
        retrieval_contract_single_table_top_k=retrieval_contract_single_table_top_k,
        max_question_corrections=max_question_corrections,
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
            # A single table no keyword search can place within the top K is
            # skipped (and replaced below), whatever the workflow says about
            # unretrievable multi-table groups — the check runs before any LLM
            # call, so it is the cheap moment to give up on a table.
            gate_unretrievable_groups=True,
            retrieval_gate_enabled=retrieval_gate_enabled,
            plan_judge_count=plan_judge_count,
            code_judge_count=code_judge_count,
            retrieval_panel=retrieval_panel,
            retrieval_contract_min_agreement=retrieval_contract_min_agreement,
            retrieval_contract_top_k_per_table=retrieval_contract_top_k_per_table,
            retrieval_contract_max_top_k=retrieval_contract_max_top_k,
            retrieval_contract_single_table_top_k=retrieval_contract_single_table_top_k,
            max_question_corrections=max_question_corrections,
        )

        # Tables that can stand in for an unretrievable one: the rest of the
        # seeded order, minus what is already sampled or already stored.
        replacements = _iter_single_table_replacements(
            csv_folder,
            exclude=set(sampled) | {
                csv_folder / f"{stem}.{extension}"
                for stem in _stored_single_table_stems(results, kind)
            },
            extension=extension, seed=seed, limit_to_n_columns=max_cols, scan_opts=scan_opts,
        )

        for st_idx, csv_path in enumerate(sampled):
            str_idx = f"st_{st_idx}"

            if _already_processed(results, kind, str_idx):
                print(f"[st_{st_idx}] Already processed — skipping.")
                continue

            def _generate(path: Path, str_idx: str = str_idx) -> dict:
                # Every table tried for this slot is its own traced group
                # under the slot's id; an unretrievable one is skipped and
                # refilled, so it is recorded as not stored.
                with trace_group(
                    str_idx, mode="single", kind=kind, tables={"Table_0": path.stem},
                ) as group:
                    content = single_agent.generate_statements(
                        path, {"Table_0": path.stem}, kind,
                        datasets_metadata.get(dataset_id_to_resource_id(path.stem)),
                        max_cols, sample_size=5,
                    )
                    status = (content.get("result") or {}).get("status")
                    group.finish(content, stored=status != UNRETRIEVABLE_STATUS)
                return content

            def _on_skip(path: Path, content: dict, st_idx: int = st_idx) -> None:
                print(
                    f"[st_{st_idx}] {path.stem} is not found within the top "
                    f"{retrieval_contract_single_table_top_k} — skipped, no entry "
                    "written; trying another table."
                )
                spent = _tokens_used(content)
                if spent:
                    __import__("time").sleep(_compute_timeout(spent))

            try:
                slot = _generate_single_table_slot(csv_path, replacements, _generate, on_skip=_on_skip)
            except memory_guard.InsufficientMemoryError as exc:
                print(f"[st_{st_idx}] Skipped, too large for the memory free: {exc} — no entry written; a later run retries it.")
                continue
            except Exception:
                logger.exception(
                    "[st_%d] %s failed to generate — no entry written; skipping.",
                    st_idx, csv_path.stem,
                )
                continue
            finally:
                # The group's tables are gone: hand their memory back (Arrow's
                # allocator keeps it otherwise, ~40 GB after a taxi table).
                memory_guard.release_unused()
            if slot.path is None:
                print(
                    f"[st_{st_idx}] No table found within the top "
                    f"{retrieval_contract_single_table_top_k} after {len(slot.skipped)} "
                    "attempt(s) — no entry written for this slot."
                )
                continue
            if slot.path != csv_path:
                print(f"[st_{st_idx}] Using {slot.path.stem} instead of {csv_path.stem}.")

            _store_generation(
                results, kind, str_idx, slot.content, {"Table_0": slot.path.stem}, slot.elapsed,
            )
            save_json(results, output_file)
            sys.stdout.flush()

            actual_tokens = _tokens_used(slot.content)
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

        try:
            with trace_group(str_idx, mode="multi", kind=kind, tables=aliases) as group:
                content = cross_agent.generate_statements(
                    dataset_paths, aliases, kind,
                    match_for_planner,
                    involved_cols, metadatas,
                    max_cols, sample_size=5
                )
                group.finish(content, stored=True)
        except memory_guard.InsufficientMemoryError as exc:
            print(f"[{idx}] Skipped, too large for the memory free: {exc} — no entry written; a later run retries it.")
            continue
        except Exception:
            logger.exception(
                "[%d] %s failed to generate — no entry written; skipping.",
                idx, ", ".join(aliases.values()),
            )
            continue
        finally:
            memory_guard.release_unused()

        generation_time = __import__("time").perf_counter() - start

        _store_generation(results, kind, str_idx, content, aliases, generation_time)
        save_json(results, output_file)
        sys.stdout.flush()

        actual_tokens = _tokens_used(content)
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
            all_matches, cfg.statement_generation.multi_table_query_count, seed=cfg.seed,
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
            None,
            lambda: _sample_single_table_datasets(
                cfg.datasets_path, single_table_query_count, cfg.datasets_format, cfg.seed,
                cfg.candidates_discovery.limit_to_n_columns, cfg.polars_opts.scan,
            ),
        )
        st_total = len(sampled)

        single_agent = SingleStatementAgent(
            cfg.llm_config_path / "litellm.yaml",
            kind,
            cfg.statement_generation.bad_tokens,
            seed=cfg.seed,
            analysis_cache_path=analysis_cache_path,
            gate_unretrievable_groups=cfg.search.gate_unretrievable_groups,
            retrieval_gate_enabled=cfg.search.retrieval_gate_enabled,
            plan_judge_count=JUDGE_MODE_COUNTS[cfg.judges.plan_mode],
            code_judge_count=JUDGE_MODE_COUNTS[cfg.judges.code_mode],
        )
        for st_idx, csv_path in enumerate(sampled):
            dataset_name = csv_path.stem
            aliases = {"Table_0": dataset_name}

            def _process_single(csv_path=csv_path, aliases=aliases):
                content = single_agent.generate_statements(
                    csv_path, aliases, kind,
                    metadata.get(dataset_id_to_resource_id(csv_path.stem)),
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

            actual_tokens = _tokens_used(content)
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
        gate_unretrievable_groups=cfg.search.gate_unretrievable_groups,
        retrieval_gate_enabled=cfg.search.retrieval_gate_enabled,
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

        actual_tokens = _tokens_used(content)
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

def _trace_run_info(cfg: OrQAConfig) -> dict:
    """What run.json records about the configuration a trace was made under."""
    def plain(value: Any) -> Any:
        try:
            return asdict(value)
        except Exception:  # noqa: BLE001 — a record, never a reason to stop
            return str(value)

    return {
        "source": cfg.source,
        "seed": cfg.seed,
        "data_path": cfg.data_path,
        "write_path": cfg.write_path,
        "datasets_path": cfg.datasets_path,
        "query_candidates_path": cfg.statement_generation.query_candidates_path,
        "queries_path": cfg.statement_generation.queries_path,
        "llm_config": cfg.llm_config_path / "litellm.yaml",
        "statement_generation": plain(cfg.statement_generation),
        "search": plain(cfg.search),
        "judges": plain(cfg.judges),
        "limit_to_n_columns": cfg.candidates_discovery.limit_to_n_columns,
    }


def generate_statements(cfg: OrQAConfig) -> None:
    """The ``generate-statements`` step, recorded in a generation trace
    (tasks.query_generation.trace, see orqa.utils.generation_trace) unless
    the workflow switches it off."""
    trace = cfg.statement_generation.trace
    if not trace.enabled or trace.path is None:
        _generate_statements(cfg)
        return
    with open_trace(trace.path, _trace_run_info(cfg), llm_io=trace.llm_io):
        _generate_statements(cfg)


def _generate_statements(cfg: OrQAConfig) -> None:
    metadata = load_normalized_datasets_metadata(cfg.normalized_metadata_filepath)
    # Built once, shared by every plan judge panel this run spins up: the
    # SAME reverse index tasks.search points at, so the plan judge's
    # retrievability check (see orqa.retrieval.gate)
    # verifies against the exact index a downstream retrieval agent would
    # actually search. Not loaded at all with the gate off — nothing else
    # uses it, and a Solr core that does not answer would otherwise stop a
    # run that asked for no gate.
    search_index = load_index(cfg) if cfg.search.retrieval_gate_enabled else None
    # The retriever panel behind the retrievability gate (None when the gate
    # is off) is built ONCE and shared across every generation call below,
    # same lifetime as search_index.
    # The index as the PORTAL searches it (the Solr core with q.op per
    # keyword_match) — used for the
    # keyword-anchor search and the panel's keyword retriever, so "found"
    # means what it means to the portal. search_index itself backs the dense
    # retriever.
    gate_index = _build_gate_index(cfg, search_index)
    retrieval_panel = _build_retrieval_panel(cfg, search_index, keyword_index=gate_index)
    for lang in ["PANDAS"]:#,"SQL"]:
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
            search_index=gate_index,
            keyword_search_top_k_coefficient=cfg.search.keyword_search_top_k_coefficient,
            multi_table_query_count=cfg.statement_generation.multi_table_query_count,
            gate_unretrievable_groups=cfg.search.gate_unretrievable_groups,
            retrieval_gate_enabled=cfg.search.retrieval_gate_enabled,
            plan_judge_count=JUDGE_MODE_COUNTS[cfg.judges.plan_mode],
            code_judge_count=JUDGE_MODE_COUNTS[cfg.judges.code_mode],
            scan_opts=cfg.polars_opts.scan,
            retrieval_panel=retrieval_panel,
            retrieval_contract_min_agreement=cfg.search.retrieval_contract.min_agreement,
            retrieval_contract_top_k_per_table=cfg.search.retrieval_contract.top_k_per_table,
            retrieval_contract_max_top_k=cfg.search.retrieval_contract.max_top_k,
            retrieval_contract_single_table_top_k=cfg.search.retrieval_contract.single_table_top_k,
            max_question_corrections=cfg.statement_generation.max_question_corrections,
        )