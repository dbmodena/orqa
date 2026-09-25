import asyncio
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Iterator, Optional

from .agent.agent import TableAnalysisAgent
from .agent.agents.StatementAgent import StatementAgent
from .agent.agents.SingleStatementAgent import SingleStatementAgent
from .benchmark.families import FamilyIndex
from .benchmark.hybrid_index import HybridDatasetIndex
from .benchmark.index import DatasetIndex, load_index, portal_view
from .benchmark.solr_index import SolrDatasetIndex
from .benchmark.questions import META_KEY, SECTION_SINGLE, get_entry, store_entry
from .benchmark.retrieval_panel import RetrieverPanel, make_llm_keyword_extractor
from .utils import (
    dataset_id_to_resource_id,
    dataset_index_shape,
    load_normalized_datasets_metadata,
    save_json,
    load_json,
)
from conf import OrQAConfig, JUDGE_MODE_COUNTS
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _build_family_index(cfg: OrQAConfig) -> Optional[FamilyIndex]:
    """The CKAN family index, shared by family-stratified sampling and the
    retrieval contract. ``None`` (never raises) when the normalized
    metadata can't be read — both features then simply degrade to their
    pre-existing unstratified/ungated behavior.
    """
    try:
        with open(cfg.normalized_metadata_filepath, encoding="utf-8") as file:
            raw_records = json.load(file)
        return FamilyIndex(raw_records, cfg.datasets_path)
    except Exception:
        logger.warning(
            "Could not build the CKAN family index; family-stratified "
            "sampling and the retrieval contract are both unavailable "
            "for this run.",
            exc_info=True,
        )
        return None


def _build_gate_index(cfg: OrQAConfig, search_index: Any) -> Any:
    """The index the gate's KEYWORD search and the anchor search run on.

    ``retrieval_contract.keyword_backend: solr`` -> the portal's own Solr core
    (see ``orqa.benchmark.solr_index``): the same retriever LakeGen queries.
    Solr being unreachable is not fatal — it warns and uses the in-process
    imitation configured by ``keyword_match``/``search_fields`` — but it IS
    loud, because that imitation is only an approximation of the portal.
    """
    contract = cfg.mcp_search.retrieval_contract
    if contract.keyword_backend == "solr" and search_index is not None:
        records = getattr(search_index, "lexical_index", search_index)
        try:
            if not isinstance(records, DatasetIndex):
                raise TypeError(
                    f"the Solr gate needs the builtin metadata index for records, got {type(records).__name__}"
                )
            solr_index = SolrDatasetIndex.build(
                records, contract.solr_core, contract.solr_url or None,
                match=contract.keyword_match,
            )
            print(
                f"[gate] keyword search runs on Solr core {contract.solr_core!r} "
                f"({solr_index.ping()} documents).", flush=True,
            )
            return solr_index
        except Exception as exc:  # unreachable Solr, unknown core, wrong backend, ...
            logger.warning(
                "Could not use Solr core %r for the retrieval gate (%s) — falling back "
                "to the in-process imitation, which only approximates the portal.",
                contract.solr_core, exc,
            )
    gate = portal_view(search_index, contract.keyword_match, contract.search_fields)
    # The KEYWORD search is the lexical half. ``portal_view`` already unwraps a
    # hybrid index when it builds an AND / field-restricted view, but with plain
    # BM25 (``any``, all fields) it hands the index back untouched — and a hybrid
    # index fuses BM25 with a semantic ranking, so the anchor search and the
    # keyword retriever would silently verify against that fusion instead.
    lexical = getattr(gate, "lexical_index", None)
    if lexical is not None:
        gate = lexical
    if gate is not None:
        print(
            f"[gate] keyword search: {type(getattr(gate, '_index', gate)).__name__}, "
            f"match={contract.keyword_match}, fields={list(contract.search_fields) or 'all'}, "
            f"retrievers={list(contract.retrievers)}.", flush=True,
        )
    return gate


def _require_index(cfg: OrQAConfig, search_index: Any) -> None:
    """Refuse to run the retrieval gate without an index.

    ``load_index`` returns ``None`` — after logging why — when the reverse index
    cannot be built (Elasticsearch not running, metadata not indexed). Every
    consumer then treats "no index" as "nothing to check", so a gate the
    workflow turned ON silently became a gate that does nothing and the run
    produced ungated questions for hours. Fail before any of that instead.
    """
    if search_index is None and cfg.mcp_search.retrieval_gate_enabled:
        raise RuntimeError(
            "The retrieval gate is enabled (tasks.mcp_search.retrieval_gate_enabled) "
            f"but the reverse index (tasks.mcp_search.backend: {cfg.mcp_search.backend!r}) "
            "could not be built or loaded, so every question would be generated "
            "UNGATED. The log above says why — for elasticsearch, start it at "
            "tasks.mcp_search.elasticsearch_url. Set retrieval_gate_enabled: false "
            "to run without the gate on purpose."
        )


def _build_retrieval_panel(
    cfg: OrQAConfig,
    search_index: Any,
    keyword_index: Optional[Any] = None,
    universe: Optional[Any] = None,
    require_enabled: bool = True,
    with_keywords: bool = True,
) -> Optional[RetrieverPanel]:
    """The retriever panel behind the retrievability gate (see
    ``orqa.agent.utility.retrievability_gate``) — ``None`` when the gate
    is disabled (``retrieval_gate_enabled`` / ``retrieval_contract.enabled``)
    or no index is available, so callers only need to check
    for ``None`` rather than re-deriving these flags themselves.

    ``require_enabled=False`` builds the panel whatever the gate switches say —
    for a tool that MEASURES retrievability (the benchmark CLI's
    ``--retrievability-report``) and must use exactly the retrievers the gate
    does. ``with_keywords=False`` skips the LLM keyword extractor (and with it
    the ``llm_keywords`` and ``hybrid_rrf`` retrievers).
    """
    if search_index is None:
        return None
    if require_enabled and (
        not cfg.mcp_search.retrieval_gate_enabled
        or not cfg.mcp_search.retrieval_contract.enabled
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
    hybrid_index = search_index if isinstance(search_index, HybridDatasetIndex) else None
    contract = cfg.mcp_search.retrieval_contract
    return RetrieverPanel(
        search_index,
        hybrid_index=hybrid_index,
        keyword_extractor=keyword_extractor,
        retrievers=contract.retrievers,
        # ``llm_keywords`` searches the way the portal does (AND over its
        # own fields) when a portal view is configured; None = the lexical
        # index, as before.
        keyword_index=keyword_index,
        rrf_k=contract.hybrid_rrf_k,
        rrf_depth=contract.hybrid_rrf_depth,
        universe=universe,
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
_TIMEOUT_SECONDS_PER_5K_TOKENS: int = 3
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
    a usage dict or a bare number)."""
    usage = content.get("token_usage")
    return sum(usage.values()) if isinstance(usage, dict) else (usage or 0)


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


def _family_stratified_order(
    files: list[Path],
    family_of: Callable[[str], str],
    seed: int,
    max_groups_per_family: int,
) -> list[Path]:
    """Round-robin ``files`` across their CKAN families (``family_of``: a
    file stem -> family id mapping), so no single family contributes more
    than ``max_groups_per_family`` entries and every family gets an equal
    turn before a second round draws from any of them again — a family with
    hundreds of near-identical siblings ("Organogram of Staff Roles &
    Salaries", 1,426 files on the UK portal) can't crowd out the rest of
    the corpus. Both the family order and each family's own file order are
    seeded, so the result is deterministic under ``seed``.
    """
    by_family: dict[str, list[Path]] = {}
    for filepath in files:
        by_family.setdefault(family_of(filepath.stem), []).append(filepath)

    rng = random.Random(seed)
    family_ids = sorted(by_family)
    rng.shuffle(family_ids)
    for family in family_ids:
        rng.shuffle(by_family[family])

    order: list[Path] = []
    cursors = dict.fromkeys(family_ids, 0)
    counts = dict.fromkeys(family_ids, 0)
    progressed = True
    while progressed:
        progressed = False
        for family in family_ids:
            if counts[family] >= max_groups_per_family:
                continue
            idx = cursors[family]
            members = by_family[family]
            if idx >= len(members):
                continue
            order.append(members[idx])
            cursors[family] = idx + 1
            counts[family] += 1
            progressed = True
    return order


def _sample_single_table_datasets(
    datasets_path: Path,
    count: int,
    extension: str = "csv",
    seed: int = 0,
    limit_to_n_columns: int | None = None,
    scan_opts: dict | None = None,
    family_of: Optional[Callable[[str], str]] = None,
    max_groups_per_family: Optional[int] = None,
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

    When ``family_of`` AND ``max_groups_per_family`` are both given, files
    are drawn in family-stratified round-robin order (see
    ``_family_stratified_order``) instead of a flat shuffle. Either left
    ``None`` (the default) keeps the plain seeded-shuffle behavior.
    """
    all_files = sorted(datasets_path.glob(f"*.{extension}"))
    stratify = family_of is not None and max_groups_per_family is not None

    if limit_to_n_columns is None:
        if stratify:
            return _family_stratified_order(all_files, family_of, seed, max_groups_per_family)[:count]
        return random.Random(seed).sample(all_files, min(count, len(all_files)))

    scan_opts = scan_opts or {}
    if stratify:
        shuffled = _family_stratified_order(all_files, family_of, seed, max_groups_per_family)
    else:
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
    family_of: Optional[Callable[[str], str]] = None,
    max_groups_per_family: Optional[int] = None,
) -> Iterator[Path]:
    """Lazily yield usable single-table candidates beyond the ones already
    sampled, in the same seeded order ``_sample_single_table_datasets`` draws
    from and under the same usability filter (rows, columns, width) — so a
    table that stands in for an unretrievable one is chosen exactly the way
    the original was, and the same on every run. ``exclude``: paths to skip
    (the sampled ones, and any table a stored entry already used)."""
    all_files = sorted(datasets_path.glob(f"*.{extension}"))
    if family_of is not None and max_groups_per_family is not None:
        ordered = _family_stratified_order(all_files, family_of, seed, max_groups_per_family)
    else:
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
    family_of: Optional[Callable[[str], str]] = None,
    max_groups_per_family: Optional[int] = None,
) -> list:
    """Cap cross-table candidate matches to ``count``, seeded like
    ``_sample_single_table_datasets`` so runs (and sibling workflows sharing
    the yaml seed) pick the same subset. ``None`` (or a count at/above the
    current total, so there is nothing to actually cap) returns
    ``all_matches`` unchanged and in its original order — sampling would
    otherwise reshuffle it via ``random.sample``, needlessly scrambling the
    index-based resume/dedup keys the caller relies on.

    When ``family_of``/``max_groups_per_family`` are both given, a match is
    capped per family across ALL of its tables: a multi-table match counts
    ONE slot against EVERY family any of its aliases belongs to, so a
    family isn't undercounted just because it only appears as one alias of
    a wider join. Either left ``None`` keeps the plain seeded-sample
    behavior.
    """
    if count is None or count >= len(all_matches):
        return all_matches
    if family_of is None or max_groups_per_family is None:
        return random.Random(seed).sample(all_matches, count)

    def families_of(match: dict) -> set[str]:
        return {family_of(dataset) for dataset in (match.get("aliases") or {}).values()}

    rng = random.Random(seed)
    shuffled = all_matches[:]
    rng.shuffle(shuffled)

    picked: list = []
    counts: dict[str, int] = {}
    leftover: list = []
    for match in shuffled:
        match_families = families_of(match)
        if any(counts.get(f, 0) >= max_groups_per_family for f in match_families):
            leftover.append(match)
            continue
        picked.append(match)
        for f in match_families:
            counts[f] = counts.get(f, 0) + 1
        if len(picked) >= count:
            break

    # A cap tight enough to leave fewer than `count` selectable matches
    # (every remaining one blocked by an already-exhausted family) is
    # filled out from whatever's left, uncapped, rather than silently
    # returning short.
    if len(picked) < count:
        picked.extend(leftover[: count - len(picked)])
    return picked


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
    max_groups_per_family: Optional[int] = None,
    family_index: Optional[FamilyIndex] = None,
    retrieval_panel: Optional[RetrieverPanel] = None,
    retrieval_contract_min_agreement: int = 2,
    retrieval_contract_top_k_per_table: int = 10,
    retrieval_contract_max_top_k: int = 20,
    retrieval_contract_single_table_top_k: int = 1,
    max_question_corrections: int = 3,
) -> list[dict]:
    bad_tokens = bad_tokens or []

    if datasets_metadata is None:
        datasets_metadata = {}

    family_of = family_index.family_id_of_stem if family_index is not None else None

    all_matches: list = load_json(candidates_file)
    all_matches = _cap_matches(
        all_matches, multi_table_query_count, seed=seed,
        family_of=family_of, max_groups_per_family=max_groups_per_family,
    )
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
                family_of=family_of, max_groups_per_family=max_groups_per_family,
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
            family_of=family_of, max_groups_per_family=max_groups_per_family,
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
            family_of=family_of, max_groups_per_family=max_groups_per_family,
        )

        for st_idx, csv_path in enumerate(sampled):
            str_idx = f"st_{st_idx}"

            if _already_processed(results, kind, str_idx):
                print(f"[st_{st_idx}] Already processed — skipping.")
                continue

            def _generate(path: Path) -> dict:
                return single_agent.generate_statements(
                    path, {"Table_0": path.stem}, kind,
                    datasets_metadata.get(dataset_id_to_resource_id(path.stem)),
                    max_cols, sample_size=5,
                )

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
            except Exception:
                logger.exception(
                    "[st_%d] %s failed to generate — no entry written; skipping.",
                    st_idx, csv_path.stem,
                )
                continue
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
            content = cross_agent.generate_statements(
                dataset_paths, aliases, kind,
                match_for_planner,
                involved_cols, metadatas,
                max_cols, sample_size=5
            )
        except Exception:
            logger.exception(
                "[%d] %s failed to generate — no entry written; skipping.",
                idx, ", ".join(aliases.values()),
            )
            continue

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
        # Family-stratified sampling (see _sample_single_table_datasets)
        # applies here too — orthogonal to the retrieval gate itself, which
        # this streaming path doesn't wire up (no search_index is passed to
        # either agent below).
        family_index = await loop.run_in_executor(None, _build_family_index, cfg)
        family_of = family_index.family_id_of_stem if family_index is not None else None
        max_groups_per_family = cfg.statement_generation.max_groups_per_family
        all_matches = _cap_matches(
            all_matches, cfg.statement_generation.multi_table_query_count, seed=cfg.seed,
            family_of=family_of, max_groups_per_family=max_groups_per_family,
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
                    family_of=family_of, max_groups_per_family=max_groups_per_family,
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
                family_of=family_of, max_groups_per_family=max_groups_per_family,
            ),
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
    # retrievability check (see orqa.agent.utility.retrievability_gate)
    # verifies against the exact index a downstream retrieval agent would
    # actually search. `None` when it can't be built (metadata not indexed
    # yet, Elasticsearch down, ...) — the check then no-ops rather than
    # blocking generation.
    search_index = load_index(cfg)
    _require_index(cfg, search_index)
    # The CKAN family index (family-stratified sampling — see FamilyIndex),
    # and the retriever panel behind the retrievability gate (None when the
    # gate is off or unbuildable) — built ONCE and shared across every
    # generation call below, same lifetime as search_index.
    family_index = _build_family_index(cfg)
    # The index as the PORTAL searches it (AND over its own fields) when
    # tasks.mcp_search.retrieval_contract asks for that — used for the
    # keyword-anchor search and the panel's keyword retriever, so "found"
    # means what it means to the portal. The unwrapped index still backs the
    # dense retriever, and is untouched for any other consumer.
    gate_index = _build_gate_index(cfg, search_index)
    retrieval_panel = _build_retrieval_panel(
        cfg,
        search_index,
        keyword_index=gate_index if gate_index is not search_index else None,
        # Solr only holds part of the metadata the embedding cache covers:
        # rank the other retrievers on the same field the portal has.
        universe=gate_index.resource_ids() if isinstance(gate_index, SolrDatasetIndex) else None,
    )
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
            keyword_search_top_k_coefficient=cfg.mcp_search.keyword_search_top_k_coefficient,
            multi_table_query_count=cfg.statement_generation.multi_table_query_count,
            gate_unretrievable_groups=cfg.mcp_search.gate_unretrievable_groups,
            retrieval_gate_enabled=cfg.mcp_search.retrieval_gate_enabled,
            plan_judge_count=JUDGE_MODE_COUNTS[cfg.judges.plan_mode],
            code_judge_count=JUDGE_MODE_COUNTS[cfg.judges.code_mode],
            scan_opts=cfg.polars_opts.scan,
            max_groups_per_family=cfg.statement_generation.max_groups_per_family,
            family_index=family_index,
            retrieval_panel=retrieval_panel,
            retrieval_contract_min_agreement=cfg.mcp_search.retrieval_contract.min_agreement,
            retrieval_contract_top_k_per_table=cfg.mcp_search.retrieval_contract.top_k_per_table,
            retrieval_contract_max_top_k=cfg.mcp_search.retrieval_contract.max_top_k,
            retrieval_contract_single_table_top_k=cfg.mcp_search.retrieval_contract.single_table_top_k,
            max_question_corrections=cfg.statement_generation.max_question_corrections,
        )