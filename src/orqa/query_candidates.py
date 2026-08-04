"""
Query Candidates Generation Stage.

Bridges candidates discovery and statement generation: random walks over the
matches graph select groups of related tables, and each group's verified
pairwise tasks are flattened into the ``relationships`` records consumed by
``create_statements`` (query_candidates.json).

Runs as the ``generate-query-candidates`` pipeline step, between
``candidates-discovery`` and ``generate-statements``.
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from conf import OrQAConfig

from .graph import matches_graph
from .schema_matching.valentine_matcher import THRESHOLD as SCHEMA_MATCH_THRESHOLD
from .utils import save_json, pd_read_dataset
from .utils.pipeline_logger import PipelineLogger

# Rows read per table: enough to judge dtypes for the case-insensitive
# check — full contents are only needed later, at statement generation.
GROUP_TABLE_NROWS = 100


def load_tasks(tasks_file: Path) -> dict:
    """Load tasks_results.json and index by (Q, R, task)."""
    tasks = {}
    with open(tasks_file) as f:
        for line in f:
            if line.strip():
                spec = json.loads(line)
                tasks[(spec["Q"], spec["R"], spec["task"])] = spec
    return tasks


# ── Match builders ────────────────────────────────────────────────────────────

def _is_string_column(df: pd.DataFrame, col: str) -> bool:
    """Return True only when the column exists and holds object/string dtype."""
    if col not in df.columns:
        return False
    return pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])


def _resolve_case_insensitive(
    task_spec: dict,
    dfs: dict[str, pd.DataFrame],
    left_alias: str,
    right_alias: str,
) -> bool:
    """
    Return True only when case_insensitive was requested AND every join key
    involved is actually a string column.  Numeric / date keys must never get
    .str.lower() / LOWER().
    """
    if not task_spec.get("case_insensitive", False):
        return False
    left_on  = task_spec.get("q_columns", [])
    right_on = task_spec.get("r_columns", [])
    df_l = dfs.get(left_alias)
    df_r = dfs.get(right_alias)
    if df_l is None or df_r is None:
        return False   # can't verify → be conservative
    return (
        all(_is_string_column(df_l, c) for c in left_on)
        and all(_is_string_column(df_r, c) for c in right_on)
    )


def _make_union_match(task_spec, df_q, df_r, alias_q, alias_r):
    """
    ``q_columns``/``r_columns`` on a union task are NOT a correspondence —
    ``q_columns`` is the LLM-proposed subset of the query table's schema, while
    ``r_columns`` (see ``matches_graph.process_edge``) is every candidate-table
    column that showed up *anywhere* in an unfiltered, un-scored COMA
    schema-matcher pass over the two FULL schemas. The two lists have
    different provenance and routinely differ wildly in length (a handful of
    curated columns vs. nearly the candidate's entire schema).

    The matcher's actual per-pair output (``task_spec["matches"]``, a list of
    ``[q_col, r_col, score]`` triples) DOES carry a real, if noisy, cross-table
    correspondence — filtered here to pairs at/above ``SCHEMA_MATCH_THRESHOLD``
    (the same cutoff already used elsewhere in the codebase to decide whether
    a schema match is trustworthy) and ranked by confidence. When nothing
    clears that bar, fall back to the disjoint originals rather than
    fabricate a pairing that isn't actually supported.
    """
    raw_matches = task_spec.get("matches") or []
    confident = sorted(
        (
            (c1, c2, float(score))
            for c1, c2, score in raw_matches
            if float(score) >= SCHEMA_MATCH_THRESHOLD
        ),
        key=lambda triple: triple[2],
        reverse=True,
    )

    if confident:
        q_columns      = [c1 for c1, _, _ in confident]
        r_columns      = [c2 for _, c2, _ in confident]
        column_scores  = [s for _, _, s in confident]
    else:
        q_columns      = task_spec.get("q_columns", [])
        r_columns      = task_spec.get("r_columns", [])
        column_scores  = []

    col_str = f"{q_columns} / {r_columns}" if q_columns or r_columns else "All columns"
    return {
        "relationship": {
            "type": "union",
            "left": alias_q,
            "right": alias_r,
            "left_cols": q_columns,
            "right_cols": r_columns,
            # Schema-match confidence per (left_cols[i], right_cols[i]) pair,
            # same length/order as left_cols/right_cols. Empty when no pair
            # cleared SCHEMA_MATCH_THRESHOLD — left_cols/right_cols are then
            # each side's independently relevant columns, NOT a correspondence.
            "column_scores": column_scores,
            "description": f"UNION: {alias_q} ∪ {alias_r} ON {col_str}",
        },
        "columns": {alias_q: set(q_columns), alias_r: set(r_columns)},
    }


def _make_join_match(task_spec, df_q, df_r, alias_q, alias_r):
    q_keys = task_spec.get("q_columns", [])
    r_keys = task_spec.get("r_columns", [])
    conditions = " AND ".join(
        f"{alias_q}.{q_keys[i]} = {alias_r}.{r_keys[i]}" for i in range(len(q_keys))
    )
    ci = _resolve_case_insensitive(
        task_spec, {alias_q: df_q, alias_r: df_r}, alias_q, alias_r
    )
    return {
        "relationship": {
            "type": "merge",
            "how": "inner",
            "left": alias_q,
            "right": alias_r,
            "left_on": q_keys,
            "right_on": r_keys,
            "case_insensitive": ci,
            "relationship": task_spec.get("relationship", ""),
            "description": f"JOIN: {alias_q} ⋈ {alias_r} ON {conditions}",
        },
        "columns": {alias_q: set(q_keys), alias_r: set(r_keys)},
    }


def _make_join_correlation_match(task_spec, df_q, df_r, alias_q, alias_r):
    q_key    = task_spec["q_key"]
    r_key    = task_spec["r_key"]
    q_target = task_spec["q_target"]
    r_target = task_spec["r_target"]
    ci = _resolve_case_insensitive(
        {"case_insensitive": task_spec.get("case_insensitive", False),
         "q_columns": [q_key], "r_columns": [r_key]},
        {alias_q: df_q, alias_r: df_r},
        alias_q,
        alias_r,
    )
    # Only the semantic pipeline actually computes this at discovery time
    # (config.py: "join-correlation tasks are verified with an actual join +
    # correlation computation") — classical/BLEND JC tasks carry no such
    # metric, so this is None there and silently omitted below.
    metrics = task_spec.get("metrics") or {}
    correlation = metrics.get("correlation")
    correlation_method = metrics.get("correlation_method")
    corr_note = (
        f" [{correlation_method} r={correlation:.2f}]"
        if correlation is not None else ""
    )
    description = (
        f"JOIN-CORRELATION: merge {alias_q} ⋈ {alias_r} "
        f"ON {alias_q}.{q_key} = {alias_r}.{r_key}"
        + (" [case-insensitive]" if ci else "")
        + f", then correlate {alias_q}.{q_target} with {alias_r}.{r_target}{corr_note}"
    )
    relationship = {
        "type": "merge_correlation",
        "how": "inner",
        "left": alias_q,
        "right": alias_r,
        "left_on": [q_key],
        "right_on": [r_key],
        "case_insensitive": ci,
        "relationship": task_spec.get("relationship", ""),
        "correlation_cols": {alias_q: q_target, alias_r: r_target},
        "description": description,
    }
    if correlation is not None:
        relationship["correlation_value"] = correlation
        relationship["correlation_method"] = correlation_method
    return {
        "relationship": relationship,
        "columns": {alias_q: {q_key, q_target}, alias_r: {r_key, r_target}},
    }


def make_match(task_spec, df_q, df_r, alias_q, alias_r):
    """Build the relationship record for one verified task.

    Returns ``{"relationship": {...spec + description...},
    "columns": {alias: set(cols)}}`` or None for unknown task types.
    """
    task = task_spec["task"]
    if task == "U":
        return _make_union_match(task_spec, df_q, df_r, alias_q, alias_r)
    if task == "J":
        return _make_join_match(task_spec, df_q, df_r, alias_q, alias_r)
    if task == "JC":
        return _make_join_correlation_match(task_spec, df_q, df_r, alias_q, alias_r)
    return None


# ── Group processing ──────────────────────────────────────────────────────────

def _relationship_fingerprint(spec: dict) -> tuple:
    """
    Direction-insensitive identity of a relationship: the same pair of tables
    matched on the same columns with the same operation type counts once, no
    matter which side the discovery task listed first.
    """
    endpoints = sorted(
        (
            (spec.get("left", ""),
             tuple(spec.get("left_on") or spec.get("left_cols") or [])),
            (spec.get("right", ""),
             tuple(spec.get("right_on") or spec.get("right_cols") or [])),
        )
    )
    correlation = tuple(sorted((spec.get("correlation_cols") or {}).items()))
    return (spec.get("type"), tuple(endpoints), correlation)


def _is_connected(aliases: dict, relationships: list[dict]) -> bool:
    """True when the relationships link every table into one component."""
    parent = {alias: alias for alias in aliases}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for spec in relationships:
        left, right = spec.get("left"), spec.get("right")
        if left in parent and right in parent:
            parent[find(left)] = find(right)

    roots = {find(alias) for alias in aliases}
    return len(roots) == 1


def process_group(
    group: dict, tasks: dict, csv_folder: Path, extension: str
) -> dict | None:
    """
    Build the flat relationship set for one group of tables.

    Every unordered pair of datasets in the group is checked for verified
    tasks — the output is a set of pairwise building blocks with no implied
    composition order: downstream planning may chain them, or build
    independent branches and compare their results.

    The group's ``operation_type`` family (["U"] or ["J", "JC"]) scopes which
    task types are considered for the whole group.

    A group is kept only when its relationships connect every table into one
    component — otherwise some table could never be legally combined.
    """
    datasets   = group["datasets"]
    operations = group["operation_type"]
    aliases    = {f"Table_{i}": datasets[i] for i in range(len(datasets))}

    log = PipelineLogger()
    dfs: dict[str, pd.DataFrame] = {}
    for alias, dataset in aliases.items():
        try:
            dfs[alias] = pd_read_dataset(
                csv_folder / f"{dataset}.{extension}",
                opts={"csv": {"nrows": GROUP_TABLE_NROWS}},
            )
        except Exception as exc:
            log.group_filtered(
                list(aliases.values()), f"could not read {dataset}.{extension} ({exc})"
            )
            return None

    relationships: list[dict] = []
    all_columns = {alias: set() for alias in aliases}
    seen: set[tuple] = set()

    alias_list = list(aliases)
    for i in range(len(alias_list)):
        for j in range(i + 1, len(alias_list)):
            alias_i, alias_j = alias_list[i], alias_list[j]
            ds_i, ds_j = aliases[alias_i], aliases[alias_j]

            for op in operations:
                # Prefer the forward orientation; fall back to the reversed
                # one — a pair contributes at most once per operation type.
                lookups = (
                    ((ds_i, ds_j, op), alias_i, alias_j),
                    ((ds_j, ds_i, op), alias_j, alias_i),
                )
                for key, alias_q, alias_r in lookups:
                    if key not in tasks:
                        continue
                    result = make_match(
                        tasks[key], dfs[alias_q], dfs[alias_r], alias_q, alias_r
                    )
                    if not result:
                        break
                    relationship = result["relationship"]
                    fingerprint = _relationship_fingerprint(relationship)
                    if fingerprint not in seen:
                        seen.add(fingerprint)
                        relationships.append(relationship)
                        for alias, cols in result["columns"].items():
                            all_columns[alias].update(cols)
                    break

    if not relationships:
        log.group_filtered(list(aliases.values()), "no relationship found")
        return None

    if not _is_connected(aliases, relationships):
        log.group_filtered(list(aliases.values()), "relationships do not connect every table")
        return None

    return {
        "aliases":          aliases,
        "relationships":    relationships,
        "columns_by_table": {k: list(v) for k, v in all_columns.items()},
    }


# ── Walk generation ───────────────────────────────────────────────────────────

def generate_walk_groups(cfg: OrQAConfig, G: matches_graph.DatasetMatchesGraph) -> list[dict]:
    """Random-walk the matches graph from each seed and return deduplicated
    table groups as a flat list of ``{seed, operation_type, datasets}``.

    A walk is treated purely as a GROUP of tables (its visit order carries no
    meaning), so walks reaching the same table set under the same operation
    family count once.
    """
    cd = cfg.candidates_discovery
    log = PipelineLogger()

    if cd.seeds_datasets_path.exists():
        with open(cd.seeds_datasets_path) as file:
            seed_ids = json.load(file)
    else:
        from .embedding_discovery.pipeline import get_seed_datasets
        seed_ids = [d[0] for d in get_seed_datasets(cfg)]

    log.walks_start(len(seed_ids))

    groups: list[dict] = []
    seen_groups: set[tuple] = set()

    for dataset_id in seed_ids:
        if dataset_id not in G._G:
            continue

        for path_length in range(1, cd.max_path_length + 1):
            for walk in G.generate_random_walks(
                dataset_id,
                cd.n_paths_for_dataset,
                path_length,
                cd.overlap_ratio_threshold,
                cd.sm_macro_avg_threshold,
                cd.sm_micro_avg_threshold,
                cfg.seed,
            ):
                if len(walk["datasets"]) < 2:
                    continue
                group_key = (
                    frozenset(walk["datasets"]),
                    tuple(sorted(walk["operation_type"])),
                )
                if group_key in seen_groups:
                    continue
                seen_groups.add(group_key)
                groups.append({"seed": dataset_id, **walk})
                log.walk_path(dataset_id, walk["datasets"], walk.get("steps", []))

    by_size = Counter(len(g["datasets"]) for g in groups)
    log.walks_summary(len(groups), dict(by_size))

    return groups


def process_all_candidates(
    groups: list[dict],
    tasks_file: Path,
    csv_folder: Path,
    output_file: Path,
    extension: str,
) -> list[dict]:
    """Process all candidate groups and write query_candidates.json."""
    tasks = load_tasks(tasks_file)
    print(f"✓ {len(tasks)} tasks, {len(groups)} groups")

    results = []
    for group in groups:
        record = process_group(group, tasks, csv_folder, extension)
        if record:
            results.append({"dataset_id": group["seed"], **record})

    save_json(results, output_file)
    print(f"✓ Saved {len(results)} results to {output_file}")
    return results


# ── Entry point (CLI via main.py) ─────────────────────────────────────────────

def generate_query_candidates(cfg: OrQAConfig) -> None:
    cd = cfg.candidates_discovery

    G = matches_graph.DatasetMatchesGraph()
    G.load(cd.matches_graph_path)

    groups = generate_walk_groups(cfg, G)
    save_json(groups, cd.candidates_path)
    print(f"✓ Saved {len(groups)} walk groups to {cd.candidates_path}")

    process_all_candidates(
        groups,
        cd.tasks_results_path,
        cfg.datasets_path,
        cfg.statement_generation.query_candidates_path,
        cfg.datasets_format,
    )
