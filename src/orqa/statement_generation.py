import asyncio
import json
import math
import random
import sys
from pathlib import Path
from typing import AsyncGenerator

import pandas as pd

from .agent.agent import StatementGenerationAgent, SingleTableStatementGenerationAgent
from .utils import load_datasets_metadata,load_normalized_datasets_metadata, load_dataset_info, save_json, load_json
from conf import OrQAConfig
from dataclasses import dataclass, field

# Timeout budget: 30 seconds per every 5 000 tokens expected.
# e.g.  5 000 tokens →  30 s
#       10 000 tokens →  60 s
#       25 000 tokens → 150 s
_TIMEOUT_SECONDS_PER_5K_TOKENS: int = 30
_TOKENS_PER_BUCKET: int = 5_000


def _compute_timeout(max_tokens: int) -> float:
    """Return the wall-clock timeout (seconds) for a call capped at *max_tokens*.

    Uses ceiling division so that even a 1-token call gets the full first bucket.
    """
    buckets = math.ceil(max_tokens / _TOKENS_PER_BUCKET)
    return buckets * _TIMEOUT_SECONDS_PER_5K_TOKENS


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

def _make_union_match(task_spec, df_q, df_r, alias_q, alias_r):
    q_columns = task_spec.get("q_columns", [])
    r_columns = task_spec.get("r_columns", [])
    involved = {alias_q: set(q_columns), alias_r: set(r_columns)}
    col_str = f"{q_columns} / {r_columns}" if q_columns or r_columns else "All columns"
    return {
        "description": f"UNION: {alias_q} ∪ {alias_r} ON {col_str}",
        "match_spec": {
            "type": "union",
            "left": alias_q,
            "right": alias_r,
            "left_cols": q_columns,
            "right_cols": r_columns,
        },
        # kept for backward compatibility
        "pandas_expr": f"concat([{alias_q}[{q_columns}], {alias_r}[{r_columns}]], ignore_index=True)",
        "columns": {k: list(v) for k, v in involved.items()},
    }


def _make_join_match(task_spec, df_r, alias_q, alias_r):
    q_keys = task_spec.get("q_columns", [])
    r_keys = task_spec.get("r_columns", [])
    involved = {alias_q: set(q_keys), alias_r: set(r_keys)}
    conditions = " AND ".join(
        f"{alias_q}.{q_keys[i]} = {alias_r}.{r_keys[i]}" for i in range(len(q_keys))
    )
    return {
        "description": f"JOIN: {alias_q} ⋈ {alias_r} ON {conditions}",
        "match_spec": {
            "type": "merge",
            "how": "inner",
            "left": alias_q,
            "right": alias_r,
            "left_on": q_keys,
            "right_on": r_keys,
            "case_insensitive": True,
            "relationship": task_spec.get("relationship", ""),
        },
        # kept for backward compatibility
        "pandas_expr": (
            f"merge(left={alias_q}, right={alias_r}, "
            f"left_on={q_keys}, right_on={r_keys}, "
            f"suffixes=('_{alias_q}', '_{alias_r}'))"
        ),
        "columns": {k: list(v) for k, v in involved.items()},
    }


def _make_join_correlation_match(task_spec, df_q, df_r, alias_q, alias_r):
    q_key    = task_spec["q_key"]
    r_key    = task_spec["r_key"]
    q_target = task_spec["q_target"]
    r_target = task_spec["r_target"]
    involved = {alias_q: {q_key, q_target}, alias_r: {r_key, r_target}}
    return {
        "description": (
            f"JOIN-CORRELATION: merge {alias_q} ⋈ {alias_r} "
            f"ON LOWER({alias_q}.{q_key}) = LOWER({alias_r}.{r_key}), "
            f"then correlate {alias_q}.{q_target} with {alias_r}.{r_target}"
        ),
        "match_spec": {
            "type": "merge_correlation",
            "how": "inner",
            "left": alias_q,
            "right": alias_r,
            "left_on": [q_key],
            "right_on": [r_key],
            "case_insensitive": True,
            "relationship": task_spec.get("relationship", ""),
            "correlation_cols": {alias_q: q_target, alias_r: r_target},
        },
        # kept for backward compatibility
        "pandas_expr": (
            f"{alias_q}.merge({alias_r}, "
            f"left_on={alias_q}['{q_key}'].str.lower(), "
            f"right_on={alias_r}['{r_key}'].str.lower(), "
            f"suffixes=('_{alias_q}', '_{alias_r}'))"
            f"[['{q_target}_{alias_q}', '{r_target}_{alias_r}']].corr()"
        ),
        "columns": {k: list(v) for k, v in involved.items()},
    }


def make_match(task_spec, df_q, df_r, alias_q, alias_r):
    task    = task_spec["task"]
    builder = _MATCH_BUILDERS.get(task)
    if builder is None:
        return None
    if task == "J":
        return builder(task_spec, df_r, alias_q, alias_r)
    return builder(task_spec, df_q, df_r, alias_q, alias_r)


_MATCH_BUILDERS = {
    "U":  _make_union_match,
    "J":  _make_join_match,
    "JC": _make_join_correlation_match,
}


# ── Match formatting ──────────────────────────────────────────────────────────

def _build_rename_map(match_specs: list[dict]) -> dict[str, dict[str, str]]:
    """
    Compute a per-table column rename map for the entire chain upfront.

    For every merge/merge_correlation spec, any column that appears as a join
    key in MORE than one table (i.e. a name collision across tables) is renamed
    to ``<original_name>_<alias>`` on its table before the chain starts.  Key
    columns that are unique across all tables are left as-is.

    Returns:
        { alias: { original_col: renamed_col, ... }, ... }
        Tables with no collisions map to an empty dict.
    """
    # Collect every key column per table across all merge specs
    table_keys: dict[str, set[str]] = {}
    for spec in match_specs:
        mtype = spec.get("type")
        if mtype not in ("merge", "merge_correlation"):
            continue
        for alias, keys in (
            (spec["left"],  spec["left_on"]),
            (spec["right"], spec["right_on"]),
        ):
            table_keys.setdefault(alias, set()).update(keys)

    # Find column names that appear in more than one table
    from collections import Counter
    col_counts: Counter = Counter()
    for keys in table_keys.values():
        col_counts.update(keys)
    colliding = {col for col, count in col_counts.items() if count > 1}

    # Also flag non-key columns that share a name across any two tables
    # (these cause the _x/_y suffix problem even for payload columns)
    all_table_cols: dict[str, set[str]] = {}
    for spec in match_specs:
        mtype = spec.get("type")
        if mtype not in ("merge", "merge_correlation"):
            continue
        for alias in (spec["left"], spec["right"]):
            if alias not in all_table_cols:
                all_table_cols[alias] = set(table_keys.get(alias, set()))
    payload_counts: Counter = Counter()
    for cols in all_table_cols.values():
        payload_counts.update(cols)
    colliding |= {col for col, count in payload_counts.items() if count > 1}

    # Build rename map: only rename columns that actually collide
    rename_map: dict[str, dict[str, str]] = {}
    for alias, cols in all_table_cols.items():
        renames = {}
        suffix = alias.lower().replace(" ", "_")
        for col in cols:
            if col in colliding:
                renames[col] = f"{col}_{suffix}"
        rename_map[alias] = renames

    return rename_map


def format_matches_for_prompt(match_specs: list[dict]) -> str:
    """
    Render a list of match_spec dicts into a clear, LLM-friendly instruction
    block using the **rename-first** pattern.

    Strategy
    --------
    For chained merges (Table_0 → Table_1 → Table_2 → …) pandas produces
    ``_x``/``_y`` suffixes whenever two tables share a column name.  By the
    third table these compound in unpredictable ways.

    The fix is to rename every ambiguous column *before* the chain starts,
    giving each column a globally unique name of the form ``<col>_<alias>``.
    The prompt instructs the LLM to:

      1. Emit one ``.rename()`` call per table (skipped when no collisions).
      2. Build the chain using the renamed key names — no suffixes needed.

    The rename map is computed once by ``_build_rename_map`` and shared across
    all steps so the LLM sees a fully consistent namespace.

    Non-merge operations (UNION) are rendered independently and appended after
    the merge chain block.
    """
    if not match_specs:
        return "(no join operations defined)"

    # ── Separate merge chain from standalone ops (union, etc.) ────────────────
    merge_specs  = [s for s in match_specs if s.get("type") in ("merge", "merge_correlation")]
    other_specs  = [s for s in match_specs if s.get("type") not in ("merge", "merge_correlation")]

    sections: list[str] = []

    # ── Merge chain block ─────────────────────────────────────────────────────
    if merge_specs:
        rename_map = _build_rename_map(merge_specs)
        is_chain   = len(merge_specs) > 1

        header_lines = [
            "MERGE CHAIN" if is_chain else "MERGE",
            "=" * (12 if is_chain else 5),
        ]
        if is_chain:
            order = " → ".join(
                dict.fromkeys(
                    alias
                    for spec in merge_specs
                    for alias in (spec["left"], spec["right"])
                )
            )
            header_lines.append(f"Join order : {order}")

        # ── Step 0: rename instructions ───────────────────────────────────────
        tables_needing_rename = {
            alias: renames
            for alias, renames in rename_map.items()
            if renames
        }
        if tables_needing_rename:
            rename_lines = ["Step 0 — rename ambiguous columns before joining:"]
            for alias, renames in tables_needing_rename.items():
                rename_expr = "{" + ", ".join(f'"{old}": "{new}"' for old, new in sorted(renames.items())) + "}"
                rename_lines.append(f"  {alias} = {alias}.rename(columns={rename_expr})")
            rename_lines.append(
                "  (Use the renamed column names in ALL subsequent steps — "
                "never the originals.)"
            )
            header_lines.append("")
            header_lines.extend(rename_lines)
        else:
            header_lines.append(
                "No column name collisions detected — no renaming required."
            )

        sections.append("\n".join(header_lines))

        # ── One block per merge step ──────────────────────────────────────────
        for step_idx, spec in enumerate(merge_specs, 1):
            mtype = spec.get("type")
            left  = spec["left"]
            right = spec["right"]
            how   = spec.get("how", "inner").upper()
            ci    = spec.get("case_insensitive", False)
            rel   = spec.get("relationship", "")

            # Resolve key names through the rename map
            left_on  = [rename_map.get(left,  {}).get(k, k) for k in spec["left_on"]]
            right_on = [rename_map.get(right, {}).get(k, k) for k in spec["right_on"]]

            step_label = (
                f"Step {step_idx} of {len(merge_specs)}"
                if is_chain else "Operation"
            )
            left_src = "result of previous step" if (is_chain and step_idx > 1) else left

            key_pairs = ", ".join(
                f"{left}.{lk} = {right}.{rk}"
                for lk, rk in zip(left_on, right_on)
            )

            # ── pandas pattern ────────────────────────────────────────────────
            left_ref = "result" if (is_chain and step_idx > 1) else left

            if mtype == "merge_correlation":
                corr_cols = spec.get("correlation_cols", {})
                lk_orig   = spec["left_on"][0]
                rk_orig   = spec["right_on"][0]
                lk        = rename_map.get(left,  {}).get(lk_orig, lk_orig)
                rk        = rename_map.get(right, {}).get(rk_orig, rk_orig)
                l_target_orig = corr_cols.get(left,  "?")
                r_target_orig = corr_cols.get(right, "?")
                l_target  = rename_map.get(left,  {}).get(l_target_orig, l_target_orig)
                r_target  = rename_map.get(right, {}).get(r_target_orig, r_target_orig)

                if ci:
                    pattern = (
                        f"{left_ref}.assign(_key={left_ref}['{lk}'].str.lower())\n"
                        f"    .merge(\n"
                        f"        {right}.assign(_key={right}['{rk}'].str.lower()),\n"
                        f"        on='_key',\n"
                        f"        how='{how.lower()}'\n"
                        f"    )\n"
                        f"    [['{l_target}', '{r_target}']]\n"
                        f"    .corr()"
                    )
                else:
                    pattern = (
                        f"{left_ref}.merge(\n"
                        f"    {right},\n"
                        f"    left_on=['{lk}'],\n"
                        f"    right_on=['{rk}'],\n"
                        f"    how='{how.lower()}'\n"
                        f")\n"
                        f"[['{l_target}', '{r_target}']].corr()"
                    )

                extra = f"   Correlation  : {l_target} (from {left}) vs {r_target} (from {right})"

            else:  # plain merge
                if ci:
                    if len(left_on) == 1:
                        pattern = (
                            f"{left_ref}.assign(_key={left_ref}['{left_on[0]}'].str.lower())\n"
                            f"    .merge(\n"
                            f"        {right}.assign(_key={right}['{right_on[0]}'].str.lower()),\n"
                            f"        on='_key',\n"
                            f"        how='{how.lower()}'\n"
                            f"    )"
                        )
                    else:
                        lk_expr = "{" + ", ".join(f"'_key_{j}': {left_ref}['{k}'].str.lower()" for j, k in enumerate(left_on)) + "}"
                        rk_expr = "{" + ", ".join(f"'_key_{j}': {right}['{k}'].str.lower()" for j, k in enumerate(right_on)) + "}"
                        on_keys = [f"_key_{j}" for j in range(len(left_on))]
                        pattern = (
                            f"{left_ref}.assign(**{lk_expr})\n"
                            f"    .merge(\n"
                            f"        {right}.assign(**{rk_expr}),\n"
                            f"        on={on_keys},\n"
                            f"        how='{how.lower()}'\n"
                            f"    )"
                        )
                else:
                    pattern = (
                        f"{left_ref}.merge(\n"
                        f"    {right},\n"
                        f"    left_on={left_on},\n"
                        f"    right_on={right_on},\n"
                        f"    how='{how.lower()}'\n"
                        f")"
                    )
                extra = ""

            block_lines = [
                f"── {step_label} ──",
                f"   Left        : {left_src}",
                f"   Right       : {right}",
                f"   Condition   : {key_pairs}",
                f"   Join type   : {how}",
                f"   Cardinality : {rel if rel else 'unspecified'}",
                f"   Case-insensitive keys : "
                + ("YES — apply .str.lower() on both sides" if ci else "no"),
            ]
            if extra:
                block_lines.append(extra)
            block_lines.append("   Pattern:")
            block_lines.extend(f"     {line}" for line in pattern.splitlines())

            sections.append("\n".join(block_lines))

    # ── Non-merge operations ──────────────────────────────────────────────────
    for i, spec in enumerate(other_specs, 1):
        mtype = spec.get("type")
        if mtype == "union":
            left       = spec["left"]
            right      = spec["right"]
            left_cols  = spec.get("left_cols",  [])
            right_cols = spec.get("right_cols", [])
            pattern = (
                f"pd.concat(\n"
                f"    [{left}[{left_cols}], {right}[{right_cols}]],\n"
                f"    ignore_index=True\n"
                f")"
            )
            block = (
                f"UNION {i}\n"
                f"   Left table  : {left}   columns: {left_cols}\n"
                f"   Right table : {right}  columns: {right_cols}\n"
                f"   Note        : column names and order must align between both sides\n"
                f"   Pattern:\n"
                + "\n".join(f"     {line}" for line in pattern.splitlines())
            )
            sections.append(block)
        else:
            sections.append(f"UNKNOWN operation type '{mtype}' — skipped")

    return "\n\n".join(sections)


# ── Match processing ──────────────────────────────────────────────────────────

def process_path(path: dict, tasks: dict, csv_folder: Path) -> dict | None:
    datasets   = path["datasets"]
    operations = path["operation_type"]
    aliases    = {f"Table_{i}": datasets[i] for i in range(len(datasets))}
    all_columns = {f"Table_{i}": set() for i in range(len(datasets))}
    path_matches     = []
    path_pandas      = []
    path_match_specs = []   # ← NEW: structured specs for prompt formatting

    for i in range(len(datasets) - 1):
        pair_matches = []
        pair_pandas  = []
        pair_specs   = []   # ← NEW

        df_i  = pd.read_csv(csv_folder / f"{datasets[i]}.csv",     low_memory=False)
        df_i1 = pd.read_csv(csv_folder / f"{datasets[i+1]}.csv",   low_memory=False)

        for op in operations:
            key          = (datasets[i], datasets[i+1], op)
            reversed_key = False
            if key not in tasks:
                key          = (datasets[i+1], datasets[i], op)
                reversed_key = True
            if key not in tasks:
                continue

            if reversed_key:
                result = make_match(tasks[key], df_i1, df_i, f"Table_{i+1}", f"Table_{i}")
            else:
                result = make_match(tasks[key], df_i,  df_i1, f"Table_{i}",  f"Table_{i+1}")

            if result:
                pair_matches.append(result["description"])
                pair_pandas.append(result["pandas_expr"])
                pair_specs.append(result["match_spec"])   # ← NEW
                for alias, cols in result["columns"].items():
                    all_columns[alias].update(cols)

        if not pair_matches:
            print(f"Filtered: no match found for Table_{i} ↔ Table_{i+1}.")
            return None

        path_matches.extend(pair_matches)
        path_pandas.extend(pair_pandas)
        path_match_specs.extend(pair_specs)   # ← NEW

    return {
        "aliases":          aliases,
        "SQL_matches":      path_matches,
        "PANDAS_matches":   path_pandas,
        "match_specs":      path_match_specs,   # ← NEW: used by format_matches_for_prompt
        "columns_by_table": {k: list(v) for k, v in all_columns.items()},
    }


def process_all_candidates(
    candidates_file: Path, tasks_file: Path,
    csv_folder: Path, output_file: Path,
) -> list[dict]:
    """Process all candidate paths and write matches.json."""
    print("Loading...")
    tasks = load_tasks(tasks_file)
    with open(candidates_file) as f:
        candidates = json.load(f)
    print(f"✓ {len(tasks)} tasks, {len(candidates)} datasets")

    results = []
    for dataset_id, path_groups in candidates.items():
        for group in path_groups:
            for path in group["paths"]:
                record = process_path(path, tasks, csv_folder)
                if record:
                    results.append({"dataset_id": dataset_id, **record})

    save_json(results, output_file)
    print(f"✓ Saved {len(results)} results to {output_file}")
    return results


# ── Statement generation ──────────────────────────────────────────────────────

def _build_match_inputs(
    match: dict, csv_folder: Path, datasets_metadata: dict, extension: str
) -> tuple[list[Path], dict, list[dict], dict]:
    """Unpack a match record into agent inputs."""
    dataset_paths, metadatas = [], []
    aliases       = {}
    involved_cols = {}
    for alias, dataset in match["aliases"].items():
        dataset_paths.append(csv_folder / f"{dataset}.{extension}")
        aliases[alias]        = dataset
        metadatas.append(datasets_metadata.get(dataset))
        involved_cols[alias]  = match["columns_by_table"].get(alias, [])
    return dataset_paths, aliases, metadatas, involved_cols

def _is_single_table_candidate(match: dict) -> bool:
    """Check if a candidate has a single dataset and no cross-table match definitions.

    A single-table candidate is identified by:
    - Having exactly one entry in the ``aliases`` mapping
    - Having no SQL or Pandas cross-table match definitions

    Args:
        match: A candidate record loaded from the matches/candidates JSON.

    Returns:
        True if the candidate is a single-table candidate, False otherwise.
    """
    aliases = match.get("aliases", {})
    if len(aliases) != 1:
        return False

    has_sql    = bool(match.get("SQL_matches"))
    has_pandas = bool(match.get("PANDAS_matches"))
    return not has_sql and not has_pandas


def _sample_single_table_datasets(
    datasets_path: Path,
    count: int,
    extension: str = "csv",
) -> list[Path]:
    """Return up to ``count`` randomly sampled dataset paths."""
    all_files = list(datasets_path.glob(f"*.{extension}"))
    return random.sample(all_files, min(count, len(all_files)))


def create_statements(
    config_path: Path,
    csv_folder: Path,
    candidates_file: Path,
    output_file: Path,
    kind: str,
    max_cols: int,
    datasets_metadata: dict | None = None,
    bad_tokens: list | None = None,
    enable_single_table: bool = False,
    single_table_query_count: int = 0,
) -> list[dict]:
    bad_tokens = bad_tokens or []

    if datasets_metadata is None:
        datasets_metadata = {}

    all_matches: list = load_json(candidates_file)
    results = load_json(output_file) if output_file.exists() else {}

    cross_agent = StatementGenerationAgent(config_path, kind, bad_tokens)

    for idx, match in enumerate(all_matches):
        if _is_single_table_candidate(match):
            continue

        dataset_paths, aliases, metadatas, involved_cols = _build_match_inputs(
            match, csv_folder, datasets_metadata, "csv"
        )

        # ── Format matches for prompt ─────────────────────────────────────────
        # Use the structured match_specs when available (new records produced by
        # process_path); fall back to the legacy PANDAS_matches string list for
        # records that pre-date this change.
        if "match_specs" in match and match["match_specs"]:
            formatted_match = format_matches_for_prompt(match["match_specs"])
        else:
            # Legacy fallback: join the raw expression strings
            formatted_match = "\n".join(match.get(f"{kind}_matches", []))

        start = __import__("time").perf_counter()

        content = cross_agent.generate_statements(
            dataset_paths, aliases, kind,
            formatted_match,
            involved_cols, metadatas,
            max_cols, sample_size=5,
        )

        generation_time = __import__("time").perf_counter() - start

        # ── Rate-limit cooldown: 30 s per 5 000 tokens actually consumed ──────
        actual_tokens = sum(content["token_usage"].values()) if isinstance(content["token_usage"], dict) else content["token_usage"]
        cooldown = _compute_timeout(actual_tokens)
        print(f"[{idx}] Consumed {actual_tokens} tokens — cooling down for {cooldown}s.")
        __import__("time").sleep(cooldown)

        result = content["result"]
        model  = content["model"].split("/")[-1]

        results.setdefault(model, {}).setdefault(kind, {})[str(idx)] = {
            "status":          "success" if result.get("queries") else "failure",
            "proposed_columns":content["proposed_columns"],
            "data":            result,
            "tokens":          content["token_usage"],
            "tables":          aliases,
            "errors":          content["errors"],
            "generation_time": generation_time,
            "avg_cols":        content["avg_cols"],
        }
        save_json(results, output_file)
        sys.stdout.flush()

    
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

    # ── load metadata (blocking, run in executor) ─────────────────────────────
    try:
        metadata = await loop.run_in_executor(
            None,
            load_normalized_datasets_metadata,cfg.metadata_path.joinpath("metadata.json"),
            None,
            cfg.source,
        )
        all_matches: list = await loop.run_in_executor(
            None, load_json, cfg.statement_generation.query_candidates_path
        )
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return

    output_file: Path = cfg.statement_generation.queries_path
    results     = load_json(output_file) if output_file.exists() else {}
    total       = len(all_matches)
    successes = failures = 0

    enable_single_table = cfg.statement_generation.enable_single_table
    single_table_query_count = cfg.statement_generation.single_table_query_count

    # ── Cross-table generation ────────────────────────────────────────────────
    cross_agent = StatementGenerationAgent(
        cfg.llm_config_path / "litellm.yaml",
        kind,
        cfg.statement_generation.bad_tokens,
    )
    for idx, match in enumerate(all_matches):
        if idx < resume_from:
            continue

        def _process_cross(match=match):
            dataset_paths, aliases, metadatas, involved_cols = _build_match_inputs(
                match, cfg.datasets_path, metadata, "csv"
            )

            # ── Format matches for prompt ─────────────────────────────────────
            if "match_specs" in match and match["match_specs"]:
                formatted_match = format_matches_for_prompt(match["match_specs"])
            else:
                formatted_match = "\n".join(match.get(f"{kind}_matches", []))

            content = cross_agent.generate_statements(
                dataset_paths, aliases, kind,
                formatted_match,
                involved_cols, metadatas,
                cfg.statement_generation.max_cols, sample_size=5,
            )
            return content, aliases

        try:
            content, aliases = await loop.run_in_executor(None, _process_cross)
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}
            return

        result = content["result"]
        status = "success" if result.get("queries") else "failure"
        model  = content["model"].split("/")[-1]

        if result.get("queries"): successes += 1
        else:                     failures  += 1

        results.setdefault(model, {}).setdefault(kind, {})[str(idx)] = {
            "status":          status,
            "data":            result,
            "tokens":          content["token_usage"],
            "tables":          aliases,
            "errors":          content["errors"],
            "generation_time": content["time_elapsed"],
            "avg_cols":        content["avg_cols"],
        }

        await loop.run_in_executor(None, save_json, results, output_file)

        # ── Rate-limit cooldown: 30 s per 5 000 tokens actually consumed ──────
        actual_tokens = sum(content["token_usage"].values()) if isinstance(content["token_usage"], dict) else content["token_usage"]
        cooldown = _compute_timeout(actual_tokens)
        print(f"[{idx}] Consumed {actual_tokens} tokens — cooling down for {cooldown}s.")
        await asyncio.sleep(cooldown)

        yield {
            "type":        "progress",
            "idx":         idx + 1,
            "total":       total,
            "successes":   successes,
            "failures":    failures,
            "status":      status,
            "aliases":     list(aliases.values()),
            "query_count": len(result.get("queries", [])),
        }

    # ── Single-table generation (random CSV sampling) ─────────────────────────
    if enable_single_table and single_table_query_count:
        sampled = await loop.run_in_executor(
            None, _sample_single_table_datasets,
            cfg.datasets_path, single_table_query_count, "csv",
        )
        st_total = len(sampled)

        single_agent = SingleTableStatementGenerationAgent(
            cfg.llm_config_path / "litellm.yaml",
            kind,
            cfg.statement_generation.bad_tokens,
        )
        for st_idx, csv_path in enumerate(sampled):
            dataset_name = csv_path.stem
            aliases = {"Table_0": dataset_name}

            def _process_single(csv_path=csv_path, aliases=aliases):
                content = single_agent.generate_statements(
                    csv_path, aliases, kind,
                    metadata.get(csv_path.stem),
                    cfg.statement_generation.max_cols, sample_size=5,
                )
                return content

            try:
                content = await loop.run_in_executor(None, _process_single)
            except Exception as exc:
                yield {"type": "error", "message": str(exc)}
                return

            result = content["result"]
            status = "success" if result.get("queries") else "failure"
            model  = content["model"].split("/")[-1]

            if result.get("queries"): successes += 1
            else:                     failures  += 1

            results.setdefault(model, {}).setdefault(kind, {})[f"st_{st_idx}"] = {
                "status":          status,
                "data":            result,
                "tokens":          content["token_usage"],
                "tables":          aliases,
                "errors":          content["errors"],
                "generation_time": content["time_elapsed"],
                "avg_cols":        content["avg_cols"],
            }

            await loop.run_in_executor(None, save_json, results, output_file)

            # ── Rate-limit cooldown: 30 s per 5 000 tokens actually consumed ──
            actual_tokens = sum(content["token_usage"].values()) if isinstance(content["token_usage"], dict) else content["token_usage"]
            cooldown = _compute_timeout(actual_tokens)
            print(f"[st_{st_idx}] Consumed {actual_tokens} tokens — cooling down for {cooldown}s.")
            await asyncio.sleep(cooldown)

            yield {
                "type":        "progress",
                "idx":         total + st_idx + 1,
                "total":       total + st_total,
                "successes":   successes,
                "failures":    failures,
                "status":      status,
                "aliases":     list(aliases.values()),
                "query_count": len(result.get("queries", [])),
            }

    yield {"type": "done", "successes": successes, "failures": failures, "total": total}


# ── Entry point (CLI via main.py) ─────────────────────────────────────────────
from orqa.candidates_generation import generate_random_walks


def generate_statements(cfg: OrQAConfig) -> None:
    generate_random_walks(cfg)
    process_all_candidates(cfg.candidates_discovery.candidates_path, cfg.candidates_discovery.tasks_results_path,cfg.datasets_path, cfg.statement_generation.query_candidates_path)
    metadata = load_normalized_datasets_metadata(cfg.normalized_metadata_filepath)
    for lang in ["SQL","PANDAS"]:
        create_statements(
            cfg.llm_config_path.joinpath("litellm.yaml"),
            cfg.datasets_path,
            cfg.statement_generation.query_candidates_path,
            cfg.statement_generation.queries_path,
            lang,
            cfg.statement_generation.max_cols,
            datasets_metadata=metadata,
            bad_tokens=cfg.statement_generation.bad_tokens,
            enable_single_table=cfg.statement_generation.enable_single_table,
            single_table_query_count=cfg.statement_generation.single_table_query_count,
        )