import asyncio
import json
import math
import random
import sys
from pathlib import Path
from typing import AsyncGenerator

import pandas as pd

from .agent.agent import StatementGenerationAgent, SingleTableStatementGenerationAgent
from .utils import load_datasets_metadata,load_normalized_datasets_metadata, load_dataset_info, save_json, load_json, prepare_dataframe
from conf import OrQAConfig
from dataclasses import dataclass, field

_TIMEOUT_SECONDS_PER_5K_TOKENS: int = 5
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
        "pandas_expr": f"concat([{alias_q}[{q_columns}], {alias_r}[{r_columns}]], ignore_index=True)",
        "columns": {k: list(v) for k, v in involved.items()},
    }


def _make_join_match(task_spec, df_r, alias_q, alias_r, df_q=None):
    q_keys = task_spec.get("q_columns", [])
    r_keys = task_spec.get("r_columns", [])
    involved = {alias_q: set(q_keys), alias_r: set(r_keys)}
    conditions = " AND ".join(
        f"{alias_q}.{q_keys[i]} = {alias_r}.{r_keys[i]}" for i in range(len(q_keys))
    )
    ci = _resolve_case_insensitive(
        task_spec,
        {alias_q: df_q, alias_r: df_r} if df_q is not None else {alias_r: df_r},
        alias_q,
        alias_r,
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
            "case_insensitive": ci,
            "relationship": task_spec.get("relationship", ""),
        },
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
    ci = _resolve_case_insensitive(
        {"case_insensitive": task_spec.get("case_insensitive", False),
         "q_columns": [q_key], "r_columns": [r_key]},
        {alias_q: df_q, alias_r: df_r},
        alias_q,
        alias_r,
    )
    return {
        "description": (
            f"JOIN-CORRELATION: merge {alias_q} ⋈ {alias_r} "
            f"ON {alias_q}.{q_key} = {alias_r}.{r_key}"
            + (" [case-insensitive]" if ci else "")
            + f", then correlate {alias_q}.{q_target} with {alias_r}.{r_target}"
        ),
        "match_spec": {
            "type": "merge_correlation",
            "how": "inner",
            "left": alias_q,
            "right": alias_r,
            "left_on": [q_key],
            "right_on": [r_key],
            "case_insensitive": ci,
            "relationship": task_spec.get("relationship", ""),
            "correlation_cols": {alias_q: q_target, alias_r: r_target},
        },
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
    task = task_spec["task"]
    if task == "U":
        return _make_union_match(task_spec, df_q, df_r, alias_q, alias_r)
    if task == "J":
        return _make_join_match(task_spec, df_r, alias_q, alias_r, df_q=df_q)
    if task == "JC":
        return _make_join_correlation_match(task_spec, df_q, df_r, alias_q, alias_r)
    return None


# ── Match formatting ──────────────────────────────────────────────────────────

def _format_matches_sql(match_specs: list[dict]) -> str:
    """
    Compact SQL-oriented description: one line per operation.
    No pandas patterns, no rename maps, no chain boilerplate.
    LOWER() note is included inline only when case_insensitive=True.
    """
    lines = []
    for spec in match_specs:
        mtype = spec.get("type")

        if mtype in ("merge", "merge_correlation"):
            left, right = spec["left"], spec["right"]
            left_on     = spec.get("left_on",  [])
            right_on    = spec.get("right_on", [])
            ci          = spec.get("case_insensitive", False)

            pairs = "  AND  ".join(
                f"{left}.{lk} = {right}.{rk}"
                for lk, rk in zip(left_on, right_on)
            )
            ci_note = "  [case-insensitive — wrap keys in LOWER()]" if ci else ""

            if mtype == "merge_correlation":
                corr = spec.get("correlation_cols", {})
                l_t  = corr.get(left,  "?")
                r_t  = corr.get(right, "?")
                lines.append(
                    f"JOIN  {pairs}{ci_note}\n"
                    f"  → correlate {left}.{l_t} with {right}.{r_t} after joining"
                )
            else:
                rel     = spec.get("relationship", "")
                rel_note = f"  [{rel}]" if rel else ""
                lines.append(f"JOIN  {pairs}{ci_note}{rel_note}")

        elif mtype == "union":
            left      = spec["left"]
            right     = spec["right"]
            left_cols  = spec.get("left_cols",  [])
            right_cols = spec.get("right_cols", [])
            lines.append(
                f"UNION  {left}{left_cols} ∪ {right}{right_cols}"
                "  (align columns by position)"
            )
        else:
            lines.append(f"# unknown operation type '{mtype}' — skipped")

    return "\n".join(lines)


def _sample_col(df: pd.DataFrame, col: str, n: int = 3) -> list:
    """Return up to n distinct non-null sample values from a column."""
    if col not in df.columns:
        return []
    vals = df[col].dropna().unique()
    return [v.item() if hasattr(v, "item") else v for v in vals[:n]]


def _format_matches_pandas(
    match_specs: list[dict],
    dfs: dict[str, pd.DataFrame] | None = None,
    involved_cols: dict[str, list[str]] | None = None,
    sample_n: int = 3,
) -> str:
    """
    Schema-first declarative prompt block for PANDAS generation.

    Tells the LLM *what* to join and *what data looks like* — not how pandas
    works.  The LLM already knows pandas; it just needs intent + shape.

    Output sections
    ---------------
    1. Links      — one line per join/union showing which columns connect tables
    2. Schema     — columns available per table, join keys marked with *,
                    correlation targets marked with ~
    3. Samples    — 3 representative values per relevant column (skipped when
                    no dataframes are supplied)

    Args:
        match_specs:   list of match_spec dicts from make_match()
        dfs:           {alias: DataFrame} — when provided, sample values are
                       included; pass None to omit the Samples section
        involved_cols: {alias: [col, ...]} — non-key columns the LLM may use;
                       when None, all columns in the df are listed
        sample_n:      number of sample values per column
    """
    if not match_specs:
        return "(no operations defined)"

    dfs           = dfs or {}
    involved_cols = involved_cols or {}

    # ── Collect per-table metadata in one pass ────────────────────────────────
    # key_cols[alias]  → set of join-key column names
    # corr_cols[alias] → set of correlation-target column names
    key_cols:  dict[str, set[str]] = {}
    corr_cols: dict[str, set[str]] = {}
    links: list[str] = []

    for spec in match_specs:
        mtype = spec.get("type")
        left, right = spec.get("left", "?"), spec.get("right", "?")

        if mtype in ("merge", "merge_correlation"):
            left_on  = spec.get("left_on",  [])
            right_on = spec.get("right_on", [])
            ci       = spec.get("case_insensitive", False)
            how      = spec.get("how", "inner").upper()

            key_cols.setdefault(left,  set()).update(left_on)
            key_cols.setdefault(right, set()).update(right_on)

            pairs    = ", ".join(f"{lk}={rk}" for lk, rk in zip(left_on, right_on))
            ci_note  = " [ci]" if ci else ""
            how_note = f" [{how}]" if how != "INNER" else ""
            link     = f"{left} → {right}  on {pairs}{ci_note}{how_note}"

            if mtype == "merge_correlation":
                cc = spec.get("correlation_cols", {})
                lt, rt = cc.get(left, "?"), cc.get(right, "?")
                corr_cols.setdefault(left,  set()).add(lt)
                corr_cols.setdefault(right, set()).add(rt)
                link += f"  correlate {lt}~{rt}"

            links.append(link)

        elif mtype == "union":
            lc = spec.get("left_cols",  [])
            rc = spec.get("right_cols", [])
            links.append(f"{left} ∪ {right}  columns {lc} → {rc}")

    # ── Section 1: Links ──────────────────────────────────────────────────────
    sections = ["Links:\n" + "\n".join(f"  {l}" for l in links)]

    # ── Section 2: Schema ─────────────────────────────────────────────────────
    # Gather all aliases mentioned across all specs (preserving order)
    seen: dict[str, None] = {}
    for spec in match_specs:
        for alias in (spec.get("left"), spec.get("right")):
            if alias:
                seen[alias] = None
    all_aliases = list(seen)

    schema_lines = ["Schema  (* = join key, ~ = correlation target):"]
    for alias in all_aliases:
        df = dfs.get(alias)
        # Columns to show: involved_cols when provided, else all df columns
        if involved_cols.get(alias):
            cols = involved_cols[alias]
        elif df is not None:
            cols = list(df.columns)
        else:
            # Fall back to whatever we know from the specs
            cols = list(
                key_cols.get(alias, set()) | corr_cols.get(alias, set())
            )

        annotated = []
        for c in cols:
            marker = ""
            if c in key_cols.get(alias, set()):
                marker = "*"
            elif c in corr_cols.get(alias, set()):
                marker = "~"
            annotated.append(f"{marker}{c}" if marker else c)

        schema_lines.append(f"  {alias}: {', '.join(annotated)}")

    sections.append("\n".join(schema_lines))

    # ── Section 3: Samples (only when dataframes are available) ──────────────
    if dfs:
        sample_lines = [f"Samples (up to {sample_n} distinct values):"]
        for alias in all_aliases:
            df = dfs.get(alias)
            if df is None:
                continue
            # Only sample columns that are relevant (keys, corr targets, involved)
            relevant = (
                key_cols.get(alias, set())
                | corr_cols.get(alias, set())
                | set(involved_cols.get(alias, []))
            )
            for col in relevant:
                vals = _sample_col(df, col, sample_n)
                if vals:
                    sample_lines.append(f"  {alias}.{col}: {vals}")

        if len(sample_lines) > 1:   # at least one column had values
            sections.append("\n".join(sample_lines))

    return "\n\n".join(sections)


def format_matches_for_prompt(
    match_specs: list[dict],
    kind: str = "PANDAS",
    dfs: dict | None = None,
    involved_cols: dict | None = None,
    sample_n: int = 3,
) -> str:
    """
    Return a compact, LLM-readable description of the match operations.

    Args:
        match_specs:   list of match_spec dicts produced by make_match().
        kind:          "SQL" or "PANDAS" — controls which formatter is used.
        dfs:           {alias: DataFrame} — PANDAS only; enables sample values
                       section.  Pass None to omit samples.
        involved_cols: {alias: [col, ...]} — PANDAS only; restricts the schema
                       section to columns the LLM should actually use.
        sample_n:      number of sample values per column in the Samples block.
    """
    if not match_specs:
        return "(no join operations defined)"
    if kind.upper() == "SQL":
        return _format_matches_sql(match_specs)
    return _format_matches_pandas(match_specs, dfs=dfs, involved_cols=involved_cols, sample_n=sample_n)


# ── Match processing ──────────────────────────────────────────────────────────

def process_path(path: dict, tasks: dict, csv_folder: Path) -> dict | None:
    datasets   = path["datasets"]
    operations = path["operation_type"]
    aliases    = {f"Table_{i}": datasets[i] for i in range(len(datasets))}
    all_columns = {f"Table_{i}": set() for i in range(len(datasets))}
    path_matches     = []
    path_pandas      = []
    path_match_specs = []

    for i in range(len(datasets) - 1):
        pair_matches = []
        pair_pandas  = []
        pair_specs   = []

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
                pair_specs.append(result["match_spec"])
                for alias, cols in result["columns"].items():
                    all_columns[alias].update(cols)

        if not pair_matches:
            print(f"Filtered: no match found for Table_{i} ↔ Table_{i+1}.")
            return None

        path_matches.extend(pair_matches)
        path_pandas.extend(pair_pandas)
        path_match_specs.extend(pair_specs)

    return {
        "aliases":          aliases,
        "SQL_matches":      path_matches,
        "PANDAS_matches":   path_pandas,
        "match_specs":      path_match_specs,
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
    """
    Unpack a match record into agent inputs.

    Returns
    -------
    dataset_paths, aliases, metadatas, involved_cols, dfs

    ``dfs`` is a {alias: DataFrame} mapping loaded here so the pandas
    formatter can pull sample values without a second CSV read.
    """
    dataset_paths, metadatas = [], []
    aliases       = {}
    involved_cols = {}
    dfs           = {}
    for alias, dataset in match["aliases"].items():
        path = csv_folder / f"{dataset}.{extension}"
        dataset_paths.append(path)
        aliases[alias]        = dataset
        metadatas.append(datasets_metadata.get(dataset))
        involved_cols[alias]  = match["columns_by_table"].get(alias, [])
        try:
            dfs[alias] = prepare_dataframe(pd.read_csv(path, low_memory=False), alias=alias)
        except Exception:
            pass   # formatter degrades gracefully without the df
    return dataset_paths, aliases, metadatas, involved_cols, dfs

def _is_single_table_candidate(match: dict) -> bool:
    """Check if a candidate has a single dataset and no cross-table match definitions."""
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


def _get_formatted_match(
    match: dict,
    kind: str,
    dfs: dict | None = None,
    involved_cols: dict | None = None,
) -> str:
    """
    Return a formatted match string for the given kind (SQL or PANDAS).
    Uses structured match_specs when available; falls back to legacy string lists.
    ``dfs`` and ``involved_cols`` are forwarded to the PANDAS formatter only.
    """
    if "match_specs" in match and match["match_specs"]:
        return format_matches_for_prompt(
            match["match_specs"], kind=kind, dfs=dfs, involved_cols=involved_cols
        )
    return "\n".join(match.get(f"{kind}_matches", []))


def _already_succeeded(results: dict, kind: str, idx: str) -> bool:
    """Return True if any model already has a 'success' entry for this idx and kind."""
    for model_data in results.values():
        entry = model_data.get(kind, {}).get(idx)
        if entry and entry.get("status") == "success":
            return True
    return False


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
    languages: list= ["English"]
) -> list[dict]:
    bad_tokens = bad_tokens or []

    if datasets_metadata is None:
        datasets_metadata = {}

    all_matches: list = load_json(candidates_file)
    results = load_json(output_file) if output_file.exists() else {}

    cross_agent = StatementGenerationAgent(config_path, kind, bad_tokens, languages=languages)

    # ── Cross-table generation ────────────────────────────────────────────────
    for idx, match in enumerate(all_matches):
        if _is_single_table_candidate(match):
            continue

        str_idx = str(idx)
        if _already_succeeded(results, kind, str_idx):
            print(f"[{idx}] Already succeeded — skipping.")
            continue

        dataset_paths, aliases, metadatas, involved_cols, dfs = _build_match_inputs(
            match, csv_folder, datasets_metadata, "csv"
        )

        formatted_match = _get_formatted_match(match, kind, dfs=dfs, involved_cols=involved_cols)

        start = __import__("time").perf_counter()

        content = cross_agent.generate_statements(
            dataset_paths, aliases, kind,
            formatted_match,
            involved_cols, metadatas,
            max_cols, sample_size=5
        )

        generation_time = __import__("time").perf_counter() - start

        actual_tokens = sum(content["token_usage"].values()) if isinstance(content["token_usage"], dict) else content["token_usage"]
        cooldown = _compute_timeout(actual_tokens)
        print(f"[{idx}] Consumed {actual_tokens} tokens — cooling down for {cooldown}s.")
        __import__("time").sleep(cooldown)

        result = content["result"]
        model  = content["model"].split("/")[-1]

        results.setdefault(model, {}).setdefault(kind, {})[str_idx] = {
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

    # ── Single-table generation ───────────────────────────────────────────────
    if enable_single_table and single_table_query_count > 0:
        print(f"\nStarting single-table generation ({single_table_query_count} datasets)...")
        sampled = _sample_single_table_datasets(csv_folder, single_table_query_count)
        single_agent = SingleTableStatementGenerationAgent(
            config_path, kind, bad_tokens, languages=languages
        )

        for st_idx, csv_path in enumerate(sampled):
            dataset_name = csv_path.stem
            aliases = {"Table_0": dataset_name}
            str_idx = f"st_{st_idx}"

            if _already_succeeded(results, kind, str_idx):
                print(f"[st_{st_idx}] Already succeeded — skipping.")
                continue

            start = __import__("time").perf_counter()

            content = single_agent.generate_statements(
                csv_path, aliases, kind,
                datasets_metadata.get(dataset_name),
                max_cols, sample_size=5,
            )

            generation_time = __import__("time").perf_counter() - start

            actual_tokens = sum(content["token_usage"].values()) if isinstance(content["token_usage"], dict) else content["token_usage"]
            cooldown = _compute_timeout(actual_tokens)
            print(f"[st_{st_idx}] Consumed {actual_tokens} tokens — cooling down for {cooldown}s.")
            __import__("time").sleep(cooldown)

            result = content["result"]
            model  = content["model"].split("/")[-1]

            results.setdefault(model, {}).setdefault(kind, {})[str_idx] = {
                "status":          "success" if result.get("queries") else "failure",
                "proposed_columns":content.get("proposed_columns"),
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
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return

    output_file: Path = cfg.statement_generation.queries_path
    results     = load_json(output_file) if output_file.exists() else {}
    total       = len(all_matches)
    successes = failures = 0

    enable_single_table      = cfg.statement_generation.enable_single_table
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
            dataset_paths, aliases, metadatas, involved_cols, dfs = _build_match_inputs(
                match, cfg.datasets_path, metadata, "csv"
            )
            formatted_match = _get_formatted_match(match, kind, dfs=dfs, involved_cols=involved_cols)
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
    #generate_random_walks(cfg)
    #process_all_candidates(
    #    cfg.candidates_discovery.candidates_path,
    #    cfg.candidates_discovery.tasks_results_path,
    #    cfg.datasets_path,
    #    cfg.statement_generation.query_candidates_path,
    #)
    metadata = load_normalized_datasets_metadata(cfg.normalized_metadata_filepath)
    for lang in ["SQL", "PANDAS"]:
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
            languages=cfg.statement_generation.detected_languages
        )