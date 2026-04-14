import asyncio
import json
import random
import sys
from pathlib import Path
from typing import AsyncGenerator

import pandas as pd

from .agent.agent import StatementGenerationAgent, SingleTableStatementGenerationAgent
from .utils import load_datasets_metadata, load_dataset_info, save_json, load_json
from conf import OrQAConfig
from dataclasses import dataclass, field


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


# ── Match processing ──────────────────────────────────────────────────────────

def process_path(path: dict, tasks: dict, csv_folder: Path) -> dict | None:
    datasets   = path["datasets"]
    operations = path["operation_type"]
    aliases    = {f"Table_{i}": datasets[i] for i in range(len(datasets))}
    all_columns = {f"Table_{i}": set() for i in range(len(datasets))}
    path_matches = []
    path_pandas  = []

    for i in range(len(datasets) - 1):
        pair_matches = []
        pair_pandas  = []

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
                for alias, cols in result["columns"].items():
                    all_columns[alias].update(cols)

        if not pair_matches:
            print(f"Filtered: no match found for Table_{i} ↔ Table_{i+1}.")
            return None

        path_matches.extend(pair_matches)
        path_pandas.extend(pair_pandas)

    return {
        "aliases":          aliases,
        "SQL_matches":      path_matches,
        "PANDAS_matches":   path_pandas,
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

    sql_matches = match.get("SQL_matches", [])
    pandas_matches = match.get("PANDAS_matches", [])
    return len(sql_matches) == 0 and len(pandas_matches) == 0



def _sample_single_table_datasets(
    csv_folder: Path,
    count: int,
    extension: str = "csv",
    seed: int | None = None,
) -> list[Path]:
    """Sample *count* random CSV files from *csv_folder*.

    If fewer files exist than *count*, all available files are returned.
    """
    all_csvs = sorted(csv_folder.glob(f"*.{extension}"))
    if not all_csvs:
        return []
    rng = random.Random(seed)
    return rng.sample(all_csvs, min(count, len(all_csvs)))


def create_statements(
    config_path: Path,
    csv_folder: Path,
    candidates_path: Path,
    output_path: Path,
    kind: str,
    max_cols: int = 15,
    datasets_metadata: dict | None = None,
    extension: str = "csv",
    bad_tokens: list | None = None,
    enable_single_table: bool = False,
    single_table_query_count: int | None = None,
) -> dict:
    """Synchronous batch statement generation (used by main.py CLI)."""
    datasets_metadata = datasets_metadata or {}
    bad_tokens        = bad_tokens or []

    all_matches = load_json(candidates_path)
    output_file = output_path
    results     = load_json(output_file) if output_file.exists() else {}
    total       = len(all_matches)
    successes = failures = 0
    idx = 0
    # ── Single-table generation (random CSV sampling) ─────────────────────────
    if enable_single_table and single_table_query_count:
        sampled = _sample_single_table_datasets(csv_folder, single_table_query_count, extension)
        st_total = len(sampled)
        st_successes = st_failures = 0
        st_failed = ['st_1', 'st_2', 'st_4', 'st_6', 'st_9', 'st_13', 'st_18', 'st_19', 'st_21', 'st_26', 'st_29', 'st_35', 'st_42', 'st_51', 'st_56', 'st_60', 'st_62', 'st_69', 'st_74', 'st_80', 'st_87', 'st_88']

        numeric_failed = ['5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '34', '37', '38', '40', '42', '44', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '92', '93', '94', '95', '96', '97', '98', '99']
        agent = SingleTableStatementGenerationAgent(config_path, kind, bad_tokens)
        for st_idx, csv_path in enumerate(sampled):
            if f"st_{st_idx}" not in st_failed:
                continue
            dataset_name = csv_path.stem
            aliases = {"Table_0": dataset_name}
            metadata = datasets_metadata.get(dataset_name)

            sys.stdout.write(
                f"\r[single-table {st_idx+1}/{st_total}]  "
                f"✅ {st_successes}   ❌ {st_failures}   "
            )

            content = agent.generate_statements(
                csv_path, aliases, kind,
                metadata, max_cols, sample_size=5,
            )

            result          = content["result"]
            status          = "success" if result.get("queries") else "failure"
            model           = content["model"].split("/")[-1]
            generation_time = content["time_elapsed"]

            if result.get("queries"): st_successes += 1
            else:                     st_failures  += 1

            # Store with a "st_" prefix to distinguish from cross-table indices
            results.setdefault(model, {}).setdefault(kind, {})[f"st_{st_idx}"] = {
                "status":          status,
                "data":            result,
                "tokens":          content["token_usage"],
                "tables":          aliases,
                "errors":          content["errors"],
                "generation_time": generation_time,
                "avg_cols":        content["avg_cols"],
            }
            save_json(results, output_file)
            sys.stdout.flush()

        successes += st_successes
        failures  += st_failures

    # ── Cross-table generation ────────────────────────────────────────────────
    agent = StatementGenerationAgent(config_path, kind, bad_tokens)
    for idx, match in enumerate(all_matches):
        if idx not in numeric_failed:
            continue
        sys.stdout.write(
            f"\r[{idx+1}/{total}]  ✅ Successes: {successes}   ❌ Failures: {failures}   "
        )

        dataset_paths, aliases, metadatas, involved_cols = _build_match_inputs(
            match, csv_folder, datasets_metadata, extension
        )
        content = agent.generate_statements(
            dataset_paths, aliases, kind,
            match[f"{kind}_matches"], involved_cols, metadatas,
            max_cols, sample_size=5,
        )

        result          = content["result"]
        status          = "success" if result.get("queries") else "failure"
        model           = content["model"].split("/")[-1]
        generation_time = content["time_elapsed"]

        if result.get("queries"): successes += 1
        else:                     failures  += 1

        results.setdefault(model, {}).setdefault(kind, {})[str(idx)] = {
            "status":          status,
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
            load_datasets_metadata,
            cfg.metadata_path / "metadata.json",
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
            content = cross_agent.generate_statements(
                dataset_paths, aliases, kind,
                match[f"{kind}_matches"], involved_cols, metadatas,
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
    metadata = load_datasets_metadata(
        cfg.metadata_path.joinpath("metadata.json"),
        None,
        source=cfg.source,
    )
    for lang in ["SQL","PANDAS"]:
        create_statements(
            cfg.llm_config_path.joinpath("litellm.yaml"),
            cfg.datasets_path,
            cfg.statement_generation.query_candidates_path,
            cfg.statement_generation.queries_path,
            lang,#cfg.statement_generation.kind,
            cfg.statement_generation.max_cols,
            datasets_metadata=metadata,
            bad_tokens=cfg.statement_generation.bad_tokens,
            enable_single_table=cfg.statement_generation.enable_single_table,
            single_table_query_count=cfg.statement_generation.single_table_query_count,
        )