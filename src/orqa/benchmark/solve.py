"""Round-trip benchmark solver — the orchestration logic (retrieval,
Valentine matching, execution, scoring). The LLM decision points it calls
into live in :class:`orqa.agent.agents.BenchmarkSolver.BenchmarkSolverAgent`.

For each question in ``generated_queries(.json|_semantic.json)``, the solver
is given ONLY the question text — never the hidden ``reference`` (ground-
truth tables/code) — and independently: (1) generates search keywords,
(2) retrieves the top-K candidate tables from the reverse index and relates
them via Valentine matching, (3) decides which candidate(s)/columns answer
the question, (4) writes code, (5) validates + executes it. The result is
then compared against the hidden ground truth two ways: table/column
selection accuracy (:func:`orqa.benchmark.questions.evaluate_table_retrieval`)
and result value/dtype accuracy (:func:`compare_results`, below) — this is
the round-trip/cycle-consistency check described in academic Text-to-SQL
benchmark-construction literature (independently re-derive an answer from
the question alone and check it against the ground truth), applied to
orqa's own generated benchmark.

Replaces the former MCP server (``orqa.benchmark.server``, retired): that
exposed the same retrieval primitives as tools for an EXTERNAL client to
drive by hand, tool call by tool call, with no code-writing, execution or
result reporting. This module runs the whole loop itself, as a workflow
step (``solve_benchmark``, wired to ``--steps solve-benchmark`` in main.py).
"""

from __future__ import annotations

import logging
import numbers
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from orqa.agent.agents.BenchmarkSolver import BenchmarkSolverAgent
from orqa.agent.validators.PandasValidator import PandasValidator
from orqa.agent.validators.SQLValidator import SQLValidator
from orqa.benchmark.index import load_index
from orqa.benchmark.questions import BenchmarkTodo, evaluate_table_retrieval, load_questions
from orqa.queries.query_execution import QueryExecutor
from orqa.schema_matching.valentine_matcher import THRESHOLD, instantiate_matcher
from orqa.utils import select_columns, summarize_large_value
from orqa.utils.pipeline_logger import PipelineLogger

logger = logging.getLogger(__name__)

# Row cap for the candidate-retrieval / Valentine-relationship pass only —
# a quick screen over up to C(top_k_tables, 2) pairs, not the figure that
# feeds the final answer comparison (see _validate_and_execute, which loads
# full, uncapped tables for both the solver's own and the reference
# execution — a row-capped read would make even CORRECT code produce a
# systematically wrong count/sum, corrupting the whole comparison).
# Ported from the retired server.py's same-named/same-value constants.
_RELATIONSHIP_ROW_CAP = 500
_RELATIONSHIP_MAX_COLUMNS = 60

# See orqa.agent.validators.QueryValidator's identical constant: a
# correlation attempted on a degenerate sample (a constant column, or too
# few overlapping rows after a join) makes numpy print a RuntimeWarning for
# a mathematically well-defined NaN result, not an actual problem. The
# nunique() precondition check below already screens out the common case
# (a constant column) before ever calling .corr(); this is the second,
# belt-and-suspenders layer for whatever edge case that check doesn't
# catch (e.g. a near-empty join) — silencing just this message family,
# not RuntimeWarning at large.
_BENIGN_NUMPY_STATS_WARNING = r"(invalid value encountered|divide by zero encountered|Degrees of freedom)"


# ── Ported from the retired orqa.benchmark.server (only consumer left) ─────

def _read_dataset(
    index, resource_id: str, seed: int, row_cap: int, limit_to_n_columns: int
) -> Optional[pd.DataFrame]:
    """Capped read for the candidate/relationship pass: first
    `limit_to_n_columns` columns (file order — the `select_columns`
    convention every other stage uses), first `row_cap` rows. Returns None
    (never raises) when the file can't be read — a single bad candidate
    should never abort the whole retrieval pass."""
    record = index.get(resource_id)
    if record is None:
        return None
    filepath = index.dataset_filepath(resource_id)
    if not filepath.exists():
        return None
    try:
        df = pd.read_csv(
            filepath, nrows=row_cap, sep=None, engine="python",
            encoding_errors="ignore", on_bad_lines="skip",
        )
    except Exception:
        logger.warning("Could not read candidate %r for relationship pass.", resource_id, exc_info=True)
        return None
    df = df[select_columns(list(df.columns), limit_to_n_columns)]
    return df


def _column_info(series: pd.Series) -> dict:
    """One column's descriptive stats for the table-selection prompt: dtype,
    null/distinct counts, and either min/max or the most frequent values."""
    n_null = int(series.isna().sum())
    non_null = series.dropna()
    info: dict[str, Any] = {
        "dtype": str(series.dtype),
        "n_null": n_null,
        "n_distinct": int(non_null.nunique()),
    }
    if non_null.empty:
        return info
    if pd.api.types.is_numeric_dtype(series):
        info["min"] = _jsonable(non_null.min())
        info["max"] = _jsonable(non_null.max())
        return info
    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    if parsed.notna().mean() > 0.9:
        info["min"] = str(parsed.min())
        info["max"] = str(parsed.max())
        return info
    top = non_null.astype(str).value_counts().head(5)
    # summarize_large_value shields an oversized value (e.g. a WKT geometry
    # cell) before it reaches the Phase 2 table-selection prompt — see
    # orqa.utils, whose docstring explains why (the same "creeps into the
    # prompt" failure mode the main pipeline's ColumnStatistics guards).
    info["top_values"] = [
        {"value": summarize_large_value(v), "count": int(c)} for v, c in top.items()
    ]
    return info


def _jsonable(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        value = round(value, 4)
    return value


def _valentine_compare(df_a: pd.DataFrame, df_b: pd.DataFrame, matcher: str, top_k: int) -> dict:
    """Valentine schema matching between two already-loaded frames: ranked
    column matches, the best-key join and its row count, the union column
    ratio, and Pearson correlations between matched numeric columns."""
    from valentine import valentine_match

    raw = valentine_match(df_a, df_b, instantiate_matcher(matcher), "table_a", "table_b")
    matches = sorted(
        (
            {"column_a": col_a, "column_b": col_b, "score": round(score, 4)}
            for ((_, col_a), (_, col_b)), score in raw.items()
        ),
        key=lambda m: m["score"],
        reverse=True,
    )

    join_info = None
    correlations = []
    if matches:
        key_a, key_b = matches[0]["column_a"], matches[0]["column_b"]
        try:
            merged = df_a.merge(df_b, left_on=key_a, right_on=key_b, suffixes=("_a", "_b"))
        except (ValueError, TypeError) as exc:
            join_info = {"left_on": key_a, "right_on": key_b, "error": str(exc)}
        else:
            join_info = {
                "left_on": key_a, "right_on": key_b,
                "score": matches[0]["score"], "joined_rows": int(len(merged)),
            }
            for m in matches[1:]:
                col_a = m["column_a"] if m["column_a"] in merged.columns else f"{m['column_a']}_a"
                col_b = m["column_b"] if m["column_b"] in merged.columns else f"{m['column_b']}_b"
                if col_a not in merged.columns or col_b not in merged.columns:
                    continue
                if not (pd.api.types.is_numeric_dtype(merged[col_a]) and pd.api.types.is_numeric_dtype(merged[col_b])):
                    continue
                # A constant column (zero variance, e.g. every joined row
                # sharing one value on this sample) makes the coefficient
                # mathematically undefined, not just unavailable — pandas
                # computes it anyway (a 0/0 division inside .corr()) and
                # numpy prints a RuntimeWarning for it, even though the
                # result is discarded right below via pd.notna(). Skip the
                # computation for a constant column instead of just
                # swallowing the warning after the fact.
                if merged[col_a].nunique(dropna=True) < 2 or merged[col_b].nunique(dropna=True) < 2:
                    continue
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=RuntimeWarning, message=_BENIGN_NUMPY_STATS_WARNING
                    )
                    pearson = merged[col_a].corr(merged[col_b])
                if pd.notna(pearson):
                    correlations.append({"column_a": m["column_a"], "column_b": m["column_b"], "pearson": round(float(pearson), 4)})

    confident_a = {m["column_a"] for m in matches if m["score"] >= THRESHOLD}
    union_ratio = round(len(confident_a) / len(df_a.columns), 4) if len(df_a.columns) else 0.0

    return {
        "column_matches": matches[:top_k],
        "join": join_info,
        "union_column_ratio": union_ratio,
        "correlations": correlations,
    }


# ── Prompt-facing formatting ────────────────────────────────────────────────

def _format_candidates(records: list[dict], dfs: dict[str, pd.DataFrame]) -> str:
    """One block per ranked candidate: title/tags/publisher plus per-column
    dtype/null/distinct stats — enough for Phase 2 to judge relevance
    without raw rows."""
    blocks = []
    for r in records:
        resource_id = r["resource_id"]
        lines = [f"### {resource_id}", f"Title: {r.get('title') or '(untitled)'}"]
        if r.get("publisher"):
            lines.append(f"Publisher: {r['publisher']}")
        if r.get("tags"):
            lines.append(f"Tags: {', '.join(r['tags'])}")
        df = dfs.get(resource_id)
        if df is None:
            lines.append("(could not be read for column inspection — CSV missing or unreadable)")
        else:
            lines.append("Columns:")
            for col in df.columns:
                info = _column_info(df[col])
                extra = ""
                if "min" in info:
                    extra = f", range {info['min']}..{info['max']}"
                elif "top_values" in info:
                    top = ", ".join(repr(v["value"]) for v in info["top_values"][:3])
                    extra = f", top values: {top}"
                lines.append(f"  - {col} ({info['dtype']}, {info['n_distinct']} distinct, {info['n_null']} null{extra})")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "(no candidates retrieved)"


def _format_relationships(pairs: list[dict]) -> str:
    lines = []
    for p in pairs:
        if p.get("error"):
            continue
        parts = [f"{p['resource_id_a']} <-> {p['resource_id_b']}:"]
        join = p.get("join")
        if join and "score" in join:
            parts.append(f"join key {join['left_on']}={join['right_on']} (score {join['score']}, {join['joined_rows']} rows on the sample)")
        if p.get("union_column_ratio"):
            parts.append(f"union_column_ratio={p['union_column_ratio']}")
        if p.get("top_correlation"):
            tc = p["top_correlation"]
            parts.append(f"correlation {tc['column_a']}~{tc['column_b']}={tc['pearson']}")
        if len(parts) > 1:
            lines.append(" ".join(parts))
    return "\n".join(lines) if lines else "(no supported relationships among these candidates)"


def _format_selected_tables(tables: list[dict], aliases: dict[str, str]) -> str:
    lines = []
    for t in tables:
        alias = aliases[t["resource_id"]]
        lines.append(f"- {alias} (from {t['resource_id']}): columns {t.get('columns_used') or '(all)'}")
    return "\n".join(lines)


# ── Validation + execution ──────────────────────────────────────────────────

def _validate_and_execute(
    question: str,
    tables_used: list[dict],
    code: str,
    expected_result_type: str,
    kind: str,
    dataframes: list[pd.DataFrame],
    table_names: list[str],
    lookup_dict: dict[str, str],
    datasets_path: Path,
    extension: str,
) -> dict:
    """Validate `code` — structural checks plus a SANDBOXED trial execution
    (separate process, timeout, memory limit — see
    orqa.agent.validators.QueryValidator._run_in_sandbox) — before trusting
    it, exactly the same validator class the main generation pipeline uses
    (orqa.agent.agents.StatementValidator._run_static_validator).

    The validator's own return value doesn't expose the executed result
    (only pass/fail + structured errors — see QueryValidator.
    validate_queries), so once it passes, execute again via the bare
    QueryExecutor to actually capture the DataFrame this function returns.
    The second run is cheap relative to validating first and — same code,
    same tables, right after a successful sandboxed trial — essentially
    never diverges from it.

    Returns {"status": "validated" | "validation_error" | "exception",
    "result_df": pd.DataFrame | None, "errors": list[str]}.
    """
    if not code or not code.strip():
        return {"status": "validation_error", "result_df": None, "errors": ["no code produced"]}

    query = {
        "id": 0,
        "question": question,
        "code": code,
        "tables": tables_used,
        "expected_result_type": expected_result_type or "table",
    }
    validator_cls = PandasValidator if kind.upper() == "PANDAS" else SQLValidator
    try:
        validator = validator_cls(dataframes, table_names, lookup_dict)
        all_valid, _legacy, good_queries, errors = validator.validate_queries({"queries": [query]})
    except Exception as exc:
        return {"status": "exception", "result_df": None, "errors": [f"{type(exc).__name__}: {exc}"]}

    if not all_valid or not good_queries:
        return {"status": "validation_error", "result_df": None, "errors": list(errors)}

    validated_query = good_queries[0]
    executor = QueryExecutor(datasets_path, extension)
    try:
        result_df = executor.execute_prepared(validated_query, kind, dict(zip(table_names, dataframes)))
    except Exception as exc:
        return {
            "status": "exception",
            "result_df": None,
            "errors": [f"passed validation but re-execution raised {type(exc).__name__}: {exc}"],
        }
    return {"status": "validated", "result_df": result_df, "errors": []}


def compare_results(reference_df: Optional[pd.DataFrame], candidate_df: Optional[pd.DataFrame]) -> dict:
    """Diagnostic checklist comparing two executed results — independent
    facts, not one collapsed pass/fail verdict, so a failure's actual
    character (wrong type vs. close-but-off vs. completely different)
    stays analyzable. Both sides, when present, are already DataFrames
    (QueryExecutor always coerces a scalar to a one-row/one-column frame).
    """
    if reference_df is None or candidate_df is None:
        return {"comparable": False, "reason": "one or both sides have no executed result"}

    ref_shape, cand_shape = tuple(reference_df.shape), tuple(candidate_df.shape)
    ref_cols, cand_cols = set(map(str, reference_df.columns)), set(map(str, candidate_df.columns))
    shared_cols = sorted(ref_cols & cand_cols)
    same_columns = ref_cols == cand_cols

    dtype_match = bool(shared_cols) and same_columns and all(
        str(reference_df[c].dtype) == str(candidate_df[c].dtype) for c in shared_cols
    )

    exact_match = False
    if same_columns and shared_cols:
        try:
            cols = sorted(shared_cols)
            a = reference_df[cols].sort_values(by=cols).reset_index(drop=True)
            b = candidate_df[cols].sort_values(by=cols).reset_index(drop=True)
            exact_match = bool(a.equals(b))
        except Exception:
            exact_match = False

    # Scalar-shaped results (both 1x1) get a numeric magnitude-of-difference
    # on top of the boolean verdicts above — "how far off", not just
    # "right or wrong", is the figure most useful for failure-mode analysis.
    value_diff = None
    if ref_shape == (1, 1) and cand_shape == (1, 1):
        ref_val, cand_val = reference_df.iloc[0, 0], candidate_df.iloc[0, 0]
        # numbers.Number (not (int, float)) so this also catches numpy
        # int64/float64 scalars — .iloc[0, 0] on a numeric column returns
        # those, not plain Python int/float, and isinstance(x, (int,
        # float)) is False for them.
        is_ref_num = isinstance(ref_val, numbers.Number) and not isinstance(ref_val, bool)
        is_cand_num = isinstance(cand_val, numbers.Number) and not isinstance(cand_val, bool)
        if is_ref_num and is_cand_num:
            abs_diff = abs(float(ref_val) - float(cand_val))
            rel_diff = (abs_diff / abs(float(ref_val))) if ref_val else (0.0 if abs_diff == 0 else None)
            value_diff = {
                "kind": "numeric",
                "absolute": round(abs_diff, 6),
                "relative": round(rel_diff, 6) if rel_diff is not None else None,
            }
        else:
            value_diff = {"kind": "other", "equal": bool(ref_val == cand_val)}

    return {
        "comparable": True,
        "reference_shape": ref_shape,
        "candidate_shape": cand_shape,
        "shape_match": ref_shape == cand_shape,
        "dtype_match": dtype_match,
        "exact_match": exact_match,
        "value_diff": value_diff,
    }


# ── Per-question loop ────────────────────────────────────────────────────────

def solve_one_question(
    record: dict,
    index,
    cfg,
    solver_agent: BenchmarkSolverAgent,
) -> dict:
    """Independently answer ONE benchmark question (never given `record["reference"]`)
    and score it against that hidden ground truth. See the module docstring
    for the full loop."""
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _accumulate(usage: dict) -> None:
        for k in usage_total:
            usage_total[k] += (usage or {}).get(k, 0)

    result: dict = {
        "id": record["id"],
        "question": record["question"],
        "usage": usage_total,
    }

    # Phase 1: keywords, in the question's own language.
    kw_result, kw_usage = solver_agent.generate_keywords(record["question"])
    _accumulate(kw_usage)
    keywords = kw_result.get("keywords") or []
    result["keywords"] = keywords
    result["detected_language"] = kw_result.get("detected_language", "")
    if not keywords:
        result["status"] = "no_keywords"
        result["table_evaluation"] = evaluate_table_retrieval(record["reference"]["tables"], {})
        return result

    # Retrieve top-K candidates.
    search_results = index.search(keywords, top_k=cfg.benchmark_solver.top_k_tables)
    result["candidates"] = [r.resource_id for r in search_results]
    if not search_results:
        result["status"] = "no_candidates"
        result["table_evaluation"] = evaluate_table_retrieval(record["reference"]["tables"], {})
        return result

    # Read each available candidate once (capped), compute pairwise
    # Valentine relationships among them.
    limit_to_n_columns = cfg.candidates_discovery.limit_to_n_columns
    dfs: dict[str, pd.DataFrame] = {}
    skipped_too_wide = []
    for r in search_results:
        if not r.csv_exists:
            continue
        df = _read_dataset(index, r.resource_id, cfg.seed, _RELATIONSHIP_ROW_CAP, limit_to_n_columns)
        if df is None:
            continue
        if df.shape[1] > _RELATIONSHIP_MAX_COLUMNS:
            skipped_too_wide.append(r.resource_id)
            continue
        dfs[r.resource_id] = df

    relationships = []
    available = [r for r in search_results if r.resource_id in dfs]
    for i, a in enumerate(available):
        for b in available[i + 1:]:
            pair = {"resource_id_a": a.resource_id, "resource_id_b": b.resource_id}
            try:
                cmp = _valentine_compare(dfs[a.resource_id], dfs[b.resource_id], cfg.benchmark_solver.matcher, top_k=1)
                pair["join"] = cmp["join"]
                pair["union_column_ratio"] = cmp["union_column_ratio"]
                pair["top_correlation"] = cmp["correlations"][0] if cmp["correlations"] else None
            except Exception as exc:
                pair["error"] = str(exc)
            relationships.append(pair)
    result["relationships_skipped_too_wide"] = skipped_too_wide

    # Phase 2: table/column selection.
    candidate_records = [r.to_dict() for r in search_results]
    ts_result, ts_usage = solver_agent.select_tables(
        record["question"],
        _format_candidates(candidate_records, dfs),
        _format_relationships(relationships),
    )
    _accumulate(ts_usage)
    tables_selected = ts_result.get("tables") or []
    expected_result_type = ts_result.get("expected_result_type", "table")
    result["table_selection"] = ts_result

    solver_tables_for_eval = {
        t["resource_id"]: t.get("columns_used") or [] for t in tables_selected if t.get("resource_id")
    }
    result["table_evaluation"] = evaluate_table_retrieval(record["reference"]["tables"], solver_tables_for_eval)

    if ts_result.get("no_viable_selection") or not tables_selected:
        result["status"] = "no_viable_selection"
        return result

    # Phase 3: code, given Phase 2's selection.
    # alias_by_resource: {resource_id: alias} — this function's own
    # lookup direction (which alias did we assign this candidate).
    # resource_by_alias: {alias: resource_id} — the OPPOSITE orientation
    # QueryExecutor.load_tables and the validator's `lookup_dict` both
    # actually require (same shape as record["reference"]["table_aliases"],
    # which is why the reference side needs no inversion below).
    alias_by_resource = {t["resource_id"]: f"Table_{i}" for i, t in enumerate(tables_selected)}
    resource_by_alias = {alias: resource_id for resource_id, alias in alias_by_resource.items()}
    kind = cfg.statement_generation.kind
    code_result, code_usage = solver_agent.write_code(
        record["question"], kind, expected_result_type,
        _format_selected_tables(tables_selected, alias_by_resource),
    )
    _accumulate(code_usage)
    code = code_result.get("code", "")
    result["code"] = code

    # Load FULL (uncapped rows/columns) tables for both sides' execution —
    # unlike the relationship pass above, a capped read here would make
    # even correct code produce a systematically wrong count/sum.
    executor = QueryExecutor(cfg.datasets_path, cfg.datasets_format)
    try:
        solver_dataframes = executor.load_tables(resource_by_alias)
    except Exception as exc:
        result["status"] = "exception"
        result["execution"] = {"solver": {"status": "exception", "errors": [f"loading selected tables: {exc}"]}}
        return result

    tables_used = [
        {"name": alias_by_resource[t["resource_id"]], "columns_involved": t.get("columns_used") or []}
        for t in tables_selected
    ]
    solver_exec = _validate_and_execute(
        record["question"], tables_used, code, expected_result_type, kind,
        list(solver_dataframes.values()), list(solver_dataframes.keys()), resource_by_alias,
        cfg.datasets_path, cfg.datasets_format,
    )

    reference = record["reference"]
    reference_exec = {"status": "exception", "result_df": None, "errors": ["no reference to execute"]}
    if reference.get("code") and reference.get("table_aliases"):
        try:
            reference_dataframes = executor.load_tables(reference["table_aliases"])
            ref_tables_used = [
                {"name": alias, "columns_involved": reference["tables"].get(resource_id, [])}
                for alias, resource_id in reference["table_aliases"].items()
            ]
            reference_exec = _validate_and_execute(
                record["question"], ref_tables_used, reference["code"],
                reference.get("expected_result_type", "table"), kind,
                list(reference_dataframes.values()), list(reference_dataframes.keys()),
                reference["table_aliases"],
                cfg.datasets_path, cfg.datasets_format,
            )
        except Exception as exc:
            reference_exec = {"status": "exception", "result_df": None, "errors": [str(exc)]}

    result["execution"] = {
        "solver": {"status": solver_exec["status"], "errors": solver_exec["errors"]},
        "reference": {"status": reference_exec["status"], "errors": reference_exec["errors"]},
    }
    if solver_exec["status"] == "validated" and reference_exec["status"] == "validated":
        result["result_evaluation"] = compare_results(reference_exec["result_df"], solver_exec["result_df"])
        result["status"] = "solved"
    else:
        result["status"] = "execution_failed"

    return result


def solve_benchmark(cfg) -> None:
    """Workflow-step entry point (see main.py's `--steps solve-benchmark`):
    prepare the reverse index for the selected city, run every not-yet-
    solved question through the round-trip solver loop, persisting
    incrementally after each one so an interrupted run resumes cleanly."""
    log = PipelineLogger()

    index = load_index(cfg)
    if index is None:
        raise RuntimeError(
            f"Could not build/load the reverse index for {cfg.source!r}; "
            "see the warning above for the underlying error."
        )

    queries_filepath = cfg.statement_generation.queries_path
    if not queries_filepath.exists():
        log.warning(f"No generated questions at {queries_filepath}; run the generate-statements step first.")
        return

    questions = load_questions(
        queries_filepath,
        language=cfg.statement_generation.target_language,
        kind=cfg.statement_generation.kind,
    )
    todo = BenchmarkTodo(cfg.benchmark_results_path)
    todo.sync(questions)
    counts = todo.counts()
    log.benchmark_start(counts["total"], counts["solved"], counts["unsolved"])

    solver_agent = BenchmarkSolverAgent(cfg.llm_config_path / "litellm.yaml")

    this_run_results: list[dict] = []
    for entry in todo.unsolved():
        qid = entry["id"]
        record = todo.get(qid)
        t0 = time.perf_counter()
        try:
            outcome = solve_one_question(record, index, cfg, solver_agent)
        except Exception as exc:
            logger.exception("solve_one_question raised for %s", qid)
            outcome = {"id": qid, "question": record.get("question", ""), "status": "exception", "errors": [str(exc)]}
        elapsed = time.perf_counter() - t0
        outcome["elapsed_s"] = round(elapsed, 1)
        todo.mark_solved(qid, outcome)
        log.benchmark_question(qid, outcome.get("question", ""), outcome, elapsed)
        this_run_results.append(outcome)

    log.benchmark_summary(this_run_results)
