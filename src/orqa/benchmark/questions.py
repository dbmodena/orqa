"""
Benchmark questions and their todo list.

The generate-statements step stores its output in
<data_path>/candidates_discovery/generated_queries.json. The file and the
benchmark results share the same structure hierarchy:

    kind -> single_table | multi_table -> query id -> question number -> content

    {
      "PANDAS": {
        "single_table": {
          "st_5": {
            "_meta": {"model", "status", "tables", "tokens", ...},
            "0": {question, code, tables, keywords, ...},
            "1": {...}
          }
        },
        "multi_table": {"0": {"_meta": {...}, "0": {...}}}
      },
      "SQL": {...}
    }

`_meta` is a reserved key at the query-id level holding what describes the
whole generation run of that candidate (model, status, token usage, table
aliases, errors, timing); the numbered keys are the individual questions
(see Query/Table in orqa.agent.utility.structured_outputs).

This module flattens that file into benchmark questions addressed by the
flat id "<query id>_<question number>" (e.g. "st_5_0") and keeps a list
of them under <data_path>/benchmark/<kind>/ (the subfolder is the
programming language kind selected in the city's workflow yaml, e.g.
benchmark/pandas or benchmark/sql):

    benchmark/<kind>/solver_results.json   one entry per question id with
                                           "solved": bool and, once solved,
                                           the stored solver `result`
                                           (orqa.benchmark.solve.solve_benchmark)

Questions carry a hidden `reference.tables` and `reference.code` (see
load_questions), checked against the solver's own independent answer by
evaluate_table_retrieval (table/column selection accuracy) and
orqa.benchmark.solve.compare_results (result value/dtype accuracy).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from orqa.utils import load_json, save_json

SECTION_SINGLE = "single_table"
SECTION_MULTI = "multi_table"

# Reserved key at the query-id level for run-wide generation metadata;
# every other key at that level is a question number.
META_KEY = "_meta"


def section_for(query_id: str) -> str:
    """Hierarchy section of a query id ("st_5" -> single_table)."""
    return SECTION_SINGLE if str(query_id).startswith("st_") else SECTION_MULTI


def pack_entry(queries: list[dict], meta: dict) -> dict:
    """Group one candidate's generated queries by question number + _meta."""
    return {META_KEY: meta, **{str(n): q for n, q in enumerate(queries)}}


def store_entry(results: dict, kind: str, query_id: str, queries: list[dict], meta: dict) -> None:
    """Insert one candidate's generation output into the hierarchy."""
    results.setdefault(kind, {}).setdefault(section_for(query_id), {})[
        str(query_id)
    ] = pack_entry(queries, meta)


def get_entry(results: dict, kind: str, query_id: str) -> Optional[dict]:
    """The stored group for a query id, or None if not generated yet."""
    return results.get(kind, {}).get(section_for(query_id), {}).get(str(query_id))


def iter_questions(payload: dict, kind: str) -> Iterator[tuple[str, str, str, dict, dict]]:
    """
    Walk one kind of a hierarchical queries/results file, yielding
    (section, query_id, question_number, question, meta).
    """
    for section, groups in (payload.get(kind) or {}).items():
        for query_id, group in groups.items():
            meta = group.get(META_KEY) or {}
            for qnum, question in group.items():
                if qnum == META_KEY:
                    continue
                yield section, query_id, qnum, question, meta


def match_plan_feedback(meta: dict, question: str) -> Optional[dict]:
    """The plan-judge history whose (final) question produced this query.

    ``_meta.result_extra.plan_feedback`` holds one entry per PLAN in the
    generation run; a query descends from exactly one of them. Matched on
    the question text of the entry itself or of any of its attempts (the
    question may have been rewritten across correction rounds — the query
    carries the FINAL wording, which the last attempt also carries).
    """
    feedback = ((meta.get("result_extra") or {}).get("plan_feedback")) or []
    for entry in feedback:
        if entry.get("question") == question:
            return entry
        for att in entry.get("attempts") or []:
            if att.get("question") == question:
                return entry
    return None


def resolve_task_types(meta: dict, question: str) -> list:
    """A query's ML-skill ``task_types`` (classification/regression/
    timeseries/causal), resolved from its plan-judge history — the plan the
    query was ultimately generated from is the LAST judge attempt's
    ``proposed_plan`` (earlier ones were revised away). Empty list when no
    matching plan-feedback entry exists (e.g. a SQL plan, or an older run
    without plan-level judging recorded).

    This is the single shared resolution the dashboard (``web_app.py``) and
    any other consumer of a query's ML-skill status should use, so they
    never silently diverge on which queries are ML-skill — the same
    one-shared-helper principle ``utils.clean_columns`` already applies to
    ``prepare_dataset``/``QueryExecutor.load_tables``.
    """
    plan_fb = match_plan_feedback(meta or {}, question)
    plan_attempts = (plan_fb or {}).get("attempts") or []
    final_plan = (plan_attempts[-1].get("proposed_plan") if plan_attempts else None) or {}
    return final_plan.get("task_types") or []


def _question_fields(query: dict, language: str) -> dict:
    """
    Pick the question/keywords in the portal's target language.

    Generation always produces the English `question` plus a
    `translated_question` in the detected portal language; for English
    portals the translation is empty or identical, so fall back to the
    plain fields.
    """
    use_translation = language.strip().lower() not in ("english", "eng", "en", "")
    question = (query.get("translated_question") or "").strip()
    keywords = query.get("translated_question_keywords") or []
    if not use_translation or not question:
        question = query.get("question", "")
        keywords = query.get("question_keywords") or []
    return {"question": question, "keywords": keywords}


def _reference_tables(query: dict, entry_tables: dict) -> dict[str, list[str]]:
    """
    Columns used by the reference query, organized by table.

    Query tables are named by alias (Table_0, ...); `entry_tables` maps the
    alias back to the real dataset name.
    """
    tables: dict[str, list[str]] = {}
    for table in query.get("tables") or []:
        alias = table.get("name", "")
        name = entry_tables.get(alias, alias)
        tables[name] = table.get("columns_involved") or []
    if not tables:
        tables = {name: [] for name in entry_tables.values()}
    return tables


def _column_scores(reference_columns: list[str], retrieved_columns: list[str]) -> dict:
    """Precision/recall/F1 of one table's submitted columns against its
    reference columns_involved, exact match (same casing as the dataset)."""
    ref_cols = set(reference_columns or [])
    ret_cols = set(retrieved_columns or [])
    correct_cols = ref_cols & ret_cols

    precision = len(correct_cols) / len(ret_cols) if ret_cols else 0.0
    recall = len(correct_cols) / len(ref_cols) if ref_cols else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "reference_columns": sorted(ref_cols),
        "retrieved_columns": sorted(ret_cols),
        "correct": sorted(correct_cols),
        "missing": sorted(ref_cols - ret_cols),
        "extra": sorted(ret_cols - ref_cols),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_table_retrieval(
    reference_tables: dict[str, list[str]],
    retrieved: dict[str, list[str]],
) -> dict:
    """
    Compare the tables (and, per table, the columns) retrieved for a
    question against its ground truth `reference.tables` (resource_id ->
    columns_involved).

    Table-level: set precision/recall/F1 over resource_ids, plus which ones
    are correct, missing (in the reference but not retrieved) or extra
    (retrieved but not in the reference) — same as before, now keyed off
    `retrieved`'s dict keys instead of a flat id list.

    Column-level: for every correctly retrieved table, precision/recall/F1
    of its submitted columns against that table's reference
    columns_involved (see _column_scores), plus a micro-average across all
    correct tables in `columns_overall`. Tables that are missing or extra
    have no meaningful column comparison and are left out of both.
    """
    reference_ids = set(reference_tables)
    retrieved_ids = set(retrieved)
    correct = reference_ids & retrieved_ids
    missing = reference_ids - retrieved_ids
    extra = retrieved_ids - reference_ids

    precision = len(correct) / len(retrieved_ids) if retrieved_ids else 0.0
    recall = len(correct) / len(reference_ids) if reference_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    columns = {
        table: _column_scores(reference_tables.get(table), retrieved.get(table))
        for table in sorted(correct)
    }
    total_correct_cols = sum(len(c["correct"]) for c in columns.values())
    total_ref_cols = sum(len(c["reference_columns"]) for c in columns.values())
    total_ret_cols = sum(len(c["retrieved_columns"]) for c in columns.values())

    col_precision = total_correct_cols / total_ret_cols if total_ret_cols else 0.0
    col_recall = total_correct_cols / total_ref_cols if total_ref_cols else 0.0
    col_f1 = (
        2 * col_precision * col_recall / (col_precision + col_recall)
        if (col_precision + col_recall)
        else 0.0
    )

    return {
        "reference_tables": sorted(reference_ids),
        "retrieved_tables": sorted(retrieved_ids),
        "correct": sorted(correct),
        "missing": sorted(missing),
        "extra": sorted(extra),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "columns": columns,
        "columns_overall": {
            "precision": round(col_precision, 4),
            "recall": round(col_recall, 4),
            "f1": round(col_f1, 4),
        },
    }


def load_questions(
    queries_filepath: Path,
    language: str = "English",
    kind: str = "PANDAS",
) -> dict[str, dict]:
    """
    Flatten generated_queries.json into {question_id: record}, keeping only
    the questions of the requested statement kind, phrased in the portal's
    target language.

    Flat ids are "<query id>_<question number>" (e.g. "st_5_0"); each record
    also carries its hierarchy coordinates (section / query_id /
    question_number) so results can be stored back in the same structure.
    """
    payload = load_json(queries_filepath)
    questions: dict[str, dict] = {}

    for section, query_id, qnum, query, meta in iter_questions(payload, kind):
        if meta.get("status") != "success":
            continue
        entry_tables = meta.get("tables") or {}
        qid = f"{query_id}_{qnum}"
        questions[qid] = {
            "id": qid,
            "section": section,
            "query_id": query_id,
            "question_number": qnum,
            **_question_fields(query, language),
            "language": language,
            "kind": kind,
            "difficulty": query.get("difficulty", ""),
            # Ground truth from the generation step, kept for later
            # comparison — not the answer the benchmarked agent sees.
            "reference": {
                "model": meta.get("model", ""),
                "tables": _reference_tables(query, entry_tables),
                "code": query.get("code", ""),
                "expected_result_type": query.get("expected_result_type", "table"),
                # alias ("Table_0", ...) -> resource_id, exactly for the
                # aliases `code` above actually references — `tables`
                # above is keyed by resource_id (for evaluate_table_
                # retrieval's comparison against the solver's own
                # resource_id-keyed selection), which loses the alias
                # `code` needs to be executed correctly. Solely for
                # orqa.benchmark.solve's reference-code execution.
                "table_aliases": {
                    t.get("name", ""): entry_tables.get(t.get("name", ""), t.get("name", ""))
                    for t in (query.get("tables") or [])
                },
            },
        }
    return questions


class BenchmarkTodo:
    """
    The persistent list of benchmark questions for one city/kind, read
    fresh from generated_queries(.json|_semantic.json) (whichever the
    workflow yaml's tasks.candidates_discovery.method selects) at
    solve-benchmark startup.

    Progress tracked per entry is "solved" (bool) plus the stored solver
    `result` once solved — solve_benchmark runs straight through every
    not-yet-solved question and persists incrementally after each one, so
    an interrupted run resumes without re-spending LLM calls on questions
    already answered. Survives restarts and re-syncs.
    """

    def __init__(self, results_path: Path):
        self.results_path = Path(results_path)
        self.todo_filepath = self.results_path / "solver_results.json"
        self._entries: dict[str, dict] = (
            load_json(self.todo_filepath) if self.todo_filepath.exists() else {}
        )

    def sync(self, questions: dict[str, dict]) -> None:
        """
        Merge freshly loaded questions into the list: new ids start not
        yet solved; ids already present keep their "solved" flag and
        stored result, only refreshing the question definition fields.
        """
        for qid, record in questions.items():
            existing = self._entries.get(qid)
            if existing:
                existing.update({k: v for k, v in record.items() if k != "id"})
                existing.setdefault("solved", False)
                continue
            self._entries[qid] = {**record, "solved": False}
        self._save()

    def list(self, solved: Optional[bool] = None) -> list[dict]:
        entries = self._entries.values()
        if solved is not None:
            entries = [e for e in entries if bool(e.get("solved")) == solved]
        return [{"id": e["id"], "question": e.get("question", "")} for e in entries]

    def get(self, qid: str) -> Optional[dict]:
        return self._entries.get(qid)

    def unsolved(self) -> list[dict]:
        """Every question not yet solved, in todo-list order."""
        return [e for e in self._entries.values() if not e.get("solved")]

    def mark_solved(self, qid: str, result: dict) -> dict:
        """Store a question's solver result and flag it solved (persisted
        immediately; re-solving an already-solved id overwrites the stored
        result)."""
        entry = self._entries.get(qid)
        if entry is None:
            raise ValueError(f"No question with id {qid!r}")
        entry["solved"] = True
        entry["result"] = result
        self._save()
        return entry

    def counts(self) -> dict[str, int]:
        solved = sum(1 for e in self._entries.values() if e.get("solved"))
        return {
            "total": len(self._entries),
            "solved": solved,
            "unsolved": len(self._entries) - solved,
        }

    def _save(self) -> None:
        save_json(self._entries, self.todo_filepath)
