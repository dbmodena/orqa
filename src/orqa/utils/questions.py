"""
Benchmark questions as stored by generate-statements.

The generate-statements step stores its output in
<data_path>/candidates_discovery/generated_queries.json, in the hierarchy

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
(see Query/Table in orqa.agent.structured_outputs).
"""

from __future__ import annotations

from typing import Iterator, Optional

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
