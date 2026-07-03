"""Shared Pydantic data models for the prompting package.

This module defines the structured-plan and column-statistics models used by the
query planner, column-statistics component, and skill gate. ``QueryLink`` is
reused from :mod:`orqa.agent.structured_outputs` rather than redefined so the
existing plan-link contract is preserved.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel

from ..structured_outputs import QueryLink


class PlanStep(BaseModel):
    """A single ordered step in a structured query plan."""

    order: int  # 1-based execution order
    op: Literal[
        "filter",
        "join",
        "union",
        "group",
        "aggregate",
        "sort",
        "select",
        "derive",
        "classification",
        "regression",
        "timeseries",
        "imputation",
        "causal",
    ]
    description: str  # human-readable intent for this step
    tables: List[str]  # aliases referenced by this step
    columns: List[str]  # concrete columns this step reads/writes
    columns_role: dict = {}  # role->columns, e.g. {"target": ["churn"], "features": [...]}
    params: dict = {}  # op-specific (e.g. {"how": "inner", "keys": [...]})


class StructuredQueryPlan(BaseModel):
    """An ordered, schema-validated decomposition of a query."""

    question: str
    question_keywords: List[str]  # max 10
    plan_keywords: List[str]  # max 10
    steps: List[PlanStep]  # ordered decomposition
    task_types: List[str]  # distinct ops requiring skills (derived from steps)
    table_links: List[QueryLink]  # retained from existing QueryPlan


class ColumnStat(BaseModel):
    """Cheap per-column statistics computed with pandas only (no LLM)."""

    column: str
    dtype: str
    cardinality: int
    null_ratio: float  # 0..1
    numeric_min: Optional[float] = None
    numeric_max: Optional[float] = None
    numeric_mean: Optional[float] = None
    top_values: List[str] = []  # top categorical values (bounded)


class TableStats(BaseModel):
    """Per-table statistics: row count plus per-column statistics."""

    alias: str
    num_rows: int
    columns: List[ColumnStat]


# ---------------------------------------------------------------------------
# Traceable output models
#
# These models supersede the flat ``Query`` in ``structured_outputs.py`` for the
# final assembled result. They capture every phase of the skill-oriented
# generation pipeline (planning, skill selection, validation, judging,
# execution, and usage) so each produced query is fully traceable back to the
# decisions that created it. They are defined here alongside
# :class:`StructuredQueryPlan` so ``assemble_result`` (task 13.2) can import them
# without introducing a circular dependency.
# ---------------------------------------------------------------------------


class TableTrace(BaseModel):
    """Trace of a single table's involvement in an assembled query."""

    name: str
    reason: str
    columns_involved: List[str]
    description: str = ""
    keywords: List[str] = []  # max 10
    translated_keywords: List[str] = []  # max 10


class PlanningTrace(BaseModel):
    """Trace of the planning phase that produced a query."""

    question: str
    question_keywords: List[str] = []
    translated_question: str = ""
    translated_question_keywords: List[str] = []
    difficulty: str
    topic: str
    story: str
    detected_language: str = ""
    query_plan: str = ""
    structured_plan: StructuredQueryPlan
    tables: List[TableTrace]


class SkillTrace(BaseModel):
    """Trace of the skill selected (if any) for a query."""

    skill_used: Optional[str] = None
    skill_version: Optional[int] = None
    reason: str = ""


class ValidationTrace(BaseModel):
    """Trace of the validation phase for a query."""

    passed: bool
    correction_cycles: int
    static_errors: List[str] = []
    final_code: str


class JudgeTrace(BaseModel):
    """Trace of the judging phase for a query."""

    approved: bool
    feedback: str = ""
    suggestions: str = ""
    violated_criteria: List[str] = []
    vagueness_check: str = ""
    requirements_check: str = ""
    response: str = ""
    translated_response: str = ""


class ExecutionTrace(BaseModel):
    """Trace of the execution phase for a query."""

    status: Literal["success", "execution_failure", "empty", "not_run"]
    error: str = ""
    row_count: Optional[int] = None
    elapsed_ms: Optional[float] = None


class UsageTrace(BaseModel):
    """Trace of LLM token usage and timings for a query."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    timings_ms: dict = {}


class TraceableQuery(BaseModel):
    """A single fully-traceable assembled query."""

    id: int
    client_id: str
    code: str
    keyword_count: int = 0
    planning: PlanningTrace
    skill: SkillTrace
    validation: ValidationTrace
    judging: JudgeTrace
    execution: ExecutionTrace
    usage: UsageTrace


class TraceableQuerySet(BaseModel):
    """The final assembled set of traceable queries."""

    queries: List[TraceableQuery]
