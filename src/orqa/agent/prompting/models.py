"""Shared Pydantic data models for the prompting package.

This module defines the structured-plan and column-statistics models used by the
query planner, column-statistics component, and skill gate. ``QueryLink`` is
reused from :mod:`orqa.agent.utility.structured_outputs` rather than redefined
so the existing plan-link contract is preserved.
"""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field

from ..utility.structured_outputs import QueryLink, Table


# Ops available to every plan (SQL and Pandas alike): plain relational/derive steps.
_SQL_OPS = Literal[
    "filter", "join", "union", "group", "aggregate", "sort", "select", "derive",
]

# ML task types available only to Pandas plans. Each one has a consolidated
# ``conf/skills/{task_type}.md``, structured into a "Generation" section
# (see ``prompts.load_skill_section``), injected unconditionally whenever it
# appears in a plan's ``task_types``, plus "Plan Judge Check"/"Code Judge
# Check" sections the plan/code judge panels draw on independently.
_ML_TASK_TYPES = Literal[
    "classification", "regression", "timeseries", "causal",
]

# The coarse SHAPE a plan promises its final result will have. Decided at
# planning time (a fifth thing the plan judge reviews), shown to the
# generation model as part of the plan, and mechanically enforced against
# the actually-executed result by the validators (see
# QueryValidator._check_expected_result_type) — for SQL and Pandas alike:
#   * "table"   — tabular data: multiple columns and/or multiple rows
#                 (a DataFrame; an indexed per-group Series also qualifies).
#   * "list"    — one ordered sequence of values (a Series/single column).
#   * "number"  — one numeric figure (count, total, average, ...).
#   * "text"    — one string value (a name, a category label, ...).
#   * "boolean" — one yes/no answer.
_RESULT_TYPES = Literal["table", "list", "number", "text", "boolean"]

_EXPECTED_RESULT_TYPE_DESCRIPTION = (
    "The SHAPE of the final answer this plan's code must produce, matching "
    "what the question asks for: 'number' for \"how many/how much...\", "
    "'boolean' for a yes/no question, 'text' for \"which single X...\", "
    "'list' for \"list the values of...\", 'table' for per-group breakdowns, "
    "rankings, or any multi-column result. This is ENFORCED: the executed "
    "result is mechanically checked against it, and a mismatch rejects the "
    "code."
)

_EXPECTED_RESULT_DESCRIPTION_DESCRIPTION = (
    "One or two sentences describing the expected result CONCRETELY: what "
    "the value(s) represent, their unit/granularity, and — for 'table'/"
    "'list' — what each row and column holds (e.g. 'one row per borough "
    "with the total number of permits issued there in 2023, sorted "
    "descending'). The code generator reads this to shape its final "
    "result variable, and the plan judges review it for consistency with "
    "the question."
)

# Shared description for `tables` on both plan types: reuses the same
# ``Table`` model the generation-time ``Query.tables`` uses, but at planning
# time ``reason`` is the authoritative, panel-judged justification for the
# table's inclusion (see PlanJudgment.table_check) — decided exactly once,
# here, never re-argued at generation or query-judgement time — so the bar
# is a concrete, articulated argument per table, not a generic one-liner.
# Coverage (every provided alias present exactly once) is enforced in code —
# see QueryPlanner.validate_plan — not left to the LLM's discretion.
_PLAN_TABLES_DESCRIPTION = (
    "One entry per table THIS question uses — EVERY alias provided in TABLE "
    "ALIASES must appear here exactly once, no more, no fewer (checked "
    "before any code is written; an omitted or invented alias fails "
    "validation immediately). Each entry's `reason` must be a "
    "well-articulated, motivated justification (2-3 sentences, not a "
    "generic one-liner) of why THIS question needs that table: its "
    "concrete role in the plan (which step(s) it feeds), the specific "
    "rows/columns/filtering the answer depends on it for, and why the "
    "question could not be answered without it. A judge panel rejects the "
    "plan if any justification is vague, generic, or does not hold — a "
    "justification that could be pasted under a different, unrelated table "
    "and still sound plausible is not articulated enough. Never 'for "
    "context', 'for completeness', or 'to enrich the analysis'. "
    "`columns_involved` is the minimal columns from that table this plan's "
    "steps actually use."
)


class SQLPlanStep(BaseModel):
    """A single ordered step in a SQL structured query plan.

    Structurally excludes ML ops — a SQL plan can never require a skill.
    """

    order: int  # 1-based execution order
    op: _SQL_OPS
    description: str  # human-readable intent for this step
    tables: List[str]  # aliases referenced by this step
    columns: List[str]  # concrete columns this step reads/writes
    columns_role: dict = {}  # role->columns, e.g. {"target": ["churn"], "features": [...]}
    params: dict = {}  # op-specific (e.g. {"how": "inner", "keys": [...]})


class PandasPlanStep(BaseModel):
    """A single ordered step in a Pandas structured query plan.

    Allows the plain relational ops plus the ML task types.
    """

    order: int  # 1-based execution order
    op: Literal[
        "filter", "join", "union", "group", "aggregate", "sort", "select", "derive",
        "classification", "regression", "timeseries", "causal",
    ]
    description: str  # human-readable intent for this step
    tables: List[str]  # aliases referenced by this step
    columns: List[str]  # concrete columns this step reads/writes
    columns_role: dict = {}  # role->columns, e.g. {"target": ["churn"], "features": [...]}
    params: dict = {}  # op-specific (e.g. {"how": "inner", "keys": [...]})


class SQLQueryPlan(BaseModel):
    """An ordered, schema-validated decomposition of a SQL query. No ``task_types``:
    a SQL plan never requires a skill."""

    question: str
    question_keywords: List[str]  # max 10
    plan_keywords: List[str]  # max 10
    steps: List[SQLPlanStep]  # ordered decomposition
    expected_result_type: _RESULT_TYPES = Field(
        default="table", description=_EXPECTED_RESULT_TYPE_DESCRIPTION
    )
    expected_result_description: str = Field(
        default="", description=_EXPECTED_RESULT_DESCRIPTION_DESCRIPTION
    )
    tables: List[Table] = Field(
        default_factory=list, description=_PLAN_TABLES_DESCRIPTION
    )
    table_links: List[QueryLink]  # retained from existing QueryPlan
    # Question-level metadata the planner owns (moved off the generation-time
    # Query model so it's decided once here, not reinvented per generation
    # call — see StatementClient.complete's plan-fields merge).
    query_plan: str = ""  # high-level natural-language description of the plan
    translated_question: str = ""
    translated_question_keywords: List[str] = []  # max 10
    detected_language: str = ""
    topic: str = ""
    story: str = ""


class PandasQueryPlan(BaseModel):
    """An ordered, schema-validated decomposition of a Pandas query."""

    question: str
    question_keywords: List[str]  # max 10
    plan_keywords: List[str]  # max 10
    steps: List[PandasPlanStep]  # ordered decomposition
    task_types: List[_ML_TASK_TYPES]  # distinct ML ops requiring a skill (derived from steps)
    expected_result_type: _RESULT_TYPES = Field(
        default="table", description=_EXPECTED_RESULT_TYPE_DESCRIPTION
    )
    expected_result_description: str = Field(
        default="", description=_EXPECTED_RESULT_DESCRIPTION_DESCRIPTION
    )
    tables: List[Table] = Field(
        default_factory=list, description=_PLAN_TABLES_DESCRIPTION
    )
    table_links: List[QueryLink]  # retained from existing QueryPlan
    # Question-level metadata the planner owns (moved off the generation-time
    # Query model so it's decided once here, not reinvented per generation
    # call — see StatementClient.complete's plan-fields merge).
    query_plan: str = ""  # high-level natural-language description of the plan
    translated_question: str = ""
    translated_question_keywords: List[str] = []  # max 10
    detected_language: str = ""
    topic: str = ""
    story: str = ""


class SQLQueryPlanSet(BaseModel):
    """A batch of independent SQL query plans produced from the same table analyses."""

    plans: List[SQLQueryPlan]


class PandasQueryPlanSet(BaseModel):
    """A batch of independent Pandas query plans produced from the same table analyses.

    Each entry is a fully independent :class:`PandasQueryPlan`: its own
    ``question``/``question_keywords`` and its own ``task_types`` derived from
    its own ``steps``. Plans in the same set commonly need *different* skills
    (e.g. one plan is a plain aggregation, another needs ``classification``),
    which is exactly why the per-task-type skill markdown is injected once per
    plan rather than once for the whole set.
    """

    plans: List[PandasQueryPlan]


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
# generation pipeline (planning, skill injection, validation, judging,
# execution, and usage) so each produced query is fully traceable back to the
# decisions that created it. They are defined here alongside
# :class:`SQLQueryPlan`/:class:`PandasQueryPlan` so ``assemble_result`` (task
# 13.2) can import them without introducing a circular dependency.
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
    structured_plan: Union[SQLQueryPlan, PandasQueryPlan]
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
    result_check: str = ""
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
