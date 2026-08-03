"""Shared Pydantic data models for the prompting package.

This module defines the structured-plan and column-statistics models used by the
query planner and column-statistics component. ``QueryLink`` is reused from
:mod:`orqa.agent.utility.structured_outputs` rather than redefined so the
existing plan-link contract is preserved.
"""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field

from ..utility.structured_outputs import QueryLink, Table


# Ops available to every plan (SQL and Pandas alike): plain relational/derive
# steps, plus `clean` — a data-quality step (drop a column, impute, cast, or
# drop rows) the planner adds explicitly when the raw per-column statistics
# (see ColumnStat's nan_count/bad_token_counts/numeric_parseable_ratio) show
# it's warranted. Tables are no longer cleaned automatically before the
# planner sees them (see utils.clean_columns/prepare_dataset), so this is
# the planner's only lever over missing/dirty/miscast values. `correlate`,
# `limit`, and `rank` (like `clean` and the outlier-preserving `derive`
# convention) have a FIXED `params` shape documented in
# _STEP_PARAMS_DESCRIPTION below, rather than the free-form meaning every
# other op's `params` carries.
_SQL_OPS = Literal[
    "filter", "join", "union", "group", "aggregate", "sort", "select", "derive",
    "clean", "correlate", "limit", "rank",
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

_DIFFICULTY_LEVELS = Literal["easy", "medium", "hard"]

_DIFFICULTY_DESCRIPTION = (
    "This plan's structural-complexity tier, per the DIFFICULTY rubric in "
    "the system prompt — judge it from `steps` alone, AFTER deciding them, "
    "never chosen first and reverse-engineered into the steps. In a "
    "multi-plan batch this is a "
    "fixed per-slot requirement (e.g. exactly one easy, one medium, one "
    "hard plan) — assign it honestly to what THIS plan's steps actually "
    "do; do not force two plans into the same tier just because their "
    "step counts look similar."
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

# Describes the `params` convention for a `clean` step, an outlier-preserving
# `derive` step, and the `correlate`/`limit`/`rank` ops (SQL and Pandas
# alike). Tables reach the planner raw — no bad-token conversion, numeric
# coercion, or null dropping has happened — so this is the ONLY place either
# cleaning decision gets made; the code generator turns each declared action
# into real code (see conf/prompts/*_statement_generation.md). The `clean`/
# `derive` conventions exist because an outlier/rare value is NEVER
# automatically noise: `clean` is for values that mean "no answer" (true
# missing data, whether already counted in `bad_token_counts` or a new one
# the planner recognizes from COLUMN STATISTICS); `derive` is for values
# that carry real information (a censoring convention, a qualitative label,
# a genuine rare-but-real extreme) and deserve to become their own explicit,
# queryable feature instead of being discarded — see the DATA QUALITY /
# CLEANING section of the prompt for the decision framework. `correlate`/
# `limit`/`rank` instead have a fixed params shape simply because each maps
# to one canonical, cross-dialect-asymmetric function (DuckDB's Pearson-only
# `corr()`, `LIMIT`, and the RANK/DENSE_RANK/ROW_NUMBER family respectively).
_STEP_PARAMS_DESCRIPTION = (
    "For a `clean` step: `params` MUST be `{\"actions\": [...]}`, one entry "
    "per column-level decision: `{\"column\": <name>, \"action\": "
    "\"drop_column\"|\"drop_rows\"|\"impute\"|\"cast\"}`. `impute` adds "
    "`\"strategy\": \"mean\"|\"median\"|\"mode\"|\"constant\"` (plus "
    "`\"value\"` when strategy is `constant`); `cast` adds `\"to\": "
    "\"numeric\"|\"string\"|\"datetime\"|\"boolean\"`. `impute`/`drop_rows` "
    "MAY also add `\"treat_as_missing\": [<literal>, ...]` — specific values "
    "(from `bad_token_counts`, `numeric_outliers`, or `minority_value_groups`) "
    "the planner has judged to mean 'no answer' even though they weren't in "
    "the configured bad-token list, e.g. an unlisted 'no data' phrasing or "
    "an isolated numeric sentinel like `-1`/`999`. "
    "For a `derive` step whose purpose is to PRESERVE an outlier/rare-value "
    "pattern as its own feature rather than discard it: `params` SHOULD be "
    "`{\"actions\": [{\"column\": <source>, \"technique\": \"flag\"|"
    "\"bucket\", \"output_column\": <name>, \"rule\": <plain-language rule, "
    "e.g. \\\"age matches '<#' shape or the literal 'minor' -> True\\\">}]}`. "
    "`flag` produces a boolean column marking matching rows; `bucket` "
    "produces a categorical column mapping value ranges/patterns to labels. "
    "`output_column` must also appear in the step's `columns` (already "
    "auto-registered for later steps, same as any other `derive` output). "
    "For BOTH step kinds: every column an action touches must also appear "
    "in the step's `columns`, and the step's `description` must cite the "
    "COLUMN STATISTICS evidence behind each decision — never act on a "
    "column with no statistical basis for it. A column marked `drop_column` "
    "may not be referenced by any later step. "
    "For a `correlate` step: `params` MUST include `\"method\": "
    "\"pearson\"|\"spearman\"|\"kendall\"` — SQL plans MUST use `\"pearson\"` "
    "(DuckDB's only native correlation aggregate; no Spearman/Kendall); "
    "Pandas plans may use any of the three. `params` MAY add "
    "`\"group_by\": [<column>, ...]` to compute the coefficient separately "
    "per group instead of once over the whole table, and MAY add "
    "`\"output_column\": <name>` naming the resulting coefficient ONLY when "
    "a later step consumes it (e.g. a `sort`/`limit` over a per-group "
    "correlation table) — omit it when the coefficient IS the plan's final "
    "answer. `columns` lists the 2+ already-existing numeric columns being "
    "correlated — `correlate` never invents a source column. "
    "For a `limit` step: `params` MUST be `{\"n\": <positive int>}` (the "
    "row cap), and MAY add `\"how\": \"head\"|\"largest\"|\"smallest\"` "
    "(default `\"head\"`) — `\"head\"` caps rows in whatever order an "
    "earlier `sort` step already produced; `\"largest\"`/`\"smallest\"` "
    "pick the top/bottom `n` rows by column(s) directly (no separate `sort` "
    "step needed), which REQUIRES `\"by\": [<column>, ...]` naming the "
    "ranking column(s) (omit `by` when `how` is `\"head\"`). "
    "For a `rank` step: `params` MUST include `\"output_column\": <name>` "
    "(the new column materializing the rank position — same "
    "forward-declaration convention as `derive`'s output columns) and "
    "`\"by\": [<column>, ...]` (the column(s) ranked on). MAY add "
    "`\"ascending\": true|false` (default `true`), `\"method\"` for tie "
    "handling — Pandas plans: `\"average\"|\"min\"|\"max\"|\"first\"|"
    "\"dense\"`; SQL plans: `\"min\"|\"dense\"|\"first\"` only (DuckDB has "
    "no native average/max tie-handling — these map directly to "
    "`RANK()`/`DENSE_RANK()`/`ROW_NUMBER()`) — and `\"group_by\": "
    "[<column>, ...]` to rank within groups rather than over the whole "
    "table. `output_column` should not also appear in `columns` for "
    "`correlate`/`rank` (list only the SOURCE columns being read there) — "
    "it is already auto-registered for later steps via `params.output_column` alone. "
    "For every other op, `params` "
    "keeps its existing free-form meaning (e.g. {\"how\": \"inner\", "
    "\"keys\": [...]} for a join)."
)


class SQLPlanStep(BaseModel):
    """A single ordered step in a SQL structured query plan."""

    order: int  # 1-based execution order
    op: _SQL_OPS
    description: str  # human-readable intent for this step
    tables: List[str]  # aliases referenced by this step
    columns: List[str]  # concrete columns this step reads/writes
    columns_role: dict = {}  # free-form role->columns annotation, e.g. {"primary": [...]}
    params: dict = Field(default_factory=dict, description=_STEP_PARAMS_DESCRIPTION)


class PandasPlanStep(BaseModel):
    """A single ordered step in a Pandas structured query plan."""

    order: int  # 1-based execution order
    op: _SQL_OPS
    description: str  # human-readable intent for this step
    tables: List[str]  # aliases referenced by this step
    columns: List[str]  # concrete columns this step reads/writes
    columns_role: dict = {}  # free-form role->columns annotation, e.g. {"primary": [...]}
    params: dict = Field(default_factory=dict, description=_STEP_PARAMS_DESCRIPTION)


class SQLQueryPlan(BaseModel):
    """An ordered, schema-validated decomposition of a SQL query."""

    question: str
    question_keywords: List[str]  # max 10
    plan_keywords: List[str]  # max 10
    steps: List[SQLPlanStep]  # ordered decomposition
    difficulty: _DIFFICULTY_LEVELS = Field(
        default="easy", description=_DIFFICULTY_DESCRIPTION
    )
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
    difficulty: _DIFFICULTY_LEVELS = Field(
        default="easy", description=_DIFFICULTY_DESCRIPTION
    )
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
    """A batch of independent Pandas query plans produced from the same table
    analyses. Each entry is a fully independent :class:`PandasQueryPlan`: its
    own ``question``/``question_keywords`` and its own ``steps``.
    """

    plans: List[PandasQueryPlan]


class ColumnStat(BaseModel):
    """Cheap per-column statistics computed with pandas only (no LLM), on the
    RAW table (before any cleaning — see utils.prepare_dataset). Feeds the
    query planner's `clean`-step decisions: which columns are actually
    missing/dirty/miscast, and by how much, rather than a pre-cleaned
    ``null_ratio`` that already silently absorbed those judgment calls.
    """

    column: str
    dtype: str
    cardinality: int
    null_ratio: float  # 0..1 — true NaN only (pandas' own NA detection)
    nan_count: int = 0  # true NaN count (same signal as null_ratio, absolute)
    bad_token_counts: dict = {}  # {literal sentinel token: count}, portal-specific
    numeric_parseable_ratio: Optional[float] = None  # object columns only; see utils._numeric_parse_ratio
    numeric_min: Optional[float] = None
    numeric_max: Optional[float] = None
    numeric_mean: Optional[float] = None
    # Numeric-dtype columns only: Tukey IQR-fenced outliers — detection only,
    # no judgment on whether they're noise (a sentinel, a data-entry error)
    # or signal (a genuine rare-but-real extreme). None when too few rows or
    # no values fall outside the fence. Shape: {"low_count", "high_count",
    # "low_examples", "high_examples", "low_fence", "high_fence"}.
    numeric_outliers: Optional[dict] = None
    # Numeric-dtype columns only: a value AT the column's own min or max that
    # repeats far more often than the column's cardinality would predict
    # under a roughly uniform spread — the classic top-/bottom-coding
    # signature in data that's ALREADY numeric (e.g. hundreds of rows
    # reading exactly 90 in an age column capped at "90 or older"). Distinct
    # from numeric_outliers above: that flags a RARE Tukey-fence extreme;
    # this flags a FREQUENT one at the boundary — the two can legitimately
    # both fire for the same value (a whole "shelf" of capped values sitting
    # beyond the fence is itself the strongest evidence of top-coding). None
    # when neither side clears the detection thresholds (see
    # ColumnStatistics.compute). Shape: any subset of {"low": {"value",
    # "count", "ratio"}, "high": {"value", "count", "ratio"}} — only the
    # side(s) that cleared the thresholds are present.
    numeric_pinned_extreme: Optional[dict] = None
    top_values: List[str] = []  # top categorical values (bounded)
    # Object-dtype columns only: the tail beyond top_values, grouped into
    # cheap structural shapes (see utils._generalize_value_shape) so a table
    # with many distinct rare values still stays compact. Bounded top-k,
    # excludes exact bad-token matches. Each entry:
    # {"pattern", "count", "examples"}.
    minority_value_groups: List[dict] = []


class TableStats(BaseModel):
    """Per-table statistics: row count plus per-column statistics."""

    alias: str
    num_rows: int
    columns: List[ColumnStat]


# ---------------------------------------------------------------------------
# Traceable output models
#
# These models supersede the flat ``Query`` in ``structured_outputs.py`` for the
# final assembled result. They capture every phase of the generation pipeline
# (planning, validation, judging, execution, and usage) so each produced
# query is fully traceable back to the decisions that created it. They are
# defined here alongside
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
    # Reliability-card payload for ML-skill (classification/regression/
    # timeseries/causal) queries — {"skill", "shape", "causal_live"}, see
    # JudgementResponseAgent._derive_reliability. None for plain
    # SQL/pandas queries with no injected skill.
    reliability: Optional[dict] = None


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
