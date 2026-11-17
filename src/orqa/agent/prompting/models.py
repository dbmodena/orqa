"""Shared Pydantic data models for the prompting package.

This module defines the structured-plan and column-statistics models used by the
query planner and column-statistics component. ``QueryLink`` is reused from
:mod:`orqa.agent.structured_outputs` rather than redefined so the
existing plan-link contract is preserved.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

from orqa.agent.planning.provenance import SOURCE_FREE_OPERATIONS
from orqa.agent.structured_outputs import QueryLink, Table


# Ops available to every plan (SQL and Pandas alike): plain relational/derive
# steps, plus `clean` — a data-quality step (drop a column, impute, cast, or
# drop rows) the planner adds explicitly when the raw per-column statistics
# (see ColumnStat: nulls, missing tokens, number text, date text) show
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
    "The difficulty tier of the QUESTION this plan answers (easy, medium or "
    "hard). It is fixed: copy the tier given with the fixed question. It is "
    "not derived from `steps` and nothing checks it against them — never "
    "shape, pad or trim the steps to reach a tier."
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
# missing data, whether already listed as a missing token or a new one
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
    "`\"strategy\": \"mean\"|\"median\"|\"mode\"|\"constant\"|\"ffill\"` (plus "
    "`\"value\"` when strategy is `constant`; `ffill` for a column COLUMN "
    "STATISTICS shows as `filled down`); `cast` adds `\"to\": "
    "\"numeric\"|\"string\"|\"datetime\"|\"boolean\"` and, for a column shown "
    "as number text or date text, `\"format\"`: that fact copied from COLUMN "
    "STATISTICS (e.g. `decimal comma; thousands dot` or `%d/%m/%Y`); for an "
    "`ambiguous`/`mixed` number text or an `(order unknown)` date text, the one "
    "reading you chose (e.g. `thousands comma` or `%m/%d/%Y`), never the "
    "undecided fact. "
    "`impute`/`drop_rows` "
    "MAY also add `\"treat_as_missing\": [<literal>, ...]` — specific stored values "
    "(from COLUMN STATISTICS: `missing tokens`, `not numbers`/`not dates` "
    "values; far-high rows are excluded or flagged by their bound instead) "
    "the planner has judged to mean 'no answer' even though they weren't in "
    "the configured missing-value list, e.g. an unlisted 'no data' phrasing or "
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
    "\"keys\": [...]} for a join). "
    "WHEREVER a step declares an output column here — `params.output_column`, "
    "`params.new_column`, or the keys of `params.aggregations` — it must ALSO "
    "appear in that step's `produces` with the column(s) it was computed "
    "from. `params` stays the fixed-shape carrier each op needs; `produces` "
    "is the authoritative record of where a new column came from."
)


# The provenance contract. A plan may legitimately reference a column no raw
# table contains (a `derive` output, an `aggregate` result, a `rank`
# position); before `produces` existed there was no way to SAY so, and
# QueryPlanner.validate_plan had to fall back on "once any step could have
# derived something, stop checking columns" — which silently accepted every
# hallucinated name after the first derivation. Declaring the origin makes
# the check exact.
#
# `sources` entries are flat strings rather than {table, column} objects: the
# alias is only ever a scope hint for RAW columns (it is ignored outright
# when the source is itself a derived column, since such a column's roots may
# span several tables and no single alias would be correct). Splitting is
# exact, not heuristic — aliases are the machine-generated `Table_N`
# vocabulary, so a source is qualified iff the text before its first "." is a
# known alias, and a column containing a literal dot can never be misread.
_PRODUCES_DESCRIPTION = (
    "Every column THIS step creates that does not already exist in a raw "
    "table — an aggregate's result, a derive's output, a rank position, a "
    "renamed column. Each entry is `{\"name\": <new column>, \"operation\": "
    "<short label: \"sum\", \"mean\", \"count\", \"flag\", \"bucket\", "
    "\"ratio\", \"year_extract\", \"rename\", ...>, \"sources\": [<column "
    "it is computed from>, ...]}`. "
    "Write a source that is a real column of a table as `Alias.column` "
    "(e.g. `Table_0.amount`); write a source an EARLIER step derived as its "
    "bare name (e.g. `total_spend`) — that column may itself come from "
    "several tables, so qualifying it would be wrong. Never guess an alias. "
    "`sources` may be empty ONLY for an operation that genuinely reads no "
    "column (" + ", ".join(sorted(SOURCE_FREE_OPERATIONS)) + ") — every "
    "other operation must name what it was computed from. "
    "DECLARE ONCE: a derived column is declared only in the step that "
    "creates it. From that step onward it behaves exactly like a real column "
    "of the data — any later step may reference it by name in `columns` or "
    "`params` for any purpose (filtering, grouping, aggregating, sorting, "
    "ranking, joining, correlating, or as the origin of a further derived "
    "column). Never re-declare it in a later step. "
    "This is checked mechanically before any judge sees the plan: a column "
    "that exists in no table and that no step declares here is rejected, as "
    "is a source that resolves to nothing."
)


class DerivedColumn(BaseModel):
    """One column a step creates that no raw table contains.

    See ``_PRODUCES_DESCRIPTION`` for the source-string convention and
    ``QueryPlanner.validate_plan`` for how origins are resolved.
    """

    name: str  # the new column this step creates
    operation: str  # short label: "sum", "flag", "bucket", "ratio", ...
    sources: List[str] = Field(
        default_factory=list,
        description=(
            "Columns this one is computed from: `Alias.column` for a raw "
            "table column, the bare name for a column an earlier step "
            "derived. Empty only for a count-like operation."
        ),
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
    produces: List[DerivedColumn] = Field(
        default_factory=list, description=_PRODUCES_DESCRIPTION
    )


class PandasPlanStep(BaseModel):
    """A single ordered step in a Pandas structured query plan."""

    order: int  # 1-based execution order
    op: _SQL_OPS
    description: str  # human-readable intent for this step
    tables: List[str]  # aliases referenced by this step
    columns: List[str]  # concrete columns this step reads/writes
    columns_role: dict = {}  # free-form role->columns annotation, e.g. {"primary": [...]}
    params: dict = Field(default_factory=dict, description=_STEP_PARAMS_DESCRIPTION)
    produces: List[DerivedColumn] = Field(
        default_factory=list, description=_PRODUCES_DESCRIPTION
    )


class SQLQueryPlan(BaseModel):
    """An ordered, schema-validated decomposition of a SQL query."""

    question: str
    question_keywords: List[str]
    plan_keywords: List[str]
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
    translated_question_keywords: List[str] = []
    detected_language: str = ""
    topic: str = ""
    story: str = ""


class PandasQueryPlan(BaseModel):
    """An ordered, schema-validated decomposition of a Pandas query."""

    question: str
    question_keywords: List[str]
    plan_keywords: List[str]
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
    translated_question_keywords: List[str] = []
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
    """Per-column facts computed with pandas only (no LLM) on the RAW table
    (before any cleaning, see utils.prepare_dataset), rendered into the
    planner, generation and correction prompts by
    ``prompts.render_column_statistics``.

    Every field holds RAW built-in Python values (int, float, str, bool; no
    numpy scalars); all display formatting (number spelling, quoting,
    cutting) happens in the renderer. A field left at its default means
    "nothing to report" and is not rendered. Which fields a column gets
    depends on its kind: numbers (``statistics.py``), text, number text,
    date text, digit codes, geometry, free text (``text_profile.py``).
    """

    column: str
    dtype: str
    cardinality: int
    null_ratio: float  # 0..1, true NaN only (pandas' own NA detection)
    nan_count: int = 0
    # Rendered type: "int", "float", "float, whole numbers", "str", "bool",
    # "mixed (str ×a, int ×b)", ...; "" lets the renderer derive it from dtype.
    type_label: str = ""

    # Domain. `values` is the COMPLETE list of (raw value, rows) in rendered
    # order; `top_values` only the most common ones.
    constant: Optional[Any] = None
    values: List[Tuple[Any, int]] = []
    top_values: List[Tuple[Any, int]] = []
    # "Most common" never stops inside a tie: the values left out that share
    # the tied count ({"values": t, "rows": c}), or, when every value has the
    # same count, that count alone (domain "k distinct, each ×c").
    top_values_tied: Optional[dict] = None
    uniform_count: Optional[int] = None
    same_as: Optional[str] = None  # identical to this earlier column
    # Structural evidence that the column names records (never a verdict):
    # {"kind": "sequence" | "unique" | "unique_distinct_rows" | "code_of" |
    # "row_key" | "fixes", "of": [columns]} — see statistics.py.
    identifier: Optional[dict] = None
    # A text column written once at the top of each block and blank below
    # (spreadsheet fill-down): {"filled": rows holding a value, "rows": rows}.
    filled_down: Optional[dict] = None
    # Whole numbers that are calendar dates (20200416): {"min", "max"} as
    # "YYYY-MM-DD"; range and quartiles are then left out.
    date_number: Optional[dict] = None
    # Text columns: distinct values that are not blank and not missing tokens
    # (the "k distinct" / "all distinct" the renderer shows; cardinality also
    # counts the blank and token spellings).
    usable_distinct: Optional[int] = None
    # Shown length of a long text value whose first 59 characters are shared
    # with another stored value: raw value -> characters to show (<= 120), so a
    # cut value stays a unique prefix.
    cut_hints: Dict[str, int] = {}

    # Numbers (numeric dtype, or object columns holding only numbers).
    # int for integer columns: an 18-digit code kept as float64 printed a
    # bound that is no stored value (NYC OFT codes).
    numeric_min: Optional[Union[int, float]] = None
    numeric_max: Optional[Union[int, float]] = None
    sequence: Optional[dict] = None  # {"step", "gaps", "row_order"}
    quartiles: List[Union[int, float]] = []
    far_high: Optional[dict] = None  # {"values": [...], "rows", "below", "block"}
    repeated_extreme: Dict[str, dict] = {}  # {"min"|"max": {"value", "rows"}}
    negative_rows: int = 0
    # Rows holding +inf/-inf (pandas reads "inf" text as float inf): left out
    # of range and quartiles, and a SUM or AVG over them is inf.
    infinite_rows: int = 0

    # Text cleanliness.
    blank: int = 0
    untrimmed: int = 0
    # Configured missing-value spellings, by raw stored spelling. Rendered as
    # their own fact only when the domain is not a complete list; listed
    # values/constant that are such tokens are named in missing_token_values.
    missing_tokens: List[Tuple[str, int]] = []
    missing_tokens_other_rows: int = 0
    # {rows, spellings} of the unlisted spellings, when the profile cut the list.
    missing_tokens_other: Optional[dict] = None
    missing_token_values: List[str] = []
    # Token spellings that look like the column's other values (letters only,
    # same case pattern: "Nan" in a first-name column), shown as such.
    missing_tokens_plausible: List[str] = []
    # Untrimmed values whose trimmed form is also stored (trimming merges them).
    untrimmed_merges: int = 0
    variants: Optional[dict] = None  # {"groups", "rows", "examples": [raw]}

    # Text kind: at most one of number / date / date_like_shape / digits /
    # geometry / text_kind / shape is set.
    number: Optional[dict] = None
    date: Optional[dict] = None
    date_like_shape: Optional[str] = None
    digits: Optional[dict] = None
    geometry: Optional[str] = None
    text_kind: Optional[str] = None  # "URLs" | "emails" | "free text"
    text_length: Optional[int] = None
    shape: Optional[str] = None
    patterns: List[Tuple[str, int]] = []
    # Usable values of a number/date/digit column that do not conform, or
    # that a code shape does not fit (`not matching`):
    # {"value", "rows"} or {"shape", "rows", "example"}.
    other: List[dict] = []
    other_rows: int = 0


class TableStats(BaseModel):
    """Per-table statistics: row count, per-column facts, and table facts
    (``structure``: empty_rows, repeated_rows, total_row, total_rows, grain,
    one_to_one, determines, determines_more, sum_columns; built-in types
    only, see ``table_structure.py``)."""

    alias: str
    num_rows: int
    columns: List[ColumnStat]
    structure: Dict[str, Any] = {}
    # The table's row count when the facts come from a uniform sample of
    # ``num_rows`` of its rows (statistics.compute_from_parquet, the largest
    # tables); None: every row.
    sampled_from: Optional[int] = None


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
    keywords: List[str] = []
    translated_keywords: List[str] = []


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
