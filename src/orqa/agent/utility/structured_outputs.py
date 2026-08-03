from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Literal, Union
from enum import Enum

class DatasetAnalysisResult(BaseModel):
    """Result of dataset analysis with UNION and JOIN suggestions"""

    UNION: List[str] = Field(
        description="List of column names that should be combined using UNION (similar columns across datasets)"
    )
    JOIN: List[str] = Field(
        description="List of column names that can be used as JOIN keys (common identifiers)"
    )


class TableAnalysis(BaseModel):
    alias: str = Field(
        ...,
        description="Table alias as provided in the prompt."
    )
    table_description: str = Field(
        ...,
        description=(
            "A concise business description of the table: what it represents, "
            "the real-world context and topic it belongs to, and what "
            "information it provides — anchored with the place and time period "
            "when they are visible in the data (e.g. 'Students expelled from "
            "New York City schools in 2016')."
        )
    )
    table_keywords: List[str] = Field(
        default_factory=list,
        description=(
            "Retrieval keywords for this table (max 10). Each is a single word "
            "or short established term — never a descriptive phrase. They are "
            "indexed in a reverse index over many portal tables and their "
            "COMBINATION must identify this table univocally: cover its "
            "specific subject, entity type, agency/organisation, place, and "
            "time period as separate keywords, avoiding generic terms shared "
            "by many tables."
        )
    )

    @field_validator("table_keywords")
    @classmethod
    def limit_table_keywords(cls, v: List[str]) -> List[str]:
        """Limit table keywords to max 10 and remove duplicates"""
        seen = set()
        unique_keywords = []
        for kw in v[:10]:  # Take only first 10
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        return unique_keywords


class TableAnalyses(BaseModel):
    tables: List[TableAnalysis] = Field(
        ...,
        description="Analysis results for each table, including alias, description, and keywords."
    )


class QueryLink(BaseModel):
    type: Literal["join", "union", "join-correlation", "other"] = Field(
        ...,
        description="The relationship type between tables for the query plan."
    )
    tables: List[str] = Field(
        ...,
        description="The aliases of the tables involved in this link."
    )
    description: str = Field(
        ...,
        description="A short description of how these tables are related and why the relationship is needed."
    )
    key_columns: List[Dict[str, str]] = Field(
        default_factory=list,
        description=(
            "The columns that MECHANICALLY establish this link — the join key "
            "(type=='join' or 'join-correlation') or the aligned columns "
            "(type=='union') — one entry per column correspondence: "
            "{table_alias: column_name, table_alias: column_name}. A composite "
            "key has one entry per column pair. Column NAMES may differ across "
            "the two tables (e.g. {'Table_1': 'dbn', 'Table_0': 'school_id'}) — "
            "never assume they share a name just because both tables are "
            "listed in `tables`. For type=='join-correlation' this is ONLY the "
            "join key used to combine the rows — see `correlated_columns` for "
            "the columns actually being correlated afterward."
        )
    )
    correlated_columns: List[Dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Only set when type == 'join-correlation': the numeric columns "
            "being correlated AFTER the join in `key_columns` — as "
            "{table_alias: column_name} pairs, distinct from the join key."
        )
    )


class QueryPlan(BaseModel):
    question: str = Field(
        ...,
        description=(
            "A business-focused natural language question that the final query "
            "should answer, phrased as an average user with no knowledge of the "
            "underlying data or of querying would ask it — while naturally "
            "containing the distinctive topical vocabulary (subject, entity "
            "type, agency, place, period) that lets a keyword extractor "
            "retrieve the right table(s) from a reverse index."
        )
    )
    query_plan: str = Field(
        ...,
        description="A high-level plan describing how the query should be constructed."
    )
    question_keywords: List[str] = Field(
        default_factory=list,
        description=(
            "The question's table-identifying retrieval terms (max 10): "
            "single words or short established terms appearing in the "
            "question whose combination matches the target table's index "
            "keywords — no phrases, no generic filler words."
        )
    )
    plan_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords extracted from the query plan (max 10)."
    )
    table_links: List[QueryLink] = Field(
        default_factory=list,
        description="Descriptions of how tables should be joined or unioned in the final query."
    )

    @field_validator("question_keywords")
    @classmethod
    def limit_question_keywords(cls, v: List[str]) -> List[str]:
        """Limit question keywords to max 10 and remove duplicates"""
        seen = set()
        unique_keywords = []
        for kw in v[:10]:  # Take only first 10
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        return unique_keywords

    @field_validator("plan_keywords")
    @classmethod
    def limit_plan_keywords(cls, v: List[str]) -> List[str]:
        """Limit plan keywords to max 10 and remove duplicates"""
        seen = set()
        unique_keywords = []
        for kw in v[:10]:  # Take only first 10
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        return unique_keywords


class TaskUnion(BaseModel):
    """UNION task: columns that can be combined with similar columns from other datasets"""

    columns: List[str] = Field(
        description="List of column names that can be combined using UNION operation. "
        "These columns have similar semantic meaning across datasets."
    )

    @field_validator("columns")
    @classmethod
    def remove_duplicate_columns(cls, v: List[str]) -> List[str]:
        """Remove duplicate column names while preserving order"""
        seen = set()
        unique_cols = []
        for col in v:
            if col not in seen:
                seen.add(col)
                unique_cols.append(col)
        return unique_cols


class TaskJoin(BaseModel):
    """JOIN task: columns that can serve as join keys"""

    columns: List[str] = Field(
        description="List of column names that can be used as JOIN keys. "
        "These are typically unique identifiers or foreign keys."
    )

    @field_validator("columns")
    @classmethod
    def remove_duplicate_columns(cls, v: List[str]) -> List[str]:
        """Remove duplicate column names while preserving order"""
        seen = set()
        unique_cols = []
        for col in v:
            if col not in seen:
                seen.add(col)
                unique_cols.append(col)
        return unique_cols


class TaskJoinCorrelation(BaseModel):
    """JOIN-CORRELATION task: a join key paired with a correlated metric"""

    join_column: str = Field(
        description="Column name to use as the JOIN key (unique identifier)"
    )
    correlation_column: str = Field(
        description="Column name that correlates with the join key "
        "(e.g., a metric or value associated with the identifier)"
    )



class Result(BaseModel):
    """Analysis result containing all identified tasks"""

    union_tasks: List[TaskUnion] = Field(
        default_factory=list,
        description="List of UNION tasks identifying columns that can be combined",
    )
    join_tasks: List[TaskJoin] = Field(
        default_factory=list,
        description="List of JOIN tasks identifying potential join keys",
    )
    join_correlation_tasks: List[TaskJoinCorrelation] = Field(
        default_factory=list,
        description="List of JOIN-CORRELATION tasks pairing join keys with correlated metrics",
    )

    @field_validator("union_tasks")
    @classmethod
    def deduplicate_union_tasks(cls, v: List[TaskUnion]) -> List[TaskUnion]:
        """Remove duplicate TaskUnion instances with the same column sets"""
        seen = set()
        unique_tasks = []
        for task in v:
            # Create a frozenset of columns for comparison (order-independent)
            col_set = frozenset(task.columns)
            if col_set not in seen:
                seen.add(col_set)
                unique_tasks.append(task)
        return unique_tasks

    @field_validator("join_tasks")
    @classmethod
    def deduplicate_join_tasks(cls, v: List[TaskJoin]) -> List[TaskJoin]:
        """Remove duplicate TaskJoin instances with the same column sets"""
        seen = set()
        unique_tasks = []
        for task in v:
            # Create a frozenset of columns for comparison (order-independent)
            col_set = frozenset(task.columns)
            if col_set not in seen:
                seen.add(col_set)
                unique_tasks.append(task)
        return unique_tasks

    @field_validator("join_correlation_tasks")
    @classmethod
    def deduplicate_join_correlation_tasks(
        cls, v: List[TaskJoinCorrelation]
    ) -> List[TaskJoinCorrelation]:
        """Remove duplicate TaskJoinCorrelation instances with the same column pairs"""
        seen = set()
        unique_tasks = []
        for task in v:
            # Create a tuple for comparison
            col_pair = (task.join_column, task.correlation_column)
            if col_pair not in seen:
                seen.add(col_pair)
                unique_tasks.append(task)
        return unique_tasks

class PairUnionTask(BaseModel):
    """UNION task over a concrete (Q, R) dataset pair: aligned column lists."""

    q_columns: List[str] = Field(
        description="Columns from the Q dataset to union, in order."
    )
    r_columns: List[str] = Field(
        description="Columns from the R dataset to union, aligned positionally "
        "with q_columns (same length, same semantic meaning per position)."
    )

    @model_validator(mode="after")
    def _aligned_lengths(self) -> "PairUnionTask":
        if len(self.q_columns) != len(self.r_columns):
            raise ValueError(
                f"q_columns ({len(self.q_columns)}) and r_columns "
                f"({len(self.r_columns)}) must have the same length"
            )
        return self


class PairJoinTask(BaseModel):
    """JOIN task over a concrete (Q, R) dataset pair: aligned join keys."""

    q_columns: List[str] = Field(
        description="Join-key columns from the Q dataset, in order."
    )
    r_columns: List[str] = Field(
        description="Join-key columns from the R dataset, aligned positionally "
        "with q_columns (q_columns[i] joins r_columns[i])."
    )

    @model_validator(mode="after")
    def _aligned_lengths(self) -> "PairJoinTask":
        if len(self.q_columns) != len(self.r_columns):
            raise ValueError(
                f"q_columns ({len(self.q_columns)}) and r_columns "
                f"({len(self.r_columns)}) must have the same length"
            )
        return self


class PairJoinCorrelationTask(BaseModel):
    """JOIN-CORRELATION task over a concrete (Q, R) pair: join keys plus the
    numeric columns to correlate after joining."""

    q_key: str = Field(description="Join-key column from the Q dataset.")
    r_key: str = Field(description="Join-key column from the R dataset.")
    q_target: str = Field(
        description="Numeric column from the Q dataset to correlate."
    )
    r_target: str = Field(
        description="Numeric column from the R dataset to correlate."
    )


class PairTaskSelection(BaseModel):
    """Operations selected for one concrete (Q, R) candidate pair."""

    union_tasks: List[PairUnionTask] = Field(
        default_factory=list,
        description="UNION operations that make sense for this pair (may be empty).",
    )
    join_tasks: List[PairJoinTask] = Field(
        default_factory=list,
        description="JOIN operations that make sense for this pair (may be empty).",
    )
    join_correlation_tasks: List[PairJoinCorrelationTask] = Field(
        default_factory=list,
        description="JOIN-CORRELATION operations that make sense for this pair (may be empty).",
    )

    @field_validator("union_tasks", "join_tasks")
    @classmethod
    def deduplicate_pair_tasks(cls, v):
        """Remove tasks with identical (q_columns, r_columns) pairings."""
        seen = set()
        unique_tasks = []
        for task in v:
            key = (tuple(task.q_columns), tuple(task.r_columns))
            if key not in seen:
                seen.add(key)
                unique_tasks.append(task)
        return unique_tasks

    @field_validator("join_correlation_tasks")
    @classmethod
    def deduplicate_pair_jc_tasks(
        cls, v: List[PairJoinCorrelationTask]
    ) -> List[PairJoinCorrelationTask]:
        seen = set()
        unique_tasks = []
        for task in v:
            key = (task.q_key, task.r_key, task.q_target, task.r_target)
            if key not in seen:
                seen.add(key)
                unique_tasks.append(task)
        return unique_tasks


class Table(BaseModel):
    name: str = Field(
        ...,
        description=(
            "The alias of the table as provided in the lookup aliases. "
            "Must exactly match one of the aliases supplied in the prompt."
        )
    )
    reason: str = Field(
        ...,
        description=(
            "A business-oriented justification for why this table is needed: "
            "its concrete role in the plan (which step(s) it feeds), and the "
            "specific rows, columns, or filtering the answer actually depends "
            "on it for — not a generic one-liner. This is decided and judged "
            "exactly once, here at PLANNING time (see PlanJudgment.table_check) "
            "— required in substance, not just present: a table's `reason` is "
            "checked for completeness and non-emptiness before any code is "
            "written, and is never re-invented at generation time (the "
            "generated query's table usage is simply copied from here)."
        )
    )
    columns_involved: List[str] = Field(
        ...,
        description=(
            "The minimal subset of columns from this table strictly necessary to answer "
            "the question. Include only columns referenced in filters, joins, aggregations, "
            "or the final output — omit anything not directly used."
        )
    )
    description: str = Field(
        default="",
        description="A short business description for this table as used by the generated query."
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Keywords extracted from the table analysis that are relevant for this query (max 10)."
    )
    translated_keywords: List[str] = Field(
        default_factory=list,
        description="Translated keywords for this table in the detected language (max 10)."
    )

    @field_validator("keywords")
    @classmethod
    def limit_keywords(cls, v: List[str]) -> List[str]:
        """Limit keywords to max 10 and remove duplicates"""
        seen = set()
        unique_keywords = []
        for kw in v[:10]:  # Take only first 10
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        return unique_keywords

    @field_validator("translated_keywords")
    @classmethod
    def limit_translated_keywords(cls, v: List[str]) -> List[str]:
        """Limit translated keywords to max 10 and remove duplicates"""
        seen = set()
        unique_keywords = []
        for kw in v[:10]:  # Take only first 10
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        return unique_keywords




class Query(BaseModel):
    """
    A structured query (SQL or Pandas) paired with a natural language question
    that expresses the intent of the query.
    """
    # NOTE: no id/client_id field here. Identity is handled entirely outside
    # the LLM: GenerationCoordinator.assign_client_ids stamps client_id onto each
    # query dict after generation (never trusting the model to mint or echo
    # one), and every downstream match (correction, judging) is positional —
    # index-based, not id-based — since generation/correction/judging always
    # process queries one at a time or in strict input order. Asking the
    # model to declare/echo an id here would be pure overhead with nothing
    # downstream ever reading it back.
    #
    # NOTE: no `tables` field here either, for the same reason. Table usage —
    # name, reason, columns_involved, description, keywords — is decided
    # exactly once, during PLANNING (see prompting.models.SQLQueryPlan/
    # PandasQueryPlan.tables, a List[Table]) and judged there by
    # PlanJudgment.table_check. StatementClient.complete stamps the plan's own
    # `tables` onto every query generated from it, the same way it already
    # does for `query_plan`/`question_keywords`/etc. — so asking the
    # generation LLM to redeclare the whole table list here would just be
    # regenerating (and risking drift from) something already decided and
    # approved upstream.
    question: str = Field(
        ...,
        description=(
            "A natural language question that the code answers. "
            "Must be phrased as an average, non-expert user would ask it, "
            "using context clues and table keywords to indicate relevant tables without naming them. "
            "The question must describe the business or analytical intent, "
            "NOT the operations or technical implementation details."
        )
    )
    # NOTE: no `difficulty` field here either, for the same reason as
    # `tables` above. Difficulty is decided exactly once, during PLANNING
    # (see prompting.models.SQLQueryPlan/PandasQueryPlan.difficulty) and
    # judged there by PlanJudgment.difficulty_check. StatementClient.complete
    # stamps the plan's own `difficulty` onto every query generated from it,
    # the same way it already does for `query_plan`/`question_keywords`/etc.
    code: str = Field(
        ...,
        description=(
            "The executable code (SQL query or Pandas operation) that answers "
            "the natural language question."
        )
    )
    # `question`, `question_keywords`, `translated_question`,
    # `translated_question_keywords`, `topic`, and `story` are all decided
    # TOGETHER during planning (see SQLQueryPlan/PandasQueryPlan in
    # prompting/models.py) — one linked bundle, not independent fields.
    # Optional/defaulted (not required with `...`) because the GENERATION
    # prompt never asks for them: StatementClient.complete's plan-fields merge
    # copies them from the plan onto every generated query after parsing, so
    # they arrive here as defaults and get overwritten immediately after.
    # The CORRECTION prompt (StatementValidator) is the one place that DOES
    # ask for them: when a correction rewrites `question`, the whole bundle
    # must be regenerated consistently with it, so the correction schema
    # needs somewhere to carry the LLM's updated values back through
    # `model_validate` — see StatementValidator._call_correction_llm_one's
    # merge, which is the only place these are read on a correction response.
    # `detected_language` is not itself rewritten on correction — it's context
    # (which language `translated_question` must be in), not linked content.
    question_keywords: List[str] = Field(
        default_factory=list,
        description=(
            "Keywords capturing the question's intent (max 10). Linked to "
            "`question` — regenerate together if you rewrite the question."
        ),
    )
    translated_question: str = Field(
        default="",
        description="`question` translated into `detected_language`. Linked to `question`.",
    )
    translated_question_keywords: List[str] = Field(
        default_factory=list,
        description="`question_keywords` translated the same way (max 10). Linked to `question`.",
    )
    detected_language: str = Field(
        default="",
        description="Detected language — context for `translated_question`, not itself linked content.",
    )
    topic: str = Field(
        default="",
        description="Short business topic/theme. Linked to `question`.",
    )
    story: str = Field(
        default="",
        description="Short business narrative behind the query. Linked to `question`.",
    )

    @field_validator("question_keywords")
    @classmethod
    def limit_question_keywords_query(cls, v: List[str]) -> List[str]:
        """Limit question keywords to max 10 and remove duplicates"""
        seen = set()
        unique_keywords = []
        for kw in v[:10]:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        return unique_keywords

    @field_validator("translated_question_keywords")
    @classmethod
    def limit_translated_question_keywords(cls, v: List[str]) -> List[str]:
        """Limit translated question keywords to max 10 and remove duplicates"""
        seen = set()
        unique_keywords = []
        for kw in v[:10]:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        return unique_keywords


class QuerySet(BaseModel):
    """
    Container for multiple queries (SQL or Pandas) generated together.
    """
    queries: List[Query] = Field(
        ...,
        description="List of queries with matching natural language questions."
    )


class ViolatedCriterion(str, Enum):
    # Code/result-focused criteria only: question quality AND table
    # justification are owned by the PLAN judge panel (PlanJudgment) and are
    # not re-judged here — the validator requires the code to use every table
    # the approved plan carries, so a code judge second-guessing table choice
    # only creates a contradiction the generator can never satisfy.
    unclear_result                = "unclear_result"
    partial_implementation        = "partial_implementation"
    over_engineering              = "over_engineering"
    silent_filter_bias            = "silent_filter_bias"
    disjointed_query              = "disjointed_query"
    trivial                       = "trivial"


class Judgment(BaseModel):
    # NOTE: no id field. JudgementResponseAgent._judge_one judges exactly one
    # query per LLM call and reads the (sole) response back positionally
    # (`judgments[0]`) — there's nothing for the model to disambiguate, so
    # asking it to declare/echo an id would be pure overhead nothing ever
    # reads back.
    #
    # Scope: the QUESTION was already reviewed by the plan judge panel — this
    # judgment covers the CODE and the executed RESULT in relation to what the
    # question asks.
    #
    # Voting is LAYERED across TWO independent layers, mirroring PlanJudgment
    # — ``plan_compliance_approval`` (does the code faithfully implement the
    # plan/question: requirements coverage, no unjustified operations, sound
    # skill-technique use) and ``present_result_approval`` (is the executed
    # result actually there and meaningful: non-empty, non-degenerate,
    # non-trivial). They are voted and aggregated independently (see
    # JudgePanel._aggregate with vote_fields) precisely so a query whose code
    # genuinely complies with the plan but whose result is empty is
    # distinguishable from a query whose CODE is wrong — the former is a
    # plan-level symptom (escalated to a one-shot plan revision by the
    # orchestrator, see agent.py's _retry_empty_results), the latter goes
    # through the normal code-correction loop. ``approved`` is derived, never
    # voted directly.
    result_check: str = Field(
        ...,
        description=(
            "One or two sentences: does the executed result actually answer "
            "the question — right shape (single figure / ranked list / "
            "per-group table) and non-degenerate values? Quote representative "
            "value(s) from the result. This is the PRESENT-RESULT layer's "
            "check text — empty, degenerate, or trivial results belong here, "
            "never under requirements_check."
        ),
    )
    requirements_check: str = Field(
        ...,
        description=(
            "Bidirectional coverage. List each core question requirement as "
            "IMPLEMENTED or MISSING in the code. List each code operation as "
            "JUSTIFIED or UNJUSTIFIED by the question, naming the specific "
            "phrase or implied need that grounds each JUSTIFIED verdict — a "
            "bare label with no grounding is not acceptable. Flag only "
            "material gaps — omit minor stylistic observations. This is the "
            "PLAN-COMPLIANCE layer's check text — whether the code correctly "
            "implements what was asked, never whether the result it produced "
            "is any good."
        ),
    )
    violated_criteria: List[ViolatedCriterion] = Field(
        default_factory=list,
        description=(
            "All triggered criteria. Empty if approved. Reject only when a flaw "
            "is unambiguous and material. `unclear_result`/`trivial` belong to "
            "the PRESENT-RESULT layer; every other criterion "
            "(`partial_implementation`/`over_engineering`/`silent_filter_bias`/"
            "`disjointed_query`) belongs to the PLAN-COMPLIANCE layer."
        ),
    )
    plan_compliance_approval: bool = Field(
        ...,
        description=(
            "Vote on the PLAN-COMPLIANCE layer: true when the code correctly "
            "and completely implements what the question asks — no missing "
            "core requirement, no unjustified operation, no "
            "disjointed/unconnected query structure. False when any of "
            "`partial_implementation`/`over_engineering`/`silent_filter_bias`/"
            "`disjointed_query` is triggered. Independent of "
            "present_result_approval: code can "
            "be a fully correct implementation of the plan and still produce "
            "an empty or degenerate result (that's a present_result failure, "
            "not a compliance one)."
        ),
    )
    present_result_approval: bool = Field(
        ...,
        description=(
            "Vote on the PRESENT-RESULT layer: true when the executed result "
            "actually answers the question — right shape, non-empty, "
            "not all-null/NaN, not a meaningless constant, no obviously "
            "corrupt values, and not something any business user could state "
            "without querying. False when `unclear_result` or `trivial` is "
            "triggered. Independent of plan_compliance_approval: a compliant, "
            "correct implementation of the plan can still legitimately "
            "produce nothing (an over-restrictive filter combination, a "
            "mismatched join) — that's a present_result failure, not a "
            "coding mistake, and is handled differently downstream."
        ),
    )
    feedback: str = Field(
        ...,
        description=(
            "What is missing in the code or should improve, in relation to what "
            "the question asks. Approved: why the result is meaningful and the "
            "code justified. Rejected: quote the exact operations or result "
            "values that triggered each criterion."
        ),
    )
    approved: bool = Field(
        default=False,
        description=(
            "Overall verdict; always plan_compliance_approval AND "
            "present_result_approval (recomputed server-side, so just set it "
            "consistently)."
        ),
    )
    response: str = Field(
        ...,
        description=(
            "3-4 sentence business insight. Must cite the actual computed value(s) from "
            "the result (specific numbers, names, dates, predicted values, or aggregates "
            "such as mean/min/max) — never a generic statement about what the result "
            "'could be used for' without stating what it actually is. Empty string if not "
            "approved."
        ),
    )
    translated_response: str = Field(
        ...,
        description="Response in the detected target language. Empty string if not approved.",
    )
    suggestions: str = Field(
        ...,
        description=(
            "Empty string if approved. If rejected: one actionable sentence per "
            "criterion prefixed [FIX QUERY], stating what to change in the code."
        ),
    )

    @model_validator(mode="after")
    def _force_consistent_verdict(self) -> "Judgment":
        """``approved`` is the AND of the two layer votes, never free —
        mirrors PlanJudgment._force_consistent_verdicts."""
        self.approved = self.plan_compliance_approval and self.present_result_approval
        return self


class Judgments(BaseModel):
    queries: List[Judgment] = Field(
        ...,
        description="One Judgment per input pair, in input order.",
    )


class PlanJudgment(BaseModel):
    """Verdict of one panel judge on a single structured query plan.

    The plan judge owns QUESTION quality, TABLE justification, the
    EXPECTED-RESULT declaration, and DIFFICULTY correspondence (the code
    judge no longer re-judges any of them): is the question concise, pinned
    to a specific topic, written like an average user seeking insights — do
    the plan's steps reflect it — does the question genuinely need every
    provided table — and does the plan's declared `difficulty` genuinely
    match what its steps structurally do?
    Deliberately small (a few flat fields) because plan panels run on small
    models: the check fields come FIRST so the model reasons before it commits
    to the votes.

    Voting is LAYERED across EIGHT independent layers — ``question_approval``
    (realistic, average-user, keyword-retrievable question), ``plan_approval``
    (steps produce exactly what the question asks), ``table_usage_approval``
    (every table justified by the question), ``expected_result_approval``,
    ``difficulty_approval`` (the declared `difficulty` matches the steps'
    structural complexity), ``convergence_approval`` (the steps converge
    into ONE final result rather than leaving independent branches
    uncombined), ``metric_combination_approval`` (any figure blended
    from 2+ tables into one output value is dimensionally sound, never a
    silent sum across incommensurate units or undifferentiated time
    periods), and ``topic_linkage_approval`` (the question names the
    table's specific real-world program/initiative when its identity
    hinges on one, rather than a generic activity that could just as
    plausibly belong to some other, unrelated program — AND, when the
    table is one of several time-vintages of the same subject that
    open-data portals routinely republish, names the specific period
    that pins down THIS vintage rather than a vague/absent one that
    could equally describe a different year's edition) — voted
    independently, and the panel majority-aggregates each
    layer on its own — the plan passes only when ALL layer majorities approve
    (see ``JudgePanel._aggregate`` with ``vote_fields``). ``approved`` is
    derived, never voted directly.
    """

    question_check: str = Field(
        ...,
        description=(
            "One or two sentences: does the question pinpoint ONE specific "
            "topic in these tables, concisely, phrased like an average "
            "non-technical user seeking an insight? Quote the anchoring "
            "term(s), or what makes it vague/verbose/technical. Also check "
            "PINPOINTING: does the question's vocabulary narrow to this "
            "table's specific topic (a domain/category qualifier, a "
            "program/agency name, a place, a period), or stay generic "
            "enough to describe many unrelated tables just as well? Judge "
            "this on the question's own specificity, not just whether "
            "question_keywords happens to retrieve the table — a generic "
            "term can retrieve it by luck of that table's own index entry "
            "while still failing to tell a reader which real-world topic "
            "it's about. Also check "
            "TEMPORAL SCOPE: if the table is "
            "tied to a fixed period (a year, a date range, a snapshot date) "
            "rather than an ongoing feed, the question must state that "
            "period, not read as if the data were current. Also check that "
            "it never narrates a `clean` step's own technical criteria "
            "(non-numeric/null exclusions, outlier filtering, bad-token "
            "literals) — that precision belongs in "
            "expected_result_description, not in what a curious "
            "non-technical user would ask."
        ),
    )
    alignment_check: str = Field(
        ...,
        description=(
            "One or two sentences: do the plan's steps, in order, produce "
            "exactly what the question asks — nothing core missing, nothing "
            "unjustified added? Name any missing or unjustified step. A "
            "`clean` step's basis must be a genuine data-quality defect "
            "(missing/corrupted/sentinel), never mere statistical rarity — "
            "dropping a `numeric_outliers`-flagged value that's otherwise a "
            "plausible, well-formed number is an unjustified filter, not "
            "hygiene, especially when it feeds a later `sum`/'total' step "
            "(it silently understates the true total)."
        ),
    )
    table_check: str = Field(
        ...,
        description=(
            "One or two sentences: is EVERY provided table genuinely needed "
            "to answer the question, per a CONCRETE, ARTICULATED "
            "justification — not a bare assertion? Name any table whose "
            "justification does not hold: topically unrelated, its "
            "contribution changes nothing in the answer, or the "
            "justification is generic/shallow boilerplate that names "
            "nothing specific about that table (apply the swap test: could "
            "the same sentence be pasted under a different table and still "
            "sound plausible?)."
        ),
    )
    unjustified_tables: List[str] = Field(
        default_factory=list,
        description=(
            "Aliases of tables whose justification does not hold — whether "
            "topically unrelated, inconsequential to the answer, or too "
            "generic/shallow to count as an articulated argument (e.g. "
            "['Table_2']). Empty when every table has a concrete, specific "
            "justification. Filling this means the QUESTION (and/or the "
            "justification itself) must be reframed so it genuinely, "
            "concretely needs the table — never that the table should be "
            "dropped."
        ),
    )
    expected_result_check: str = Field(
        ...,
        description=(
            "One or two sentences: do the plan's `expected_result_type` "
            "(table/list/number/text/boolean) and "
            "`expected_result_description` match what the question actually "
            "asks for and what the steps produce? Name the specific mismatch "
            "if any: a \"how many...\" question promising a table, a "
            "per-group breakdown promising a single number, or a "
            "description that contradicts the steps' final output."
        ),
    )
    difficulty_check: str = Field(
        ...,
        description=(
            "One or two sentences: the payload's `computed_difficulty` "
            "block gives `structural_tier`/`data_engineering_tier` already "
            "computed by CODE from this plan's `steps` (step counts, "
            "chaining, branching, group/aggregation cardinality) — take "
            "`structural_tier` as given, never re-derive it. Your only job "
            "is the two things code cannot decide, both on the "
            "DATA-ENGINEERING axis: (1) PATTERN DISTINCTNESS — when that "
            "axis is `medium` from 2+ technique 'flag'/'bucket' derive "
            "steps, do they actually preserve genuinely DIFFERENT "
            "messy-data patterns, or double-mark the same signal? (2) "
            "BUCKET-LOGIC REALISM — when "
            "`dq_hard_pending_your_confirmation` is true (a bucket step "
            "already structurally feeds a group/aggregate), does its "
            "branching (3+ categories) reflect real domain judgment, or an "
            "arbitrary split? Downgrade the data-engineering tier one "
            "level for either failure, then compare `max(structural_tier, "
            "data_engineering_tier)` to the declared `difficulty`. Name the "
            "concrete mismatch using `computed_difficulty.breakdown`. The "
            "fix is always in the STEPS, never in relabeling — do not write "
            "'adjust/change the difficulty to X' anywhere in this check or "
            "in `suggestions`, even as one of two options alongside a real "
            "steps-based fix."
        ),
    )
    convergence_check: str = Field(
        ...,
        description=(
            "One or two sentences: do the plan's steps converge into ONE "
            "final result, or do two or more independently-produced "
            "branches (each from its own table(s), sharing no combining "
            "step) get left standing side by side under one question? "
            "Name the missing combining step if any (a join/union on a "
            "shared key, a correlation/comparison, or a final synthesis "
            "step) — always passes (say so briefly) for a single-table or "
            "single-chain plan."
        ),
    )
    metric_combination_check: str = Field(
        ...,
        description=(
            "One or two sentences, only when a derive/aggregate step blends "
            "figures from 2+ tables into ONE output value (a 'combined "
            "total', a blended score): is the combination dimensionally "
            "sound, or does it silently sum raw values on incommensurate "
            "units/scales (e.g. a COUNT + a geometric-area SUM + a dollar "
            "SUM), fold different time periods' figures into one "
            "undifferentiated total that erases which period contributed "
            "what, OR sum same-unit figures (e.g. two plain counts) from "
            "conceptually unrelated categories/processes with no natural "
            "shared identity — ask whether a domain expert would recognize "
            "the SUM ITSELF as one coherent, nameable quantity, and treat a "
            "generic thematic label ('combined civic activity') as NOT "
            "sufficient evidence it is one (apply the same swap test as "
            "Check 3: would this exact justification sound equally "
            "plausible pasted over a different arbitrary trio of columns?). "
            "Comparing/contrasting/ratio-ing figures ACROSS different time "
            "periods is NOT a flaw when that comparison is the question's "
            "own insight and each period's figure stays distinct in the "
            "output — only an opaque additive sum across periods is. Name "
            "the specific unsound combination if any — always passes (say "
            "so briefly) for a plan with no cross-table blended metric."
        ),
    )
    topic_linkage_check: str = Field(
        ...,
        description=(
            "One or two sentences: does the table's own description/"
            "keywords show its identity hinges on ONE specific NAMED "
            "program/initiative/agency, narrower than the generic activity "
            "it falls under? If so, does the question actually name that "
            "specific program, or does it only describe the generic "
            "activity in a way that could just as plausibly belong to a "
            "different, unrelated program? SEPARATELY: is this table one "
            "of several time-vintages of the same subject that open-data "
            "portals routinely republish (an annual snapshot, a periodic "
            "refresh)? If so, does the question state the SPECIFIC period "
            "that pins this table's vintage down, or only a vague/generic "
            "time reference ('in recent years', 'historically') or none at "
            "all, that could equally describe a different year's edition? "
            "Judge both independently of Check 1 — a question can be "
            "concise, concrete, and even keyword-retrievable while still "
            "leaving a reader unable to tell WHICH specific real-world "
            "program or WHICH time-vintage it's about. Always passes (say "
            "so briefly) when the table has no single named program "
            "distinguishing it from its generic category and is not one of "
            "several time-vintages of the same subject, or when the "
            "question already names the specific program and period."
        ),
    )
    question_approval: bool = Field(
        ...,
        description=(
            "Vote on the QUESTION layer (Check 1): true when the question "
            "reads like an average, non-technical user wrote it AND its "
            "topical vocabulary overlaps the tables' keywords enough to be "
            "retrievable."
        ),
    )
    plan_approval: bool = Field(
        ...,
        description=(
            "Vote on the PLAN layer (Check 2): true when the steps, in "
            "order, produce exactly what the question asks — nothing core "
            "missing, nothing unjustified added."
        ),
    )
    table_usage_approval: bool = Field(
        ...,
        description=(
            "Vote on the TABLE-USAGE layer (Check 3): true when every "
            "provided table is genuinely required by the question. Must be "
            "false whenever unjustified_tables is non-empty."
        ),
    )
    expected_result_approval: bool = Field(
        default=True,
        description=(
            "Vote on the EXPECTED-RESULT layer (Check 4): true when the "
            "plan's expected_result_type and expected_result_description "
            "match both the question's ask and the steps' final output. "
            "False when the promised shape contradicts either (e.g. a "
            "\"how many...\" question with expected_result_type 'table', "
            "or a per-borough breakdown promising 'number'). The declared "
            "shape is mechanically enforced against the executed result "
            "downstream, so a wrong declaration here dooms every "
            "generation attempt."
        ),
    )
    difficulty_approval: bool = Field(
        default=True,
        description=(
            "Vote on the DIFFICULTY layer (Check 5): true when the declared "
            "`difficulty` equals `max(computed_difficulty.structural_tier, "
            "data_engineering_tier)` — the structural half is a given, "
            "code-computed fact; the data-engineering half may be one tier "
            "lower than `computed_difficulty` claims if pattern-distinctness "
            "or bucket-logic-realism (see `difficulty_check`) doesn't hold. "
            "False on any resulting mismatch. This layer NEVER approves "
            "changing the difficulty label itself — a mismatch is always "
            "fixed by adjusting the STEPS to match the label, since in a "
            "batch the label is a fixed per-slot assignment (one easy, one "
            "medium, one hard)."
        ),
    )
    convergence_approval: bool = Field(
        default=True,
        description=(
            "Vote on the CONVERGENCE layer (Check 6): true when the plan's "
            "steps converge into ONE final result — via a join/union on a "
            "shared key, a correlation or comparison, or a final "
            "synthesizing step — OR the plan is single-table/single-chain "
            "(this layer always passes then). False when the plan computes "
            "two or more genuinely independent results from separate "
            "branches and never combines them, leaving them reported side "
            "by side under one question. Independent of Check 3: every "
            "table can be individually well-justified (table_usage_approval "
            "true) while nothing ever ties their outputs together "
            "(convergence_approval false)."
        ),
    )
    metric_combination_approval: bool = Field(
        default=True,
        description=(
            "Vote on the METRIC-COMBINATION layer (Check 7): true when "
            "either no step blends figures from 2+ tables into one output "
            "value, OR it does and the combination is dimensionally AND "
            "conceptually sound. False when a step sums/averages raw "
            "values on incommensurate units/scales (a count + an area + a "
            "dollar sum), sums figures from genuinely different time "
            "periods into one undifferentiated total, OR sums same-unit "
            "figures (e.g. two plain counts) from unrelated categories/"
            "processes with no natural shared identity — a generic "
            "thematic label alone does not establish one. NOT false merely "
            "because the plan spans multiple time periods or tables — only "
            "when their figures are summed away into one opaque number "
            "rather than kept distinct or combined via a dimensionally-"
            "valid ratio/rate. Independent of Check 6: branches can "
            "converge via a sound join (convergence_approval true) through "
            "an unsound additive step (metric_combination_approval false)."
        ),
    )
    topic_linkage_approval: bool = Field(
        default=True,
        description=(
            "Vote on the TOPIC-LINKAGE layer (Check 8): true when the "
            "table has no single named program/initiative distinguishing "
            "it from its generic category (or the question names that "
            "specific program), AND either the table is not one of several "
            "time-vintages of the same subject or the question states the "
            "specific period that pins this table's vintage down. False "
            "when the table's identity hinges on a specific named program/"
            "initiative/agency but the question only ever describes the "
            "generic activity it falls under, OR when this table is one of "
            "several period-vintages of the same subject (an annual "
            "snapshot, a periodic refresh — per its own description) but "
            "the question states no period, or only a vague one ('in "
            "recent years', 'historically') that wouldn't distinguish this "
            "vintage from a different year's edition — even if that "
            "phrasing is concrete and retrievable. Independent of Check 1: "
            "a question can pass question_approval (concise, retrievable, "
            "no jargon, and even mentioning *some* period so it doesn't "
            "read as live data) while still failing here, because "
            "retrievability and live/current phrasing are about whether a "
            "search index finds the table and whether the tense is right — "
            "not whether a reader of the question alone would know which "
            "real-world program or which specific year's edition it's "
            "about."
        ),
    )
    approved: bool = Field(
        default=False,
        description=(
            "Overall verdict; always question_approval AND plan_approval "
            "AND table_usage_approval AND expected_result_approval AND "
            "difficulty_approval AND convergence_approval AND "
            "metric_combination_approval AND topic_linkage_approval "
            "(recomputed server-side, so just set it consistently)."
        ),
    )

    feedback: str = Field(
        ...,
        description=(
            "What is missing or should improve. Approved: one sentence on why "
            "the question, plan, and table usage are sound. Rejected: the "
            "specific flaw per failed layer, quoting the offending question "
            "part, step, or table justification."
        ),
    )
    suggestions: str = Field(
        ...,
        description=(
            "Empty string if approved. If rejected: one actionable sentence "
            "per failed layer — a rewrite of the question and/or the concrete "
            "step fix. For an unjustified table, ALWAYS suggest reframing the "
            "question so the table becomes necessary; NEVER suggest dropping "
            "the table (the code is required to use every table). For "
            "a wrong expected-result layer, state the correct "
            "expected_result_type and/or the corrected description — "
            "whichever of the declaration or the steps is actually wrong. "
            "For a difficulty mismatch, name the concrete step(s)/table(s) "
            "to add, remove, or simplify so the plan's actual "
            "complexity genuinely matches its assigned tier (e.g. 'cut this "
            "down to a single filter and one aggregate to match `easy`', or "
            "'add a second grouped aggregation to match `medium`') — NEVER "
            "suggest changing the `difficulty` value itself; it is a fixed "
            "per-slot batch assignment, not yours (or the planner's) to "
            "reassign here. Relabeling is always wrong, even when the steps "
            "genuinely read as a different tier — the fix is still to "
            "simplify/compound the STEPS back down to the declared tier, "
            "never to relabel, and this holds even if you also name a "
            "legitimate steps-based fix in the SAME sentence — state ONLY "
            "the steps-based fix, with no relabeling option alongside it. "
            "For a convergence failure, name "
            "the specific combining step the plan is missing (a join/union "
            "on a shared key, a correlation/comparison, or a final "
            "synthesis step) — never suggest dropping either branch, and "
            "never suggest changing the question instead of adding the "
            "missing step. For a metric-combination failure, name the "
            "specific dimensionally-invalid combination and the fix — "
            "report the components as separate columns instead of summing "
            "them, or replace the additive sum with a dimensionally-sound "
            "rate/ratio/normalized index; never suggest dropping a table "
            "(that is Check 3's territory, not this one's). For a "
            "topic-linkage failure, name the table's specific program/"
            "initiative/agency (from its description or keywords) and say "
            "the question's own prose must weave it in — not just "
            "question_keywords — while still reading like an average "
            "user's question."
        ),
    )

    @model_validator(mode="after")
    def _force_consistent_verdicts(self) -> "PlanJudgment":
        """The overall verdict and the table layer are derived, never free:
        ``approved`` is the AND of the eight layer votes, and flagged
        unjustified tables force the table layer down — so a judge can't
        e.g. list an unjustified table while voting the layer up."""
        if self.unjustified_tables:
            self.table_usage_approval = False
        self.approved = (
            self.question_approval
            and self.plan_approval
            and self.table_usage_approval
            and self.expected_result_approval
            and self.difficulty_approval
            and self.convergence_approval
            and self.metric_combination_approval
            and self.topic_linkage_approval
        )
        return self