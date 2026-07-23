from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Literal, Union
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
    type: Literal["join", "union", "correlation", "other"] = Field(
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
    columns: List[str] = Field(
        default_factory=list,
        description="Columns that participate in the join or union relationship."
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
    difficulty: str = Field(
        ...,
        description=(
            "The difficulty level of the query: easy, medium, or hard. "
            "Reflects the complexity of the analytical operations and structure."
        )
    )
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
    meaningless_prediction_target = "meaningless_prediction_target"
    trivial                       = "trivial"
    # The code's implementation of the plan's APPROVED skill technique is
    # itself wrong/incomplete (e.g. an unencoded categorical feature passed to
    # TabPFN, lag features built on unsorted data, a causal estimate missing
    # one of the two counterfactual predictions) — distinct from
    # `meaningless_prediction_target` (a bad TARGET choice) and from
    # `partial_implementation` (a QUESTION requirement, not a skill-technique
    # one). See the injected per-skill "Skill Check" section in judge.md.
    skill_misuse                  = "skill_misuse"


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
    result_check: str = Field(
        ...,
        description=(
            "One or two sentences: does the executed result actually answer "
            "the question — right shape (single figure / ranked list / "
            "per-group table) and non-degenerate values? Quote representative "
            "value(s) from the result."
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
            "material gaps — omit minor stylistic observations."
        ),
    )
    violated_criteria: List[ViolatedCriterion] = Field(
        default_factory=list,
        description=(
            "All triggered criteria. Empty if approved. Reject only when a flaw "
            "is unambiguous and material."
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
        ...,
        description="True only if all checks pass and violated_criteria is empty.",
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


class Judgments(BaseModel):
    queries: List[Judgment] = Field(
        ...,
        description="One Judgment per input pair, in input order.",
    )


class PlanJudgment(BaseModel):
    """Verdict of one panel judge on a single structured query plan.

    The plan judge owns QUESTION quality, TABLE justification, and SKILL
    justification (the code judge no longer re-judges any of them): is the
    question concise, pinned to a specific topic, written like an average
    user seeking insights — do the plan's steps reflect it — does the
    question genuinely need every provided table — and does the plan's use
    of an ML skill, or its deliberate absence, match what the question asks
    for?
    Deliberately small (a few flat fields) because plan panels run on small
    models: the check fields come FIRST so the model reasons before it commits
    to the votes.

    Voting is LAYERED: ``question_approval`` (realistic, average-user,
    keyword-retrievable question), ``plan_approval`` (steps produce exactly
    what the question asks), ``table_usage_approval`` (every table justified
    by the question), and ``skill_approval`` (the plan's skill usage, or lack
    thereof, is justified by the question) are voted independently, and the
    panel majority-aggregates each layer on its own — the plan passes only
    when ALL layer majorities approve (see ``JudgePanel._aggregate`` with
    ``vote_fields``). ``approved`` is derived, never voted directly.
    """

    question_check: str = Field(
        ...,
        description=(
            "One or two sentences: does the question pinpoint ONE specific "
            "topic in these tables, concisely, phrased like an average "
            "non-technical user seeking an insight? Quote the anchoring "
            "term(s), or what makes it vague/verbose/technical."
        ),
    )
    alignment_check: str = Field(
        ...,
        description=(
            "One or two sentences: do the plan's steps, in order, produce "
            "exactly what the question asks — nothing core missing, nothing "
            "unjustified added? Name any missing or unjustified step."
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
    skill_check: str = Field(
        ...,
        description=(
            "One or two sentences: does the plan's use of an ML/predictive "
            "skill (task_types: classification, regression, timeseries, "
            "causal) — or its deliberate absence — match what the question "
            "asks for? Name the specific mismatch: a skill bolted on where "
            "the question is a plain aggregation/filter/lookup, a skill "
            "missing where the question explicitly asks for a prediction, "
            "forecast, or an intervention's causal effect, or a skill of "
            "the wrong type for what's asked (e.g. regression where the "
            "question wants a category). If the system prompt included one "
            "or more 'Skill Check' sections for this plan's proposed "
            "skill(s), apply those checks too — they cover failure modes "
            "specific to that skill that this general description can't."
        ),
    )
    predictor_check: str = Field(
        ...,
        description=(
            "One or two sentences, for a plan with a classification/"
            "regression/causal step only (always passes for plain plans and "
            "for timeseries, which predicts from the target's OWN history, "
            "not other columns): does each named feature/predictor/covariate "
            "have a plausible, explainable real-world link to the target — "
            "one a lay person would nod along to — or does the combination "
            "read as arbitrary numeric/categorical columns bolted together "
            "because they happened to be in the same table? Name the "
            "specific implausible pairing if any (e.g. predicting a system-"
            "wide total from two unrelated single-entity metrics, or one "
            "age-group's outcome from an unrelated age-group's share)."
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
    skill_approval: bool = Field(
        ...,
        description=(
            "Vote on the SKILL layer (Check 4): true when the plan's use of "
            "an ML skill, or its deliberate absence, is justified by the "
            "question — a skill present only where genuinely asked for, and "
            "absent only where none is needed."
        ),
    )
    predictor_approval: bool = Field(
        default=True,
        description=(
            "Vote on the PREDICTOR layer (Check 5): true when every named "
            "feature/predictor/covariate has a plausible link to the "
            "target, OR the plan is plain/timeseries (this layer always "
            "passes for those). False when classification/regression/"
            "causal predictors read as arbitrary column-grabbing with no "
            "stated or obvious real-world mechanism connecting them to the "
            "target. Independent of Check 4: a plan can genuinely need a "
            "prediction (skill_approval true) yet use implausible inputs "
            "for it (predictor_approval false)."
        ),
    )
    expected_result_approval: bool = Field(
        default=True,
        description=(
            "Vote on the EXPECTED-RESULT layer (Check 6): true when the "
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
    approved: bool = Field(
        default=False,
        description=(
            "Overall verdict; always question_approval AND plan_approval "
            "AND table_usage_approval AND skill_approval AND "
            "predictor_approval AND expected_result_approval (recomputed "
            "server-side, so just set it consistently)."
        ),
    )

    feedback: str = Field(
        ...,
        description=(
            "What is missing or should improve. Approved: one sentence on why "
            "the question, plan, table usage, skill usage, and predictor "
            "choice are sound. Rejected: the specific flaw per failed layer, "
            "quoting the offending question part, step, table justification, "
            "skill mismatch, or implausible predictor pairing."
        ),
    )
    suggestions: str = Field(
        ...,
        description=(
            "Empty string if approved. If rejected: one actionable sentence "
            "per failed layer — a rewrite of the question and/or the concrete "
            "step fix. For an unjustified table, ALWAYS suggest reframing the "
            "question so the table becomes necessary; NEVER suggest dropping "
            "the table (the code is required to use every table). For an "
            "unjustified skill layer, either add/retype the specific "
            "task_type or reframe the question, whichever actually matches "
            "the plan's real intent — but for a BOLTED-ON skill, name BOTH "
            "the removal (drop the task_type/step) AND its concrete plain-"
            "operation replacement (which group/aggregate/derive step now "
            "produces the answer over the same rows); dropping the skill "
            "with no replacement named is not an actionable suggestion. For "
            "an implausible predictor layer, name which specific feature(s) "
            "to drop or replace with a column that has a real, stated "
            "connection to the target — keep the skill, fix the inputs. For "
            "a wrong expected-result layer, state the correct "
            "expected_result_type and/or the corrected description — "
            "whichever of the declaration or the steps is actually wrong."
        ),
    )

    @model_validator(mode="after")
    def _force_consistent_verdicts(self) -> "PlanJudgment":
        """The overall verdict and the table layer are derived, never free:
        ``approved`` is the AND of the six layer votes, and flagged
        unjustified tables force the table layer down — so a judge can't
        e.g. list an unjustified table while voting the layer up."""
        if self.unjustified_tables:
            self.table_usage_approval = False
        self.approved = (
            self.question_approval
            and self.plan_approval
            and self.table_usage_approval
            and self.skill_approval
            and self.predictor_approval
            and self.expected_result_approval
        )
        return self