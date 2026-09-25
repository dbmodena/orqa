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
            "Retrieval keywords for this table. Each is a single word "
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
        """Remove duplicate table keywords, preserving order. No count cap."""
        seen = set()
        unique_keywords = []
        for kw in v:
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
            "The question's table-identifying retrieval terms: "
            "single words or short established terms appearing in the "
            "question whose combination matches the target table's index "
            "keywords — no phrases, no generic filler words."
        )
    )
    plan_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords extracted from the query plan."
    )
    table_links: List[QueryLink] = Field(
        default_factory=list,
        description="Descriptions of how tables should be joined or unioned in the final query."
    )

    @field_validator("question_keywords")
    @classmethod
    def limit_question_keywords(cls, v: List[str]) -> List[str]:
        """Remove duplicate question keywords, preserving order. No count cap."""
        seen = set()
        unique_keywords = []
        for kw in v:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        return unique_keywords

    @field_validator("plan_keywords")
    @classmethod
    def limit_plan_keywords(cls, v: List[str]) -> List[str]:
        """Remove duplicate plan keywords, preserving order. No count cap."""
        seen = set()
        unique_keywords = []
        for kw in v:
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
        description="Keywords extracted from the table analysis that are relevant for this query."
    )
    translated_keywords: List[str] = Field(
        default_factory=list,
        description="Translated keywords for this table in the detected language."
    )

    @field_validator("keywords")
    @classmethod
    def limit_keywords(cls, v: List[str]) -> List[str]:
        """Remove duplicate keywords, preserving order. No count cap."""
        seen = set()
        unique_keywords = []
        for kw in v:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        return unique_keywords

    @field_validator("translated_keywords")
    @classmethod
    def limit_translated_keywords(cls, v: List[str]) -> List[str]:
        """Remove duplicate translated keywords, preserving order. No count cap."""
        seen = set()
        unique_keywords = []
        for kw in v:
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
    # `tables` above. Difficulty is the slot's target tier, judged on the
    # QUESTION by the question judge and pinned onto the plan (see
    # prompting.models.SQLQueryPlan/PandasQueryPlan.difficulty); nothing on the
    # plan side estimates or votes on it. StatementClient.complete
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
            "Keywords capturing the question's intent. Linked to "
            "`question` — regenerate together if you rewrite the question."
        ),
    )
    translated_question: str = Field(
        default="",
        description="`question` translated into `detected_language`. Linked to `question`.",
    )
    translated_question_keywords: List[str] = Field(
        default_factory=list,
        description="`question_keywords` translated the same way. Linked to `question`.",
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
        """Remove duplicate question keywords, preserving order. No count cap."""
        seen = set()
        unique_keywords = []
        for kw in v:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        return unique_keywords

    @field_validator("translated_question_keywords")
    @classmethod
    def limit_translated_question_keywords(cls, v: List[str]) -> List[str]:
        """Remove duplicate translated question keywords, preserving order. No count cap."""
        seen = set()
        unique_keywords = []
        for kw in v:
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


# ── Benchmark solver (orqa.benchmark.solve) ─────────────────────────────────
#
# Three single-shot phases — keyword generation, table selection, code
# writing — for the round-trip benchmark solver: given ONLY a question (never
# the hidden ground truth), independently retrieve candidate tables and
# answer it, so the answer can be compared back against the ground truth
# (see orqa.benchmark.questions.evaluate_table_retrieval and
# orqa.benchmark.solve.compare_results). Deliberately three separate calls,
# not one — mirrors the QueryPlanner -> StatementClient split used by
# generation, without the judge/correction machinery layered on top of it
# there: a benchmark solver measures raw capability against retrieved,
# not-guaranteed-relevant context, so no phase retries itself here.

class SearchKeywords(BaseModel):
    """Phase 1: retrieval keywords extracted from a benchmark question."""

    detected_language: str = Field(
        ...,
        description=(
            "The natural language the question is written in (e.g. "
            "'English', 'Spanish', 'Italian', 'French') — your own read of "
            "it, not assumed from any other context."
        )
    )
    keywords: List[str] = Field(
        default_factory=list,
        description=(
            "Retrieval keywords extracted from the question, IN "
            "detected_language (the reverse index only matches that "
            "language): entities, topics, measures, place and time "
            "expressions. Single words or short established terms, never "
            "descriptive phrases."
        )
    )

    @field_validator("keywords")
    @classmethod
    def dedupe_keywords(cls, v: List[str]) -> List[str]:
        """Remove duplicate keywords, preserving order. No count cap."""
        seen = set()
        unique = []
        for kw in v:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        return unique


class SolverTableUsage(BaseModel):
    """One retrieved table the solver decided to use, and which of its
    columns for."""

    resource_id: str = Field(
        ...,
        description="The dataset's resource id, exactly as returned by the reverse index search."
    )
    columns_used: List[str] = Field(
        default_factory=list,
        description="The columns from this table actually needed to answer the question."
    )


# Same 5 shapes as prompting.models._RESULT_TYPES, duplicated here (not
# imported) to avoid a structured_outputs <-> prompting.models import cycle
# (prompting.models already imports Table/QueryLink FROM this module).
_SOLVER_RESULT_TYPES = Literal["table", "list", "number", "text", "boolean"]


class TableSelection(BaseModel):
    """Phase 2: which retrieved candidate table(s) — not all of them are
    necessarily relevant — actually answer the question, and the shape of
    that answer."""

    tables: List[SolverTableUsage] = Field(
        default_factory=list,
        description=(
            "The candidate table(s) that actually answer the question, "
            "each with the columns used from it. Candidates not listed "
            "here were judged irrelevant and must be ignored in Phase 3 — "
            "not every retrieved table has to be used."
        )
    )
    expected_result_type: _SOLVER_RESULT_TYPES = Field(
        ...,
        description=(
            "The SHAPE of the final answer: number (\"how many/how "
            "much...\"), boolean (yes/no), text (\"which single X...\"), "
            "list (one ordered sequence of values), or table (per-group "
            "breakdowns, rankings, any multi-column result)."
        )
    )
    reasoning: str = Field(
        default="",
        description="1-2 sentences: why these table(s)/columns, and why the others (if any) were rejected."
    )
    no_viable_selection: bool = Field(
        default=False,
        description=(
            "True when NONE of the retrieved candidates can answer the "
            "question — retrieval missed. When true, `tables` may be "
            "empty; never force a selection you don't believe answers "
            "the question just to populate this field."
        )
    )


class SolverCode(BaseModel):
    """Phase 3: the code answering the question, given Phase 2's table/column
    selection — code only, table usage was already decided."""

    code: str = Field(
        ...,
        description=(
            "Executable code (Pandas or SQL, per the requested kind) that "
            "answers the question using exactly the tables/columns Phase 2 "
            "selected — no other tables. Pandas: end in an assignment/"
            "expression whose value is the answer; bracket/getitem column "
            "access only, never dot access. SQL: a single query aliasing "
            "table variables by their given names."
        )
    )


class PlanJudgment(BaseModel):
    """Verdict of one panel judge on a query plan whose QUESTION is frozen —
    the question was written and approved upstream
    (``orqa.agent.agents.QuestionStage``: retrievability, readability, topic
    linkage, grounding, table necessity), so none of that is asked here. Four
    layers, majority-aggregated one by one (``JudgePanel._aggregate`` with
    ``vote_fields``): ``plan_approval`` (the steps produce exactly what the
    question asks), ``table_usage_approval`` (every table's role is
    justified), ``expected_result_approval`` (the declared result is the
    natural conclusion of the steps AND accounts for every analysis the plan
    performs — two uncombined branches are exactly a declaration that fails
    to account for both) and ``metric_combination_approval`` (any figure
    blended from 2+ tables into one output value is dimensionally sound).
    DIFFICULTY is deliberately absent: it is the slot's target, judged on the
    QUESTION upstream, never estimated or voted on at the plan. Deliberately small (a few flat fields) because plan panels run
    on small models: the check fields come FIRST so the model reasons before
    it commits to the votes. ``approved`` is derived, never voted."""

    alignment_check: str = Field(
        ...,
        description=(
            "1-2 sentences applying Check 2 (steps produce exactly what the "
            "question asks). Name the missing or unjustified step, or state "
            "that none applies."
        ),
    )
    table_check: str = Field(
        ...,
        description=(
            "1-2 sentences applying Check 3 (every table genuinely required, "
            "per the swap test). Name each table whose justification fails, "
            "or state that none does."
        ),
    )
    unjustified_tables: List[str] = Field(
        default_factory=list,
        description=(
            "Aliases (e.g. ['Table_2']) of tables failing Check 3; empty when "
            "every table is justified. The question is frozen, so the fix is "
            "in the table's `reason` and the steps — never a reworded question."
        ),
    )
    expected_result_check: str = Field(
        ...,
        description=(
            "1-2 sentences applying Check 4 (the declared result matches the "
            "question, is the natural conclusion the steps build toward, and "
            "accounts for EVERY analysis the plan performs). Name the specific "
            "mismatch or the unaccounted-for result, or state that none applies."
        ),
    )
    metric_combination_check: str = Field(
        ...,
        description=(
            "1-2 sentences applying Check 5 (a figure blended from 2+ tables "
            "into one output value is dimensionally and conceptually sound). "
            "Name the unsound combination, or state briefly that it passes."
        ),
    )
    plan_approval: bool = Field(
        ..., description="Vote on Check 2 (steps match the question). See alignment_check."
    )
    table_usage_approval: bool = Field(
        ...,
        description=(
            "Vote on Check 3 (every table required). Must be false whenever "
            "unjustified_tables is non-empty."
        ),
    )
    expected_result_approval: bool = Field(
        default=True, description="Vote on Check 4. See expected_result_check."
    )
    metric_combination_approval: bool = Field(
        default=True,
        description=(
            "Vote on Check 5; always true when no step blends figures across "
            "tables. See metric_combination_check."
        ),
    )
    approved: bool = Field(
        default=False,
        description="Derived: the AND of the four votes (recomputed server-side).",
    )
    feedback: str = Field(
        ...,
        description=(
            "Approved: one sentence on why all layers hold. Rejected: the "
            "specific flaw per failed layer, quoting the offending step, "
            "justification or declaration."
        ),
    )
    suggestions: str = Field(
        ...,
        description=(
            "Empty when approved. Otherwise one actionable sentence per failed "
            "layer. Never suggest dropping a table, dropping a branch, "
            "changing `difficulty`, or changing the (frozen) question."
        ),
    )

    @model_validator(mode="after")
    def _force_consistent_verdicts(self) -> "PlanJudgment":
        if self.unjustified_tables:
            self.table_usage_approval = False
        self.approved = (
            self.plan_approval
            and self.table_usage_approval
            and self.expected_result_approval
            and self.metric_combination_approval
        )
        return self


class QuestionTableReason(BaseModel):
    """Why a question needs ONE of its tables — the writer's own claim, handed
    to the question judge, which tests it (Check 4, table necessity)."""

    alias: str = Field(
        ...,
        description="The table alias exactly as listed in TABLE ALIASES.",
    )
    reason: str = Field(
        ...,
        description=(
            "ONE sentence naming what this table contributes to THIS "
            "question: the specific measure, group, filter or period the "
            "question asks about that comes from it. Concrete enough that it "
            "could not be pasted under a different table; never boilerplate "
            "such as 'for context' or 'provides relevant data'. May name "
            "columns and values (it is never shown to the asker)."
        ),
    )


class QuestionDraft(BaseModel):
    """One candidate question for one slot — see ``orqa.agent.agents.
    QuestionGenerator``. Deliberately just the question and why it needs each
    table: no steps, no result shape, no keyword list. The retrieval keywords
    are the verified anchor the question was BUILT FROM, recomputed
    deterministically from the final text (``question_gate.covered_terms``),
    never authored here. ``table_reasons`` go to the question judge with the
    question (they are not passed on to the planner)."""

    slot: int = Field(
        ...,
        description="The slot number this question answers, exactly as listed in the prompt.",
    )
    question: str = Field(
        ...,
        description=(
            "The question, in plain everyday words as an average user "
            "would write it — never a column header, code or identifier — "
            "built around the anchor terms of the slot that read naturally, "
            "in their exact wording."
        ),
    )
    table_reasons: List[QuestionTableReason] = Field(
        ...,
        description=(
            "Exactly one entry per table alias in TABLE ALIASES (also for a "
            "single table): why this question needs that table. Written "
            "after the question and consistent with it."
        ),
    )
    question_keywords: List[str] = Field(
        default_factory=list,
        description=(
            "Only when the slot lists NO anchor terms: 3-6 distinctive "
            "single words or short terms that literally appear in the "
            "question. Leave empty otherwise."
        ),
    )

    @field_validator("question_keywords")
    @classmethod
    def dedupe(cls, v: List[str]) -> List[str]:
        seen: set = set()
        out = []
        for item in v:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out


class QuestionDraftSet(BaseModel):
    """One :class:`QuestionDraft` per slot listed in the prompt."""

    questions: List[QuestionDraft] = Field(default_factory=list)


class QuestionJudgment(BaseModel):
    """Verdict of one panel judge on a candidate QUESTION, before any plan
    exists.

    Owns what can be judged from the question and the tables alone —
    readability for an average user, topic linkage, whether the tables can
    ground it, whether it needs every table, and whether it demands about as
    much analysis as its slot's target difficulty. Retrievability is NOT voted
    here: it is measured against the real retriever panel before the judges
    are ever called (``orqa.agent.utility.question_gate``). Step alignment,
    result coherence and metric soundness are the PLAN judge's, later.

    Voted in five independent layers and majority-aggregated per layer (see
    ``JudgePanel._aggregate`` with ``vote_fields``); ``approved`` is derived.
    Check fields come first so a small model reasons before it votes.
    """

    readability_check: str = Field(
        ...,
        description=(
            "1-2 sentences applying Check 1 (reads like an average, "
            "non-technical open-data user; one clear ask). Name and quote "
            "the flaw, or state that none applies."
        ),
    )
    topic_check: str = Field(
        ...,
        description=(
            "1-2 sentences applying Check 2 (names the specific program/"
            "initiative/agency the table is about, not just the generic "
            "activity; states a fixed period). Name the vague phrasing, or "
            "state that none applies."
        ),
    )
    grounding_check: str = Field(
        ...,
        description=(
            "1-2 sentences applying Check 3 (every measure, entity, place "
            "and period the question asks about exists in these tables). "
            "Name what is missing, or state that none is."
        ),
    )
    table_check: str = Field(
        ...,
        description=(
            "1-2 sentences applying Check 4 (answering needs every listed "
            "table, as each table's stated reason claims). Name each table "
            "the question does not need or whose reason is missing, generic "
            "or not what the question asks for, or state that none is."
        ),
    )
    difficulty_check: str = Field(
        ...,
        description=(
            "1-2 sentences applying Check 5 (the question demands about as "
            "much analysis as its slot's target difficulty). Count the "
            "analytical operations it needs (subset, group, aggregate, rank/"
            "top-N, correlate) and compare with the slot's tier, or state "
            "that the data cannot support the tier."
        ),
    )
    unjustified_tables: List[str] = Field(
        default_factory=list,
        description=(
            "Aliases of tables the question does not genuinely need; empty "
            "when every table is needed (always empty for a single table)."
        ),
    )
    readability_approval: bool = Field(..., description="Vote on Check 1.")
    topic_linkage_approval: bool = Field(default=True, description="Vote on Check 2.")
    grounding_approval: bool = Field(..., description="Vote on Check 3.")
    table_necessity_approval: bool = Field(
        default=True,
        description=(
            "Vote on Check 4; must be false whenever unjustified_tables is "
            "non-empty. Always true for a single table."
        ),
    )
    difficulty_approval: bool = Field(
        ...,
        description=(
            "Vote on Check 5: false when the question needs clearly less "
            "(or clearly more) analysis than its slot's difficulty asks for."
        ),
    )
    approved: bool = Field(
        default=False,
        description="Derived: the AND of the five votes (recomputed server-side).",
    )
    feedback: str = Field(
        ...,
        description=(
            "Approved: one sentence on why all layers hold. Rejected: the "
            "specific flaw per failed layer, quoting the offending words."
        ),
    )
    suggestions: str = Field(
        ...,
        description=(
            "Empty when approved. Otherwise one actionable sentence per "
            "failed layer describing how to REWRITE THE QUESTION (for the "
            "difficulty layer: which analysis to add or drop). Never "
            "suggest dropping a required retrieval-anchor term."
        ),
    )

    @model_validator(mode="after")
    def _force_consistent_verdicts(self) -> "QuestionJudgment":
        if self.unjustified_tables:
            self.table_necessity_approval = False
        self.approved = (
            self.readability_approval
            and self.topic_linkage_approval
            and self.grounding_approval
            and self.table_necessity_approval
            and self.difficulty_approval
        )
        return self


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
            "PRESENT-RESULT layer's check text (Check 2): 1-2 sentences on "
            "whether the executed result answers the question, quoting "
            "representative value(s). Empty/degenerate/trivial results belong "
            "here, never under requirements_check."
        ),
    )
    requirements_check: str = Field(
        ...,
        description=(
            "PLAN-COMPLIANCE layer's check text (Check 1): the bidirectional "
            "IMPLEMENTED/MISSING and JUSTIFIED/UNJUSTIFIED mapping, with the "
            "grounding phrase named for each JUSTIFIED verdict."
        ),
    )
    violated_criteria: List[ViolatedCriterion] = Field(
        default_factory=list,
        description=(
            "All triggered criteria, per the criterion table in the "
            "instructions; empty if approved."
        ),
    )
    plan_compliance_approval: bool = Field(
        ...,
        description=(
            "Vote on Check 1 (code implements what the question asks). "
            "See requirements_check."
        ),
    )
    present_result_approval: bool = Field(
        ...,
        description=(
            "Vote on Check 2 (the executed result actually answers the "
            "question). See result_check."
        ),
    )
    feedback: str = Field(
        ...,
        description=(
            "Approved: why the result is meaningful and the code justified. "
            "Rejected: quote the exact operations or result values that "
            "triggered each criterion."
        ),
    )
    approved: bool = Field(
        default=False,
        description=(
            "Derived: plan_compliance_approval AND present_result_approval "
            "(recomputed server-side, so just set it consistently)."
        ),
    )
    response: str = Field(
        ...,
        description=(
            "3-4 sentence business insight citing the actual computed value(s) "
            "from the result. Empty string if not approved."
        ),
    )
    translated_response: str = Field(
        ...,
        description="Response in the detected target language. Empty string if not approved.",
    )
    suggestions: str = Field(
        ...,
        description=(
            "Empty if approved. If rejected: one actionable sentence per "
            "criterion prefixed [FIX QUERY]. Never propose removing a table or "
            "its join/union."
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
