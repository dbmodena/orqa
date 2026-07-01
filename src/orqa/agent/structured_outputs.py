from pydantic import BaseModel, Field, field_validator
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
        description="A concise business description of the table and its role in the final query."
    )
    table_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords that capture the most important concepts in this table (max 10)."
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
        description="A business-focused natural language question that the final query should answer, phrased as a non-expert user would ask."
    )
    query_plan: str = Field(
        ...,
        description="A high-level plan describing how the query should be constructed."
    )
    question_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords that capture the intent of the question (max 10)."
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
            "A business-oriented justification for why this table is needed. "
            "Explain what unique information it contributes to answering the question "
            "and why the query cannot be answered without it."
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
    #id: int = Field(..., description=(
    #    "Incremental identifier of the query."
    #))
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
    translated_question:str = Field(
        ...,
        description=(
            "A natural language question that the code answers translated in the detected language."
        )
    )
    topic: str = Field(
        ...,
        description=(
            "A short business topic or theme that summarizes the query's primary analytical concern. "
            "Use this field as a reference alongside keywords when planning, questioning, and interpreting results."
        )
    )
    story: str = Field(
        ...,
        description=(
            "A short narrative describing the business insight or storyline behind this query. "
            "This should capture why the query matters and what story it tells about the data."
        )
    )
    detected_language : str = Field(
        ...,
        description=(
            "Detected language from the provided dataset. "
        )
    )
    code: str = Field(
        ...,
        description=(
            "The executable code (SQL query or Pandas operation) that answers "
            "the natural language question."
        )
    )
    tables: List[Table] = Field(
        ...,
        description="List of tables involved in the query and why they're employed in the query."
    )
    query_plan: str = Field(
        default="",
        description="A high-level plan describing how the query was constructed."
    )
    question_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords associated with the question and query intent (max 10). Should include keywords from tables."
    )
    translated_question_keywords: List[str] = Field(
        default_factory=list,
        description="Translated keywords associated with the question in the detected language (max 10)."
    )

    @field_validator("question_keywords")
    @classmethod
    def limit_question_keywords_query(cls, v: List[str]) -> List[str]:
        """Limit question keywords to max 10 and remove duplicates"""
        seen = set()
        unique_keywords = []
        for kw in v[:10]:  # Take only first 10
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
        for kw in v[:10]:  # Take only first 10
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
    vocabulary_mismatch    = "vocabulary_mismatch"
    too_broad              = "too_broad"
    unclear_result         = "unclear_result"
    partial_implementation = "partial_implementation"
    over_engineering       = "over_engineering"
    silent_filter_bias     = "silent_filter_bias"
    unjustified_table      = "unjustified_table"
    disjointed_query       = "disjointed_query"


class Judgment(BaseModel):
    id: int = Field(..., description="Query identifier copied from input.")
    vagueness_check: str = Field(
        ...,
        description=(
            "Start with YES or NO (does the question contain no domain-specific "
            "anchoring term at all?), then one sentence quoting the term(s) that "
            "anchor or fail to anchor it."
        ),
    )
    requirements_check: str = Field(
        ...,
        description=(
            "Bidirectional coverage. List each core question requirement as "
            "IMPLEMENTED or MISSING. List each query operation as JUSTIFIED or "
            "UNJUSTIFIED. Flag only material gaps — omit minor stylistic observations."
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
            "Approved: why the result is meaningful and complexity justified. "
            "Rejected: quote the exact terms or operations that triggered each criterion."
        ),
    )
    approved: bool = Field(
        ...,
        description="True only if all checks pass and violated_criteria is empty.",
    )
    response: str = Field(
        ...,
        description="3-4 sentence business insight. Empty string if not approved.",
    )
    translated_response: str = Field(
        ...,
        description="Response in the detected target language. Empty string if not approved.",
    )
    suggestions: str = Field(
        ...,
        description=(
            "Empty string if approved. If rejected: one sentence per criterion "
            "prefixed [FIX QUESTION], [FIX QUERY], or [FIX QUESTION & QUERY]."
        ),
    )


class Judgments(BaseModel):
    queries: List[Judgment] = Field(
        ...,
        description="One Judgment per input pair, in input order.",
    )