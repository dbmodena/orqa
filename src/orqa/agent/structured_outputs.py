from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Union

class DatasetAnalysisResult(BaseModel):
    """Result of dataset analysis with UNION and JOIN suggestions"""

    UNION: List[str] = Field(
        description="List of column names that should be combined using UNION (similar columns across datasets)"
    )
    JOIN: List[str] = Field(
        description="List of column names that can be used as JOIN keys (common identifiers)"
    )


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
            "The question must describe the business or analytical intent, "
            "NOT the operations or technical implementation details."
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




class QuerySet(BaseModel):
    """
    Container for multiple queries (SQL or Pandas) generated together.
    """
    queries: List[Query] = Field(
        ...,
        description="List of queries with matching natural language questions."
    )


class Judgment(BaseModel):
    """
    Structured judgement
    """
    id: int = Field(..., description=(
        "Identifier of the query."
    ))
    Feedback: str = Field(
        ...,
        description=(
            "Natural language feedback to the generated query."
        )
    )
    Approved: bool = Field(..., description=(
        "Binary classification of the generated query."
    ))
    Response: str = Field(
        ...,
        description=(
            "Natural language response to the generated query."
        )
    )


class Judgments(BaseModel):
    queries: List[Judgment] = Field(
        ...,
        description="List of queries with matching natural language questions."
    )