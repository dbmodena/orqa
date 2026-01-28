from pydantic import BaseModel, Field, field_validator
from typing import List


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




class SQLQuery(BaseModel):
    
    question: str
    query: str
    tables: List[str]
    columns: List[str]
    difficulty: int



class SQLQuery(BaseModel):
    """
    A structured SQL query paired with a natural language question
    that expresses the intent of the query.
    """
    id: str = Field(
        ...,
        description="Stable unique identifier for the query (UUID or deterministic hash)."
    )
    question: str = Field(
        ...,
        description=(
            "A natural language question that the SQL query answers. "
            "The question must describe the business or analytical intent, "
            "NOT the SQL operations or schema structure."
        )
    )

    query: str = Field(
        ...,
        description="The SQL query that answers the natural language question."
    )

    tables: List[str] = Field(
        ...,
        description="List of table names used in the SQL query."
    )

    columns: List[str] = Field(
        ...,
        description="List of column names referenced in the SQL query."
    )

    difficulty: int = Field(
        ...,
        ge=1,
        le=5,
        description="Query complexity from 1 (simple) to 5 (very complex)."
    )


class SQLQuerySet(BaseModel):
    """
    Container for multiple SQL queries generated together.
    """

    queries: List[SQLQuery] = Field(
        ...,
        description="List of SQL queries with matching natural language questions."
    )