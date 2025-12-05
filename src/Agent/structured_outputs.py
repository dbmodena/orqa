from pydantic import BaseModel, Field, ValidationError
from typing import List

class CSVAnalysisResult(BaseModel):
    """Result of CSV analysis with UNION and JOIN suggestions"""
    UNION: List[str] = Field(
        description="List of column names that should be combined using UNION (similar columns across datasets)"
    )
    JOIN: List[str] = Field(
        description="List of column names that can be used as JOIN keys (common identifiers)"
    )



class Task(BaseModel):
    """Definition of a Task(JOIN, UNION or JOIN-CORRELATION) over a set of columns""" 
    Task: str = Field("Type of task to execute over a specific set of table columns.")
    columsn: List[str] = Field(
        description="List of column names that can be used to execute the chosen task."
    )


# Define specific task classes
class TaskUnion(BaseModel):
    """UNION task: columns that can be combined with similar columns from other datasets"""
    columns: List[str] = Field(
        description="List of column names that can be combined using UNION operation. "
                    "These columns have similar semantic meaning across datasets."
    )

class TaskJoin(BaseModel):
    """JOIN task: columns that can serve as join keys"""
    columns: List[str] = Field(
        description="List of column names that can be used as JOIN keys. "
                    "These are typically unique identifiers or foreign keys."
    )

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
        description="List of UNION tasks identifying columns that can be combined"
    )
    join_tasks: List[TaskJoin] = Field(
        default_factory=list,
        description="List of JOIN tasks identifying potential join keys"
    )
    join_correlation_tasks: List[TaskJoinCorrelation] = Field(
        default_factory=list,
        description="List of JOIN-CORRELATION tasks pairing join keys with correlated metrics")