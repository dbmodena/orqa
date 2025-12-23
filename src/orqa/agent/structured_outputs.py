from pydantic import BaseModel, Field, field_validator, conint, model_validator
from typing import List,Set, Tuple
from itertools import combinations


class CSVAnalysisResult(BaseModel):
    """Result of CSV analysis with UNION and JOIN suggestions"""
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
    
    @field_validator('columns')
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
    
    @field_validator('columns')
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
        description="List of UNION tasks identifying columns that can be combined"
    )
    join_tasks: List[TaskJoin] = Field(
        default_factory=list,
        description="List of JOIN tasks identifying potential join keys"
    )
    join_correlation_tasks: List[TaskJoinCorrelation] = Field(
        default_factory=list,
        description="List of JOIN-CORRELATION tasks pairing join keys with correlated metrics"
    )
    
    @field_validator('union_tasks')
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
    
    @field_validator('join_tasks')
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
    
    @field_validator('join_correlation_tasks')
    @classmethod
    def deduplicate_join_correlation_tasks(cls, v: List[TaskJoinCorrelation]) -> List[TaskJoinCorrelation]:
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





class TablePairScore(BaseModel):
    left_table_id: str = Field(..., description="First table identifier")
    right_table_id: str = Field(..., description="Second table identifier")
    score: conint(ge=1, le=10)

    def normalized_pair(self) -> Tuple[str, str]:
        if self.left_table_id == self.right_table_id:
            raise ValueError("Self-pairs are not allowed")
        return tuple(sorted((self.left_table_id, self.right_table_id)))


class Match(BaseModel):
    table_ids: List[str] = Field(
        ...,
        description="List of all tables participating in the match"
    )
    table_pair_scores: List[TablePairScore] = Field(
        default_factory=list,
        description="Pairwise match scores between tables"
    )

    @model_validator(mode="after")
    def validate_all_pairs_present(self):
        tables: Set[str] = set(self.table_ids)

        if len(tables) < 2:
            raise ValueError("At least two distinct tables are required")

        # Expected unordered pairs
        expected_pairs: Set[Tuple[str, str]] = {
            tuple(sorted(pair)) for pair in combinations(tables, 2)
        }

        # Provided pairs
        provided_pairs: Set[Tuple[str, str]] = set()
        for pair_score in self.table_pair_scores:
            pair = pair_score.normalized_pair()
            if pair in provided_pairs:
                raise ValueError(f"Duplicate score provided for pair {pair}")
            provided_pairs.add(pair)

        missing = expected_pairs - provided_pairs
        extra = provided_pairs - expected_pairs

        if missing:
            raise ValueError(
                f"Missing scores for table pairs: {sorted(missing)}"
            )

        if extra:
            raise ValueError(
                f"Unexpected pairs provided: {sorted(extra)}"
            )

        return self
