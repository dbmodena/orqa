from typing import Any, Literal, Optional

import pandas as pd
from valentine import algorithms as schema_matchers
from valentine import valentine_match

DOCUMENT_TYPE = "csv"
THRESHOLD = 0.5
MAX_WORKERS = 12


def instantiate_matcher(name: str, **kwargs) -> schema_matchers.BaseMatcher:
    match name:
        case "jaccard_distance":
            return schema_matchers.JaccardDistanceMatcher(**kwargs)
        case "coma":
            params = {"use_instances": False, "java_xmx": "2048m"} | kwargs
            return schema_matchers.Coma(**params)  # ty: ignore
        case "cupid":
            return schema_matchers.Cupid(**kwargs)
        case "similarity_flooding":
            return schema_matchers.SimilarityFlooding(**kwargs)
        case "distribution_based":
            return schema_matchers.DistributionBased(**kwargs)
        case _:
            raise ValueError(f"Unknown schema matcher: {name}")


def schema_matching(
    matcher: schema_matchers.BaseMatcher,
    task: Literal["U", "J", "MJ", "JC"],
    Q: pd.DataFrame,
    R: pd.DataFrame,
    q_columns: list[str],
    r_columns: Optional[list[str]] = None,
    q_key: Optional[str] = None,
    r_key: Optional[str] = None,
    q_target: Optional[str] = None,
    r_target: Optional[str] = None,
    verbose: bool = False,
) -> tuple[dict, float, float]:
    """
    Compute an average schema matching score with Valentine matchers.

    The idea is that, for pair of tables with similar schema, the global
    average matching score should be high and similar to the score computed on
    the portion of specified columns, identified from previous steps.

    Instead, for tables with dissimilar schema, the global average matching score
    should be low, and only when focusing on the selected subsets of columns it
    raises to an higher average score.
    """
    if verbose:
        print("Computing Schema Matching...")
    matches: dict[Any, float] = valentine_match(Q, R, matcher, "Q", "R")
    matches = {(x, y): s for ((_, x), (_, y)), s in matches.items()}
    if verbose:
        print("Done.")

    if not matches:
        return matches, 0, 0
    global_avg = sum(matches.values()) / len(matches)
    specific_avg = 0
    filtered_values = []

    match task:
        case "U":
            filtered_values = [
                value
                for (col1, col2), value in matches.items()
                if col1 in q_columns  # and col2 in r_columns
            ]

            specific_avg = (
                sum(filtered_values) / len(filtered_values) if filtered_values else 0
            )
        case "J" | "MJ":
            assert isinstance(r_columns, list)
            filtered_values = [
                value
                for (col1, col2), value in matches.items()
                if col1 in q_columns and col2 in r_columns
            ]

            specific_avg = (
                sum(filtered_values) / len(filtered_values) if filtered_values else 0
            )
        case "JC":
            filtered_values = [
                value
                for (col1, col2), value in matches.items()
                if (col1 == q_key and col2 == r_key)
                or (col1 == q_target and col2 == r_target)
            ]

            specific_avg = (
                sum(filtered_values) / len(filtered_values) if filtered_values else 0
            )
        case _:
            raise ValueError(f"Invalid task: {task}")

    return matches, global_avg, specific_avg
