import inspect
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
            # pyproject only pins valentine>=0.4.1 (no upper bound), and
            # java_xmx was added to Coma.__init__ in a later release than
            # that floor — an older installed valentine raises
            # "unexpected keyword argument 'java_xmx'" on this hardcoded
            # default. Only pass params the installed Coma actually accepts,
            # so this stays compatible across valentine versions instead of
            # assuming one exact signature.
            accepted = set(inspect.signature(schema_matchers.Coma.__init__).parameters)
            params = {k: v for k, v in params.items() if k in accepted}
            return schema_matchers.Coma(**params)
        case "cupid":
            return schema_matchers.Cupid(**kwargs)
        case "similarity_flooding":
            return schema_matchers.SimilarityFlooding(**kwargs)
        case "distribution_based":
            return schema_matchers.DistributionBased(**kwargs)
        case _:
            raise ValueError(f"Unknown schema matcher: {name}")


# valentine 1.0.0 rewrote valentine_match()'s call shape: it now takes a
# LIST of DataFrames (computing all pairs at once) instead of two positional
# DataFrames, and returns results keyed by a ColumnPair namedtuple
# (.source_column/.target_column) instead of nested ((table, col), (table,
# col)) tuples. pyproject only pins valentine>=0.4.1 (no upper bound), so
# whichever generation is actually installed is detected here — via the
# first parameter's name, "dfs" (batch) vs "df1" (pairwise) — rather than
# assumed, since different environments have resolved different majors.
_IS_BATCH_API = next(iter(inspect.signature(valentine_match).parameters), None) == "dfs"


def _valentine_match_all(
    Q: pd.DataFrame, R: pd.DataFrame, matcher: "schema_matchers.BaseMatcher"
) -> dict[tuple[str, str], float]:
    """Run valentine_match(Q, R, matcher) and return {(q_col, r_col): score},
    regardless of which valentine API generation is installed (see
    _IS_BATCH_API above). Table name labels are never surfaced to callers
    either way — only column names survive into the returned keys.
    """
    if _IS_BATCH_API:
        raw = valentine_match([Q, R], matcher)
        return {(cp.source_column, cp.target_column): score for cp, score in raw.items()}

    # Pre-1.0.0: two positional DataFrames, nested-tuple keys. Not passing
    # explicit table-name args — every version's own default label is at
    # least two characters (needed since valentine builds column guids by
    # indexing the second character of the table name; a single-letter
    # label crashes it), and the label is discarded below regardless.
    raw = valentine_match(Q, R, matcher)
    return {(x, y): s for ((_, x), (_, y)), s in raw.items()}


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
    matches: dict[Any, float] = _valentine_match_all(Q, R, matcher)
    if verbose:
        print("Done.")

    if not matches:
        return matches, 0, 0
    global_avg = sum(matches.values()) / len(matches)
    
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
