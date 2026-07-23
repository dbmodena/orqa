"""
Pair verification for candidates discovery.

Two checks run per nominated (Q, R) pair:
- Valentine schema matching gates the pair before any LLM tokens are spent.
- An actual join + correlation computation verifies join-correlation tasks
  (Valentine has no notion of correlation).
"""

import logging
import time
from typing import Literal, Optional

import pandas as pd
import polars as pl

from ..schema_matching.valentine_matcher import instantiate_matcher, schema_matching

logger = logging.getLogger(__name__)


def verify_pair_schema(
    Q: pd.DataFrame,
    R: pd.DataFrame,
    matcher_name: str,
    matcher_kwargs: dict | None = None,
) -> dict:
    """Run Valentine over the full pair and return the match evidence.

    Returns ``{"matches": [(q_col, r_col, score), ...] sorted by score desc,
    "sm_macro_avg": float, "sm_micro_avg": float, "sm_n_matches": int,
    "sm_time": float}``. ``sm_micro_avg`` here is the best single column
    match — the pair-level analogue of the old task-specific average, useful
    to keep a pair alive on one excellent join-key match.
    """
    matcher = instantiate_matcher(matcher_name, **(matcher_kwargs or {}))
    start = time.time()
    matches, global_avg, _ = schema_matching(
        matcher, "U", Q, R, q_columns=list(Q.columns)
    )
    elapsed = time.time() - start

    match_list = sorted(
        ((q_col, r_col, round(score, 3)) for (q_col, r_col), score in matches.items()),
        key=lambda m: -m[2],
    )
    best = match_list[0][2] if match_list else 0.0
    return {
        "matches": match_list,
        "sm_macro_avg": round(global_avg, 3),
        "sm_micro_avg": best,
        "sm_n_matches": len(match_list),
        "sm_time": round(elapsed, 3),
    }


def passes_schema_gate(
    evidence: dict, macro_threshold: float, micro_threshold: float
) -> bool:
    """A pair proceeds when the schemas match globally OR one column pair
    matches strongly (an excellent join key must not be killed by a low
    global average)."""
    return (
        evidence["sm_macro_avg"] >= macro_threshold
        or evidence["sm_micro_avg"] >= micro_threshold
    )


def _as_numeric(df: pl.DataFrame, column: str) -> Optional[pl.Series]:
    """Return the column cast to numeric, trying progressively laxer casts."""
    series = df.get_column(column)
    if series.dtype.is_numeric():
        return series

    casting_exprs = [
        pl.col(column).cast(pl.Float32),
        pl.col(column)
        .str.strip_chars()
        .str.replace_all(r"[£,*]", "", literal=False)
        .cast(pl.Float32),
        pl.col(column).cast(pl.Float32, strict=False),
    ]
    for expr in casting_exprs:
        try:
            return (
                df.lazy().with_columns(expr).select(column).collect().get_column(column)
            )
        except pl.exceptions.InvalidOperationError:
            continue
    return None


def check_join_correlation(
    Q: pl.DataFrame,
    R: pl.DataFrame,
    q_key: str,
    r_key: str,
    q_target: str,
    r_target: str,
    threshold: float,
    method: Literal["pearson", "spearman"] = "pearson",
    min_joined_rows: int = 10,
) -> Optional[float]:
    """Join Q and R on their keys and correlate the target columns.

    Returns the absolute correlation when the join yields enough rows and
    ``|corr| >= threshold``; None otherwise (the JC task is then dropped).
    """
    q_targets = _as_numeric(Q, q_target)
    r_targets = _as_numeric(R, r_target)
    if q_targets is None or r_targets is None:
        return None

    left = pl.DataFrame({q_key: Q.get_column(q_key), "__q_target": q_targets})
    # Distinct column names on the R side so self-pair joins can't collide.
    right = pl.DataFrame({"__r_key": R.get_column(r_key), "__r_target": r_targets})

    try:
        joined = left.join(
            right,
            left_on=pl.col(q_key).cast(pl.String),
            right_on=pl.col("__r_key").cast(pl.String),
            how="inner",
        ).drop_nulls(["__q_target", "__r_target"])
    except Exception as exc:
        logger.debug("JC join failed on %s=%s: %s", q_key, r_key, exc)
        return None

    if joined.height < min_joined_rows:
        return None

    corr = joined.select(
        pl.corr("__q_target", "__r_target", method=method)
    ).item()
    if corr is None:
        return None

    corr = abs(float(corr))
    return corr if corr >= threshold else None
