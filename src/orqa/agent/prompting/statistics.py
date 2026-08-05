"""Cheap per-column statistics computed with pandas only (no LLM).

``ColumnStatistics.compute`` produces a :class:`TableStats` describing a single
DataFrame: its row count plus a :class:`ColumnStat` for every column. The
statistics (cardinality, null ratio, dtype, numeric aggregates, bounded top-k
categorical values, IQR-fenced numeric outliers, pinned-ceiling/floor
detection, and grouped minority/tail values) feed the query planner prompt.
None of these statistics judge whether an outlier is noise or signal — that
call is left entirely to the query planner (see the DATA QUALITY / CLEANING
section of ``conf/prompts/query_planner.md``); this module only detects and
bounds.

The computation is deterministic and idempotent: computing statistics twice on
the same DataFrame yields equal results.
"""

from typing import List, Optional, Sequence

import pandas as pd
from pandas.api import types as ptypes

from ...utils import (
    _generalize_value_shape,
    _numeric_parse_ratio,
    _strip_numeric_formatting,
    summarize_large_value,
)
from .models import ColumnStat, TableStats


class ColumnStatistics:
    """Computes cheap per-column statistics using pandas only (no LLM)."""

    @staticmethod
    def compute(
        df: pd.DataFrame,
        alias: str,
        top_k: int = 5,
        max_card_scan: int = 100_000,
        bad_tokens: Optional[Sequence[str]] = None,
        pinned_extreme_min_cardinality: int = 20,
        pinned_extreme_min_rows: int = 30,
        pinned_extreme_min_count: int = 20,
        pinned_extreme_min_ratio: float = 5.0,
    ) -> TableStats:
        """Compute per-column statistics for ``df``.

        Args:
            df: The DataFrame to summarise. Expected to be the RAW view (see
                ``utils.prepare_dataset``) — no bad-token conversion, numeric
                coercion, or null dropping has been applied to it — so the
                stats below reflect what the query planner actually sees and
                must decide on via a ``clean`` step.
            alias: The table alias recorded on the returned ``TableStats``.
            top_k: Maximum number of top categorical values to record per column.
            max_card_scan: Upper bound on the number of rows scanned for
                cardinality/top-value computation. When the frame is larger, a
                deterministic head slice is used to keep the computation cheap.
            bad_tokens: Portal-specific missing-value sentinel literals (e.g.
                "n/a", "not available") to count per column, case-insensitively,
                on top of true NaN. ``None``/empty skips this entirely.
            pinned_extreme_min_cardinality: A numeric column's (unbounded)
                distinct-value count must reach this before pinned-extreme
                detection even runs — guards against false-positiving on a
                naturally-repeating low-cardinality column (e.g. a 1-5 rating
                scale, where every value is EXPECTED to repeat a lot).
            pinned_extreme_min_rows: Minimum non-null rows before pinned-extreme
                detection runs (mirrors numeric_outliers' own len(non_null)>=8
                guard, just a higher bar since this check reasons about
                relative frequency, which needs more points to be meaningful).
            pinned_extreme_min_count: A boundary value's exact-match count must
                clear this absolute floor to flag, regardless of ratio.
            pinned_extreme_min_ratio: AND clear this multiple of the
                uniform-spread expectation (non_null_count / cardinality) —
                must be a clear outlier in FREQUENCY, not merely "somewhat
                more common than average."

        Returns:
            A :class:`TableStats` with the row count and one :class:`ColumnStat`
            per column.
        """
        lowered_bad_tokens = (
            {str(t).strip().lower() for t in bad_tokens} if bad_tokens else set()
        )
        num_rows = int(len(df))

        # Bound the scan for cardinality/top-values on very large frames using a
        # deterministic head slice so repeated calls stay idempotent.
        scan_df = df if num_rows <= max_card_scan else df.head(max_card_scan)

        columns: List[ColumnStat] = []
        for column in df.columns:
            series = df[column]
            scan_series = scan_df[column]

            dtype = str(series.dtype)
            cardinality = int(scan_series.nunique(dropna=True))

            if num_rows > 0:
                null_ratio = float(series.isna().mean())
            else:
                null_ratio = 0.0
            # Guard against floating point drift outside the 0..1 range.
            null_ratio = min(1.0, max(0.0, null_ratio))

            numeric_min = None
            numeric_max = None
            numeric_mean = None
            numeric_outliers: Optional[dict] = None
            numeric_pinned_extreme: Optional[dict] = None
            top_values: List[str] = []
            minority_value_groups: List[dict] = []

            is_numeric = ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(
                series
            )

            if is_numeric:
                non_null = series.dropna()
                if not non_null.empty:
                    numeric_min = float(non_null.min())
                    numeric_max = float(non_null.max())
                    numeric_mean = float(non_null.mean())

                    # Tukey IQR fencing: flags values statistically far from
                    # this column's OWN spread (not a fixed cutoff), without
                    # judging whether they're a data-entry error, a sentinel
                    # ("-1"/"999" for missing), or a genuine rare-but-real
                    # extreme — that call belongs to the query planner.
                    # Needs a handful of points for quartiles to be meaningful.
                    if len(non_null) >= 8:
                        q1 = float(non_null.quantile(0.25))
                        q3 = float(non_null.quantile(0.75))
                        iqr = q3 - q1
                        low_fence = q1 - 1.5 * iqr
                        high_fence = q3 + 1.5 * iqr
                        low_vals = non_null[non_null < low_fence]
                        high_vals = non_null[non_null > high_fence]
                        if not low_vals.empty or not high_vals.empty:
                            numeric_outliers = {
                                "low_count": int(low_vals.shape[0]),
                                "high_count": int(high_vals.shape[0]),
                                "low_examples": sorted(low_vals.unique().tolist())[:top_k],
                                "high_examples": sorted(
                                    high_vals.unique().tolist(), reverse=True
                                )[:top_k],
                                "low_fence": low_fence,
                                "high_fence": high_fence,
                            }

                    # Pinned-ceiling/floor detection: a value AT this
                    # column's own min or max that repeats far more than a
                    # roughly uniform spread would predict — the top-/
                    # bottom-coding signature in data that's ALREADY
                    # numeric (e.g. hundreds of rows reading exactly 90 in
                    # an age column capped at "90 or older"). Unlike the
                    # Tukey fence above (flags a RARE extreme), this flags
                    # a FREQUENT one — the two checks are independent and
                    # can both fire for the same value. Two guards against
                    # false-positiving on a naturally-repeating
                    # low-cardinality column (e.g. a 1-5 rating scale):
                    # the cardinality floor below (skips it entirely) and
                    # the count/ratio floors inside the loop (a value must
                    # be a clear frequency outlier, not just "common").
                    if (
                        len(non_null) >= pinned_extreme_min_rows
                        and cardinality >= pinned_extreme_min_cardinality
                    ):
                        expected = len(non_null) / cardinality
                        pinned: dict = {}
                        for side, bound in (("low", numeric_min), ("high", numeric_max)):
                            count = int((non_null == bound).sum())
                            ratio = (count / expected) if expected > 0 else 0.0
                            if (
                                count >= pinned_extreme_min_count
                                and ratio >= pinned_extreme_min_ratio
                            ):
                                pinned[side] = {
                                    "value": bound,
                                    "count": count,
                                    "ratio": round(ratio, 2),
                                }
                        numeric_pinned_extreme = pinned or None
            else:
                # Bounded top-k categorical values, most frequent first. Ties are
                # broken deterministically by pandas' stable ordering.
                value_counts = scan_series.value_counts(dropna=True)
                # summarize_large_value shields an oversized value (e.g. a
                # WKT geometry) — this feeds straight into the planner AND
                # generation prompts (see QueryPlanner._render_statistics /
                # prompts.render_column_statistics), never execution.
                top_values = [
                    summarize_large_value(str(idx)) for idx in value_counts.head(top_k).index
                ]

                # The TAIL beyond top_values, grouped into cheap structural
                # shapes (see _generalize_value_shape) rather than listed raw —
                # stays small regardless of how many distinct rare values a
                # table has. Excludes already-configured bad tokens (fully
                # reported via bad_token_counts already, no need to duplicate)
                # AND values that themselves parse as a plain number (already
                # fully represented in aggregate by numeric_parseable_ratio —
                # without this, a rare-but-ordinary number like a single "87"
                # collapses to the same trivial "#" shape as every other
                # number, and that bucket's combined count can crowd
                # genuinely distinctive shapes out of the top-k). Object
                # dtype only: this is about rare literal strings, not e.g.
                # rare datetime/categorical-dtype values.
                if series.dtype == object and cardinality > top_k:
                    tail = value_counts.iloc[top_k:]
                    tail_is_numeric = pd.to_numeric(
                        _strip_numeric_formatting(tail.index.to_series().astype(str)),
                        errors="coerce",
                    ).notna()
                    groups: dict = {}
                    for (raw_value, count), is_numeric_like in zip(
                        tail.items(), tail_is_numeric
                    ):
                        if is_numeric_like:
                            continue
                        raw_str = str(raw_value)
                        if lowered_bad_tokens and raw_str.strip().lower() in lowered_bad_tokens:
                            continue
                        shape = _generalize_value_shape(raw_str)
                        bucket = groups.setdefault(shape, {"count": 0, "examples": []})
                        bucket["count"] += int(count)
                        if len(bucket["examples"]) < 3:
                            bucket["examples"].append(summarize_large_value(raw_str))
                    ranked = sorted(
                        groups.items(), key=lambda kv: kv[1]["count"], reverse=True
                    )
                    minority_value_groups = [
                        {"pattern": shape, "count": info["count"], "examples": info["examples"]}
                        for shape, info in ranked[:top_k]
                    ]

            nan_count = int(series.isna().sum())

            # Portal-specific missing-value literals, counted separately from
            # true NaN — only meaningful on object columns (numeric dtypes
            # can't hold a stray string token in the first place).
            bad_token_counts: dict = {}
            if lowered_bad_tokens and series.dtype == object:
                lowered_values = series.dropna().astype(str).str.strip().str.lower()
                for token in lowered_bad_tokens:
                    n = int((lowered_values == token).sum())
                    if n:
                        bad_token_counts[token] = n

            # Fraction of an object column's values that would parse as
            # numeric once cleaned — flags columns numeric-in-disguise
            # ("1,314", "34.10%", "$500") without ever coercing them.
            numeric_parseable_ratio = (
                _numeric_parse_ratio(series)
                if (not is_numeric and series.dtype == object)
                else None
            )

            columns.append(
                ColumnStat(
                    column=str(column),
                    dtype=dtype,
                    cardinality=cardinality,
                    null_ratio=null_ratio,
                    nan_count=nan_count,
                    bad_token_counts=bad_token_counts,
                    numeric_parseable_ratio=numeric_parseable_ratio,
                    numeric_min=numeric_min,
                    numeric_max=numeric_max,
                    numeric_mean=numeric_mean,
                    numeric_outliers=numeric_outliers,
                    numeric_pinned_extreme=numeric_pinned_extreme,
                    top_values=top_values,
                    minority_value_groups=minority_value_groups,
                )
            )

        return TableStats(alias=alias, num_rows=num_rows, columns=columns)
