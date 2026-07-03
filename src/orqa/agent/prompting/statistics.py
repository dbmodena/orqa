"""Cheap per-column statistics computed with pandas only (no LLM).

``ColumnStatistics.compute`` produces a :class:`TableStats` describing a single
DataFrame: its row count plus a :class:`ColumnStat` for every column. The
statistics (cardinality, null ratio, dtype, numeric aggregates, and bounded
top-k categorical values) feed the query planner prompt and the skill gate.

The computation is deterministic and idempotent: computing statistics twice on
the same DataFrame yields equal results.
"""

from typing import List

import pandas as pd
from pandas.api import types as ptypes

from .models import ColumnStat, TableStats


class ColumnStatistics:
    """Computes cheap per-column statistics using pandas only (no LLM)."""

    @staticmethod
    def compute(
        df: pd.DataFrame,
        alias: str,
        top_k: int = 5,
        max_card_scan: int = 100_000,
    ) -> TableStats:
        """Compute per-column statistics for ``df``.

        Args:
            df: The DataFrame to summarise.
            alias: The table alias recorded on the returned ``TableStats``.
            top_k: Maximum number of top categorical values to record per column.
            max_card_scan: Upper bound on the number of rows scanned for
                cardinality/top-value computation. When the frame is larger, a
                deterministic head slice is used to keep the computation cheap.

        Returns:
            A :class:`TableStats` with the row count and one :class:`ColumnStat`
            per column.
        """
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
            top_values: List[str] = []

            is_numeric = ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(
                series
            )

            if is_numeric:
                non_null = series.dropna()
                if not non_null.empty:
                    numeric_min = float(non_null.min())
                    numeric_max = float(non_null.max())
                    numeric_mean = float(non_null.mean())
            else:
                # Bounded top-k categorical values, most frequent first. Ties are
                # broken deterministically by pandas' stable ordering.
                value_counts = scan_series.value_counts(dropna=True)
                top_values = [str(idx) for idx in value_counts.head(top_k).index]

            columns.append(
                ColumnStat(
                    column=str(column),
                    dtype=dtype,
                    cardinality=cardinality,
                    null_ratio=null_ratio,
                    numeric_min=numeric_min,
                    numeric_max=numeric_max,
                    numeric_mean=numeric_mean,
                    top_values=top_values,
                )
            )

        return TableStats(alias=alias, num_rows=num_rows, columns=columns)
