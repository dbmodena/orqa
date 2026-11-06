"""Per-column and per-table facts computed with pandas only (no LLM).

``ColumnStatistics.compute`` turns one RAW DataFrame (``utils.prepare_dataset``:
no missing-token conversion, no coercion, no null dropping) into a
:class:`TableStats`: the row count, one :class:`ColumnStat` per column and the
table facts (``structure``). ``prompts.render_column_statistics`` renders it
into the planner, generation and correction prompts.

This module decides each column's type and orchestrates the profiles:

- numbers (numeric dtypes, or object columns holding only numbers): the
  numeric facts below (``numeric_profile``);
- text (object columns of str, and the str values of a mixed column):
  ``text_profile.profile_text``;
- bools and datetimes: their domain (and the parsed date range);
- table facts (empty/repeated rows, total rows, grain, one-to-one,
  many-to-one, sum columns, fill-down columns) and identical columns:
  ``table_structure.profile_table``;
- the identifier fact (``_identifiers``): the structural evidence that a
  whole-number column names records (a sequence, unique values, a code
  paired one-to-one with a text column, a part of the row key, a column
  whose value fixes others), from the column profile and the table facts;
- whole numbers that are calendar dates (``_date_as_number``: 20200416).

Every column is factorized ONCE (``pd.factorize``, NaN as code -1); the
frequency table, the column profiles and the table facts all reuse those
codes, so the string work scales with DISTINCT values and nothing re-reads
the rows per fact. Every statistic covers every row: a head slice would
misstate a large file's domain (a value first seen after row 100k would be
missing).

The facts only DETECT and BOUND: none of them judges whether a value is noise
or signal (a far-high value can be a sentinel, a total row or a genuine
record) -- that call belongs to the query planner (DATA QUALITY section of
``conf/prompts/query_planner.md``). Values are stored raw (built-in Python
types); all formatting happens in the renderer. The computation is
deterministic: fixed seeds, stable sorts, row-count thresholds only.
"""

import logging
import numbers
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .models import ColumnStat, TableStats
from .table_structure import _YEAR_AXIS, _py as _builtin, profile_table
from .text_profile import date_coverage, parse_number_text, profile_text

logger = logging.getLogger(__name__)

# The numeric thresholds below were measured on 1,693 numeric columns of 400
# real portal tables (80 per portal, loaded like production) and re-checked on
# a second 294-table sample; the counts quoted come from those runs.

# Up to this many distinct values the complete value list IS the distribution
# and supplies the exact filter literals; range and quartiles would only
# repeat it.
_LIST_MAX_CARD = 10
# The head of a discrete column is shown whole up to this many distinct
# values (median top-5 coverage is 0.71 at 6-20 distinct values, 0.25 at
# 21-100) or when the top values cover at least half of the rows ...
_DISCRETE_MAX_CARD = 20
_HEAD_MIN_COVERAGE = 0.5
# ... otherwise only spikes (zero inflation, caps, defaults): a 5% share put 0
# in the top values of 340 of the 356 columns with that much zero inflation,
# while the head of a continuous column (311 columns with top-5 coverage
# under 10%) is values seen 2-8 times that cost ~50 tokens and say nothing.
_SPIKE_SHARE = 0.05
# Quartiles locate the bulk when the top values do not already show it. The
# mean was dropped: it lay outside [p25, p75] in 16% of columns, and the
# median sat in the bottom 10% of [min, max] in 31%.
_QUARTILE_MAX_COVERAGE = 0.8
# Record-id sequences: at least this many distinct whole values filling at
# least half of one evenly stepped run (72 detected, 70 true ids/row
# numbers). The 2 false positives were one row per year (NYC Year
# 1995..2022, Paris annee 2004..2025), where "never correlate" would be
# wrong, hence the year-axis exclusion.
_SEQUENCE_MIN_CARD = 20
# Far high: the widest ratio gap among the 11 largest distinct values, at
# least 3x, with at most 5% of the rows above it. Fires on 2.8% of columns
# (0 of 37 untrue on the second sample), where the Tukey 1.5 IQR fence fired
# on 65% and flagged a median 5.3% of rows.
_FAR_HIGH_SCAN = 11
_FAR_HIGH_MIN_RATIO = 3.0
_FAR_HIGH_MAX_SHARE = 0.05
_FAR_HIGH_SHOWN = 3
# Values above the gap that are whole and span < 1% of their magnitude are a
# separate code/id block (Paris budget keys 6588341..6588373 above 754888,
# UK AphiaId 999000881..999000932): 6 of 37 far-high firings.
_BLOCK_MAX_SPAN = 0.01
# A value AT the min or max repeating >= 20 times and >= 5x the mean count of
# its 3 inward neighbours is a floor/ceiling/default, not the head of a
# decaying distribution. Fires on 1.1% of columns; the older "frequent
# boundary" rule fired on 10%, mostly 0 already in the top values, and a
# quarter of its firings were no spike at all.
_REPEATED_MIN_ROWS = 20
_REPEATED_MIN_FACTOR = 5.0
_REPEATED_NEIGHBOURS = 3
# ... and only where the values are dense: the 3 inward neighbours lie within
# 0.25 x IQR of the extreme. At the edge of a code list the neighbours are
# sparse, so an ordinary code beat them 5x (fact audit: span/IQR 0.84, 1.63
# and 1.90 on three account-code columns, Paris article 205, Valencia
# Concepto 913 and Articulos 91; 0.002-0.042 on true measure floors such as
# Valencia Valor 100; about 0.1 on a 90+ age cap).
_REPEATED_MAX_SPAN_IQR = 0.25
# Integers above 2**53 are not exact in float64: no gcd/step arithmetic.
_EXACT_FLOAT_INT = 2**53
# Factorize codes are int32 below this many distinct values: int64 codes for
# every column cost 6.3 GB on a 113M-row table (8 bytes per row and column,
# all held until the table facts are done).
_INT32_CODES_MAX_CARD = 2**31
# Identifier fact (SPEC_V4 A). A whole-number column distinct in every row is
# an id only from this many non-null rows: 39 such columns of >= 1000 rows
# in 2,503 real candidate columns were all ids on inspection (civkey,
# n_sq_vo, BoroCT2010, Propref, Pay_By_Cel ...), while below it unique whole
# columns are often quantities (HABITANTES 107 distinct in 107 rows, 32
# daily counts of a pedestrian counter). The weaker evidence (a code paired
# with a name, a row-key part, a determinant) has no row minimum.
_IDENTIFIER_MIN_ROWS = 1000
# ... and at least this many distinct values: a 0/100 flag or a yes/no code
# names nothing.
_IDENTIFIER_MIN_CARD = 3
# Date as number (SPEC_V4 D4): whole values that all read as YYYYMMDD in
# this range (8 columns in the identifier sweep, all dates: NYC PUBLICATION
# DATE 20200416, Bologna data_isti 19000101..20120315). At least 3 distinct
# values, as in the sweep that calibrated the rule.
_DATE_NUMBER_RANGE = (18000101, 21001231)
_DATE_NUMBER_MIN_CARD = 3


class ColumnStatistics:
    """Computes the column and table facts of one table (pandas only)."""

    @staticmethod
    def compute(
        df: pd.DataFrame,
        alias: str,
        top_k: int = 5,
        bad_tokens: Optional[Sequence[str]] = None,
        source: Optional[Union[str, Path]] = None,
    ) -> TableStats:
        """Compute the statistics of ``df``.

        Args:
            df: The RAW table (see ``utils.prepare_dataset``), so the facts
                describe what the planner must decide on via a ``clean`` step.
            alias: The table alias recorded on the returned ``TableStats``.
            top_k: How many values a numeric "most common" fact shows at
                most (fewer when the cut falls inside a tie), and the head
                whose row coverage decides quartiles / most common.
            bad_tokens: The configured missing-value spellings ("n/a", "-",
                ...), matched case-insensitively after trimming. Reported as
                missing tokens, never converted: a token can be a real
                category, the planner decides.
            source: The file ``df`` was loaded from (``utils.prepare_dataset``:
                its first ``len(df)`` rows, any column selection). A parquet
                file gives the per-column frequency tables through Arrow
                (``profile_input``, from 1M rows: 22 s instead of 36 s on a
                113M-row table, no Python object hashed per row); a column whose Arrow input disagrees
                with the frame (dtype, a sampled cell) takes the pandas route.
                The facts are the same either way.

        Returns:
            A :class:`TableStats` with one :class:`ColumnStat` per column, in
            column order, under its exact name.
        """
        tokens = {str(t).strip().lower() for t in bad_tokens} if bad_tokens else set()
        num_rows = int(len(df))
        if any(not isinstance(c, str) for c in df.columns):
            # The table facts name columns by label, ColumnStat by str: int
            # labels 0, 1 gave a one-to-one on [0, 1] beside columns "0" and
            # "1", and the renderer failed on it ('int' object has no
            # attribute 'strip'). utils.clean_columns already makes labels
            # str in production. A shallow copy: no data is copied.
            df = df.copy(deep=False)
            df.columns = [str(c) for c in df.columns]

        columns: List[ColumnStat] = []
        codes: Dict[str, np.ndarray] = {}
        uniques: Dict[str, list] = {}
        kinds: Dict[str, str] = {}
        measures: Dict[str, np.ndarray] = {}
        empty_codes: Dict[str, set] = {}
        views: Dict[str, Optional[dict]] = {}
        arrow = _arrow_inputs(source, df) if source is not None and num_rows >= _ARROW_MIN_ROWS else {}
        for label in df.columns:
            series = df[label]
            given = arrow.pop(label, None)
            if given is not None:
                col_codes, col_uniques, hashable, counts = given.codes, given.uniques, given.hashable, given.counts
            else:
                col_codes, col_uniques, hashable = _factorize(series, label)
                counts = np.bincount(col_codes[col_codes >= 0], minlength=len(col_uniques))
            del given
            nan_count = num_rows - int(counts.sum())
            base = {
                "column": str(label),
                "dtype": str(series.dtype),
                "cardinality": int(len(col_uniques)),
                "null_ratio": nan_count / num_rows if num_rows else 0.0,
                "nan_count": nan_count,
            }
            stat = None
            if hashable:
                try:
                    kind, type_label, profile, measure, empties, view = _profile_column(
                        series, col_codes, col_uniques, counts, tokens, top_k
                    )
                    stat = ColumnStat(**base, type_label=type_label, **profile)
                except Exception:
                    # One column's failure must not cost the table the other
                    # columns' facts (and the planner its prompt): the column
                    # keeps its counts and enters the table facts as "other"
                    # (codes only, no measure).
                    logger.exception("Column statistics: profiling column %r failed; keeping its counts only", label)
            if stat is None:
                kind, measure, empties, view = "other", None, set(), None
                stat = ColumnStat(**base, type_label=_unprofiled_type(series.dtype))
            codes[label] = col_codes
            uniques[label] = col_uniques.tolist()
            kinds[label] = kind
            empty_codes[label] = empties
            views[label] = view
            if measure is not None:
                measures[label] = measure
            columns.append(stat)

        try:
            structure, same_as = profile_table(df, codes, uniques, kinds, measures, empty_codes, seed=0)
        except Exception:
            # The column facts stand on their own: a failure in the table
            # facts drops only those (no structure, no "same values as").
            logger.exception("Column statistics: table facts of %s failed; keeping the column facts only", alias)
            structure, same_as = {}, {}

        # A whole-number column that a table fact uses as a key is a code:
        # its quartiles and a repeated min/max read as a quantity's (fact
        # audit: quartiles of census blocks, of arrondissements 75001..75020
        # and of account codes; "repeated min 1" on a census block that is
        # part of the grain). Its range stays: it bounds the code space.
        keys = _key_columns(structure)
        for i, label in enumerate(df.columns):
            if label in keys and kinds[label] == "int":
                columns[i] = columns[i].model_copy(update={"quartiles": [], "repeated_extreme": {}})

        # The identifier fact (SPEC_V4 A): a column that names records keeps
        # its range and its frequent values, but its quartiles, far-high
        # values, repeated extremes and negative count are a quantity's facts
        # and would read as one. The header names the single-column row key.
        labels = list(df.columns)
        identifiers = _identifiers(labels, columns, views, kinds, structure, same_as)
        for i, label in enumerate(labels):
            if label in identifiers:
                columns[i] = columns[i].model_copy(update={
                    "identifier": identifiers[label], "quartiles": [], "far_high": None,
                    "repeated_extreme": {}, "negative_rows": 0,
                })
        row_key = _row_key_column(labels, identifiers, structure)
        if row_key is not None:
            structure["row_key_col"] = row_key

        # A fill-down column (SPEC_V4 B, found by the table facts) carries its
        # fact on the column line too.
        position = {label: i for i, label in enumerate(df.columns)}
        for label, fill in (structure.get("fill_down") or {}).items():
            if label in position:
                columns[position[label]] = columns[position[label]].model_copy(
                    update={"filled_down": {"filled": int(fill["filled"]), "rows": int(fill["rows"])}}
                )

        # An identical column says only "same values as <first>" (plus its
        # nulls): repeating its facts would cost tokens and hide the identity.
        for later, first in same_as.items():
            base = columns[position[later]]
            columns[position[later]] = ColumnStat(
                column=base.column,
                dtype=base.dtype,
                cardinality=base.cardinality,
                null_ratio=base.null_ratio,
                nan_count=base.nan_count,
                type_label=base.type_label,
                same_as=str(first),
            )

        return TableStats(alias=alias, num_rows=num_rows, columns=columns, structure=structure)


# Rows per column compared between the Arrow input and the frame.
_ARROW_CHECK_ROWS = 64
# Below this many rows pandas' factorize is faster than reading the file
# again (281 real tables under 1M rows: 21.3 s through Arrow, 19.3 s not).
_ARROW_MIN_ROWS = 1_000_000


def _arrow_inputs(source: Union[str, Path], df: pd.DataFrame) -> Dict[str, Any]:
    """The Arrow frequency tables of ``df``'s columns read from ``source``,
    by label; only the columns that match the frame (same dtype, the same
    value in ``_ARROW_CHECK_ROWS`` spread rows). Empty when ``source`` is not
    a parquet file or cannot be read: those columns take the pandas route."""
    path = Path(source)
    if path.suffix.lower() != ".parquet" or not len(df.columns) or df.columns.duplicated().any():
        return {}
    try:
        from .profile_input import profile_parquet

        inputs = profile_parquet(path, [str(c) for c in df.columns], limit_to_n_rows=len(df))
    except Exception:
        logger.warning("Column statistics: Arrow input of %s failed; using the frame", path.name, exc_info=True)
        return {}
    n = len(df)
    if inputs.num_rows != n:
        logger.warning("Column statistics: %s has %d rows, the frame %d; using the frame", path.name, inputs.num_rows, n)
        return {}
    rows = np.unique(np.linspace(0, n - 1, num=min(n, _ARROW_CHECK_ROWS)).astype(np.int64)) if n else np.array([], np.int64)
    out: Dict[str, Any] = {}
    for given in inputs.columns:
        series = df[given.name]
        if given.dtype != str(series.dtype) or not _same_cells(given, series, rows):
            logger.warning(
                "Column statistics: column %r of %s: Arrow input differs from the frame (%s vs %s); using the frame",
                given.name, path.name, given.dtype, series.dtype,
            )
            continue
        out[given.name] = given
    return out


def _same_cells(given, series: pd.Series, rows: np.ndarray) -> bool:
    """The frame's cells at ``rows`` are the values the Arrow codes name
    (null where the code is -1)."""
    codes = given.codes[rows]
    cells = series.iloc[rows]
    if not given.hashable:
        return bool(((codes < 0) == cells.isna().to_numpy()).all())
    for code, cell in zip(codes.tolist(), cells.tolist()):
        if code < 0:
            if not _is_null(cell):
                return False
            continue
        value = given.uniques[code]
        try:
            if _is_null(cell) or not (value is cell or bool(value == cell)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _is_null(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # array-like cells
        return False


def _key_columns(structure: dict) -> set:
    """The columns a table fact uses as a key: the grain parts, the
    one-to-one members and the left side of each "each a has one b"."""
    keys = set(structure.get("grain", {}).get("cols", []))
    for pair in structure.get("one_to_one", []):
        keys.update(pair["cols"])
    keys.update(a for a, _ in structure.get("determines", []))
    return keys


def _identifiers(
    labels: Sequence[str],
    columns: Sequence[ColumnStat],
    views: Dict[str, Optional[dict]],
    kinds: Dict[str, str],
    structure: dict,
    same_as: Dict[str, str],
) -> Dict[str, dict]:
    """The identifier fact of each column it fires on (SPEC_V4 A), as
    ``{"kind", "of"}``: structural evidence that the column names records,
    never a verdict (the planner judges by the name and the values too; a
    measure constant per category reads as "code of" or "fixes").

    Candidates are whole-number columns (kind "int") and plain number text
    whose conforming values are all whole (no currency sign, percent or unit:
    those spell a quantity), with >= 3 distinct values, outside the calendar
    year axis, not dates as numbers, not an identical column. The first
    evidence that holds, in this order, is the fact:

    1. ``sequence`` (a gcd-stepped run of distinct whole values): "sequence";
    2. distinct in every non-null row, >= 1000 rows: "unique";
    3. distinct in every DISTINCT row (card == rows - repeated rows),
       >= 1000 rows: "unique_distinct_rows" (NYC IM_INCIDENT_KEY: 6,026,076
       distinct, 7,915 repeated rows); never "near-unique" (# Total Days
       2467/2475 is a measure);
    4. one-to-one with a text column that is not number text, exactly (no
       ``except`` rows): "code_of" <the text columns of the class>;
    5. a part of the multi-column grain: "row_key" <the grain columns>;
    6. the left side of "each a has one b": "fixes" <the b columns>.
    """
    by_label = {label: columns[i] for i, label in enumerate(labels)}
    repeated = int((structure.get("repeated_rows") or {}).get("rows", 0))
    grain = list((structure.get("grain") or {}).get("cols", []))
    pairs = [list(pair["cols"]) for pair in structure.get("one_to_one", []) if not pair.get("except")]
    fixes = {a: list(bs) for a, bs in structure.get("determines", [])}
    out: Dict[str, dict] = {}
    for label in labels:
        stat, view = by_label[label], views.get(label)
        if label in same_as or not _identifier_candidate(stat, view, kinds[label]):
            continue
        kind, of = None, []
        if stat.sequence:
            kind = "sequence"
        elif view["rows"] >= _IDENTIFIER_MIN_ROWS and view["card"] == view["rows"]:
            kind = "unique"
        elif view["rows"] >= _IDENTIFIER_MIN_ROWS and repeated and view["card"] == view["rows"] - repeated:
            kind = "unique_distinct_rows"
        else:
            names = [
                c
                for cols in pairs
                if label in cols
                for c in cols
                if c != label and kinds.get(c) == "text" and c in by_label and by_label[c].number is None
            ]
            if names:
                kind, of = "code_of", names
            elif label in grain:
                kind, of = "row_key", grain
            elif label in fixes:
                kind, of = "fixes", fixes[label]
        if kind:
            out[label] = {"kind": kind, "of": [str(c) for c in of]}
    return out


def _identifier_candidate(stat: ColumnStat, view: Optional[dict], kind: str) -> bool:
    """Whether a column can carry the identifier fact (see ``_identifiers``)."""
    if view is None or not view["whole"] or view["card"] < _IDENTIFIER_MIN_CARD or stat.date_number:
        return False
    if kind == "text":
        if any(key in (stat.number or {}) for key in ("prefix", "suffix", "unit")):
            return False
    elif kind != "int":
        return False
    return not (view["min"] >= _YEAR_AXIS[0] and view["max"] <= _YEAR_AXIS[1])


def _row_key_column(labels: Sequence[str], identifiers: Dict[str, dict], structure: dict) -> Optional[str]:
    """The column the header names as the row key (``one row per <col>``):
    the leftmost column distinct in every (distinct) row or a sequence, when
    no multi-column grain is reported (the grain already says what a row
    is)."""
    if structure.get("grain"):
        return None
    for label in labels:
        if identifiers.get(label, {}).get("kind") in ("sequence", "unique", "unique_distinct_rows"):
            return str(label)
    return None


def _numeric_view(uniques, counts) -> Optional[dict]:
    """What the identifier fact reads of a numeric column, from its distinct
    finite values: ``{"whole", "card", "rows", "min", "max"}`` (rows = rows
    holding a finite value), or None when it holds none."""
    as_float = np.asarray(uniques, dtype=np.float64)
    finite = np.isfinite(as_float)
    if not finite.any():
        return None
    v = as_float[finite]
    return {
        "whole": bool(np.all(v == np.round(v))),
        "card": int(finite.sum()),
        "rows": int(np.asarray(counts, dtype=np.int64)[finite].sum()),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def _number_text_view(parsed: np.ndarray, counts) -> Optional[dict]:
    """``_numeric_view`` of a number-text column, from the number each
    distinct string parses to (NaN where it does not conform): two spellings
    of one number ("1" and "01") are one distinct value."""
    finite = np.isfinite(parsed)
    if not finite.any():
        return None
    distinct = np.unique(parsed[finite])
    return {
        "whole": bool(np.all(distinct == np.round(distinct))),
        "card": int(len(distinct)),
        "rows": int(np.asarray(counts, dtype=np.int64)[finite].sum()),
        "min": float(distinct[0]),
        "max": float(distinct[-1]),
    }


def _date_as_number(uniques) -> Optional[dict]:
    """``{"min", "max"}`` as YYYY-MM-DD when every finite value of a
    whole-number column reads as a YYYYMMDD date (SPEC_V4 D4): all in
    ``_DATE_NUMBER_RANGE``, month 1-12, day 1-31, and parsed by strptime
    ``%Y%m%d`` (20200230 is no date), over >= 3 distinct values."""
    as_float = np.asarray(uniques, dtype=np.float64)
    v = as_float[np.isfinite(as_float)]
    lo, hi = _DATE_NUMBER_RANGE
    if len(v) < _DATE_NUMBER_MIN_CARD or v.min() < lo or v.max() > hi or not bool(np.all(v == np.round(v))):
        return None
    ints = v.astype(np.int64)
    month, day = ints // 100 % 100, ints % 100
    if not bool(np.all((month >= 1) & (month <= 12) & (day >= 1) & (day <= 31))):
        return None
    parsed = pd.to_datetime(pd.Series(ints).astype(str), format="%Y%m%d", errors="coerce")
    if parsed.isna().any():
        return None
    return {"min": parsed.min().strftime("%Y-%m-%d"), "max": parsed.max().strftime("%Y-%m-%d")}


def _factorize(series: pd.Series, label: str):
    """``(codes, uniques, hashable)``: the column factorized once, NaN as code
    -1, uniques in first-seen order. Codes are int32 below 2**31 distinct
    values (``_INT32_CODES_MAX_CARD``); this module only compares, counts
    (bincount) and indexes with them, and table_structure widens them where
    it combines codes into keys. ``hashable`` is False when pandas could not
    hash the cells and the codes come from their text form."""
    try:
        codes, uniques = pd.factorize(series, use_na_sentinel=True)
        hashable = True
    except TypeError:
        # A parquet list or struct column loads as arrays or dicts
        # ("unhashable type: 'numpy.ndarray'"): their text form still gives
        # the column its counts and its partition of the rows.
        logger.exception("Column statistics: column %r holds unhashable values; keeping its counts only", label)
        codes, uniques = pd.factorize(series.astype(str).where(series.notna()), use_na_sentinel=True)
        hashable = False
    dtype = np.int32 if len(uniques) < _INT32_CODES_MAX_CARD else np.int64
    return codes.astype(dtype, copy=False), uniques, hashable


def _unprofiled_type(dtype) -> str:
    """Type label of a column whose profile failed, from its dtype alone: ""
    lets the renderer derive int / float / bool / datetime; an object
    column's values were never classified, so "str" would claim text."""
    return "object" if dtype == object else ""


def _profile_column(series, codes, uniques, counts, tokens, top_k):
    """Decide one column's type from its distinct non-null values and profile it.

    Returns ``(kind, type_label, profile, measure, empty_codes, view)``: the
    kind passed to ``table_structure`` ("int" | "float" | "text" | "bool" |
    "datetime" | "other"), the rendered type, the ColumnStat fields of the
    profile, the float64 row array of a numeric or number-text column (else
    None), the codes of blank / missing-token values (text only), and the
    numeric view the identifier fact reads (``_numeric_view``; None when the
    column holds no number).
    """
    dtype = series.dtype
    if ptypes.is_bool_dtype(dtype):
        return "bool", "bool", _domain(uniques, counts), None, set(), None
    if ptypes.is_numeric_dtype(dtype):
        return _numeric_column(uniques, counts, codes, top_k, integer_dtype=ptypes.is_integer_dtype(dtype))
    if ptypes.is_datetime64_any_dtype(dtype):
        return "datetime", "datetime", _date_range(uniques), None, set(), None
    if not (dtype == object or isinstance(dtype, pd.StringDtype)):
        return "other", str(dtype), {}, None, set(), None

    # Object columns: the distinct values' Python types decide. pandas'
    # inference runs in C over the distinct values; only an unusual mix walks
    # them in Python. The loaders do not produce one: CSV object columns are
    # str-only or bools with blanks, and the 18.4k UK/NYC parquet schemas hold
    # only string, int64, double and bool columns.
    inferred = ptypes.infer_dtype(uniques, skipna=True)
    if inferred == "empty":  # all null: nothing to profile
        return "text", "str", {}, None, set(), None
    if inferred == "boolean":
        # pandas loads a bool column that has nulls as object (CSV blanks,
        # parquet nulls): df[col] and ~col raise on its NaN, and astype(bool)
        # counts NaN as True (fact audit, Bologna centro_storico: 111318
        # "true" rows instead of 37947). The type says so.
        label = "bool with nulls" if bool((codes < 0).any()) else "bool"
        return "bool", label, _domain(uniques, counts), None, set(), None
    if inferred in ("integer", "floating", "mixed-integer-float"):
        return _numeric_column(uniques, counts, codes, top_k, integer_dtype=inferred == "integer")
    if inferred == "string":
        return "text", "str", *_text_profile(uniques, counts, codes, tokens, all_str=True)

    rows_by_type: Dict[str, int] = {}
    for value, count in zip(uniques, counts):
        name = _type_name(value)
        rows_by_type[name] = rows_by_type.get(name, 0) + int(count)
    if "str" not in rows_by_type:
        return "other", str(dtype), {}, None, set(), None
    # Most rows first; ties keep first-seen order (sorted is stable).
    mix = sorted(rows_by_type.items(), key=lambda kv: -kv[1])
    label = "mixed (" + ", ".join(f"{name} ×{rows}" for name, rows in mix) + ")"
    return "text", label, *_text_profile(uniques, counts, codes, tokens, all_str=False)


def _numeric_column(uniques, counts, codes, top_k: int, integer_dtype: bool):
    """A numeric column's ``_profile_column`` result: its kind and type, the
    numeric facts, its measure and its view. A whole-number column whose
    values all read as YYYYMMDD gets ``date_number`` (SPEC_V4 D4) in place of
    its range, quartiles and sequence: 20200101..20200131 is a calendar, and
    the sequence fact would call it a record id."""
    kind, label = _numeric_type(uniques, integer_dtype=integer_dtype)
    profile = numeric_profile(uniques, counts, codes, top_k)
    if kind == "int":
        dates = _date_as_number(uniques)
        if dates:
            for key in ("numeric_min", "numeric_max", "quartiles", "sequence"):
                profile.pop(key, None)
            profile["date_number"] = dates
    return kind, label, profile, _measure(uniques, codes), set(), _numeric_view(uniques, counts)


def _numeric_type(uniques, integer_dtype: bool):
    """(kind, type label) of a numeric column. A float column whose finite
    values are all whole is mostly a NaN-padded integer column (241 of 1,693
    numeric columns, 202 of them with NaN): it is an "int" kind for the table
    facts, and its type says so, since astype(int) fails on its NaN."""
    if integer_dtype:
        return "int", "int"
    values = np.asarray(uniques, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size and bool(np.all(values == np.round(values))):
        return "int", "float, whole numbers"
    return "float", "float"


def _type_name(value: Any) -> str:
    """The short Python type name a mixed column's type label shows."""
    if isinstance(value, str):
        return "str"
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, numbers.Integral):
        return "int"
    if isinstance(value, numbers.Real):
        return "float"
    return type(value).__name__


def _measure(uniques, codes) -> np.ndarray:
    """The float64 row array of a numeric column, NaN for null and +/-inf
    (not a number a sum or a comparison can use). Built from the distinct
    values: a NaN appended after them is what code -1 (null) indexes."""
    values = np.asarray(uniques, dtype=np.float64)
    values = np.where(np.isfinite(values), values, np.nan)
    return np.append(values, np.nan)[codes]


def _domain(uniques, counts) -> dict:
    """Constant or complete value list of a bool column (most rows first,
    ties in first-seen order)."""
    order = np.argsort(-counts, kind="stable")
    items = [(_builtin(uniques[i]), int(counts[i])) for i in order]
    if not items:
        return {}
    if len(items) == 1:
        return {"constant": items[0][0]}
    return {"values": items}


def _date_range(uniques) -> dict:
    """Parsed range of a datetime column: YYYY-MM-DD, with HH:MM when some
    value has a time of day (a date stored at 22:00/23:00 is a local date
    kept in UTC, which the bare date would hide)."""
    stamps = pd.DatetimeIndex(uniques)
    if stamps.empty:
        return {}
    if stamps.tz is not None:
        # UTC for everybody: a zoned column is reported on the UTC clock.
        stamps = stamps.tz_convert("UTC").tz_localize(None)
    fmt = "%Y-%m-%d %H:%M" if bool((stamps.normalize() != stamps).any()) else "%Y-%m-%d"
    date = {"min": stamps.min().strftime(fmt), "max": stamps.max().strftime(fmt)}
    coverage = date_coverage(pd.Series(stamps))
    if coverage:
        date["coverage"] = coverage
    return {"date": date}


def _text_profile(uniques, counts, codes, tokens, all_str: bool):
    """Profile the str values of a text column (``text_profile``) and derive
    what the table facts need from it: the number-text measure and the codes
    of blank / missing-token values. A mixed column (``all_str`` False) is
    profiled on its str values only; its type label counts the others.

    Returns ``(profile, measure, empty_codes, view)``: ``view`` is the
    numeric view of a number-text column with one format (``_number_text_view``),
    else None.
    """
    values = pd.Series(np.asarray(uniques, dtype=object), dtype=object)
    text = values
    if all_str:
        str_pos = np.arange(len(text))
    else:
        is_str = values.map(type).eq(str).to_numpy()
        str_pos = np.flatnonzero(is_str)
        text = values.where(is_str)
    # Frequency table of the str values: count desc, ties first-seen.
    order = str_pos[np.argsort(-counts[str_pos], kind="stable")]
    freq = pd.Series(counts[order], index=pd.Index(text.to_numpy()[order], dtype=object))
    profile = profile_text(freq, tokens, rng_seed=0)
    if not all_str:
        if "usable_distinct" in profile:
            # The numbers/bools of a mixed column are usable distinct values too.
            profile["usable_distinct"] += len(uniques) - len(str_pos)
        # A constant, complete list or uniform count of the str values is not
        # the column's: the renderer counts a listed domain over every
        # non-null row, so [True, False, "yes"] read 'constant "yes" ×3' for
        # one "yes" row and [1, 2, "a", "b"] listed "a", "b" as every value.
        # The domain falls back to the distinct count; a missing token the
        # list held stays a fact, at its rows.
        listed_tokens = profile.pop("missing_token_values", [])
        for key in ("constant", "values", "uniform_count"):
            profile.pop(key, None)
        if listed_tokens:
            profile["missing_tokens"] = [(t, int(freq.loc[t])) for t in listed_tokens]

    stripped = text.str.strip()
    empty = (stripped == "") | stripped.str.lower().isin(tokens)
    empties = set(np.flatnonzero(empty.to_numpy()).tolist())

    # A number-text column is a measure for the table facts (sum columns,
    # total rows) once parsed with its own format (numbers stored as numbers
    # in a mixed column keep their value); an ambiguous or mixed format has
    # no single parse, so it is not one.
    number = profile.get("number")
    measure, view = None, None
    if number and not number.get("ambiguous") and not number.get("mixed"):
        parsed = parse_number_text(values, number).to_numpy(dtype=np.float64)
        measure = np.append(parsed, np.nan)[codes]
        view = _number_text_view(parsed, counts)
    return profile, measure, empties, view


def numeric_profile(uniques, counts, codes, top_k: int = 5) -> dict:
    """Numeric facts of one column, from its frequency table.

    Args:
        uniques: The column's distinct non-null values (any numeric dtype, or
            Python int/float objects), as ``pd.factorize`` returns them.
        counts: Row count of each distinct value (aligned with ``uniques``).
        codes: Row -> index into ``uniques`` (-1 = null); read only for the
            row order of a sequence.
        top_k: Size of the "most common" head.

    Returns:
        The ColumnStat fields to set (absent = nothing to report): constant
        | values (complete list, count desc, ties value asc) | numeric_min,
        numeric_max, then sequence, quartiles, uniform_count | top_values
        and top_values_tied, far_high, repeated_extreme; negative_rows,
        infinite_rows. Values are raw (built-in) values of the column, so an
        int column's bounds and whole quartiles are ints; quartiles of a
        whole column are observed values, of a non-whole column the
        unrounded linear quantile.

    Every fact but the domain (constant, complete list) reads the distinct
    FINITE values sorted once (+/-inf rows are not numbers a planner can
    bound), so the cost scales with distinct values, not rows.
    """
    raw_values = np.asarray(uniques)
    as_float = raw_values.astype(np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    finite = np.flatnonzero(np.isfinite(as_float))
    # Sort the stored values, not their float64 copies: integers above 2**53
    # collide or swap in float64, so the 18-digit NYC OFT codes showed a max
    # of 570429510300525888 for the stored 570429510300525908, and 15 ids
    # 10**18+1..+15 a range and quartiles of 10**18. Object arrays compare
    # Python ints exactly.
    key = as_float[finite] if raw_values.dtype.kind == "f" else raw_values[finite]
    pos = finite[np.argsort(key, kind="stable")]
    v = as_float[pos]
    n = counts[pos]
    card = len(v)

    def raw(i: int) -> Any:
        return _builtin(raw_values[pos[i]])

    # +/-inf are stored values (pandas reads "inf"/"Infinity" text as float
    # inf), so the constant and the complete list hold them, at the ends of
    # the value order: [7]*10 + [inf]*10 read "constant 7", and 50 inf rows
    # beside 1, 2, 3 read "3 values, each once".
    domain = np.concatenate([np.flatnonzero(as_float == -np.inf), pos, np.flatnonzero(as_float == np.inf)])
    if len(domain) == 0:
        return {}
    if len(domain) == 1:
        return {"constant": _builtin(raw_values[domain[0]])}

    out: dict = {}
    nn = int(n.sum())
    negative = int(n[v < 0].sum())
    # A few negatives in a mostly non-negative column (refunds, reversals, -1
    # sentinels); a column mostly below zero (longitudes) is just its range.
    if 0 < negative < nn / 2:
        out["negative_rows"] = negative

    # An all-distinct measure with 11-20 values is NOT listed (unlike
    # categorical text): twelve amounts each seen once say less than their
    # range and quartiles, at several times the tokens.
    if len(domain) <= _LIST_MAX_CARD:
        rows = counts[domain]
        order = np.argsort(-rows, kind="stable")  # count desc, ties value asc
        out["values"] = [(_builtin(raw_values[domain[i]]), int(rows[i])) for i in order]
        return out

    # Past the list, the inf rows would be in no fact at all.
    infinite = int(counts.sum()) - nn
    if infinite:
        out["infinite_rows"] = infinite

    by_count = np.argsort(-n, kind="stable")  # count desc, ties value asc
    whole = bool(np.all(v == np.round(v)))
    all_distinct = card == nn
    sequence = None
    if all_distinct and whole and card >= _SEQUENCE_MIN_CARD:
        sequence = _sequence(v, codes, as_float)

    out["numeric_min"] = raw(0)
    out["numeric_max"] = raw(card - 1)
    shown = set()  # sorted positions already shown with their rows
    if sequence:
        out["sequence"] = sequence
    else:
        head = by_count[:top_k]
        coverage = float(n[head].sum()) / nn
        if coverage < _QUARTILE_MAX_COVERAGE:
            out["quartiles"] = _quartiles(v, n, nn, whole, raw)
        if n[0] >= 2 and bool(np.all(n == n[0])):
            # Every value has the same count (Paris arrondissement: 20 values
            # x13, a balanced panel): any "most common" head would be an
            # arbitrary cut by value order. Count 1 is "all distinct".
            out["uniform_count"] = int(n[0])
        else:
            if card <= _DISCRETE_MAX_CARD or coverage >= _HEAD_MIN_COVERAGE:
                keep = [i for i in head if n[i] >= 2]
            else:
                keep = [i for i in head if n[i] >= 2 and n[i] >= _SPIKE_SHARE * nn]
            keep, tied = _outside_tie(keep, by_count, n)
            if keep:
                out["top_values"] = [(raw(i), int(n[i])) for i in keep]
                shown.update(int(i) for i in keep)
            if tied:
                out["top_values_tied"] = tied

    far = _far_high(v, n, nn)
    if far:
        j, rows = far
        above = v[j:]
        block = (
            len(above) >= 2
            and bool(np.all(above == np.round(above)))
            and bool(above[-1] - above[0] < _BLOCK_MAX_SPAN * above[-1])
        )
        if block:
            # A block renders as its bounds: largest first, like the values.
            shown_values = [raw(card - 1), raw(j)]
        else:
            shown_values = [raw(i) for i in range(card - 1, max(j, card - _FAR_HIGH_SHOWN) - 1, -1)]
        out["far_high"] = {"values": shown_values, "rows": rows, "below": raw(j - 1), "block": block}
        shown.update(range(j, card))

    repeated = {
        side: {"value": raw(i), "rows": int(n[i])}
        for side, i in _repeated_extremes(v, n, nn, whole, out.get("quartiles"))
        if i not in shown
    }
    if repeated:
        out["repeated_extreme"] = repeated
    return out


def _outside_tie(keep: List[int], by_count: np.ndarray, n: np.ndarray):
    """``(keep, tied)``: the "most common" head (a prefix of ``by_count``)
    without a cut inside a tie. When the last kept count equals the next
    value's, every kept value with that count is dropped and ``tied`` says
    how many values of the column share it (``{"values", "rows"}``): which
    of them a 5-item cut kept depended only on value order, so the list
    read as more frequent than the rest (fact audit: 2.4 and 3 x6 hidden
    behind 1.4, 1.6, 1.8 x6 in a 70-row Valencia index column)."""
    k = len(keep)
    if k == 0 or k == len(by_count) or n[by_count[k]] != n[keep[-1]]:
        return keep, None
    count = n[keep[-1]]
    return [i for i in keep if n[i] != count], {"values": int((n == count).sum()), "rows": int(count)}


def _repeated_extremes(
    v: np.ndarray, n: np.ndarray, nn: int, whole: bool, quartiles: Optional[List[Union[int, float]]]
):
    """``(side, i)`` for the min (i = 0) and max (i = card - 1) that repeat
    like a floor, ceiling or default: >= 20 rows, >= 5x the mean count of the
    3 inward distinct values, at least the column's mean rows per distinct
    value (a code below its mean is no spike: NYC census block 1 x68 against
    124 rows per block), and dense inward values (within 0.25 x IQR). A zero
    minimum is never one: it is the domain bound of counts and amounts, not
    a cap (Bologna euro_2: 0 x24 are real zero vehicle counts)."""
    card = len(v)
    inward = {
        "min": (0, np.arange(1, _REPEATED_NEIGHBOURS + 1)),
        "max": (card - 1, np.arange(card - 2, card - 2 - _REPEATED_NEIGHBOURS, -1)),
    }
    for side, (i, near) in inward.items():
        if (
            n[i] < _REPEATED_MIN_ROWS
            or n[i] < _REPEATED_MIN_FACTOR * float(n[near].mean())
            or n[i] * card < nn
            or (side == "min" and v[i] == 0)
        ):
            continue
        if quartiles is None:
            quartiles = _quartiles(v, n, nn, whole)
        if abs(v[near[-1]] - v[i]) <= _REPEATED_MAX_SPAN_IQR * (quartiles[2] - quartiles[0]):
            yield side, i


def _sequence(v: np.ndarray, codes: np.ndarray, as_float: np.ndarray) -> Optional[dict]:
    """A record id / row number: distinct whole values (sorted ``v``) filling
    at least half of one gcd-stepped run from min to max, outside the year
    axis. Its min/max read like any quantity's; the planner must know never
    to sum, average or correlate it."""
    if np.abs(v).max() >= _EXACT_FLOAT_INT or (v[0] >= _YEAR_AXIS[0] and v[-1] <= _YEAR_AXIS[1]):
        return None
    ints = v.astype(np.int64)
    step = int(np.gcd.reduce(np.diff(ints)))
    gaps = int((ints[-1] - ints[0]) // step) + 1 - len(ints)
    if gaps > len(ints):
        return None
    rows = as_float[codes[codes >= 0]]
    steps = np.diff(rows[np.isfinite(rows)])
    if (steps > 0).all():
        row_order = "ascending"
    elif (steps < 0).all():
        row_order = "descending"
    else:
        row_order = "unsorted"
    return {"step": step, "gaps": gaps, "row_order": row_order}


def _quartiles(
    v: np.ndarray, n: np.ndarray, nn: int, whole: bool, raw: Optional[Callable[[int], Any]] = None
) -> List[Union[int, float]]:
    """p25, p50, p75 over the rows. A whole column gets the OBSERVED value at
    rank floor(q*(nn-1)): years, codes and counts stay real values (3
    significant digits would have changed a quartile in 395 of 930 whole
    columns), as stored (``raw(i)``: an int column keeps its ints; without
    ``raw``, its float64 value, for arithmetic only). A non-whole column gets
    numpy's linear quantile, unrounded (the renderer rounds it to the IQR's
    precision)."""
    cum = np.cumsum(n)
    out = []
    for q in (0.25, 0.5, 0.75):
        rank = q * (nn - 1)
        low = int(np.floor(rank))
        i = int(np.searchsorted(cum, low + 1))
        if whole:
            out.append(raw(i) if raw else float(v[i]))
        else:
            j = int(np.searchsorted(cum, low + 2)) if low + 1 < nn else i
            out.append(float(v[i] + (rank - low) * (v[j] - v[i])))
    return out


def _far_high(v: np.ndarray, n: np.ndarray, nn: int) -> Optional[tuple]:
    """``(j, rows)``: the widest ratio gap among the largest distinct values,
    ``v[j-1]`` below it and ``rows`` rows above it (a sentinel, a total row,
    another id space or a genuine large record: the planner decides). Only on
    non-negative columns, where a ratio of magnitudes means something."""
    if v[0] < 0:
        return None
    card = len(v)
    best = None
    # Gaps between consecutive values among the top _FAR_HIGH_SCAN, scanned
    # from the top: on equal ratios the highest gap wins.
    for j in range(card - 1, max(card - _FAR_HIGH_SCAN, 0), -1):
        lo, hi = v[j - 1], v[j]
        if lo <= 0:
            continue
        ratio = hi / lo
        rows = int(n[j:].sum())
        if (
            ratio >= _FAR_HIGH_MIN_RATIO
            and rows <= max(1, _FAR_HIGH_MAX_SHARE * nn)
            and (best is None or ratio > best[0])
        ):
            best = (ratio, j, rows)
    return (best[1], best[2]) if best else None
