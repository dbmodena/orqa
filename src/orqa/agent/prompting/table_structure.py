"""Table facts for the column-statistics block: how rows and columns relate.

``profile_table`` reports, from the per-column codes ``ColumnStatistics.compute``
factorizes once, the facts a planner gets wrong when it only sees one column
at a time (SPEC section 4; measured on 331 real tables in the research):

- fully empty rows and rows that repeat an earlier row (counts are off),
  with the evidence that tells a double entry from a real repeat;
- a grand-total row or rollup total rows mixed with the detail rows (every
  sum over them doubles: exactly 2x in all 7 grand-total tables found);
- the multi-column grain, ``one row per (sezione, numero_lista)`` (counting
  rows instead of entities: 8,996 "sections" instead of 692), and a part
  numbered within the others (polling station 1..k per arrondissement);
- identical columns, one-to-one columns (a code and its name are ONE
  dimension, also when a few shifted rows break it), many-to-one ``each a
  has one b`` (a number repeated on every row of its group double counts
  when summed), and sum columns (a total added to its parts double counts);
- a fill-down column (SPEC v4 section B: a minister named once above the
  meetings of their block, a district above its settlements): its blank
  cells hold the value above them, so grouping or counting per value needs
  a forward fill first, and the repeated rows are counted again with it
  filled (two blank-minister rows equal in every other cell are no repeat
  once their ministers differ).

Everything is language-free: arithmetic on numbers and equality of codes, no
word lists, no column-name semantics. A fact fires only when it holds on
every row (sum columns: on >= 98% of complete rows) AND is unlikely to hold
by accident; each accident guard below carries the false-positive rate the
critics measured without it (the rollup search's grouping heuristics
excepted: carried over from the research unmeasured, and said so where they
are defined).

Relations are tested on code arrays, never on strings: "a determines b"
holds when one value of b is seen per value of a, which one scatter
``m[code_a] = code_b`` and one gather check in O(rows) without hashing (null,
code -1, lands in the last slot of ``m``: null is a value). A relation that
holds on the table holds on every subset of its rows, so candidates are first
tested on nested seeded samples and only the survivors on all rows; for these
exact relations that gives the same answer as testing everything on all rows.
Codes may arrive as int32 (statistics.py keeps them so under 2**31 values);
every combination of codes (row keys, leave-one-out keys) is built in int64,
because int32 x + 1 and key * width wrap silently under NumPy 2 promotion.
"""

import math
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from pandas.util import hash_array

# Nested screening samples (prefixes of one seeded random row order). The
# grain is screened on the larger one and at most _GRAIN_MAX_VERIFY survivors
# are checked on all rows (SPEC 4.3): more survivors than that means the
# minimal key is not unique, so no grain is reported.
_PRESCREEN_ROWS = 5_000
_SCREEN_ROWS = 200_000
_GRAIN_MAX_VERIFY = 3
# Sum-column candidates (up to ~6,000 per 20-measure table) are screened on
# this many seeded rows at a loose 90% match, then the 98% rule is checked on
# all rows. A column matching on >= 98% of its rows cannot fall under 90% on
# the sample (that is > 5 standard deviations even with 100 complete rows),
# so the screen only skips work.
_SUM_SCREEN_ROWS = 3_000
_SUM_SCREEN_MATCH = 0.9
_SUM_MATCH = 0.98

# "Equals the sum": published totals and sum columns are rounded copies of
# their parts. 0.2% relative (plus 0.011 for two-decimal rounding of each
# part, sum columns) kept every real total in the research sample and no
# false one.
_SUM_REL_TOL = 0.002
_SUM_ABS_TOL = 0.011

# Totals mixed with detail rows are a spreadsheet layout; the rollup search
# groups the whole table once per dimension, so both searches stop here.
_TOTAL_MAX_ROWS = 100_000

# Many-to-one chance guard: log10 of the probability that every multi-row
# group of a is pure in b if a and b were independent. A 20-column table runs
# up to 380 directed tests: the research's -3 allowed ~0.4 accidental edges
# per table (the critics found 50 of 520 edges above -3); -5 allows 0.004.
_CHANCE_LOG10 = -5.0
_MAX_EDGES = 6

# Grain accident guard: colliding row pairs expected if the key's columns
# were independent, C(n,2) x prod over the columns of P(two DISTINCT rows
# share the value) (null a value). A row paired with itself is no collision:
# counting it (sum p_v^2) adds ~n/(2k) per k-valued column, so a year plus a
# near-unique int count passed as the grain of a 600-row cube (15.8 instead
# of 0.77), hiding (year, region, sex) at 285. Re-measured on 719 real
# tables: the 12 accidental keys the self-pair formula let through all have
# E <= 2.71 (LCG x Registered_Patients 1.22, titolo x subcategoria 1.71,
# Organisation ID x Start period 2.71); the weakest real grain kept is 10.3,
# the referendum (sezione, risposta) 225. A threshold of 5 lets in three
# accidental keys (three int measures 6.18, ENUMBER x geometry 8.28, 9.07)
# and makes three real grains ambiguous; 10 leaves nine weak real grains
# (E 4.8-9.8, Año x Mes on 33 rows 9.55) unreported: absent, not wrong.
_GRAIN_MIN_EXPECTED_COLLISIONS = 10.0

# Values of a calendar-year column, as statistics._YEAR_AXIS (which can
# import this one: statistics imports this module, so not the reverse). A
# year is one axis shared by every group, never numbered within one.
_YEAR_AXIS = (1800, 2100)

# A text column whose cells average more than this many characters is prose,
# never a dimension: as a determinant or a key part it is unique by accident.
_LONG_TEXT_CHARS = 100

# Fact audit (~340 facts on 150 real tables) thresholds, each with the case
# it was measured on:
# - repeated rows: a column distinct on >= 90% of the non-empty rows
#   identifies rows, so rows identical even in it are double entries (a
#   MULTIPOLYGON 98.8% distinct, a transaction number 99.3% distinct); with
#   no such column identical rows are plausibly real events (37.7% of 20,841
#   parking tickets: same street, day and offence).
_IDENTIFYING_SHARE = 0.9
# - rollup totals: share of the marker rows that must hold the sum of their
#   group. One NYC marker (Indicator Sub-Category null) covered 540 rows, of
#   which 72 were alone in their group and 76 were no sum (at most 73%
#   pass). In a genuine HIV cube SEX "All" passes on 2,186 of 2,192 rows;
#   RACE/ETHNICITY "All" on 1,348 of 1,528 (176 differ from the sum of the
#   races, so "drop the All rows, then sum" is not safe there).
_ROLLUP_MARKER_PASS = 0.9
# - rollup search heuristics, carried over from the r_structure research
#   unmeasured (no false-positive rate was recorded for them; the marker
#   test above is what the audit measured). Whole-number measures with 2 to
#   max(60, 2% of rows) values join the dimensions in the second pass:
_ROLLUP_LOW_INT_VALUES = (60, 0.02)
#   a dimension with more than max(50, 20% of rows) values is no rollup axis:
_ROLLUP_MAX_DIM_VALUES = (50, 0.2)
#   groups of >= 3 rows (a total and two parts) must cover >= 30% of rows:
_ROLLUP_MIN_GROUP = 3
_ROLLUP_MIN_COVERED = 0.3
#   a measure adds up over a dimension when its largest value is the sum of
#   the others in >= 80% of the groups (and >= 2 of them)...
_ROLLUP_FIT_SHARE = 0.8
#   ...and >= 90% of those largest rows share one value of it (the marker).
_ROLLUP_MARKER_SHARE = 0.9
# - nested grain part: share of the other parts' groups whose minimum is the
#   same value (census block 1000 in 96% of 2,325 tracts, polling station 1
#   in 100% of arrondissements; at most 36% for every other part of the 10
#   audited grains).
_NESTED_MIN_SHARE = 0.9
# - near one-to-one: rows whose a is not the most frequent a of its b group.
#   <= 2% flagged exactly the 3 code/name pairs broken by shifted CSV rows
#   (0.32%, 0.46%, 1.1%) among 30 audited relations; every genuine
#   hierarchy had >= 11%.
_NEAR_ONE_TO_ONE = 0.02
# - ...and those rows are anomalies only when most of them sit in rare a
#   values (a group of at most a quarter of the median a group): 72-100% of
#   the exception rows of the 3 shifted pairs, 0-3.5% for the 3 pairs of a
#   chemist table under 2% where distinct practices share a name or a
#   building (139 rows in practices of 20-64 rows, median 35).
_RARE_GROUP_SHARE = 0.25
# - one-to-one support: the repeated groups other than the largest hold >= 1%
#   of the rows ("1" x1250 next to 5 shifted rows made a constant and its
#   label a "pair").
_PAIR_GROUP_SHARE = 0.01

# Fill-down columns (SPEC v4 section B): a text column holding a value on a
# few rows, each heading the blank rows below it. Measured on 1,374 real
# tables (nz_filldown/features.jsonl, rule v2 fired 6: 2 true, 4 false):
# with every term below the sample fires on Minister 2/16, Appointing
# Minister 2/8 and District 6/26 and on nothing else, and the reference case
# (a UK ministers' meetings table, Minister 5/72) fires. The terms are
# evaluated on the NON-EMPTY rows (rows holding some value), in file order,
# as the features were: the reference table has 73 rows, one fully empty.
# - filled rows F: 2 <= F <= 25% of rows (description 42/120 = 35% and
#   internet_1 80/189 = 42% are sparse columns);
_FILL_MIN_FILLED = 2
_FILL_MAX_FILLED_SHARE = 0.25
# - void (null or blank) rows >= 6 and >= 50% of rows (both follow from the
#   F terms above; kept as the spec writes them);
_FILL_MIN_VOID = 6
_FILL_MIN_VOID_SHARE = 0.5
# - every filled cell starts a run: consecutive filled cells never hold the
#   same value. This is what drops EndDate 4/22 ("14/12/2018" twice in a
#   row) and NYC PRE_TYPE 2876/122244 ("AVE", "AVE": a sparse column of a
#   street table sorted by name), and 7 more sparse columns of sorted tables
#   (PropertyAddress5 1716/8136, DOCKET 1199/7080, ...).
# - each filled cell starts a block running to the next filled cell: rows
#   in blocks of length 1 <= 10% of rows, median block length >= 3. The spec
#   writes "<= 10% of blocks"; the reference case has 2 such blocks of 5 (a
#   one-meeting minister at row 57, the last minister at row 71), so the
#   share is taken over the rows (2 of 72); description (17 of 120 rows) and
#   internet_1 (32 of 189) still fail it, as the spec says.
_FILL_MAX_SINGLE_ROW_SHARE = 0.1
_FILL_MIN_MEDIAN_BLOCK = 3
# - >= 2 distinct filled values (Deliverable "Yes" x10 is a flag);
_FILL_MIN_DISTINCT = 2
# - the void rows are otherwise dense: the median over the other columns of
#   their filled share on those rows >= 0.5, so a sparse sheet region
#   (MEMBERS ALLOWANCES, 0.29) or an empty-row region holds nothing to fill
#   down to.
_FILL_MIN_DENSITY = 0.5


def profile_table(
    df: pd.DataFrame,
    codes: Dict[str, np.ndarray],
    uniques: Dict[str, Sequence],
    kinds: Dict[str, str],
    measures: Dict[str, np.ndarray],
    empty_codes: Dict[str, Set[int]],
    seed: int = 0,
) -> Tuple[dict, Dict[str, str]]:
    """Compute the table facts of ``df`` from its per-column codes.

    Args:
        df: The raw table (only its length and column order are read).
        codes: Column -> ``pd.factorize(series, use_na_sentinel=True)`` codes
            (int32, or int64 past 2**31 distinct values; -1 = NaN). Widen to
            int64 before any arithmetic on them (key(), _loo_keys).
        uniques: Column -> raw distinct values in code (first-seen) order.
        kinds: Column -> "int" (numeric, all finite values whole), "float"
            (numeric, some non-whole), "text", "bool", "datetime" or "other".
        measures: Column -> float64 row array (NaN where not a number) for
            numeric and number-text columns.
        empty_codes: Column -> codes of blank or missing-token values.
        seed: Seed of the screening samples. Screening never changes an
            exact relation; only the grain's verification cap can make the
            result depend on it (and the sum-column screen, by a > 5-sigma
            miss).

    Returns:
        ``(structure, same_as)``. ``structure`` holds only the facts that
        fire: ``empty_rows``; ``repeated_rows`` {rows, largest, unique_col
        (None: no column identifies rows), after_fill (only with fill-down
        columns: the repeats once they are forward-filled; rows is 0 when
        repeats exist only after the fill)}; ``fill_down`` {col: {filled,
        rows (the non-empty rows)}} and ``fill_down_cols`` [cols];
        ``total_row`` {measure, value, label_col, label}; ``total_rows``
        [{col, value, rows (marker rows holding the sum of their group), of
        (all marker rows), group [cols]}]; ``grain`` {cols (parents first),
        nested {part: [parents]}}; ``one_to_one`` [{cols, except (rows
        breaking it; omitted when exact)}]; ``determines`` [[a, [bs]]] and
        ``determines_more`` (edges cut by the cap); ``sum_columns`` [[t,
        [parts]]]. Every list is in column order, built-in types only.
        ``same_as`` maps each column identical to an earlier one to the
        first.
    """
    t = _Table(df, codes, uniques, kinds, measures, empty_codes, seed)
    structure: Dict[str, Any] = {}
    # No columns (every header cell blank, dropped by clean_columns): no row
    # holds a value, and the row hash has no column to start from.
    if t.n == 0 or not t.cols:
        return structure, {}

    empty = t.empty_rows()
    if empty.any():
        structure["empty_rows"] = int(empty.sum())
    fill = _fill_down(t, empty)
    repeated = t.repeated_rows(empty)
    if fill:
        # The repeats a second time with the fill-down columns filled: rows
        # equal only because both left the cell blank are no repeat once
        # their values differ (the reference table: 2 -> 0), and a row
        # repeating its block head without the label becomes one (0 -> k).
        after = t.repeated_rows(empty, filled=tuple(fill))
        after_rows = after["rows"] if after else 0
        if repeated is None and after_rows:
            repeated = {"rows": 0, "largest": 1, "unique_col": None}
        if repeated is not None:
            repeated["after_fill"] = after_rows
    if repeated:
        structure["repeated_rows"] = repeated
    if fill:
        structure["fill_down"] = fill
        structure["fill_down_cols"] = list(fill)

    # An identical column is an alias of the first: it takes no part in any
    # other relation (a duplicated measure would otherwise "confirm" a total
    # row on its own, and every relation would be listed twice).
    same_as = _identical_columns(t)
    cols = [c for c in t.cols if c not in same_as]

    total = _grand_total(t, cols)
    rollups = _rollup_totals(t, cols, has_grand_total=total is not None)
    if total is not None:
        row, measure, value = total
        label_col, label = _total_label(t, cols, row, rollups)
        structure["total_row"] = {
            "measure": measure,
            "value": value,
            "label_col": label_col,
            "label": label,
        }
    if rollups:
        structure["total_rows"] = [
            {"col": col, "value": _py(t.value(col, code)), "rows": rows, "of": of, "group": group}
            for col, code, rows, of, group in rollups
        ]

    partitions = _same_partition_pairs(t, cols)
    grain = _grain(t, cols, partitions)
    if grain:
        structure["grain"] = grain
    classes = _one_to_one(t, partitions)
    determines, near, more = _many_to_one(t, cols, classes)
    pairs = [{"cols": cls} for cls in classes] + [
        {"cols": sorted((a, b), key=t.pos.get), "except": k} for a, b, k in near
    ]
    if pairs:
        structure["one_to_one"] = sorted(pairs, key=lambda p: [t.pos[c] for c in p["cols"]])
    if determines:
        structure["determines"] = determines
        if more:
            structure["determines_more"] = more
    sums = _sum_columns(t, cols)
    if sums:
        structure["sum_columns"] = sums
    return structure, same_as


class _Table:
    """Per-column facts every detector reads, computed once."""

    def __init__(self, df, codes, uniques, kinds, measures, empty_codes, seed):
        self.n = n = int(len(df))
        self.cols = list(df.columns)
        self.pos = {c: i for i, c in enumerate(self.cols)}
        self.codes = codes
        self.uniques = uniques
        self.kinds = kinds
        self.measures = measures
        self.empty_codes = {c: set(empty_codes.get(c, ())) for c in self.cols}
        self.card = {c: len(uniques[c]) for c in self.cols}
        # counts[c][k] = rows holding value k, with the null count in the
        # LAST slot, so counts[c][codes[c]] is every row's group size.
        self.counts = {}
        for c in self.cols:
            x = codes[c]
            cnt = np.bincount(x[x >= 0], minlength=self.card[c])
            self.counts[c] = np.append(cnt, n - int(cnt.sum()))
        self.nulls = {c: int(self.counts[c][-1]) for c in self.cols}
        # Distinct values counting null as one: the size of the partition of
        # the rows the column induces.
        self.cardp = {c: self.card[c] + (self.nulls[c] > 0) for c in self.cols}
        rng = np.random.default_rng(seed)
        self._order = (
            rng.choice(n, _SCREEN_ROWS, replace=False) if n > _SCREEN_ROWS else rng.permutation(n)
        )
        self._samples: Dict[int, np.ndarray] = {}
        self._sample_codes: Dict[Tuple[Any, int], np.ndarray] = {}
        self._long: Dict[Any, bool] = {}
        self._fractional: Dict[Any, bool] = {}

    # -- samples ---------------------------------------------------------
    def stages(self) -> List[Optional[int]]:
        """Screening sample sizes below ``n``, then None = all rows."""
        return [k for k in (_PRESCREEN_ROWS, _SCREEN_ROWS) if k < self.n] + [None]

    def rows(self, k: Optional[int]) -> Optional[np.ndarray]:
        """Sorted seeded sample of ``k`` rows (None = all rows); the samples
        are prefixes of one random order, so each contains the smaller."""
        if k is None:
            return None
        if k not in self._samples:
            self._samples[k] = np.sort(self._order[:k])
        return self._samples[k]

    def col(self, c, k: Optional[int] = None) -> np.ndarray:
        if k is None:
            return self.codes[c]
        if (c, k) not in self._sample_codes:
            self._sample_codes[(c, k)] = self.codes[c][self.rows(k)]
        return self._sample_codes[(c, k)]


    # -- column properties -----------------------------------------------
    def value(self, c, code: int):
        return None if code == -1 else self.uniques[c][code]

    def blank_codes(self, c) -> Set[int]:
        u = self.uniques[c]
        return {k for k in self.empty_codes[c] if isinstance(u[k], str) and not u[k].strip()}

    def is_long(self, c) -> bool:
        if self.kinds[c] != "text":
            return False
        if c not in self._long:
            u = self.uniques[c]
            if isinstance(u, pd.Index) and isinstance(u.dtype, pd.StringDtype):
                # A "str" column's values stay Arrow strings (statistics._compute).
                lengths = u.str.len().to_numpy(dtype=np.int64, na_value=0)
            else:
                lengths = np.fromiter(
                    (len(v) if isinstance(v, str) else len(str(v)) for v in u), dtype=np.int64, count=len(u)
                )
            rows = self.counts[c][:-1]
            filled = int(rows.sum())
            self._long[c] = bool(filled) and float(lengths @ rows) / filled > _LONG_TEXT_CHARS
        return self._long[c]

    def is_fractional(self, c) -> bool:
        """A fractional float, or number text holding a non-whole number
        (a coordinate stored as text is as accidental a key as a float)."""
        if self.kinds[c] == "float":
            return True
        if self.kinds[c] != "text" or c not in self.measures:
            return False
        if c not in self._fractional:
            x = self.measures[c]
            x = x[np.isfinite(x)]
            self._fractional[c] = bool((x != np.round(x)).any())
        return self._fractional[c]

    def is_measure(self, c, min_card: int) -> bool:
        return c in self.measures and self.card[c] >= min_card

    # -- relations ---------------------------------------------------------
    def determines(self, a, b) -> bool:
        """Rows with the same a value share their b value (null is a value):
        screened on the samples, then checked on all rows."""
        return all(self._determines_on(a, b, k) for k in self.stages())

    def _determines_on(self, a, b, k: Optional[int]) -> bool:
        ca, cb = self.col(a, k), self.col(b, k)
        m = np.empty(self.card[a] + 1, dtype=np.int64)
        m[ca] = cb
        return bool(np.array_equal(m[ca], cb))

    def mapping(self, a, b) -> np.ndarray:
        """b's code for each of a's codes, on a relation a -> b that holds."""
        m = np.full(self.card[a] + 1, -1, dtype=np.int64)
        m[self.codes[a]] = self.codes[b]
        return m[:-1]

    def key(
        self,
        cols: Sequence,
        k: Optional[int] = None,
        rows: Optional[np.ndarray] = None,
        codes: Optional[Dict[Any, np.ndarray]] = None,
    ) -> np.ndarray:
        """One int64 per row identifying its combination of values in ``cols``
        (null is a value), re-coded densely whenever the code space would
        overflow. Codes are widened first: int32 codes wrap silently here.
        ``codes`` replaces the table's own (the forward-filled columns of
        the after-fill repeat count)."""
        key, size = None, 1
        for c in cols:
            if codes is not None:
                x = codes[c] if rows is None else codes[c][rows]
            else:
                x = self.col(c, k) if rows is None else self.codes[c][rows]
            x = x.astype(np.int64, copy=False)
            width = self.card[c] + 1
            if key is None:
                key, size = x + 1, width
                continue
            if size * width >= 2**62:
                key, distinct = pd.factorize(key)
                size = len(distinct)
            key = key * width + (x + 1)
            size *= width
        return key

    def is_key(self, cols: Sequence, k: Optional[int] = None) -> bool:
        key = self.key(cols, k)
        return len(pd.unique(key)) == len(key)

    # -- rows ----------------------------------------------------------------
    def void(self, c) -> np.ndarray:
        """Per code slot (null last): True for null and blank values (a
        missing token is a value here: a configured token can be a real
        category)."""
        flag = np.zeros(self.card[c] + 1, dtype=bool)
        flag[-1] = True
        flag[list(self.blank_codes(c))] = True
        return flag

    def empty_rows(self) -> np.ndarray:
        """Rows whose every cell is null or blank."""
        empty = np.ones(self.n, dtype=bool)
        for c in self.cols:
            empty &= self.void(c)[self.codes[c]]
            if not empty.any():
                break
        return empty

    def filled_codes(self, c) -> np.ndarray:
        """``codes[c]`` forward-filled: every void cell (null or blank) takes
        the last filled cell above it, as ``Series.ffill`` would; the rows
        above the first filled cell take the first row's (void) code."""
        x = self.codes[c]
        held = np.where(self.void(c)[x], 0, np.arange(self.n))
        np.maximum.accumulate(held, out=held)
        return x[held]

    def repeated_rows(self, empty: np.ndarray, filled: Sequence = ()) -> Optional[dict]:
        """Non-empty rows equal in every column to an earlier row, with what
        tells a double entry from a real repeat (a bare count steered a
        planner to deduplicate 38% of real parking tickets): ``largest``,
        the biggest group of identical rows (463 copies of an embedded
        header row), and ``unique_col``, the most distinct column that
        identifies rows (see _IDENTIFYING_SHARE) and holds a value on every
        repeated row, or None when no column identifies rows. With
        ``filled``, those columns are forward-filled first (the fill-down
        columns; ``unique_col`` still reads the stored cells)."""
        # A column holding n distinct values rules repeats out without the
        # row hash (the hash is the costliest step: 7 s on 64M rows).
        if any(self.card[c] == self.n for c in self.cols):
            return None
        codes = self.codes
        if filled:
            codes = {c: self.filled_codes(c) if c in filled else self.codes[c] for c in self.cols}
        rows = np.flatnonzero(~empty)
        sub = rows if empty.any() else None
        # One 64-bit hash per row, grouped once by np.unique (SPEC v4 G3:
        # 0.59 s against 3.0 s for pd.factorize on a 7M-row table): the
        # exact radix key re-factorized whenever its code space passed
        # 2**62, then was factorized again (14.3 s each of the 31.5 s on a
        # 113M-row table). Equal rows hash alike, so a row alone in its hash
        # group repeats nothing; a collision can only merge different rows,
        # and the exact key on the candidate rows splits them again.
        h = None
        for c in self.cols:
            x = hash_array(codes[c] if sub is None else codes[c][sub])
            if h is None:
                h = x
            else:
                h *= np.uint64(1000003)
                h ^= x
        _, inverse, counts = np.unique(h, return_inverse=True, return_counts=True)
        if len(counts) == len(h):
            return None
        cand = rows[counts[inverse] >= 2]
        group, distinct = pd.factorize(self.key(self.cols, rows=cand, codes=codes))
        sizes = np.bincount(group)
        repeated = cand[sizes[group] >= 2]
        if not len(repeated):
            return None
        unique_col = None
        for c in self.cols:
            # Rows "identical" in a cell they all leave empty share nothing.
            if self.card[c] >= _IDENTIFYING_SHARE * len(rows) and not self.void(c)[self.codes[c][repeated]].any():
                if unique_col is None or self.card[c] > self.card[unique_col]:
                    unique_col = c
        # Rows outside the candidates are distinct from every row.
        return {"rows": int(len(cand) - len(distinct)), "largest": int(sizes.max()), "unique_col": unique_col}


def _py(v):
    """A raw value as a built-in Python type (numpy scalars, timestamps)."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def _classes(pairs: Sequence[Tuple[Any, Any]], pos: Dict[Any, int]) -> List[List[Any]]:
    """Merge pairs sharing a column into classes; columns and classes in
    column order."""
    parent: Dict[Any, Any] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb, key=pos.get)] = min(ra, rb, key=pos.get)
    groups: Dict[Any, List[Any]] = {}
    for c in parent:
        groups.setdefault(find(c), []).append(c)
    return sorted((sorted(g, key=pos.get) for g in groups.values()), key=lambda g: pos[g[0]])


# ---------------------------------------------------------------------------
# SPEC v4 B: fill-down columns
# ---------------------------------------------------------------------------
def _fill_down(t: _Table, empty: np.ndarray) -> Dict[Any, dict]:
    """Text columns whose few values head the blank rows below them, as
    ``{col: {"filled": F, "rows": m}}`` in column order, ``m`` the non-empty
    rows. Every term is language-free (see the ``_FILL_*`` constants) and
    reads the non-empty rows in file order:

    - the first row is filled (a leading gap is a sparse column: APPEALDATE
      lead 68, BUA24NMW lead 2147);
    - 2 <= F <= 25% of rows; void (null or blank) rows >= 6 and >= 50%;
    - each filled cell starts a block running to the next filled cell: rows
      in blocks of length 1 <= 10% of rows, median block length >= 3;
    - the void rows are otherwise dense: the median over the other columns
      of their filled share on those rows >= 0.5;
    - every filled cell starts a run: consecutive filled cells never hold
      the same value (compared stripped, on the distinct heads only), and
      >= 2 distinct filled values.

    The structural terms come first: the string work runs only on the few
    columns that pass them, over their distinct head values.
    """
    rows = np.flatnonzero(~empty)
    m = int(len(rows))
    text = [c for c in t.cols if t.kinds[c] == "text"]
    if m < _FILL_MIN_FILLED + _FILL_MIN_VOID or len(t.cols) < 2 or not text:
        return {}
    sub: Any = rows if m < t.n else slice(None)
    voids: Dict[Any, np.ndarray] = {}

    def void_of(c) -> np.ndarray:
        """Per non-empty row: True where c is null or blank (the other
        columns' masks are built only when a candidate reaches the density
        term)."""
        if c not in voids:
            voids[c] = t.void(c)[t.codes[c][sub]]
        return voids[c]

    found: Dict[Any, dict] = {}
    for c in text:
        v = void_of(c)
        heads = np.flatnonzero(~v)
        f = int(len(heads))
        if f < _FILL_MIN_FILLED or f > _FILL_MAX_FILLED_SHARE * m:
            continue
        if m - f < max(_FILL_MIN_VOID, _FILL_MIN_VOID_SHARE * m):
            continue
        if heads[0] != 0:
            continue
        lengths = np.diff(np.append(heads, m))
        if int((lengths == 1).sum()) > _FILL_MAX_SINGLE_ROW_SHARE * m:
            continue
        if float(np.median(lengths)) < _FILL_MIN_MEDIAN_BLOCK:
            continue
        void_rows = np.flatnonzero(v)
        shares = [1.0 - float(void_of(o)[void_rows].mean()) for o in t.cols if o != c]
        if float(np.median(shares)) < _FILL_MIN_DENSITY:
            continue
        head_codes, inverse = np.unique(t.codes[c][sub][heads], return_inverse=True)
        u = t.uniques[c]
        stripped, distinct = pd.factorize(np.array([str(u[k]).strip() for k in head_codes], dtype=object))
        values = stripped[inverse]
        if len(distinct) < _FILL_MIN_DISTINCT or bool((values[1:] == values[:-1]).any()):
            continue
        found[c] = {"filled": f, "rows": m}
    return found


# ---------------------------------------------------------------------------
# 4.4 identical columns
# ---------------------------------------------------------------------------
def _same_values(t: _Table, a, b) -> bool:
    """Equal distinct values in code order (numbers compared as float64,
    anything else as str). Equal CODES alone prove nothing: pd.factorize
    numbers values by first appearance, so any two columns that split the
    rows the same way get equal codes (every pair of all-distinct columns,
    every code/name pair: 98.9% of equal-code pairs in the critics' sample
    held different values)."""
    ua, ub = t.uniques[a], t.uniques[b]
    if len(ua) != len(ub) or not np.array_equal(t.codes[a], t.codes[b]):
        return False
    if t.kinds[a] in ("int", "float") and t.kinds[b] in ("int", "float"):
        return all(float(x) == float(y) for x, y in zip(ua, ub))
    return all(str(x) == str(y) for x, y in zip(ua, ub))


def _identical_columns(t: _Table) -> Dict[Any, Any]:
    same_as: Dict[Any, Any] = {}
    firsts: List[Any] = []
    for b in t.cols:
        for a in firsts:
            if (
                t.kinds[a] == t.kinds[b]
                and t.card[a] == t.card[b] >= 2
                and t.nulls[a] == t.nulls[b]
                # Equal codes put the nulls on the same rows, so both or
                # neither column is mostly null: two near-empty columns
                # share their blanks, not a meaning.
                and t.nulls[a] < 0.9 * t.n
                and _same_values(t, a, b)
            ):
                same_as[b] = a
                break
        else:
            firsts.append(b)
    return same_as


# ---------------------------------------------------------------------------
# 4.5 one-to-one (and the equal partitions the grain collapses)
# ---------------------------------------------------------------------------
def _same_partition_pairs(t: _Table, cols: Sequence) -> List[Tuple[Any, Any]]:
    """Column pairs that split the rows identically (null as a value):
    a -> b with as many groups on both sides is a bijection. All-distinct
    columns are skipped (they trivially pair with each other)."""
    pairs = []
    for a, b in combinations(cols, 2):
        if t.cardp[a] == t.cardp[b] and 2 <= t.cardp[a] < t.n and t.determines(a, b):
            pairs.append((a, b))
    return pairs


def _pair_ok(t: _Table, a, b) -> bool:
    """Columns that can name one dimension together."""
    if any(t.card[c] < 2 or t.nulls[c] > 0.1 * t.n or t.is_fractional(c) for c in (a, b)):
        return False
    # Two measures in a fixed ratio (a SUM and an AVG over 3 days) split the
    # rows identically without naming the same thing.
    return not (t.is_measure(a, 21) and t.is_measure(b, 21))


def _one_to_one(t: _Table, partitions: Sequence[Tuple[Any, Any]]) -> List[List[Any]]:
    """Classes of columns that split the rows identically (exact pairs)."""
    if t.n < 10:
        return []
    keep = []
    for a, b in partitions:
        if not _pair_ok(t, a, b):
            continue
        # Same values under another type (1 and "1"): neither an alias nor a
        # code/name pair worth a fact.
        if t.kinds[a] != t.kinds[b] and _same_values(t, a, b):
            continue
        if _pair_support(t, a, b):
            keep.append((a, b))
    return _classes(keep, t.pos)


def _pair_support(t: _Table, a, b) -> bool:
    """On a relation a -> b: >= 50% of all rows in multi-row groups of a,
    over >= 2 groups, the groups other than the largest holding >= 1% of
    the rows (see _PAIR_GROUP_SHARE), once the null/blank/missing-token
    group of either column is set aside: a pair whose only repeated value
    is "-" in both columns (every other value once) says nothing about
    codes and names. (Not 1% per group: (School, School Name) holds its 538
    rows in 124 repeated groups of 2-5 rows, none reaching 1%.)"""
    rows = t.counts[a][:-1].copy()
    rows[list(t.empty_codes[a])] = 0
    b_of_a = t.mapping(a, b)
    rows[b_of_a == -1] = 0
    rows[np.isin(b_of_a, list(t.empty_codes[b]))] = 0
    multi = rows >= 2
    covered = int(rows[multi].sum())
    return (
        int(multi.sum()) >= 2
        and covered >= 0.5 * t.n
        and covered - int(rows.max()) >= _PAIR_GROUP_SHARE * t.n
    )


def _near_one_to_one(t: _Table, a, b) -> Optional[int]:
    """On a relation a -> b: the rows breaking b -> a (whose a is not the
    most frequent a of their b group) when they are few anomalies, so a and
    b are one dimension broken by them, not a hierarchy; else None. 153
    rows (shifted CSV rows: a ward-name fragment in the district code
    column) turned a district code/name pair into "each LAD16CD has one
    LAD16NM". See _NEAR_ONE_TO_ONE and _RARE_GROUP_SHARE."""
    sizes = t.counts[a]
    to_b = np.full(len(sizes), -1, dtype=np.int64)
    to_b[t.codes[a]] = t.codes[b]  # null a lands in the last slot; null b is -1
    slot = to_b + 1
    # One modal a per b group (the first of a tie); every other a breaks b -> a.
    order = np.lexsort((-sizes, slot))
    first = np.ones(len(order), dtype=bool)
    first[1:] = slot[order][1:] != slot[order][:-1]
    minority = np.ones(len(sizes), dtype=bool)
    minority[order[first]] = False
    k = int(sizes[minority].sum())
    if not k or k > _NEAR_ONE_TO_ONE * t.n:
        return None
    rare = sizes <= max(1.0, _RARE_GROUP_SHARE * float(np.median(sizes[:-1])))
    return k if 2 * int(sizes[minority & rare].sum()) >= k else None


# ---------------------------------------------------------------------------
# 4.6 many-to-one
# ---------------------------------------------------------------------------
def _chance_log10(t: _Table, a, b) -> float:
    """log10 P(every multi-row group of a is pure in b | a, b independent):
    sum over a's groups g with >= 2 rows of log10(sum_v p_v^|g|), p_v = b's
    value shares (null a value). Terms are <= 0, so it stops once past the
    guard."""
    p = t.counts[b] / t.n
    p = p[p > 0]
    sizes, groups = np.unique(t.counts[a][t.counts[a] >= 2], return_counts=True)
    total = 0.0
    for size, count in zip(sizes, groups):
        total += int(count) * math.log10(max(float(np.sum(p ** int(size))), 1e-300))
        if total <= _CHANCE_LOG10:
            break
    return total


def _many_to_one(
    t: _Table, cols: Sequence, classes: Sequence[List[Any]]
) -> Tuple[List[list], List[Tuple[Any, Any, int]], int]:
    """``(facts, near, more)``: the "each a has one b" facts [[a, [bs]]]
    (at most _MAX_EDGES edges), the near one-to-one pairs (a, b, rows
    breaking b -> a) taken out of them, and the number of edges the cap cut
    (an absent fact must mean nothing to report: the cap hid 5 exact edges
    of a 14,412-row pharmacy table)."""
    n = t.n

    def source_ok(a) -> bool:
        if t.cardp[a] >= n or t.nulls[a] > 0.1 * n or t.is_fractional(a) or t.is_long(a):
            return False
        sizes = t.counts[a]
        multi = sizes >= 2
        return int(multi.sum()) >= 3 and int(sizes[multi].sum()) >= 0.5 * n

    def target_ok(b) -> bool:
        # A near-constant b is "determined" by anything (critics: 43 of 520
        # edges, e.g. a count column -> a flag that is 99.9% one value).
        top = int(t.counts[b][:-1].max()) if t.card[b] else 0
        return t.card[b] >= 2 and t.nulls[b] <= 0.1 * n and (t.card[b] >= 3 or top < 0.9 * n)

    rep = {c: cls[0] for cls in classes for c in cls}
    sources = [a for a in cols if source_ok(a)]
    targets = [b for b in cols if target_ok(b)]
    edges = set()
    for a in sources:
        for b in targets:
            if t.cardp[a] <= t.cardp[b] or rep.get(a, a) == rep.get(b, b):
                continue
            if t.determines(a, b) and _chance_log10(t, a, b) <= _CHANCE_LOG10:
                edges.add((rep.get(a, a), rep.get(b, b)))
    nodes = {c for e in edges for c in e}
    reduced = sorted(
        (
            (a, b)
            for a, b in edges
            if not any((a, c) in edges and (c, b) in edges for c in nodes if c not in (a, b))
        ),
        key=lambda e: (t.pos[e[0]], t.pos[e[1]]),
    )
    # The reduction runs before near pairs leave the edges, so "each w has
    # one a" stands for w -> b too when a and b are a near pair.
    near, kept = [], []
    for a, b in reduced:
        k = _near_one_to_one(t, a, b)
        if k is not None and _pair_ok(t, a, b) and _pair_support(t, a, b):
            near.append((a, b, k))
        else:
            kept.append((a, b))
    facts: List[list] = []
    for a, b in kept[:_MAX_EDGES]:
        if facts and facts[-1][0] == a:
            facts[-1][1].append(b)
        else:
            facts.append([a, [b]])
    return facts, near, max(0, len(kept) - _MAX_EDGES)


# ---------------------------------------------------------------------------
# 4.3 grain
# ---------------------------------------------------------------------------
def _grain(t: _Table, cols: Sequence, partitions: Sequence[Tuple[Any, Any]]) -> Optional[dict]:
    """The unique minimal combination of 2-3 dimension columns that is
    unique per row, as {"cols", "nested"} (see _nesting). Combinations whose
    uniqueness could be an accident (see _GRAIN_MIN_EXPECTED_COLLISIONS) are
    set aside BEFORE the one-minimal-key check: the accidental key
    (sezione, voti_validi), E = 1.9 (6.0 in another year's table), otherwise
    hid the grain (sezione, risposta), E = 225, of a 902-row referendum
    table. A single all-distinct column needs no fact: its distinct count
    already equals the row count."""
    n = t.n
    if n < 20:
        return None
    cand = [
        c
        for c in cols
        if t.kinds[c] in ("text", "bool", "datetime", "int")
        and 2 <= t.cardp[c] <= 0.99 * n
        and not t.is_fractional(c)
        and not t.is_long(c)
        # Number text with many values is a measure, not a dimension.
        and not (t.kinds[c] == "text" and t.is_measure(c, 51))
    ]
    # Columns splitting the rows identically are interchangeable in a key
    # (a code and its name): keep the first of each class.
    in_cand = set(cand)
    merged = {c for cls in _classes([p for p in partitions if set(p) <= in_cand], t.pos) for c in cls[1:]}
    reps = [c for c in cand if c not in merged]
    # P(two distinct rows share c's value); counts holds null in its last slot.
    collide = {c: float(np.sum(t.counts[c].astype(np.float64) * (t.counts[c] - 1))) / (n * (n - 1)) for c in reps}

    for size in (2, 3):
        combos = [
            combo
            for combo in combinations(reps, size)
            # At most one entity-like column (more distinct values than half
            # the rows): two of them are unique together by accident.
            if sum(t.cardp[c] > n / 2 for c in combo) <= 1
            and math.prod(t.cardp[c] for c in combo) >= n
            and math.comb(n, 2) * math.prod(collide[c] for c in combo) >= _GRAIN_MIN_EXPECTED_COLLISIONS
        ]
        keys = _unique_combos(t, combos)
        if len(keys) > 1:
            return None
        if keys:
            return _nesting(t, keys[0])
    return None


def _nesting(t: _Table, key: Sequence) -> dict:
    """The grain as {"cols": parents first, "nested": {part: [parents]}}.

    A whole-number part that restarts inside the groups of the other parts
    is numbered WITHIN them: it identifies nothing alone (polling station
    1..k per arrondissement: COUNT(DISTINCT num_bv) gave 95 sectors of 897;
    census block 1000.. per tract: 304 of 37,588). It restarts when its
    minimum is one value in most groups (see _NESTED_MIN_SHARE) while the
    groups differ in size; in a (near) full cross one size covers most
    groups and every part restarts (a year x month panel), so no size may
    cover half of them (station groups: 16 sizes over 20 arrondissements,
    largest share 10%; tract groups: largest share 8%). A calendar year is
    never numbered within: it is one axis shared by every group, and 40
    regions all reporting from 2010 but stopping at different years made
    "year numbered within region" (see _YEAR_AXIS)."""
    nested = []
    for b in key:
        if t.kinds[b] != "int" or b not in t.measures:
            continue
        x = t.measures[b]
        x = x[np.isfinite(x)]
        if len(x) and _YEAR_AXIS[0] <= x.min() and x.max() <= _YEAR_AXIS[1]:
            continue
        group, distinct = pd.factorize(t.key([c for c in key if c != b]))
        if np.bincount(np.bincount(group)).max() >= 0.5 * len(distinct):
            continue
        mins = pd.Series(t.measures[b]).groupby(group).min().value_counts()
        if len(mins) and mins.iloc[0] >= _NESTED_MIN_SHARE * len(distinct):
            nested.append(b)
    cols = [c for c in key if c not in nested] + nested
    return {"cols": cols, "nested": {b: [c for c in cols if c != b] for b in nested}}


def _unique_combos(t: _Table, combos: Sequence[tuple]) -> List[tuple]:
    """The combos unique per row, up to 2 (two already make the minimal key
    ambiguous). Above 200,000 rows the combos unique on the seeded sample are
    verified on all rows only when at most _GRAIN_MAX_VERIFY survive;
    otherwise the survivors are returned unverified (more than one: no
    grain)."""
    survivors = list(combos)
    if t.n > _PRESCREEN_ROWS:
        survivors = [c for c in survivors if t.is_key(c, _PRESCREEN_ROWS)]
    if t.n > _SCREEN_ROWS:
        screened = []
        for c in survivors:
            if t.is_key(c, _SCREEN_ROWS):
                screened.append(c)
                if len(screened) > _GRAIN_MAX_VERIFY:
                    return screened
        survivors = screened
    keys = []
    for c in survivors:
        if t.is_key(c):
            keys.append(c)
            if len(keys) == 2:
                break
    return keys


# ---------------------------------------------------------------------------
# 4.2 total rows
# ---------------------------------------------------------------------------
def _grand_total(t: _Table, cols: Sequence) -> Optional[Tuple[int, Any, Any]]:
    """A row whose value equals the sum of all other rows, as ``(row,
    measure, stored value)``: the value of a measure no other row shares, so
    excluding ``measure = value`` drops exactly that row.

    Per measure, only the largest |value| can be the total. A row is accepted
    when >= 2 measures put their total on it, or it is the only measure (with
    >= 5 other non-zero rows), or the row's label cells are all blank (with
    >= 2 other non-zero rows): one measure alone can equal the sum of 2
    others by chance."""
    n = t.n
    if n < 3 or n > _TOTAL_MAX_ROWS:
        return None
    meas = [c for c in cols if t.is_measure(c, 3)]
    hits: Dict[int, List[Tuple[Any, int]]] = {}
    for m in meas:
        x = t.measures[m]
        ok = np.flatnonzero(np.isfinite(x))
        if len(ok) < 3:
            continue
        v = x[ok]
        r = int(np.argmax(np.abs(v)))
        nonzero = int(np.count_nonzero(v))
        if v[r] == 0 or nonzero < 3:
            continue
        if abs(v[r] - (v.sum() - v[r])) <= _SUM_REL_TOL * abs(v[r]) + 1e-9:
            hits.setdefault(int(ok[r]), []).append((m, nonzero - 1))
    labels = [c for c in cols if t.kinds[c] in ("text", "bool", "datetime") and c not in t.measures]

    def labels_blank(r: int) -> bool:
        return bool(labels) and all(
            t.codes[c][r] == -1 or int(t.codes[c][r]) in t.blank_codes(c) for c in labels
        )

    accepted = [
        (-len(found), r)
        for r, found in hits.items()
        if len(found) >= 2
        or (len(meas) == 1 and found[0][1] >= 5)
        or (labels_blank(r) and found[0][1] >= 2)
    ]
    if not accepted:
        return None
    row = min(accepted)[1]
    for m, _ in hits[row]:
        code = int(t.codes[m][row])
        x = t.measures[m]
        if t.counts[m][code] == 1 and int(np.count_nonzero(x == x[row])) == 1:
            return row, m, _py(t.uniques[m][code])
    return None


def _total_label(t: _Table, cols: Sequence, row: int, rollups) -> Tuple[Any, Any]:
    """The first text column whose value at the total row sits on that row
    only (or only on detected total rows), so filtering it out drops the total
    and nothing else. A value every row shares ("2015-16" in a one-year table)
    or a number in a number-text column is no label; a word in a mostly
    numeric column ("Grand Total" among district numbers) is."""
    totals = np.zeros(t.n, dtype=bool)
    totals[row] = True
    for col, code, *_ in rollups:
        totals |= t.codes[col] == code
    for c in cols:
        if t.kinds[c] != "text":
            continue
        code = int(t.codes[c][row])
        if code == -1 or code in t.empty_codes[c] or not isinstance(t.uniques[c][code], str):
            continue
        if c in t.measures and np.isfinite(t.measures[c][row]):
            continue
        if t.counts[c][code] == 1 or totals[t.codes[c] == code].all():
            return c, _py(t.uniques[c][code])
    return None, None


def _loo_keys(t: _Table, dims: Sequence) -> List[Tuple[np.ndarray, int]]:
    """For each dim, the dense code of the combination of all OTHER dims,
    in O(k) factorizations (prefix and suffix products)."""

    def dense(key):
        codes, distinct = pd.factorize(key)
        return codes.astype(np.int64), len(distinct)

    def join(left, right):
        (a, ca), (b, cb) = left, right
        return dense(a * cb + b)

    one = (np.zeros(t.n, dtype=np.int64), 1)
    # Widened first: int32 codes times a combination count wrap silently.
    singles = [(t.codes[d].astype(np.int64) + 1, t.card[d] + 1) for d in dims]
    prefix = [one]
    for s in singles:
        prefix.append(join(prefix[-1], s))
    suffix = [one]
    for s in reversed(singles):
        suffix.append(join(s, suffix[-1]))
    suffix.reverse()
    return [join(prefix[i], suffix[i + 1]) for i in range(len(dims))]


def _rollup_totals(t: _Table, cols: Sequence, has_grand_total: bool) -> List[Tuple[Any, int, int, int, List[Any]]]:
    """Rollup rows (a cube's "All" rows): for a dimension H, group the rows by
    every OTHER dimension; when in >= 80% of the groups (>= 2 groups) the
    largest value equals the sum of the others, the argmax rows are totals and
    the H value >= 90% of them share is their marker.

    The marker is then tested on EVERY row carrying it (see
    _ROLLUP_MARKER_PASS): finding it on the argmax rows of the groups that
    fit says nothing about the marker rows alone in their group or in a group
    they do not sum, and "exclude the total rows" deleted whole indicators.
    Returns ``(H, marker code, rows that hold the sum, marker rows, group
    columns)`` (code -1 = a null marker), in column order.

    Dimensions are the non-measure text/bool/date columns, then (only if those
    find nothing) also the whole-number columns with few values, then (only
    if that finds nothing) those columns but one, which is the measure: a
    cube's count with <= 60 values joined the dimensions and left nothing to
    sum (All = M + F in 24 of 24 groups went unreported)."""
    n = t.n
    # Two groups of a total and two parts.
    if n < 6 or n > _TOTAL_MAX_ROWS:
        return []
    meas = [c for c in cols if t.is_measure(c, 3)]
    text_dims = [c for c in cols if t.kinds[c] in ("text", "bool", "datetime") and c not in t.measures]
    few, share = _ROLLUP_LOW_INT_VALUES
    low_ints = {c for c in cols if t.kinds[c] == "int" and c in t.measures and 2 <= t.cardp[c] <= max(few, share * n)}
    passes = [text_dims, [c for c in cols if c in low_ints or c in text_dims]]
    passes += [[c for c in cols if (c in low_ints and c != m) or c in text_dims] for m in meas if m in low_ints]
    tried = set()
    for dims in passes:
        dims = [d for d in dims if t.cardp[d] >= 2]
        values = [m for m in meas if m not in dims]
        # A pass equal to an earlier one (one low-int measure: its own pass
        # is the text pass) found nothing already.
        if not dims or not values or tuple(dims) in tried:
            continue
        tried.add(tuple(dims))
        V = np.column_stack([np.where(np.isfinite(t.measures[m]), t.measures[m], 0.0) for m in values])
        frame = pd.DataFrame(V)
        found = []
        for H, (g, cg) in zip(dims, _loo_keys(t, dims)):
            if t.cardp[H] > max(_ROLLUP_MAX_DIM_VALUES[0], _ROLLUP_MAX_DIM_VALUES[1] * n):
                continue
            size = np.bincount(g, minlength=cg)
            big = size >= _ROLLUP_MIN_GROUP
            if not big.any() or size[big].sum() < _ROLLUP_MIN_COVERED * n:
                continue
            grouped = frame.groupby(g, sort=True)
            S = grouped.sum().to_numpy()
            MX = grouped.max().to_numpy()
            NZ = (frame != 0).groupby(g, sort=True).sum().to_numpy()
            valid = big[:, None] & (NZ >= 2) & (MX > 0)
            fits = (np.abs(2 * MX - S) <= _SUM_REL_TOL * np.abs(MX) + 1e-6) & valid
            nvalid, nfit = valid.sum(axis=0), fits.sum(axis=0)
            # The measures that add up over H's groups: the marker rows are
            # tested on these only (a rate or a percentage never sums).
            additive = [
                j
                for j in range(len(values))
                if nvalid[j]
                and (
                    (nfit[j] / nvalid[j] >= _ROLLUP_FIT_SHARE and nfit[j] >= 2)
                    or (nfit[j] == nvalid[j] and len(dims) == 1)
                )
            ]
            # One dimension: the whole table is one group, so one measure's
            # fit is one coincidence (a sparse vote column's 2 = 1 + 1 called
            # a whole zone, 23 of 85 polling stations, "total rows"); two
            # measures must agree, as for the grand total.
            if not additive or (len(dims) == 1 and len(additive) < 2):
                continue
            best = max(additive, key=lambda j: nfit[j])
            argmax_rows = frame[best].groupby(g, sort=True).idxmax().to_numpy()[fits[:, best]]
            markers, rows = np.unique(t.codes[H][argmax_rows], return_counts=True)
            # One dimension only: the "group" is the whole table, which the
            # grand total already reports.
            if rows.max() < _ROLLUP_MARKER_SHARE * len(argmax_rows) or (len(dims) == 1 and has_grand_total):
                continue
            code = int(markers[np.argmax(rows)])
            passing, of = _marker_rows_holding_sums(t, g, cg, t.codes[H] == code, [values[j] for j in additive])
            if passing >= _ROLLUP_MARKER_PASS * of:
                found.append((H, code, passing, of, [d for d in dims if d != H]))
        if found:
            return found
    return []


def _marker_rows_holding_sums(
    t: _Table, g: np.ndarray, cg: int, marked: np.ndarray, additive: Sequence
) -> Tuple[int, int]:
    """``(rows, of)``: of the ``of`` marker rows, the ``rows`` whose group
    (rows equal in every other dimension, codes ``g``) has other rows and
    whose marker values add up to the sum of those other rows on some
    additive measure. A measure is evidence for a group only when every
    marker row and at least one other row hold a number there (a NYC
    indicator reported in Count leaves Dollars empty: 0 = 0 there proves
    nothing), while a group of zeros is a total of zeros. Summing the marker
    rows of a group keeps tables stacking two versions of a year (two "All"
    rows per group, 576 of 2,192 in a NYC HIV cube) from failing."""
    weight = marked.astype(np.float64)
    markers = np.bincount(g, weights=weight, minlength=cg)
    others = np.bincount(g, minlength=cg) - markers
    holds = np.zeros(cg, dtype=bool)
    for m in additive:
        x = t.measures[m]
        finite = np.isfinite(x)
        v = np.where(finite, x, 0.0)
        in_markers = np.bincount(g, weights=v * weight, minlength=cg)
        in_others = np.bincount(g, weights=v, minlength=cg) - in_markers
        numbers = np.bincount(g, weights=finite * weight, minlength=cg) == markers
        numbers &= np.bincount(g, weights=finite * (1.0 - weight), minlength=cg) > 0
        holds |= numbers & (np.abs(in_markers - in_others) <= _SUM_REL_TOL * np.abs(in_markers) + 1e-6)
    holds &= (markers > 0) & (others > 0)
    return int(markers[holds].sum()), int(marked.sum())


# ---------------------------------------------------------------------------
# 4.7 sum columns
# ---------------------------------------------------------------------------
def _sum_residual(total: np.ndarray, parts: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """|total - sum(parts)| on the complete rows, and the complete-row mask."""
    s = parts[0]
    for p in parts[1:]:
        s = s + p
    ok = np.isfinite(total) & np.isfinite(s)
    return np.abs(total[ok] - s[ok]), ok


def _sum_fit(total: np.ndarray, parts: Sequence[np.ndarray]) -> float:
    """Share of complete rows where total == sum(parts) within tolerance."""
    diff, ok = _sum_residual(total, parts)
    if len(diff) < 3:
        return 0.0
    return float((diff <= _SUM_REL_TOL * np.abs(total[ok]) + _SUM_ABS_TOL).mean())


def _sum_holds(total: np.ndarray, parts: Sequence[np.ndarray]) -> bool:
    diff, ok = _sum_residual(total, parts)
    complete = len(diff)
    if complete < max(3, 0.5 * len(total)):
        return False
    tv = total[ok]
    if (diff <= _SUM_REL_TOL * np.abs(tv) + _SUM_ABS_TOL).mean() < _SUM_MATCH:
        return False
    # Three non-zero totals at least: "Aug-12 = Jan-12 + Apr-13" held on the
    # 3 rows where all three month columns parse, 0 = 0 + 0, 32 = 19 + 13
    # and 42 = 23 + 19, a coincidence among the thousands of (total, parts)
    # tests of a 20-measure table.
    if np.count_nonzero(tv) < max(3, 0.1 * complete):
        return False
    # An identity that is exact (zero up to float64 rounding) on >= 98% of
    # the rows has no tolerance for a small part to ride in: votants =
    # exprimes + blancs + nuls held on 218 of 218 rows with blancs and nuls
    # under 1% of votants, and the weight rule below dropped it.
    exact = (diff <= 1e-9 * np.maximum(np.abs(tv), 1.0)).mean() >= _SUM_MATCH
    with np.errstate(all="ignore"):
        for p in parts:
            pv = p[ok]
            if np.count_nonzero(pv) < 0.1 * complete:
                return False
            # Every part must carry weight: a part 100x smaller than the
            # total "matches" inside the relative tolerance (an id or a year
            # plus a big number: "Date = SchoolYear + Enrolled").
            if not exact and np.nanmedian(np.abs(pv) / np.maximum(np.abs(tv), 1e-12)) < 0.01:
                return False
    return True


def _sum_columns(t: _Table, cols: Sequence) -> List[list]:
    """t = a + b (any two other measures) or t = a run of >= 3 consecutive
    measures, first match per total, all parts listed."""
    num = [c for c in cols if t.is_measure(c, 2)]
    if len(num) < 3:
        return []
    sample = t.rows(_SUM_SCREEN_ROWS) if t.n > _SUM_SCREEN_ROWS else None
    screen = {c: t.measures[c] if sample is None else t.measures[c][sample] for c in num}
    facts = []
    for total in num:
        others = [c for c in num if c != total]
        runs = [tuple(others[i:j]) for i in range(len(others)) for j in range(i + 3, len(others) + 1)]
        for parts in list(combinations(others, 2)) + runs:
            if _sum_fit(screen[total], [screen[p] for p in parts]) < _SUM_SCREEN_MATCH:
                continue
            if _sum_holds(t.measures[total], [t.measures[p] for p in parts]):
                facts.append([total, list(parts)])
                break
    return facts
