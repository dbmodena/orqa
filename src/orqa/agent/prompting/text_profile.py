"""Text-column facts for the column statistics block (SPEC v3, section 3).

``profile_text`` turns the frequency table of a text column (distinct raw
strings with their row counts) into the text-specific ``ColumnStat`` fields:
the domain (constant / complete list / most common), cleanliness (blank,
untrimmed, missing tokens, case/space variants), at most ONE kind (number
text, date text, date-like text, digit code, geometry, URLs, emails, free
text, time text, code shape) with the facts needed to parse it, and the
values a typed column's parse would turn into NULL or a code shape does not
fit.

Cost scales with DISTINCT values, never rows, and per-value Python work is
bounded: a column can hold millions of distinct strings (a UK GlobalID column
holds 3M UUIDs). Kinds are DETECTED on at most 4000 distinct values (the 2000
most frequent plus 2000 seeded random others, their row counts scaled to
estimate the whole), then CONFIRMED by a vectorised pass (pandas ``.str``)
over every distinct value, so every count, range and non-conforming value the
block states is exact, not an estimate.

Language-free by design: character classes, Unicode categories, digit runs
and arithmetic only; no word lists, no portal config. Month-name dates
("2020 Mar 16 04:55:46 PM") are typed through strptime directives (%b, %B,
%p): the parsing engine's names decide, never a list of ours. Values
stay exactly as stored (never stripped), so a quoted value in the block is a
real stored literal; all display formatting happens in
``prompts.render_column_statistics``. Everything returned is a built-in
Python type.
"""

import itertools
import json
import calendar
import re
import unicodedata

import numpy as np
import pandas as pd

# Kind detection classifies at most this many distinct values in Python
# (half the most frequent, half seeded random); ~1.5 µs each.
DETECT_CAP = 4000
# Bound for the per-value work that is not classification: case/space variant
# keys (NFKC + casefold, the most frequent values) and the free-text / pattern
# screening sample.
BOUND_CAP = 20000
# A kind must cover half the usable rows, both in the sample and, for the
# parsed kinds, in the full conformity pass: measured on 389 typed columns,
# 386 conform >= 50% and the 3 that do not (Valencia amounts mixing
# "19056,68" with "7,834,81") got a wrong format from a minority of rows.
KIND_MIN = 0.5
# 8-digit identifiers look like YYYYMMDD; the compact date reading needs a
# near-unanimous column.
COMPACT_MIN = 0.9
LIST_K = 20  # untyped and number text: list every value up to this many
# An untyped column of up to 40 values is still listed whole when the rendered
# list ("value" ×n, ...) fits 400 chars. Audit: Bologna eta (21 age bands,
# ~250 chars) and Valencia GRUPOS (21 bands, 303 chars) fell one value past
# the 20-value cut to most common + patterns, which hid the open-ended bands
# "014 e prima", "110 e oltre" and "? 90".
SHORT_LIST_K = 40
SHORT_LIST_CHARS = 400
TYPED_LIST_K = 5  # any other kind: list every value up to this many
# An all-distinct column is listed only when short: its values repeat the
# table sample, so a long list buys nothing.
DISTINCT_LIST_CHARS = 200
TOP_K = 5
SPIKE = 5.0  # a typed column's most common value needs >= 5x the mean rows per value
OTHER_K = 3
# Missing-token spellings listed by name; the rest are counted (rows and
# spellings). Audit: NYC STREET_HIGHWAY listed 8 spellings holding 0.02% of
# the rows, ~60 tokens.
MISSING_K = 4
ZERO_LED_MIN = 0.01  # >= 1% zero-led all-digit rows: a cast would destroy the zeros
WIDTH_MIN = 0.9
SHAPE_MIN = 0.9
SHAPE_CHARS = 40
PATTERN_MIN = 0.05
PATTERNS_K = 2
# A pattern is a family: at least 2 distinct (trimmed) member values. Audit:
# `# Month converted to SNA` ×144 (NYC) and `Barrio: # EL CARMEN` ×39
# (Valencia) were each ONE stored value, restating a most-common entry as a
# numbered family that does not exist. (Two members always hold more rows
# than the larger of them.)
PATTERN_MEMBERS = 2
FREE_TEXT_LEN = 40
FREE_TEXT_SPACES = 4
# Prose rarely repeats: at most 2 usable rows per distinct value. Audit: long
# labels passed the length/space gate and were called free text with no value
# listed: 58 Valencia error labels ×661 rows each, 571 Bologna intervention
# types (195 rows each on average), 25 school names ×8.5 rows.
FREE_TEXT_REPEAT = 2
TIME_K = 3
TIME_MIN_VALUES = 10
REPEAT_MIN_ROWS = 20
REPEAT_RATIO = 5.0
OFFSETS_K = 3  # ISO offsets are listed when the column holds at most this many
# A date column's most common value is a placeholder only when it covers 1%
# of the non-null rows: "12/01/2021 12:00:05 AM" ×21 out of 6M rows was ~60
# tokens of noise on every big NYC table.
DATE_TOP_MIN = 0.01
# Calendar coverage of a parsed date column needs 3 distinct dates over a
# span of at least 2 months.
COVERAGE_MIN_DATES = 3
COVERAGE_MIN_MONTHS = 2
# A decimal reading on fewer rows than max(5, 1% of the usable rows), and on
# under half the numbers, is a stray, not a format: it never makes the column
# "mixed" or "ambiguous", and its values are not numbers. Audit: one "9,4"
# among 2191 NYC budget values made the column "decimal dot and comma mixed",
# dropped its range and listed the standard "4,196.00" as not numbers.
MINORITY_ROWS = 5
MINORITY_SHARE = 0.01
# The renderer shows a value longer than CUT_LONG chars as its first CUT_KEEP
# (after rstrip) plus "…", matched with LIKE 'prefix%'. When that prefix is
# shared, the value gets a longer cut (at most CUT_MAX). Audit: Bologna note
# listed "ATTENZIONE: IN ALCUNI GIORNI (ES: FESTE PATRONALI, PREFESTI"… twice
# (×714 and ×51), and the prefix matched 642 stored values.
CUT_LONG = 60
CUT_KEEP = 59
CUT_MAX = 120
# A word after a space is a unit only on half the conforming rows. Audit:
# Paris house numbers took "A" (91 of 14000 rows, beside glued b/t and spaced
# P/Z suffixes) as a unit, so "230 A" cast to 230 while "18 P" became NULL.
UNIT_MIN = 0.5
# Number text at or above 1e15 is not a measure: float64 keeps every integer
# exact only below 2**53 (~9.0e15), and such values are glued lists or long
# ids. Audit (3 of 11,916 sampled text columns): NYC ZIPCode
# "11,362,113,641,136,711,..." read as thousands groups (max 1.1e49) and Paris
# rv_num_ref (max 2.5e35) crashed the renderer's bound formatting past 1e26;
# NYC CommunityBoard "305,310,311,313,315,318,410" gave a max that no row
# stores, below the true one. Those values are not numbers, and a 20-digit
# id column falls back to its code shape.
NUMBER_MAX = 1e15

# Unicode space separators (category Zs): the thousands "space" of French and
# Spanish exports is often NBSP (U+00A0) or narrow NBSP (U+202F).
_ZS = " \u00a0\u1680\u2000-\u200a\u202f\u205f\u3000"
_SPC = f"[{_ZS}]"

# ---- number text -------------------------------------------------------------
# One optional symbol before the digits, the signed digit body, and whatever
# follows; _number_affix decides whether the symbols make a measured quantity
# ("$1,234", "84.80 %", "12 kg") or something else ("<5", "250+", "E06000054").
# The gap is possessive: the suffix takes spaces too, so after a failed match
# handing it the gap's spaces one at a time cannot succeed, and it cost 0.24 s
# on a 10k-char value with a space run (quadratic, 1 s at 20k). Same matches
# and groups as before on 200k random strings.
_AFFIX = re.compile(
    rf"(?P<pre>[^0-9+\-.,{_ZS}]?){_SPC}*"
    rf"(?P<num>[+-]?(?:[0-9](?:[0-9.,{_ZS}]*[0-9])?|[.,][0-9]+))"
    rf"(?P<gap>{_SPC}*+)(?P<suf>[^0-9]*)"
)
_NUM_CLASSES = [(k, re.compile(p)) for k, p in [
    ("int", r"[+-]?[0-9]+"),
    ("thou_comma", r"[+-]?[0-9]{1,3}(?:,[0-9]{3}){2,}"),
    ("thou_dot", r"[+-]?[0-9]{1,3}(?:\.[0-9]{3}){2,}"),
    ("comma3", r"[+-]?[1-9][0-9]{0,2},[0-9]{3}"),
    ("point3", r"[+-]?[1-9][0-9]{0,2}\.[0-9]{3}"),
    ("point_dec_thou", r"[+-]?[0-9]{1,3}(?:,[0-9]{3})+\.[0-9]+"),
    ("comma_dec_thou", r"[+-]?[0-9]{1,3}(?:\.[0-9]{3})+,[0-9]+"),
    ("point_dec", r"[+-]?[0-9]*\.[0-9]+"),
    ("comma_dec", r"[+-]?[0-9]*,[0-9]+"),
    ("space_int", rf"[+-]?[0-9]{{1,3}}(?:{_SPC}[0-9]{{3}})+"),
    ("space_point", rf"[+-]?[0-9]{{1,3}}(?:{_SPC}[0-9]{{3}})+\.[0-9]+"),
    ("space_comma", rf"[+-]?[0-9]{{1,3}}(?:{_SPC}[0-9]{{3}})+,[0-9]+"),
]]
# Classes in which the comma reads as a decimal mark (or a lone 3-digit
# group): dropped when comma lists ("29,30,37") outweigh them.
_COMMA_DECIMAL = ("comma_dec", "comma_dec_thou", "space_comma", "comma3")
_COMMA_LIST = re.compile(r"[+-]?[0-9]+(?:,[0-9]+){2,}")
# The two readings that make a column "mixed": a comma decimal mark, against
# a dot decimal mark or comma thousands groups.
_COMMA_SIDE = ("comma_dec", "comma_dec_thou", "space_comma")
_POINT_SIDE = ("point_dec", "point_dec_thou", "space_point", "thou_comma")
# Digits followed by a space and a word (captured), whatever the word. The
# digit body ends on a non-space and the spaces after it are possessive, so a
# space run is read once: the body and the gap both taking spaces cost 0.45 s
# on a 10k-char value (quadratic, 1.8 s at 20k). Same matches as before.
_WORD_AFTER = rf"[^0-9]?{_SPC}*[+-]?[0-9](?:[0-9.,{_ZS}]*[0-9.,])?{_SPC}++([^\W\d_]+)"

# ---- date text ----------------------------------------------------------------
_TIME = (
    r"(?:(?P<tsep>[T ])[0-9]{1,2}:[0-9]{2}(?P<sec>:[0-9]{2})?(?P<frac>[.,][0-9]{1,9})?"
    r"(?P<ampm>\s?[AaPp]\.?[Mm]\.?)?)?"
)
_D_YMD = re.compile(
    r"(?P<y>[0-9]{4})(?P<s>[-/.])(?P<m>[0-9]{1,2})(?P=s)(?P<d>[0-9]{1,2})" + _TIME
    + r"(?P<tz>Z|[+-][0-9]{2}(?::?[0-9]{2})?)?"
)
_D_XY = re.compile(r"(?P<a>[0-9]{1,2})(?P<s>[-/.])(?P<b>[0-9]{1,2})(?P=s)(?P<y>[0-9]{4}|[0-9]{2})" + _TIME)
_D_COMPACT = r"(?:1[89]|20)[0-9]{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12][0-9]|3[01])"
_D_YM = re.compile(r"(?P<y>(?:1[89]|20)[0-9]{2})(?P<s>[-/.])(?P<m>0?[1-9]|1[0-2])")
_D_MY = re.compile(r"(?P<m>0?[1-9]|1[0-2])(?P<s>[-/.])(?P<y>(?:1[89]|20)[0-9]{2})")
# The ISO grammar both pandas (format="ISO8601") and DuckDB (TRY_CAST AS
# TIMESTAMP) parse: a dot fraction, and an offset only after the seconds and
# without a space (DuckDB returns NULL for "10:00Z" and "10:00:00 +00:00").
_ISO = (
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}(?:[T ][0-9]{1,2}:[0-9]{2}"
    r"(?::[0-9]{2}(?:\.[0-9]{1,9})?(?:Z|[+-][0-9]{2}(?::?[0-9]{2})?)?)?)?"
)
# The offset mask of the UTC recipes (SPEC v4 C1): a value whose clock ends
# in "Z" or "±hh[:mm]" is parsed with utc=True, the others naive, each
# subset in its own call, so the parse depends neither on row order nor on
# the machine's zone (pandas 2.3 lets a naive value inherit the offset of the
# value before it when both are in one call). The group captures the offset.
_ISO_OFFSET_MASK = re.compile(r"\d\d:\d\d(?::\d\d(?:\.\d+)?)?\s*(Z|[+-]\d\d(?::?\d\d)?)$")
# Month-name formats (SPEC v4 D1), detected through strptime directives: a
# sampled value is bucketed by its character shape (a digit -> 9, a letter
# run -> a letter for its length, a whitespace run -> one space) into the one candidate
# format of that shape, and counts as a date only when pandas parses it with
# that format, so the engine's month names decide and no value outside these
# shapes costs more than its shape. pandas to_datetime(format=) and DuckDB
# try_strptime accept the same strings: verified on "2020 Mar 16 04:55:46 PM"
# (%Y %b %d %I:%M:%S %p) and "May 26 2023  8:56AM" / "Jun  7 2018  9:58AM"
# (%b %d %Y %I:%M%p: a format space matches any whitespace run in both).
_MONTH_DATES = ("%b %d %Y", "%d %b %Y", "%Y %b %d", "%B %d %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y")
_MONTH_TIMES = ("", " %H:%M", " %H:%M:%S", " %I:%M %p", " %I:%M:%S %p", " %I:%M%p", " %I:%M:%S%p")
_MONTH_FORMATS = tuple(d + t for d in _MONTH_DATES for t in _MONTH_TIMES)
_MONTH_MAX_LEN = 40
_MONTH_GATE = re.compile(r"\w{1,9}\s")  # every candidate starts with a token and a space
_DIGIT, _LETTERS, _WS = re.compile(r"[0-9]"), re.compile(r"[^\W\d_]+"), re.compile(r"\s+")
_SHAPE_OF = {  # a letter run of n letters is the n-th letter: "Mar" -> c, "March" -> e, "AM" -> b
    "%Y": ("9999",), "%d": ("9", "99"), "%I": ("9", "99"), "%H": ("9", "99"), "%M": ("99",), "%S": ("99",),
    "%b": ("c",), "%B": tuple("defghi"), "%p": ("b",),
}
# A clock time with no date part ("16:56", "7:05:30"). Audit: UK Start/End
# Time (every value a valid HH:MM) were "date-like text, format not detected";
# DuckDB TRY_CAST AS DATE/TIMESTAMP gives NULL there, AS TIME works.
_D_TIME = re.compile(r"(?P<h>[0-9]{1,2}):(?P<m>[0-9]{2})(?::(?P<s>[0-9]{2}))?")
_DIRECTIVES = {
    "%Y": "[0-9]{4}", "%y": "[0-9]{2}", "%m": "[0-9]{1,2}", "%d": "[0-9]{1,2}",
    "%H": "[0-9]{1,2}", "%I": "[0-9]{1,2}", "%M": "[0-9]{2}", "%S": "[0-9]{2}",
    "%f": "[0-9]{1,9}", "%p": "[AaPp][Mm]", "%b": r"[^\W\d_]{3}", "%B": r"[^\W\d_]{3,9}",
}
# Character shapes (digit -> 9, letter run -> a): a clock group, or a
# day-month-year group with a 4-digit year, marks date-like text whose
# format was not detected ("2023 Jan 27 12:00:00 AM").
_CLOCK = re.compile(r"(?<!9)9{1,2}:99(?!9)")
_DATE_GROUP = re.compile(r"(?<!9)(?:9999(?P<a>[-/.])99?(?P=a)99?|99?(?P<b>[-/.])99?(?P=b)9999)(?!9)")
# A bare clock shape is a time of day (or a duration), never date-like text.
_TIME_SHAPE = re.compile(r"9{1,2}:99(?::99)?")
# The seconds part of a clock in a strptime format (with its fraction).
_SECONDS = re.compile(r":%S(?:[.,]%f)?")

# ---- other kinds ----------------------------------------------------------------
_WKT = re.compile(r"\s*(MULTI)?(POINT|LINESTRING|POLYGON|GEOMETRYCOLLECTION)\s*[ZM]{0,2}\s*\(", re.I)
# Only a geometry type makes GeoJSON: any JSON object has a "type" key
# ({"type": "museum"} made a 2-value category "geometry GeoJSON museum" and
# hid its values). Greedy, so a Feature reports its geometry's type and a
# properties "type" after it is skipped. On 8,883 brace-led Paris/NYC values
# the labels were unchanged (Polygon 4478, Point 3019, LineString 935, ...).
_GEOJSON = re.compile(r'\{.*"type"\s*:\s*"((?:Multi)?(?:Point|LineString|Polygon)|GeometryCollection)"', re.S)
# The separator is atomic: the next token never starts with a space, so its
# first match is the only one that can succeed; backtracking through
# `\s*[,; ]\s*` cost 0.7 s on a 10k-char value (quadratic, 2.8 s at 20k).
_COORD = re.compile(r"\(?\s*[-+]?[0-9]{1,3}\.[0-9]{3,}(?>\s*[,; ]\s*)[-+]?[0-9]{1,3}\.[0-9]{3,}\s*\)?")
_URL = re.compile(r"(?:https?|ftp)://\S+|www\.\S+", re.I)
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^\W\d_]{2,}")

_NO_TOP = ("geometry", "URLs", "emails")


def profile_text(freq: pd.Series, bad_tokens: set[str], rng_seed: int = 0) -> dict:
    """Text facts of one column from its frequency table.

    Args:
        freq: index = distinct raw str values (exactly as stored), values =
            row counts, ordered count desc with ties first-seen; NaN excluded.
        bad_tokens: configured missing-value tokens, stripped and lowercased.
        rng_seed: seed of the bounded samples (deterministic output).

    Returns:
        ColumnStat field values for the text-specific fields (constant,
        values, top_values, top_values_tied, uniform_count, usable_distinct,
        cut_hints, blank, untrimmed, untrimmed_merges, missing_tokens,
        missing_tokens_other_rows, missing_tokens_other {rows, spellings},
        missing_tokens_plausible, missing_token_values, variants, number,
        date, date_like_shape, digits, geometry, text_kind, text_length,
        shape, patterns, other, other_rows); keys with nothing to report are
        omitted (usable_distinct is always set). The date dict holds format
        (ISO, "ISO with UTC offset" or a strptime string), example, also,
        alt_format, years, min, max, min_rows, max_rows, time, unit,
        offset_rows, offsets and coverage; the parsed range is UTC, a value
        without an offset read as UTC.
    """
    out: dict = {}
    if freq.empty:
        return out
    raw = pd.Series(np.asarray(freq.index, dtype=object))
    cnt = freq.to_numpy(dtype=np.int64)
    st = raw.str.strip()
    blank = (st == "").to_numpy()
    missing = np.zeros(len(raw), dtype=bool)
    if bad_tokens:
        missing = st.str.lower().isin(bad_tokens).to_numpy() & ~blank
    usable = ~blank & ~missing
    ui = np.flatnonzero(usable)
    usable_rows = int(cnt[ui].sum())
    all_distinct = len(ui) > 1 and bool((cnt[ui] == 1).all())

    # The cardinality counts blank and token spellings too. Audit: 152 rows
    # of " " made 22 all-distinct Welsh names read "23 distinct".
    out["usable_distinct"] = len(ui)
    if blank.any():
        out["blank"] = int(cnt[blank].sum())

    # conform: the values that fit the kind (number, date, digits) or the
    # code shape; parsed: a number-text value's number (NaN elsewhere).
    kind, conform, flagged, parsed = None, None, None, None
    if len(ui):
        si, smult = _bounded(ui, DETECT_CAP, rng_seed)
        sample = st.to_numpy()[si].tolist()
        weights = cnt[si] * smult
        ev = _classify(sample, weights)
        for cand in _candidates(ev):
            if cand == "number":
                found = _confirm_number(st, cnt, usable, usable_rows, ev)
            elif cand == "digits":
                found = _confirm_digits(st, cnt, usable, usable_rows)
            elif cand == "date":
                found = _confirm_date(raw, st, cnt, usable, usable_rows, ev)
            else:  # geometry / URLs / emails: no parse, the sample decides
                kind = cand
                if cand == "geometry":
                    out["geometry"] = max(ev["geo"], key=ev["geo"].get)
                else:
                    out["text_kind"] = cand
                break
            if found:
                kind, (fact, conform, flagged, parsed) = cand, found
                out[cand] = fact
                break
        # Free text, date-like text and code shapes describe columns too
        # large to list: a listed column's values already show their form.
        listable = np.flatnonzero(~blank)
        if kind is None and len(ui) > LIST_K and not _lists_every_value(raw, cnt, listable, all_distinct, None, {}):
            kind, conform = _untyped_kind(out, st, cnt, ui, usable_rows, sample, weights, ev, rng_seed)

    # Padding is a whitespace slip only on a value that fits the column's
    # kind or shape; on one that does not, it is part of a structural defect
    # reported with the non-conforming values. Audit: the 53 "untrimmed" UK
    # ward codes were exactly the 53 column-shifted rows (' Marske and New
    # Marske"'), which a trim would have kept as codes.
    untrimmed = usable & (st.to_numpy() != raw.to_numpy())
    if conform is not None:
        untrimmed &= conform
    if untrimmed.any():
        out["untrimmed"] = int(cnt[untrimmed].sum())
        # Trimming merges a padded spelling into its stored trimmed twin, so
        # per-spelling counts undercount. Audit: Valencia DESCRIPCION, 354
        # padded spellings, 29484 of their 33812 rows have a stored twin
        # ("SEGURIDAD SOCIAL " and "SEGURIDAD SOCIAL").
        merges = int(st[untrimmed].isin(raw).sum())
        if merges:
            out["untrimmed_merges"] = merges

    _domain(out, raw, cnt, blank, missing, ui, usable_rows, all_distinct, kind, conform, flagged, parsed)
    if kind is None and len(ui) > LIST_K and "values" not in out:
        pats = _patterns(st, cnt, ui, usable_rows, rng_seed)
        if pats:
            out["patterns"] = pats
    if conform is not None:
        _not_conforming(out, raw, st, cnt, usable & ~conform)
    if kind != "geometry":  # NFKC of multi-kilobyte shapes buys nothing: no values are listed
        variants = _variants(raw, st, cnt, ui)
        if variants:
            if "values" in out:  # every spelling is listed already
                variants.pop("examples")
            out["variants"] = variants
    hints = _cut_hints(out, raw)
    if hints:
        out["cut_hints"] = hints
    return out


def parse_number_text(series: pd.Series, number: dict) -> pd.Series:
    """Parse a text column with its ``number`` fact (as returned by
    ``profile_text``) into float64, NaN where a value does not conform.

    Ambiguous and mixed separators give all NaN: the profile states no range
    for them because no single reading is right, and a measure built on a
    guess would mislead the table facts that sum it. Numbers stored as
    numbers in a mixed column keep their value."""
    out = np.full(len(series), np.nan)
    if not number or number.get("ambiguous") or number.get("mixed"):
        return pd.Series(out, index=series.index)
    codes, uniques = pd.factorize(series, use_na_sentinel=True)
    u = pd.Series(np.asarray(uniques, dtype=object))
    if pd.api.types.infer_dtype(u, skipna=True) == "string":
        isstr = np.ones(len(u), dtype=bool)
    else:
        isstr = np.array([isinstance(v, str) for v in u], dtype=bool)
    vals = np.full(len(u), np.nan)
    st = u[isstr].str.strip()
    ok = st.str.fullmatch(_number_regex(number)).to_numpy(dtype=bool)
    pos = np.flatnonzero(isstr)
    vals[pos[ok]] = _number_values(st[ok], number)
    for i in np.flatnonzero(~isstr):  # numbers stored as numbers (rare: mixed columns)
        v = u.iat[i]
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)):
            vals[i] = float(v)
    has = codes >= 0
    out[has] = vals[codes[has]]
    return pd.Series(out, index=series.index)


# ---------------------------------------------------------------------------
# Sampling and classification
# ---------------------------------------------------------------------------


def _bounded(idx: np.ndarray, cap: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """At most ``cap`` of ``idx`` (in frequency order): the most frequent
    half plus a seeded random half of the rest, with the multiplier that
    makes their row counts estimate the whole."""
    if len(idx) <= cap:
        return idx, np.ones(len(idx))
    head, rest = idx[: cap // 2], idx[cap // 2:]
    pick = np.sort(np.random.default_rng(seed).choice(len(rest), cap - cap // 2, replace=False))
    mult = np.concatenate([np.ones(len(head)), np.full(len(pick), len(rest) / len(pick))])
    return np.concatenate([head, rest[pick]]), mult


def _classify(values: list, weights: np.ndarray) -> dict:
    """Row-weighted evidence of the sampled (stripped, usable) values. A value
    counts for one kind only, in this order: date, number, geometry, URL,
    email; a compact YYYYMMDD value also counts as an integer, because the
    date reading only holds for a near-unanimous column."""
    ev = {
        "tot": float(weights.sum()), "date": {}, "xy": {}, "ym": {}, "compact": 0.0,
        "num": 0.0, "classes": {}, "affix": {}, "list": 0.0, "alldig": 0.0, "lead0": 0.0,
        "geo": {}, "URLs": 0.0, "emails": 0.0,
    }
    month_keys = _month_name_keys(values)
    for s, w in zip(values, weights):
        key = month_keys.get(s) or _date_key(s, ev)
        if key == ("C8",):
            ev["compact"] += w
        elif key:
            ev["date"][key] = ev["date"].get(key, 0.0) + w
            continue
        if _number_evidence(s, w, ev):
            continue
        g = _geometry(s)
        if g:
            ev["geo"][g] = ev["geo"].get(g, 0.0) + w
        elif _URL.fullmatch(s):
            ev["URLs"] += w
        elif _EMAIL.fullmatch(s):
            ev["emails"] += w
    _reject_span_labels(ev)
    return ev


def _date_key(s: str, ev: dict):
    """The date format family of one value, or None. X/Y/year values record
    their first and second fields (day/month order), year-month values their
    parts (span-label check)."""
    m = _D_YMD.fullmatch(s)
    if m:
        iso = (
            m["s"] == "-" and len(m["m"]) == 2 and len(m["d"]) == 2 and not m["ampm"]
            and (not m["frac"] or m["frac"][0] == ".") and (not m["tz"] or m["sec"])
        )
        if iso:
            return ("ISO",)
        if m["tz"]:  # an offset outside the ISO grammar: never rendered as %z
            return None
        return ("S", f"%Y{m['s']}%m{m['s']}%d" + _time_format(m))
    m = _D_XY.fullmatch(s)
    if m:
        key = ("XY", m["s"], "%Y" if len(m["y"]) == 4 else "%y", _time_format(m))
        a, b = ev["xy"].get(key, (0, 0))
        ev["xy"][key] = (max(a, int(m["a"])), max(b, int(m["b"])))
        return key
    if re.fullmatch(_D_COMPACT, s):
        return ("C8",)
    m = _D_TIME.fullmatch(s)
    if m and int(m["h"]) < 24 and int(m["m"]) < 60 and int(m["s"] or 0) < 60:
        return ("TIME", "%H:%M:%S" if m["s"] else "%H:%M")
    for rx, kind in ((_D_YM, "YM"), (_D_MY, "MY")):
        m = rx.fullmatch(s)
        if m and not (m["s"] == "." and len(m["m"]) == 1):  # "2019.5" is a decimal
            key = (kind, m["s"])
            if kind == "YM":
                ev["ym"].setdefault(key, set()).add((int(m["y"]), int(m["m"])))
            return key
    return None


def _time_format(m: re.Match) -> str:
    if not m["tsep"]:
        return ""
    f = m["tsep"] + ("%I" if m["ampm"] else "%H") + ":%M"
    if m["sec"]:
        f += ":%S"
    if m["frac"]:
        f += m["frac"][0] + "%f"
    if m["ampm"]:
        f += (" " if m["ampm"][0].isspace() else "") + "%p"
    return f


def _reject_span_labels(ev: dict) -> None:
    """"2010/11" is a school or fiscal year, not November 2010: a year-month
    reading whose every value has second part == (first part + 1) mod 100 is
    dropped (those values then count as nothing)."""
    for key, pairs in list(ev["ym"].items()):
        if all(m == (y + 1) % 100 for y, m in pairs):
            ev["date"].pop(key, None)


def _number_evidence(s: str, w: float, ev: dict) -> bool:
    m = _AFFIX.fullmatch(s)
    if not m:
        return False
    num = m["num"]
    cls = next((k for k, rx in _NUM_CLASSES if rx.fullmatch(num)), None)
    if cls is None:
        if _COMMA_LIST.fullmatch(num):
            ev["list"] += w
        return False
    affix = _number_affix(m["pre"], m["gap"], m["suf"])
    if affix is None:
        return False
    ev["num"] += w
    ev["classes"][cls] = ev["classes"].get(cls, 0.0) + w
    if affix != ("", "", ""):
        ev["affix"][affix] = ev["affix"].get(affix, 0.0) + w
    if s.isascii() and s.isdigit():
        ev["alldig"] += w
        if len(s) > 1 and s[0] == "0":
            ev["lead0"] += w
    return True


def _number_affix(pre: str, gap: str, suf: str):
    """(prefix, suffix, unit) when the symbols around the digits make a
    measured quantity: a currency sign (Unicode Sc) before or after, % or ‰
    after, a letter unit set off by a space ("12 kg"). Anything else -- a
    comparison or approximation sign ("<5", "≥30", "~4"), a trailing "+"
    ("250+"), letters glued to the digits ("E06000054", "6C") -- is not a
    plain number: a censored bound or a code must not be cast."""
    if pre and unicodedata.category(pre) != "Sc":
        return None
    if not suf:
        return (pre, "", "")
    if len(suf) == 1 and (suf in "%‰" or unicodedata.category(suf) == "Sc"):
        return (pre, suf, "")
    if gap and suf.isalpha():
        return (pre, "", suf)
    return None


def _geometry(s: str):
    m = _WKT.match(s)
    if m:
        return "WKT " + ((m.group(1) or "") + m.group(2)).upper()
    if s[:1] == "{":
        m = _GEOJSON.match(s)
        if m:
            return "GeoJSON " + m.group(1)
    if _COORD.fullmatch(s):
        return "coordinate pairs"
    return None


def _candidates(ev: dict) -> list:
    """Kinds to confirm, in order; the first that passes its full check wins.
    A kind needs half the sampled usable rows (the compact date 90%)."""
    tot = ev["tot"]
    if not tot:
        return []
    if ev["compact"] >= COMPACT_MIN * tot:
        ev["date"] = {("C8",): ev["compact"]}
        return ["date"]
    if sum(ev["date"].values()) >= KIND_MIN * tot:
        return ["date"]
    if ev["num"] >= KIND_MIN * tot:  # compact-date lookalikes are counted as integers here
        if ev["alldig"] and ev["lead0"] >= ZERO_LED_MIN * ev["alldig"]:
            return ["digits", "number"]
        return ["number"]
    if ev["geo"] and sum(ev["geo"].values()) >= KIND_MIN * tot:
        return ["geometry"]
    for kind in ("URLs", "emails"):
        if ev[kind] >= KIND_MIN * tot:
            return [kind]
    return []


# ---------------------------------------------------------------------------
# Number text
# ---------------------------------------------------------------------------


def _number_format(ev: dict) -> tuple[dict, bool]:
    """Separators from the row-weighted classes of the sample, plus whether a
    lone 3-digit group ("1,234") still needs the full-column integer check.
    The format follows the row majority: a stray reading (see MINORITY_ROWS)
    is dropped, so its values fail the format's grammar (not numbers)."""
    g = dict(ev["classes"])
    comma_dec = sum(g.get(k, 0.0) for k in _COMMA_SIDE)
    if ev["list"] > comma_dec:  # "29,30,37" outweighs "33,34": the comma separates list items
        g = {k: w for k, w in g.items() if k not in _COMMA_DECIMAL}
    floor = max(MINORITY_ROWS, MINORITY_SHARE * ev["tot"])

    def stray(rows: float) -> bool:
        return 0 < rows < floor and 2 * rows < ev["num"]

    comma_side = sum(g.get(k, 0.0) for k in _COMMA_SIDE)
    point_side = sum(g.get(k, 0.0) for k in _POINT_SIDE)
    if comma_side and point_side and stray(min(comma_side, point_side)):
        drop = _COMMA_SIDE if comma_side < point_side else _POINT_SIDE
        g = {k: w for k, w in g.items() if k not in drop}
    comma_dec = sum(g.get(k, 0.0) for k in _COMMA_SIDE)
    point_dec = g.get("point_dec", 0) + g.get("point_dec_thou", 0) + g.get("space_point", 0)
    thou_comma = g.get("thou_comma", 0) + g.get("point_dec_thou", 0)
    thou_dot = g.get("thou_dot", 0) + g.get("comma_dec_thou", 0)
    space = g.get("space_int", 0) + g.get("space_point", 0) + g.get("space_comma", 0)
    comma3, point3 = g.get("comma3", 0), g.get("point3", 0)
    fmt: dict = {}
    check = False
    if comma_dec and (point_dec or thou_comma):
        fmt["mixed"] = True
    elif comma_dec or (thou_dot and not thou_comma and not point_dec):
        fmt["decimal"] = ","
        th = "." if (thou_dot or point3) else "space" if space else None
        if th:
            fmt["thousands"] = th
    elif point_dec or thou_comma:
        th = "," if (thou_comma or comma3) else "space" if space else None
        if th:
            fmt["thousands"] = th
    elif comma3 and not stray(comma3):
        fmt["thousands"], check = ",", True
    elif point3 and not stray(point3):
        fmt["decimal"], fmt["thousands"], check = ",", ".", True
    elif space:
        fmt["thousands"] = "space"
    if ev["affix"]:
        pre, suf, unit = max(ev["affix"], key=ev["affix"].get)
        for k, v in (("prefix", pre), ("suffix", suf), ("unit", unit)):
            if v:
                fmt[k] = v
    return fmt, check


def _number_regex(number: dict) -> str:
    """Grammar of the values a number fact parses: optional prefix, signed
    digit body with the stated separators, optional suffix or unit. Bare
    numbers (no symbol) always conform."""
    sep = number.get("thousands") or number.get("ambiguous")
    th = {",": ",", ".": r"\.", "space": _SPC}.get(sep)
    whole = "[0-9]+" if th is None else rf"(?:[0-9]{{1,3}}(?:{th}[0-9]{{3}})+|[0-9]+)"
    if number.get("ambiguous"):
        core = whole
    elif number.get("mixed"):
        # Every reading that votes for "mixed" (_COMMA_SIDE, _POINT_SIDE)
        # conforms, grouped ones included: an ungrouped grammar listed the
        # "4,862.79" that made a column mixed as not numbers, and 600
        # "12,345.67" beside 400 "123,45" lost the number kind altogether.
        core = (
            r"(?:[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]{1,3}(?:\.[0-9]{3})+(?:,[0-9]+)?"
            rf"|[0-9]{{1,3}}(?:{_SPC}[0-9]{{3}})+(?:[.,][0-9]+)?|[0-9]+(?:[.,][0-9]+)?|[.,][0-9]+)"
        )
    else:
        dec = "," if number.get("decimal") == "," else r"\."
        core = rf"(?:{whole}(?:{dec}[0-9]+)?|{dec}[0-9]+)"
    rx = rf"(?P<n>[+-]?{core})"
    if number.get("prefix"):
        rx = rf"(?:{re.escape(number['prefix'])}{_SPC}*)?" + rx
    if number.get("suffix"):
        rx += rf"(?:{_SPC}*{re.escape(number['suffix'])})?"
    if number.get("unit"):
        rx += rf"(?:{_SPC}+{re.escape(number['unit'])})?"
    return rx


def _number_values(st: pd.Series, number: dict) -> np.ndarray:
    """float64 of stripped values that match ``_number_regex(number)``; NaN
    when not finite or not below NUMBER_MAX."""
    core = st
    if number.get("prefix") or number.get("suffix") or number.get("unit"):
        core = core.str.replace("^" + _number_regex(number) + r"\Z", r"\g<n>", regex=True)
    th = number.get("thousands")
    if th == "space":
        core = core.str.replace(_SPC, "", regex=True)
    elif th:
        core = core.str.replace(th, "", regex=False)
    if number.get("decimal") == ",":
        core = core.str.replace(",", ".", regex=False)
    v = pd.to_numeric(core, errors="coerce").to_numpy(dtype=float)
    v[~(np.abs(v) < NUMBER_MAX)] = np.nan  # also NaN and inf
    return v


def _confirm_number(st, cnt, usable, usable_rows, ev):
    """Full conformity pass of the chosen number format: exact range, and no
    number kind when under half the usable rows parse. Returns (fact,
    conforming mask, None, parsed numbers) or None."""
    number, check = _number_format(ev)
    ok = st.str.fullmatch(_number_regex(number)).to_numpy(dtype=bool) & usable
    if "unit" in number and not _unit_holds(st, cnt, usable, ok, number["unit"]):
        number = {k: x for k, x in number.items() if k != "unit"}
        ok = st.str.fullmatch(_number_regex(number)).to_numpy(dtype=bool) & usable
    if cnt[ok].sum() < KIND_MIN * usable_rows:
        return None
    if number.get("mixed"):
        return number, ok, None, None
    idx = np.flatnonzero(ok)
    v = _number_values(st.iloc[idx], number)
    good = ~np.isnan(v)
    ok[idx[~good]] = False
    if cnt[ok].sum() < KIND_MIN * usable_rows:
        return None
    idx, v = idx[good], v[good]
    if check:
        # A lone 3-digit group is a thousands separator only when every
        # integer written without a separator stays below 1000 (one would
        # carry it); with no such integer at all, nothing tells.
        bare = ~st.iloc[idx].str.contains("[.,]").to_numpy()
        if not bare.any() or (np.abs(v[bare]) >= 1000).any():
            sep = number["thousands"]
            number = {k: x for k, x in number.items() if k not in ("decimal", "thousands")}
            number["ambiguous"] = sep
            ok = st.str.fullmatch(_number_regex(number)).to_numpy(dtype=bool) & usable
            return (number, ok, None, None) if cnt[ok].sum() >= KIND_MIN * usable_rows else None
    parsed = np.full(len(st), np.nan)
    parsed[idx] = v
    shown = _split_bare(number, st.iloc[idx], cnt[idx], v)
    number["min"], number["max"] = float(shown.min()), float(shown.max())
    return number, ok, None, parsed


def _unit_holds(st, cnt, usable, ok, unit: str) -> bool:
    """A word after the digits is a unit only when it is on half the
    conforming rows and no non-conforming value puts another word after its
    digits: "2009 Fall" beside "2010 Spring" (NYC planting season, 41% of the
    rows) is a label, and so is "230 A" beside "18 P" (Paris house numbers)."""
    with_unit = ok & st.str.contains(rf"{_SPC}{re.escape(unit)}$").to_numpy(dtype=bool)
    if cnt[with_unit].sum() < UNIT_MIN * cnt[ok].sum():
        return False
    words = st[usable & ~ok].str.extract("^" + _WORD_AFTER + "$", expand=False).dropna()
    return not (words.str.casefold() != unit.casefold()).any()


def _split_bare(number: dict, st: pd.Series, counts: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The numbers the range states. When a prefix or suffix is on most
    conforming rows, the conforming values without it are another scale
    reported as number["bare"] {rows, min, max}, and the range covers the
    values that carry it. Audit: NYC Percentage held "85%" and the fraction
    0.855 for the same indicator, so "suffix %; range 0.072..89" merged two
    scales and a `> 50` filter treated 0.855 as 0.855%."""
    has = np.ones(len(st), dtype=bool)
    if "prefix" in number:
        has &= st.str.startswith(number["prefix"]).to_numpy(dtype=bool)
    if "suffix" in number:
        has &= st.str.endswith(number["suffix"]).to_numpy(dtype=bool)
    bare_rows = int(counts[~has].sum())
    if not bare_rows or bare_rows * 2 >= counts.sum():
        return v
    number["bare"] = {"rows": bare_rows, "min": float(v[~has].min()), "max": float(v[~has].max())}
    return v[has]


# ---------------------------------------------------------------------------
# Digit codes
# ---------------------------------------------------------------------------


def _confirm_digits(st, cnt, usable, usable_rows):
    ok = st.str.fullmatch("[0-9]+").to_numpy(dtype=bool) & usable
    if cnt[ok].sum() < KIND_MIN * usable_rows:
        return None
    idx = np.flatnonzero(ok)
    digits = st.iloc[idx]
    lens = digits.str.len().to_numpy()
    by_len = np.bincount(lens, weights=cnt[idx])
    fact: dict = {}
    if by_len.max() >= WIDTH_MIN * by_len.sum():
        fact["width"] = int(by_len.argmax())
    else:
        fact["widths"] = [int(lens.min()), int(lens.max())]
    fact["zero_led"] = int(cnt[idx[digits.str.match("0[0-9]").to_numpy(dtype=bool)]].sum())
    return fact, ok, None, None


# ---------------------------------------------------------------------------
# Date text
# ---------------------------------------------------------------------------


def _strptime_regex(fmt: str) -> str:
    """Grammar of the values a strptime format reads. A format space matches
    any whitespace run, as in pandas and DuckDB ("Jun  7 2018  9:58AM" fits
    "%b %d %Y %I:%M%p")."""
    return "".join(
        _DIRECTIVES.get(p) or "".join(r"\s+" if t.isspace() else re.escape(t) for t in re.findall(r"\s+|\S", p))
        for p in re.split(r"(%[a-zA-Z])", fmt)
    )


def _format_shapes(fmt: str) -> list:
    """Every character shape a month-name format can take (1- or 2-digit
    day and hour, 4- to 9-letter full month name)."""
    alts = []
    for p in re.split(r"(%[a-zA-Z])", fmt):
        if p in _SHAPE_OF:
            alts.append(_SHAPE_OF[p])
        elif p:
            alts.append((_WS.sub(" ", p),))
    return ["".join(c) for c in itertools.product(*alts)]


_MONTH_BY_SHAPE = {s: f for f in _MONTH_FORMATS for s in _format_shapes(f)}  # the shapes never clash


def _month_shape(s: str) -> str:
    return _WS.sub(" ", _LETTERS.sub(lambda m: chr(96 + min(len(m.group()), 26)), _DIGIT.sub("9", s)))


def _month_name_keys(values: list) -> dict:
    """value -> ("S", format) for the sampled values a month-name format
    parses: bucketed by shape, each bucket parsed once (vectorised). A %b
    bucket joins the %B bucket of the same skeleton when both parse: %B
    reads "May" too, so "March 5, 2020" beside "May 5, 2020" is one format."""
    buckets: dict = {}
    for s in values:
        if len(s) <= _MONTH_MAX_LEN and _MONTH_GATE.match(s):
            fmt = _MONTH_BY_SHAPE.get(_month_shape(s))
            if fmt:
                buckets.setdefault(fmt, []).append(s)
    parsed = {}
    for fmt, vals in buckets.items():
        ok = pd.to_datetime(pd.Series(vals, dtype=object), format=fmt, errors="coerce").notna().to_numpy()
        if ok.any():
            parsed[fmt] = [s for s, k in zip(vals, ok) if k]
    keys = {}
    for fmt, vals in parsed.items():
        full = fmt.replace("%b", "%B")
        target = full if "%b" in fmt and full in parsed else fmt
        for s in vals:
            keys[s] = ("S", target)
    return keys


def _date_family(key: tuple) -> tuple:
    """Formats one column writes interchangeably: a clock with or without
    its seconds part, a 2- or 4-digit year. A family is confirmed as one
    date kind. Audit: Valencia air quality Hora ("14:00" beside "14:00:00")
    listed 96360 of its 201480 rows as not times, and NYC Start Date listed
    "7/1/2006" as not dates beside "%m/%d/%y" values. DuckDB TRY_CAST AS TIME
    parses both clocks, and try_strptime both years when the %y format is
    listed first (%Y reads "4/5/14" as year 14)."""
    kind = key[0]
    if kind == "TIME":
        return ("TIME",)
    if kind == "S":
        return ("S", _SECONDS.sub("", key[1]))
    if kind == "XY":
        return ("XY", key[1], _SECONDS.sub("", key[3]))
    return key


def _date_descriptor(key: tuple, xy: tuple) -> dict:
    """Display format, alternative (unknown day/month order), grammar and
    pandas parse formats of a date format; ``xy`` holds the largest first
    and second fields of an X/Y/year family (its day/month order)."""
    kind = key[0]
    if kind == "ISO":
        return {"format": "ISO", "regex": _ISO, "day": True}
    if kind == "S":
        return {"format": key[1], "regex": _strptime_regex(key[1]), "parse": [key[1]], "day": True}
    if kind == "C8":
        return {"format": "%Y%m%d", "regex": _D_COMPACT, "parse": ["%Y%m%d"], "day": True}
    if kind == "TIME":
        return {"format": key[1], "regex": _strptime_regex(key[1]), "parse": [key[1]], "day": False, "time_only": True}
    if kind in ("YM", "MY"):
        s = key[1]
        month = "(?:0[1-9]|1[0-2])" if s == "." else "(?:0?[1-9]|1[0-2])"
        year = "(?:1[89]|20)[0-9]{2}"
        fmt, rx = (f"%Y{s}%m", year + re.escape(s) + month) if kind == "YM" else (f"%m{s}%Y", month + re.escape(s) + year)
        return {"format": fmt, "regex": rx, "parse": [fmt], "day": False}
    _, s, year, tfmt = key
    dm, md = f"%d{s}%m{s}{year}{tfmt}", f"%m{s}%d{s}{year}{tfmt}"
    a, b = xy
    desc = {"regex": _strptime_regex(dm), "day": True}
    if a > 12 >= b:
        desc.update(format=dm, parse=[dm])
    elif b > 12 >= a:
        desc.update(format=md, parse=[md])
    else:  # no field above 12, or both: the order is unknown
        desc.update(format=dm, alt_format=md, parse=[dm, md])
    return desc


def _confirm_date(raw, st, cnt, usable, usable_rows, ev):
    """Full conformity pass of the date format family with the most sampled
    rows: its most common format is the column's, its other formats conform
    too (parsed in turn) and are stated as date["also"] with their rows.
    Returns (fact, conforming mask, raw positions shown as repeated extremes,
    None) or None."""
    families: dict = {}
    for key, w in ev["date"].items():
        families.setdefault(_date_family(key), {})[key] = w
    members = max(families.values(), key=lambda f: sum(f.values()))
    keys = sorted(members, key=members.get, reverse=True)  # stable: ties keep sample order
    fields = [ev["xy"][k] for k in keys if k[0] == "XY"]
    xy = (max(a for a, _ in fields), max(b for _, b in fields)) if fields else (0, 0)
    descs = [_date_descriptor(k, xy) for k in keys]
    desc = descs[0]
    match = st.str.fullmatch("|".join(f"(?:{d['regex']})" for d in descs)).to_numpy(dtype=bool) & usable
    idx = np.flatnonzero(match)
    sub = st.iloc[idx]
    found = None
    if desc["format"] == "ISO":
        # Values with and without an offset are parsed apart, split by the
        # offset mask of the UTC recipes: pandas 2.3 (format="ISO8601",
        # utc=True) gives a naive value the offset of the offset value before
        # it, which put 12 local midnights at 22:00 (a false "time always
        # 22:00") and the max a day early. The range is UTC, a naive value
        # read as UTC, whatever the row order or the machine's zone.
        found = sub.str.extract(_ISO_OFFSET_MASK, expand=False)
        off = found.notna().to_numpy()
        parts = []
        if off.any():
            parts.append(pd.to_datetime(sub[off], format="ISO8601", utc=True, errors="coerce").dt.tz_localize(None))
        if (~off).any():
            parts.append(pd.to_datetime(sub[~off], format="ISO8601", errors="coerce"))
        parsed = pd.concat(parts).reindex(sub.index) if parts else pd.Series(pd.NaT, index=sub.index, dtype="datetime64[ns]")
    else:
        # pandas parses only the values written in the given format (a 4-digit
        # %Y never reads "14", %H:%M never "10:00:30"), so the order is free.
        # Each further format parses only the values still unparsed (the
        # parse is per value; re-parsing all 22.5M distinct timestamps of a
        # 113M-row table with the %B twin of a %b format cost 13 s).
        formats = [f for d in descs for f in d["parse"]]
        parsed = pd.to_datetime(sub, format=formats[0], errors="coerce")
        for alt in formats[1:]:
            missing = parsed.isna().to_numpy()
            if not missing.any():
                break
            parsed[missing] = pd.to_datetime(sub[missing], format=alt, errors="coerce")
    good = parsed.notna().to_numpy()
    idx, parsed = idx[good], parsed[good]
    need = COMPACT_MIN if keys[0] == ("C8",) else KIND_MIN
    if cnt[idx].sum() < need * usable_rows:
        return None
    ok = np.zeros(len(st), dtype=bool)
    ok[idx] = True
    fact = {"format": desc["format"]}
    if desc.get("time_only"):
        fact["time_only"] = True
    if found is not None:
        offset_rows, offsets = _iso_offsets(found[good], cnt[idx])
        if offset_rows:
            fact["format"] = "ISO with UTC offset"
            fact["offset_rows"] = offset_rows
            if offsets:
                fact["offsets"] = offsets
        # "ISO" alone hides whether a time part, a fraction or an offset is
        # stored; one real value shows it (a reader test wrote
        # format="%Y-%m-%d" for "2018-01-01T00:00:00.000" and parsed nothing).
        # A quoted value is a stored literal: the first one stored unpadded,
        # else the padded value as stored (the stripped " 2024-01-01T10:00:00"
        # was quoted, a literal no row holds).
        unpadded = (raw.iloc[idx] == st.iloc[idx]).to_numpy()
        fact["example"] = raw.iat[idx[int(unpadded.argmax())] if unpadded.any() else idx[0]]
    if "alt_format" in desc:
        fact["alt_format"] = desc["alt_format"]
    also = []
    for d in descs[1:]:
        rows = int(cnt[idx[sub.iloc[good].str.fullmatch(d["regex"]).to_numpy(dtype=bool)]].sum())
        if rows:
            also.append({"format": d["format"], "rows": rows})
    if also:
        fact["also"] = sorted(also, key=lambda a: -a["rows"])
    flagged = _date_facts(fact, desc, parsed, cnt[idx], idx)
    return fact, ok, flagged, None


def _iso_offsets(found: pd.Series, counts: np.ndarray) -> tuple:
    """(rows carrying a UTC offset, their distinct offsets) from the offsets
    the mask extracted ("Z" as "+00:00"), most rows first; the list is empty
    when there are more than 3. The generator keys its UTC recipe on the
    rows: an offset on every value, on some, or on none. Audit: Bologna
    transit counts were all "+00:00" with the range starting at 22:00/23:00
    UTC (local midnight); "with UTC offset" alone hid that every value is
    UTC, and grouping by UTC day made a phantom first day/month."""
    has = found.notna().to_numpy()
    if not has.any():
        return 0, []
    labels = {o: _offset_label(o) for o in found[has].unique()}
    rows = pd.Series(counts[has]).groupby(found[has].map(labels).to_numpy(), sort=False).sum()
    offsets = [] if len(rows) > OFFSETS_K else [str(o) for o in rows.sort_values(ascending=False, kind="stable").index]
    return int(counts[has].sum()), offsets


def _offset_label(offset: str) -> str:
    """"Z", "+01", "+0100", "+01:00" -> "+00:00" / "+01:00" style."""
    if offset == "Z":
        return "+00:00"
    digits = offset[1:].replace(":", "")
    return f"{offset[0]}{digits[:2]}:{digits[2:4] or '00'}"


def _date_facts(fact: dict, desc: dict, parsed: pd.Series, counts: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Range, repeated extremes, constant clock time, first/last-day-only
    and calendar coverage facts of the parsed values; returns the raw
    positions shown as repeated extremes (not repeated as most common
    values). A time of day has only its range (as stored: HH:MM or HH:MM:SS)
    and repeated extremes."""
    t = parsed.dt
    secs = (t.hour * 3600 + t.minute * 60 + t.second).to_numpy()
    clocks = np.unique(secs)
    time_varies = len(clocks) > 1
    lo, hi = parsed.min(), parsed.max()
    flagged = np.zeros(0, dtype=np.int64)
    if "alt_format" in fact:
        fact["years"] = [int(lo.year), int(hi.year)]
    else:
        if fact.get("time_only"):
            # As stored; a column holding both clocks shows the seconds when
            # one is not :00 (Hora "14:00" beside "14:00:00" keeps HH:MM).
            shown = "%H:%M:%S" if "also" in fact and (secs % 60).any() else fact["format"]
        else:
            shown = "%Y-%m-%d %H:%M" if time_varies else "%Y-%m-%d"
        fact["min"], fact["max"] = lo.strftime(shown), hi.strftime(shown)
        expect = counts.sum() / len(counts)
        for side, val in (("min", lo), ("max", hi)):
            hit = (parsed == val).to_numpy()
            rows = int(counts[hit].sum())
            if rows >= REPEAT_MIN_ROWS and rows >= REPEAT_RATIO * expect:
                fact[side + "_rows"] = rows
                flagged = np.concatenate([flagged, idx[hit]])
    if fact.get("time_only"):
        return flagged
    # A constant non-midnight clock (Valencia: 22:00 on every value) is a
    # local date stored as UTC; all-midnight says nothing beyond "dates".
    if len(counts) >= TIME_MIN_VALUES and len(clocks) <= TIME_K and clocks.any() and not (clocks % 60).any():
        fact["time"] = [f"{c // 3600:02d}:{c % 3600 // 60:02d}" for c in clocks]
    if desc["day"] and not time_varies and "alt_format" not in fact and len(counts) >= 3:
        if (t.day == 1).all():
            fact["unit"] = "month" if (t.month != 1).any() else "year"
        elif t.is_month_end.all() and (t.month != t.month.iloc[0]).any():
            # Audit: Valencia FECHA holds only month ends (68 months, 178027
            # rows) and had no granularity fact, so it read as daily data.
            fact["unit"] = "month_end"
    if "alt_format" not in fact and fact.get("unit") != "year":
        # Monthly data (first or last days of months, year-month formats)
        # is covered by months; a column of 1 Januaries says so already.
        coverage = date_coverage(parsed, months_only=not desc["day"] or "unit" in fact)
        if coverage:
            fact["coverage"] = coverage
    return flagged


def date_coverage(parsed: pd.Series, months_only: bool = False):
    """Calendar coverage of parsed dates (SPEC v4 D3): "every day present"
    when the distinct calendar days fill the span, else "every month present"
    or "k of n months present" (k distinct year-months, n months in the span);
    None under 3 distinct dates or a span under 2 months. ``months_only``
    skips the day reading (monthly data)."""
    days = parsed.dt.normalize()
    lo, hi = days.min(), days.max()
    if pd.isna(lo):
        return None
    # lo + COVERAGE_MIN_MONTHS months (day clamped to the month's end, as
    # pd.DateOffset), on Python dates: pandas overflows past 2262-04-11.
    year, month = divmod(lo.month - 1 + COVERAGE_MIN_MONTHS, 12)
    year, month = lo.year + year, month + 1
    if hi.date() < lo.date().replace(year=year, month=month, day=min(lo.day, calendar.monthrange(year, month)[1])):
        return None
    distinct_days = int(days.nunique())
    if distinct_days < COVERAGE_MIN_DATES:
        return None
    # Calendar dates, not a Timedelta: a span over 292 years (1700..2100
    # placeholders) overflows pandas' Timedelta.
    if not months_only and distinct_days == (hi.date() - lo.date()).days + 1:
        return "every day present"
    k = int((parsed.dt.year * 12 + parsed.dt.month).nunique())
    n = (hi.year - lo.year) * 12 + hi.month - lo.month + 1
    return "every month present" if k == n else f"{k} of {n} months present"


# ---------------------------------------------------------------------------
# Untyped columns: free text, date-like, code shape, patterns
# ---------------------------------------------------------------------------


def _char_shapes(s: pd.Series) -> pd.Series:
    """Character shapes of the first 40 characters: a digit -> 9, a run of
    letters -> a, anything else kept."""
    return s.str.slice(0, SHAPE_CHARS).str.replace(r"\d", "9", regex=True).str.replace(r"[^\W\d_]+", "a", regex=True)


def _shape_regex(shape: str) -> str:
    """Grammar of the first 40 characters of the values with this shape (a
    shape never holds two adjacent `a`, so the match is exact). One fullmatch
    costs 0.56 s on 3M distinct codes; recomputing their shapes, 2.5 s."""
    return "".join(r"\d" if c == "9" else r"[^\W\d_]+" if c == "a" else re.escape(c) for c in shape)


def _untyped_kind(out, st, cnt, ui, usable_rows, sample, weights, ev, seed):
    """Free text, else date-like text, else a code shape. Returns (kind,
    conform): conform marks the values that fit a code shape (None for the
    other kinds); (None, None) when none fits."""
    bi, bm = _bounded(ui, BOUND_CAP, seed)
    sb = st.iloc[bi]
    lens = sb.str.len().to_numpy()
    w = cnt[bi] * bm
    if (
        usable_rows <= FREE_TEXT_REPEAT * len(ui)
        and (lens * w).sum() >= FREE_TEXT_LEN * w.sum()
        and (sb.str.count(" ").to_numpy() * w).sum() >= FREE_TEXT_SPACES * w.sum()
    ):
        # Weighted by rows like every other number in the block (the median
        # over distinct values said 93 chars where the row median is 30).
        order = np.argsort(lens, kind="stable")
        half = np.searchsorted(np.cumsum(w[order]), w.sum() / 2)
        out["text_kind"], out["text_length"] = "free text", int(lens[order][half])
        return "free text", None
    tot = ev["tot"]
    shapes = pd.Series(weights).groupby(_char_shapes(pd.Series(sample, dtype=object)).to_numpy(), sort=False).sum()
    datey = np.array(
        [bool((_CLOCK.search(s) or _DATE_GROUP.search(s)) and not _TIME_SHAPE.fullmatch(s)) for s in shapes.index],
        dtype=bool,
    )
    if shapes[datey].sum() >= KIND_MIN * tot:
        out["date_like_shape"] = str(shapes[datey].idxmax())
        return "date-like", None
    if sum(ev["date"].values()) < KIND_MIN * tot:
        top = shapes.idxmax()
        if shapes[top] >= SHAPE_MIN * tot and "9" in top and not _CLOCK.search(top):
            # Checked over every usable value, so the values outside the
            # shape are reported exactly. Audit: they were the rows that
            # mattered: the open age bands "014 e prima" / "110 e oltre", the
            # latest month stored as "MAY 2026" in an `a-99` month column,
            # and 53 column-shifted UK rows.
            conform = np.zeros(len(st), dtype=bool)
            conform[ui] = st.iloc[ui].str.slice(0, SHAPE_CHARS).str.fullmatch(_shape_regex(top)).to_numpy(dtype=bool)
            if cnt[conform].sum() >= SHAPE_MIN * usable_rows:
                out["shape"] = str(top)
                return "shape", conform
    return None, None


def _collapse(s: pd.Series) -> pd.Series:
    """Digit runs -> '#', first 40 characters (the backticked shapes)."""
    return s.str.replace("[0-9]+", "#", regex=True).str.slice(0, SHAPE_CHARS)


def _patterns(st, cnt, ui, usable_rows, seed) -> list:
    """At most 2 digit-run shapes covering >= 5% of usable rows over at least
    2 distinct trimmed values. Screened on the bounded sample at half the
    threshold, then counted exactly over every usable value only when a
    shape qualifies: collapsing 3M distinct UUIDs costs ~2 s and never
    yields a pattern."""

    def grouped(pos: np.ndarray, rows: np.ndarray) -> pd.DataFrame:
        values = st.iloc[pos]
        g = pd.DataFrame({"shape": _collapse(values).to_numpy(), "value": values.to_numpy(), "rows": rows})
        return g.groupby("shape", sort=False).agg(rows=("rows", "sum"), members=("value", "nunique"))

    bi, bm = _bounded(ui, BOUND_CAP, seed)
    g = grouped(bi, cnt[bi] * bm)
    if len(bi) < len(ui):
        if not ((g["rows"] >= PATTERN_MIN / 2 * usable_rows) & g.index.str.contains("#")).any():
            return []
        g = grouped(ui, cnt[ui])
    g = g[(g["rows"] >= PATTERN_MIN * usable_rows) & g.index.str.contains("#") & (g["members"] >= PATTERN_MEMBERS)]
    g = g.iloc[np.argsort(-g["rows"].to_numpy(), kind="stable")]
    return [(str(s), int(n)) for s, n in g["rows"].head(PATTERNS_K).items()]


# ---------------------------------------------------------------------------
# Domain, missing tokens, non-conforming values, variants
# ---------------------------------------------------------------------------


def _lists_every_value(raw, cnt, listable, all_distinct, kind, facts: dict) -> bool:
    """Whether the domain is the complete list of the listable (non-blank)
    values, missing tokens included; ``facts`` holds the kind facts found."""
    k = len(listable)
    if kind == "geometry":
        return False
    number = facts.get("number")
    if number and ("ambiguous" in number or "mixed" in number):
        # No range is stated, so a short list is the only view of the values.
        # Audit: NYC FEAT_CODE (10 codes "4,900".."4,985") and SUB_CODE (19)
        # showed no value at all, and the planner could not filter a code.
        return k <= LIST_K
    if all_distinct:
        # Only categorical text: a short all-distinct list of names gives
        # exact literals, while dates/numbers/codes are told by their range
        # or shape (a 12-month column listed AND given a range said
        # everything twice).
        return kind is None and k <= LIST_K and sum(len(raw.iat[i]) for i in listable) <= DISTINCT_LIST_CHARS
    if kind is None:
        if k <= LIST_K:
            return True
        if k > SHORT_LIST_K:
            return False
        items = (len(json.dumps(raw.iat[i], ensure_ascii=False)) + len(f" ×{cnt[i]}") for i in listable)
        return sum(items) + len(", ") * (k - 1) <= SHORT_LIST_CHARS
    if kind == "number" or facts.get("date", {}).get("time_only"):
        # A short number-text column is mostly a code ("4,900", "2007"), a
        # short time column a set of clock times (NYC opening hours: 13
        # values, "05:00" x233 closing): the stored literals are what a
        # filter needs, next to the format.
        return k <= LIST_K
    return k <= TYPED_LIST_K


def _domain(out, raw, cnt, blank, missing, ui, usable_rows, all_distinct, kind, conform, flagged, parsed) -> None:
    """Constant / complete list / most common, and the missing tokens.

    Configured missing tokens count as values: inside a complete list they
    stay at their count (named in missing_token_values), because a token can
    be a real category (NYC BOO_CORRECTION: "-" x3096717 and "+" x73044 are
    a sign flag); only outside a complete list are they a fact of their own."""
    listable = np.flatnonzero(~blank)
    k = len(listable)
    complete = k == 1
    if complete:
        out["constant"] = raw.iat[listable[0]]
    elif k and _lists_every_value(raw, cnt, listable, all_distinct, kind, out):
        out["values"] = [(raw.iat[i], int(cnt[i])) for i in listable]
        complete = True
    if complete:
        tokens = [raw.iat[i] for i in listable if missing[i]]
        if tokens:
            out["missing_token_values"] = tokens
        return
    mi = np.flatnonzero(missing)
    if len(mi):
        out["missing_tokens"] = [(raw.iat[i], int(cnt[i])) for i in mi[:MISSING_K]]
        if len(mi) > MISSING_K:
            rest = mi[MISSING_K:]
            out["missing_tokens_other_rows"] = int(cnt[rest].sum())
            out["missing_tokens_other"] = {"rows": int(cnt[rest].sum()), "spellings": int(len(rest))}
        plausible = _plausible_tokens([raw.iat[i] for i in mi[:MISSING_K]], raw, cnt, ui)
        if plausible:
            out["missing_tokens_plausible"] = plausible
    if kind in _NO_TOP or all_distinct or not len(ui):
        return
    if cnt[0] > 1 and (cnt == cnt[0]).all():
        # No value is more common than another: the domain says "each ×c".
        # Audit: Paris arrondissement (20 values ×13) listed 5 of them as
        # "most common".
        out["uniform_count"] = int(cnt[0])
        return
    least = 2.0
    if kind:  # typed: only spikes
        least = max(least, SPIKE * usable_rows / len(ui))
    if kind == "date":  # a placeholder covers rows; "12/01/2021 12:00:05 AM" ×21 of 6M is noise
        least = max(least, DATE_TOP_MIN * cnt.sum())
    keep = np.ones(len(ui), dtype=bool)
    if conform is not None:
        keep &= conform[ui]
    if flagged is not None and len(flagged):
        keep &= ~np.isin(ui, flagged)
    pos = ui[keep]
    if parsed is not None:
        # Number text: one number stored in several spellings is one value,
        # stored as the number. Audit: Valencia CRED. INICIALES showed "0"
        # ×50391 where 0 is on 51115 rows ("0" + "0,00"), and ranked 100
        # above 10000 (1040 vs 1136 rows once merged).
        g = pd.DataFrame({"v": parsed[pos], "rows": cnt[pos], "pos": pos})
        g = g.groupby("v", sort=False).agg(rows=("rows", "sum"), pos=("pos", "min"))
        g = g.iloc[np.lexsort((g["pos"].to_numpy(), -g["rows"].to_numpy()))]
        values, rows = [float(v) for v in g.index], g["rows"].to_numpy()
    else:
        values, rows = [raw.iat[i] for i in pos[:TOP_K]], cnt[pos]
    rows = rows[rows >= least]
    n, tied = _untied_head(rows)
    if n:
        out["top_values"] = [(values[i], int(rows[i])) for i in range(n)]
    if tied:
        out["top_values_tied"] = tied


def _plausible_tokens(spellings: list, raw, cnt, ui) -> list:
    """Token spellings that look like the column's own values: letters only,
    Title case ("Nan", "None"), in a column whose usable rows are mostly Title
    case. Audit: NYC FIRST "Nan" ×1 is a first name among "Ryan", "Evan".
    Upper and lower case are no evidence: an all-caps column spells its
    markers in caps too (measured on 924 letters-only tokens in NYC/UK/ODS
    columns: 239 upper-case matches, nearly all markers such as "NULL" ×9209
    in all-caps address lines)."""
    titled = [s for s in spellings if s.strip().isalpha() and s.strip().istitle() and not s.strip().isupper()]
    if not titled:
        return []
    usable = raw.iloc[ui].str.strip()
    title_rows = cnt[ui][(usable.str.istitle() & ~usable.str.isupper()).to_numpy(dtype=bool)].sum()
    return titled if 2 * title_rows >= cnt[ui].sum() else []


def _cut_hints(out: dict, raw: pd.Series) -> dict:
    """raw value -> shown length, for every long value the block shows whose
    cut prefix (first 59 chars after rstrip) another stored value shares:
    the shortest length at which no other stored value starts with it, at
    most 120 (a value that is itself the start of another is shown whole)."""
    shown = [out["constant"]] if isinstance(out.get("constant"), str) else []
    for key in ("values", "top_values", "missing_tokens"):
        shown += [v for v, _ in out.get(key, []) if isinstance(v, str)]
    shown += [g.get("value", g.get("example")) for g in out.get("other", [])]
    shown += (out.get("variants") or {}).get("examples", [])
    long = [v for v in dict.fromkeys(shown) if len(v) > CUT_LONG and len(v.rstrip()) > CUT_KEEP]
    if not long:
        return {}
    heads = raw.str.slice(0, CUT_KEEP).to_numpy()
    hints = {}
    for v in long:
        whole = v.rstrip()
        rivals = raw[(heads == whole[:CUT_KEEP]) & (raw != v).to_numpy()]
        if rivals.empty:
            continue
        # The prefix grows unique monotonically: bisect the shortest length.
        lo, hi = CUT_KEEP + 1, min(len(whole), CUT_MAX)
        while lo < hi:
            mid = (lo + hi) // 2
            if rivals.str.startswith(whole[:mid]).any():
                lo = mid + 1
            else:
                hi = mid
        hints[v] = lo
    return hints


def _untied_head(rows: np.ndarray) -> tuple[int, dict]:
    """How many of the ranked counts (desc) to list, at most 5, never
    stopping inside a tie: when the last listed count equals the next one,
    every listed value with that count is dropped and the tie is reported as
    {"values": all values with that count, "rows": the count}. Audit: Bologna
    candidato showed 4 of the 676 candidates ×452 as "most common"; Valencia
    GRUPOS 4 of 20 age bands ×19, picked by file order."""
    n = min(TOP_K, len(rows))
    if n == len(rows) or rows[n - 1] != rows[n]:
        return n, {}
    c = rows[n - 1]
    return int(np.argmax(rows == c)), {"values": int((rows == c).sum()), "rows": int(c)}


def _not_conforming(out, raw, st, cnt, nc_mask) -> None:
    """Usable values a typed column's parse turns into NULL, or that its code
    shape does not fit: at most 3 groups by rows; a one-value group shows the
    value, else its digit-run shape and the most frequent raw example."""
    nc = np.flatnonzero(nc_mask)
    if not len(nc):
        return
    out["other_rows"] = int(cnt[nc].sum())
    g = pd.DataFrame({"shape": _collapse(st.iloc[nc]).to_numpy(), "rows": cnt[nc], "pos": nc})
    g = g.groupby("shape", sort=False).agg(rows=("rows", "sum"), n=("rows", "size"), pos=("pos", "min"))
    g = g.iloc[np.lexsort((g["pos"].to_numpy(), -g["rows"].to_numpy()))].head(OTHER_K)
    out["other"] = [
        {"value": raw.iat[pos], "rows": int(rows)} if n == 1
        else {"shape": str(shape), "rows": int(rows), "example": raw.iat[pos]}
        for shape, rows, n, pos in zip(g.index, g["rows"], g["n"], g["pos"])
    ]


def _variants(raw, st, cnt, ui) -> dict:
    """One value spelled several ways (case, Unicode compatibility form,
    inner whitespace) among the most frequent usable values. Padding is not a
    variant (``untrimmed`` reports it); accents are kept (stripping them can
    merge different words). Examples are raw stored spellings of the
    largest group, most rows first."""
    head = ui[:BOUND_CAP]
    spell = st.to_numpy()[head]
    keys = [" ".join(unicodedata.normalize("NFKC", s).casefold().split()) for s in spell]
    g = pd.DataFrame({"key": keys, "spell": spell, "rows": cnt[head], "pos": head})
    g = g.groupby(["key", "spell"], sort=False).agg(rows=("rows", "sum"), pos=("pos", "min")).reset_index()
    g = g[g["key"].duplicated(keep=False)]
    if g.empty:
        return {}
    per_key = g.groupby("key", sort=False)["rows"].agg(["sum", "max"])
    top_key = per_key["sum"].idxmax()
    top = g[g["key"] == top_key].sort_values(["rows", "pos"], ascending=[False, True], kind="stable")
    return {
        "groups": int(len(per_key)),
        "rows": int((per_key["sum"] - per_key["max"]).sum()),
        "examples": [raw.iat[i] for i in top["pos"].head(3)],
    }
