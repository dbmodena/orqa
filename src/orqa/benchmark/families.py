"""CKAN dataset families and the distinguishing facets that pick one member
of a family out from its siblings.

A *family* is a CKAN ``dataset_id``: the group of files (``resource_id``s)
published together under one dataset entry. 83% of UK gold tables belong to
a family with other files (median 38, max 133 — "Organogram of Staff Roles &
Salaries" alone has 1,426), so retrieval that only finds the FAMILY has not
yet found the FILE. :class:`FamilyIndex` maps resource ids to families and
back to the on-disk file stem; :func:`distinguishing_facets` finds the
smallest set of human-statable details (a period, a qualifying word, a
data-scope value) that a question needs in order to single the gold file out
from its siblings; :func:`missing_facets` checks whether a candidate question
actually states them.
"""

from __future__ import annotations

import calendar
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ..utils import dataset_id_to_resource_id
from .index import tokenize

# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


class FamilyIndex:
    """``dataset_id -> [resource_id]`` (and back), built from normalized
    metadata records plus a scan of the datasets folder for the on-disk
    file stem each resource id maps to.

    A record with no ``dataset_id`` is its own singleton family (keyed by
    its own resource id) — CKAN records always carry one, but this keeps the
    index total over any record shape a caller hands it (e.g. Socrata/ODS
    metadata, which has no dataset grouping at all).
    """

    def __init__(self, records: Iterable[dict], datasets_path: Path):
        self.datasets_path = Path(datasets_path)

        self._family_of_resource: dict[str, str] = {}
        self._members: dict[str, list[str]] = defaultdict(list)
        for record in records:
            resource_id = record.get("resource_id") or record.get("dataset_id")
            if not resource_id:
                continue
            family_id = record.get("dataset_id") or resource_id
            self._family_of_resource[resource_id] = family_id
            self._members[family_id].append(resource_id)

        self._stem_of_resource: dict[str, str] = {}
        self._resource_of_stem: dict[str, str] = {}
        if self.datasets_path.exists():
            for filepath in sorted(self.datasets_path.iterdir()):
                if not filepath.is_file():
                    continue
                stem = filepath.stem
                resource_id = dataset_id_to_resource_id(stem)
                self._stem_of_resource[resource_id] = stem
                self._resource_of_stem[stem] = resource_id

    def family_id(self, resource_id: str) -> str:
        return self._family_of_resource.get(resource_id, resource_id)

    def family_id_of_stem(self, stem: str) -> str:
        resource_id = self._resource_of_stem.get(stem) or dataset_id_to_resource_id(stem)
        return self.family_id(resource_id)

    def members(self, family_id: str) -> list[str]:
        """Every resource id of this family, including the query itself."""
        return list(self._members.get(family_id, [family_id]))

    def siblings(self, resource_id: str) -> list[str]:
        """The OTHER resource ids sharing ``resource_id``'s family."""
        family = self.family_id(resource_id)
        return [r for r in self._members.get(family, []) if r != resource_id]

    def family_size(self, resource_id: str) -> int:
        family = self.family_id(resource_id)
        return len(self._members.get(family, [resource_id]))

    def stem(self, resource_id: str) -> Optional[str]:
        """The on-disk file stem for this resource id, when a local file exists."""
        return self._stem_of_resource.get(resource_id)


# ---------------------------------------------------------------------------
# Temporal extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Period:
    """An inclusive calendar span, normalized so two differently-worded
    mentions of the SAME period ("Q1 2022" / "January to March 2022")
    compare equal. ``raw`` (the matched text) is display-only and excluded
    from equality/hash on purpose — it's what makes that cross-wording
    equality possible.
    """

    start: date
    end: date
    raw: str = field(default="", compare=False)

    @property
    def span_days(self) -> int:
        return (self.end - self.start).days

    def __str__(self) -> str:
        if self.start == self.end:
            return self.start.strftime("%-d %B %Y") if hasattr(self.start, "strftime") else str(self.start)
        if (self.start.month, self.start.day) == (1, 1) and (self.end.month, self.end.day) == (12, 31):
            if self.start.year == self.end.year:
                return str(self.start.year)
            return f"{self.start.year} to {self.end.year}"
        if self.start.day == 1 and self.end.day == calendar.monthrange(self.end.year, self.end.month)[1]:
            if self.start.year == self.end.year and self.start.month == self.end.month:
                return f"{calendar.month_name[self.start.month]} {self.start.year}"
        return f"{self.start.isoformat()} to {self.end.isoformat()}"


_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_SEP = r"(?:to|-|–|—)"

_ISO_DATE_RE = re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b")
_MONTH_RANGE_YEAR_RE = re.compile(
    rf"\b(?P<m1>{_MONTH_ALT})\.?\s+{_SEP}\s+(?P<m2>{_MONTH_ALT})\.?\s+(?P<y>\d{{4}})\b",
    re.IGNORECASE,
)
_DAY_MONTH_YEAR_RE = re.compile(
    rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{_MONTH_ALT})\.?,?\s+(?P<y>\d{{4}})\b",
    re.IGNORECASE,
)
_WEEK_RANGE_RE = re.compile(
    rf"\bweeks?\s+(?P<w1>\d{{1,2}})\s*{_SEP}\s*(?P<w2>\d{{1,2}})\s+(?P<y>\d{{4}})\b",
    re.IGNORECASE,
)
_QUARTER_RE = re.compile(r"\bQ(?P<q>[1-4])[\s-]+(?P<y>\d{4})\b", re.IGNORECASE)
_YEAR_RANGE_RE = re.compile(rf"\b(?P<y1>\d{{4}})\s*{_SEP}\s*(?P<y2>\d{{4}})\b")
_FISCAL_YEAR_RE = re.compile(r"\b(?P<y1>\d{4})[/-](?P<y2>\d{2})\b")
_MONTH_YEAR_RE = re.compile(
    rf"\b(?P<month>{_MONTH_ALT})\.?\s+(?P<y>\d{{4}})\b", re.IGNORECASE
)
_BARE_YEAR_RE = re.compile(r"\b(?P<y>(?:19|20)\d{2})\b")

# Tried in priority order (most specific / longest first) so a shorter
# submatch of an already-accepted span (e.g. "March 2022" inside "January to
# March 2022") never adds a spurious extra fact — see _extract below.
_PATTERNS: list[re.Pattern] = [
    _MONTH_RANGE_YEAR_RE,
    _DAY_MONTH_YEAR_RE,
    _ISO_DATE_RE,
    _WEEK_RANGE_RE,
    _QUARTER_RE,
    _YEAR_RANGE_RE,
    _FISCAL_YEAR_RE,
    _MONTH_YEAR_RE,
    _BARE_YEAR_RE,
]


def _month_end(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _period_from_match(pattern: re.Pattern, m: re.Match) -> Optional[Period]:
    raw = m.group(0)
    try:
        if pattern is _ISO_DATE_RE:
            y, mo, d = int(m["y"]), int(m["m"]), int(m["d"])
            day = date(y, mo, d)
            return Period(day, day, raw)
        if pattern is _DAY_MONTH_YEAR_RE:
            y = int(m["y"])
            mo = _MONTHS[m["month"].lower()]
            d = int(m["d"])
            day = date(y, mo, d)
            return Period(day, day, raw)
        if pattern is _MONTH_RANGE_YEAR_RE:
            y = int(m["y"])
            m1 = _MONTHS[m["m1"].lower()]
            m2 = _MONTHS[m["m2"].lower()]
            return Period(date(y, m1, 1), date(y, m2, _month_end(y, m2)), raw)
        if pattern is _MONTH_YEAR_RE:
            y = int(m["y"])
            mo = _MONTHS[m["month"].lower()]
            return Period(date(y, mo, 1), date(y, mo, _month_end(y, mo)), raw)
        if pattern is _WEEK_RANGE_RE:
            y = int(m["y"])
            w1, w2 = int(m["w1"]), int(m["w2"])
            if w1 > w2:
                w1, w2 = w2, w1
            start = date.fromisocalendar(y, w1, 1)
            end = date.fromisocalendar(y, w2, 7)
            return Period(start, end, raw)
        if pattern is _QUARTER_RE:
            y = int(m["y"])
            q = int(m["q"])
            m1 = (q - 1) * 3 + 1
            m2 = m1 + 2
            return Period(date(y, m1, 1), date(y, m2, _month_end(y, m2)), raw)
        if pattern is _YEAR_RANGE_RE:
            y1, y2 = int(m["y1"]), int(m["y2"])
            if y1 > y2:
                y1, y2 = y2, y1
            return Period(date(y1, 1, 1), date(y2, 12, 31), raw)
        if pattern is _FISCAL_YEAR_RE:
            y1 = int(m["y1"])
            y2_short = int(m["y2"])
            y2 = (y1 - y1 % 100) + y2_short
            if y2 <= y1:
                y2 += 100
            return Period(date(y1, 1, 1), date(y2, 12, 31), raw)
        if pattern is _BARE_YEAR_RE:
            y = int(m["y"])
            return Period(date(y, 1, 1), date(y, 12, 31), raw)
    except ValueError:
        # An out-of-range calendar value (Feb 30, week 60, ...): not a real
        # date, so it contributes no fact rather than raising.
        return None
    return None


def temporal_facts(text: str) -> set[Period]:
    """Extract every non-overlapping temporal mention from ``text``.

    Patterns are tried most-specific first, and a match overlapping an
    already-accepted (earlier-tried) span is discarded — so "31 December
    2021" is one day-level fact, not also a redundant "December 2021"
    month-level one, and "January to March 2022" is one range, not also a
    bare "2022".
    """
    consumed: list[tuple[int, int]] = []
    facts: set[Period] = set()
    for pattern in _PATTERNS:
        for m in pattern.finditer(text):
            start, end = m.span()
            if any(start < c_end and end > c_start for c_start, c_end in consumed):
                continue
            period = _period_from_match(pattern, m)
            if period is None:
                continue
            consumed.append((start, end))
            facts.add(period)
    return facts


def _generalizations(period: Period) -> list[Period]:
    """``period`` itself plus its containing month- and year-spans, coarsest
    first, deduplicated by span. Lets the facet search consider a fact at
    several granularities without re-parsing text.
    """
    y = period.start.year
    year_span = Period(date(y, 1, 1), date(y, 12, 31), str(y))
    candidates = [period, year_span]
    if period.start.month == period.end.month and period.start.year == period.end.year:
        mo = period.start.month
        candidates.append(
            Period(date(y, mo, 1), date(y, mo, _month_end(y, mo)), f"{calendar.month_name[mo]} {y}")
        )
    seen: set[tuple[date, date]] = set()
    ordered: list[Period] = []
    for candidate in sorted(candidates, key=lambda p: -p.span_days):
        key = (candidate.start, candidate.end)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)
    return ordered


# ---------------------------------------------------------------------------
# Distinguishing facets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Facet:
    """One detail a question must state to rule a sibling out.

    ``kind`` is ``"temporal"``, ``"word"``, or ``"scope"``. ``label`` is the
    human-readable rendering handed to the planner prompt.
    """

    kind: str
    label: str
    period: Optional[Period] = None
    word: Optional[str] = None
    column: Optional[str] = None
    value: Optional[str] = None


def _record_text(record: dict) -> str:
    return " ".join(
        str(record.get(key) or "")
        for key in ("resource_name", "resource_description", "temporal_coverage")
    )


def _singular(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es") and token[-3] in "sxz":
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _normalized_tokens(text: str) -> set[str]:
    return {_singular(t) for t in tokenize(text)}


_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "by", "with", "from", "as", "is", "are", "csv", "data", "dataset",
    "file", "files",
})


def _word_candidates(record: dict) -> list[str]:
    tokens = tokenize(f"{record.get('resource_name') or ''} {record.get('resource_description') or ''}")
    seen: list[str] = []
    seen_set: set[str] = set()
    for token in tokens:
        if token in _STOPWORDS or token.isdigit() or len(token) < 3:
            continue
        if token in seen_set:
            continue
        seen_set.add(token)
        seen.append(token)
    return seen


def _greedy_temporal_facets(
    gold: dict, siblings: dict[str, dict], remaining: set[str]
) -> tuple[list[Facet], set[str]]:
    gold_periods = temporal_facts(_record_text(gold))
    sibling_generalizations: dict[str, set[tuple[date, date]]] = {
        sid: {
            (g.start, g.end)
            for period in temporal_facts(_record_text(record))
            for g in _generalizations(period)
        }
        for sid, record in siblings.items()
    }

    candidates: dict[tuple[date, date], Period] = {}
    for period in gold_periods:
        for g in _generalizations(period):
            candidates.setdefault((g.start, g.end), g)
    # Coarsest first: the widest span that still distinguishes is preferred,
    # so the question only has to state as much precision as is necessary.
    ordered = sorted(candidates.values(), key=lambda p: -p.span_days)

    facets: list[Facet] = []
    for candidate in ordered:
        if not remaining:
            break
        span = (candidate.start, candidate.end)
        ruled_out = {
            sid for sid in remaining
            if span not in sibling_generalizations.get(sid, set())
        }
        if not ruled_out:
            continue
        facets.append(Facet(kind="temporal", label=f"period: {candidate}", period=candidate))
        remaining -= ruled_out
    return facets, remaining


def _greedy_word_facets(
    gold: dict, siblings: dict[str, dict], remaining: set[str]
) -> tuple[list[Facet], set[str]]:
    gold_words = _word_candidates(gold)
    gold_word_set = set(gold_words)
    sibling_tokens: dict[str, set[str]] = {
        sid: set(tokenize(f"{record.get('resource_name') or ''} {record.get('resource_description') or ''}"))
        for sid, record in siblings.items()
    }

    facets: list[Facet] = []
    available = list(gold_words)
    while remaining and available:
        best_word = None
        best_ruled_out: set[str] = set()
        for word in available:
            ruled_out = {sid for sid in remaining if word not in sibling_tokens.get(sid, set())}
            if len(ruled_out) > len(best_ruled_out):
                best_word, best_ruled_out = word, ruled_out
        if best_word is None or not best_ruled_out:
            break
        facets.append(Facet(kind="word", label=f"detail: {best_word}", word=best_word))
        remaining -= best_ruled_out
        available.remove(best_word)
    return facets, remaining


def _greedy_scope_facets(
    gold_scope: dict, sibling_scopes: dict[str, dict], remaining: set[str]
) -> tuple[list[Facet], set[str]]:
    facets: list[Facet] = []
    available = list(gold_scope.items())
    scoped = {sid for sid in remaining if sid in sibling_scopes}
    while scoped and available:
        best_col, best_val, best_ruled_out = None, None, set()
        for col, val in available:
            ruled_out = {
                sid for sid in scoped
                if sibling_scopes.get(sid, {}).get(col) != val
            }
            if len(ruled_out) > len(best_ruled_out):
                best_col, best_val, best_ruled_out = col, val, ruled_out
        if best_col is None or not best_ruled_out:
            break
        facets.append(Facet(kind="scope", label=f"{best_col}: {best_val}", column=best_col, value=str(best_val)))
        remaining -= best_ruled_out
        scoped -= best_ruled_out
        available.remove((best_col, best_val))
    return facets, remaining


def distinguishing_facets(
    gold: dict,
    siblings: dict[str, dict],
    gold_scope: Optional[dict] = None,
    sibling_scopes: Optional[dict[str, dict]] = None,
) -> dict:
    """The smallest set of details a question needs to single ``gold`` out
    from ``siblings`` (``{resource_id: metadata_record}``).

    Greedy set cover, tried in order: temporal facts, then word facets from
    the gold file's own name/description, then — only for whatever is still
    unresolved, and only when ``gold_scope``/``sibling_scopes`` (single-valued
    column facts, see ``table_analyzer.scope_values``) are supplied — data
    scope facets. A sibling still unresolved after all three is a *residual
    sibling*: no metadata or scope detail rules it out, so it stays an
    acceptable answer rather than an unmet requirement (see
    ``orqa.agent.utility.retrievability_gate``).

    Returns ``{"facets": [Facet, ...], "residual_siblings": [resource_id, ...]}``.
    """
    remaining = set(siblings)

    temporal, remaining = _greedy_temporal_facets(gold, siblings, remaining)
    word, remaining = _greedy_word_facets(gold, siblings, remaining)
    facets = temporal + word

    if remaining and gold_scope and sibling_scopes:
        scope, remaining = _greedy_scope_facets(gold_scope, sibling_scopes, remaining)
        facets += scope

    return {"facets": facets, "residual_siblings": sorted(remaining)}


def missing_facets(question: str, facets: Sequence[Facet]) -> list[Facet]:
    """Which of ``facets`` the question text fails to state.

    Word/scope facets: a tokenized match (with simple plural stripping) —
    every token of the facet's value must appear among the question's own
    tokens. Temporal facets: the question must state a period fully
    CONTAINED in the facet's span (an exact match, or something more
    specific — "30 September 2022" satisfies a "2022" facet; "2022" alone
    does not satisfy a "30 September 2022" facet).
    """
    if not facets:
        return []
    question_tokens = _normalized_tokens(question)
    question_periods = temporal_facts(question)

    missing: list[Facet] = []
    for facet in facets:
        if facet.kind == "temporal":
            covered = any(
                qp.start >= facet.period.start and qp.end <= facet.period.end
                for qp in question_periods
            )
            if not covered:
                missing.append(facet)
            continue
        value = facet.word if facet.word is not None else (facet.value or "")
        value_tokens = _normalized_tokens(value)
        if value_tokens and not value_tokens.issubset(question_tokens):
            missing.append(facet)
    return missing
