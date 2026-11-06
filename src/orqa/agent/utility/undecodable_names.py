"""Column names that hold U+FFFD, the Unicode replacement character.

Some published CSVs are not UTF-8: a Windows-1252 "£" is the single byte
0xA3. Where such a file was turned into parquet with a lossy UTF-8 decode,
every undecodable byte became U+FFFD. In the UK snapshot (2026-11-06) that is
1,037 headers in 584 of 15,205 tables, nearly all money columns: "Total £" is
stored as "Total \\ufffd", "Value (£)" as "Value (\\ufffd)". The original byte
is gone, and the name with U+FFFD is the real name: LakeGen re-runs the
reference code on the same files, so OrQA cannot rename the column.

A model reading "Total �" writes the symbol it believes was meant ("Total £")
and keeps doing so when told the real name, because it takes U+FFFD for a
display glitch (seen 2026-11-06: one plan re-requested four times with
"did you mean: Total �", then degraded to the free-text fallback). So:

* where a name is a LABEL (plan steps, ``columns_involved``), a written name
  that matches exactly one damaged header, each U+FFFD standing for one
  character, is resolved to that header (:func:`resolve_undecodable_name`,
  :func:`resolve_plan_names`). Nothing is guessed: the match is unique or
  nothing changes;
* where a name is CODE, the code runs as written, so the model is told how to
  type it: the ASCII escape ``\\ufffd`` in a Python or JSON string
  (:func:`escaped_literal`), the character itself in SQL (DuckDB has no
  escape for it in identifiers).
"""

from __future__ import annotations

import copy
import json
import re
from functools import lru_cache
from typing import Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "REPLACEMENT_CHAR",
    "is_undecodable",
    "escaped_literal",
    "undecodable_matches",
    "resolve_undecodable_name",
    "resolve_plan_names",
]

REPLACEMENT_CHAR = "�"

# What one U+FFFD may stand for in a written name: the escape typed as text
# (a model writing \\ufffd inside a JSON string), any non-ASCII character
# (the £, é, ñ that was lost) or one ASCII punctuation mark/symbol (a "?" or
# "$" in its place). Never an ASCII letter, digit or space: "Total s" is not a
# spelling of "Total �", it is another name.
_STANDS_FOR = r"(?:\\ufffd|[^\x00-\x7f]|[!-/:-@\[-`{-~])"


def is_undecodable(name: object) -> bool:
    """Whether a column name holds U+FFFD."""
    return isinstance(name, str) and REPLACEMENT_CHAR in name


def escaped_literal(name: str) -> str:
    """The name as a double-quoted string literal with every U+FFFD written as
    the ASCII escape ``\\ufffd``: valid in Python and in JSON, and equal to the
    real name once read (``"Total \\ufffd"``)."""
    return json.dumps(name, ensure_ascii=False).replace(REPLACEMENT_CHAR, "\\ufffd")


@lru_cache(maxsize=4096)
def _pattern(name: str) -> "re.Pattern[str]":
    return re.compile(_STANDS_FOR.join(re.escape(part) for part in name.split(REPLACEMENT_CHAR)))


def undecodable_matches(written: str, names: Iterable[object]) -> List[str]:
    """Every damaged name in ``names`` that ``written`` spells, each U+FFFD
    standing for one character (see ``_STANDS_FOR``), sorted. ``[]`` when
    ``written`` is itself one of the names."""
    if not isinstance(written, str) or not written:
        return []
    damaged = {n for n in names if is_undecodable(n)}
    if written in damaged:
        return []
    return sorted(n for n in damaged if _pattern(n).fullmatch(written))


def resolve_undecodable_name(written: str, names: Iterable[object]) -> Optional[str]:
    """The one damaged name ``written`` spells, or ``None`` when it spells
    none or several."""
    matches = undecodable_matches(written, names)
    return matches[0] if len(matches) == 1 else None


def resolve_plan_names(
    raw_plan: dict, known_columns: Mapping[str, Iterable[object]]
) -> Tuple[dict, List[Tuple[str, str]]]:
    """A copy of a raw plan dict in which every column reference that names no
    real column but spells exactly one damaged one is replaced by it.

    Looked at: each step's ``columns``, ``columns_role`` keys, ``params``
    (keys and string values, at any depth) and ``produces[].sources``, and each
    ``tables[].columns_involved``. An ``Alias.column`` reference resolves
    against that alias's columns only. A name some step declares in
    ``produces`` is never touched, nor is prose (descriptions, the question).

    Returns ``(plan, changes)``, ``changes`` being ``(written, real)`` pairs in
    first-seen order. The plan is returned unchanged, and not copied, when no
    table has a damaged column.
    """
    by_alias = {alias: {str(c) for c in cols or ()} for alias, cols in (known_columns or {}).items()}
    every = set().union(*by_alias.values()) if by_alias else set()
    if not isinstance(raw_plan, dict) or not any(is_undecodable(c) for c in every):
        return raw_plan, []

    plan = copy.deepcopy(raw_plan)
    steps = [s for s in (plan.get("steps") or []) if isinstance(s, dict)]
    declared = {
        entry.get("name")
        for step in steps
        for entry in (step.get("produces") or [])
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    changes: dict = {}

    def fix(text):
        if not isinstance(text, str) or text in every or text in declared:
            return text
        if "." in text:
            head, tail = text.split(".", 1)
            if head in by_alias and tail:
                if tail in by_alias[head] or tail in declared:
                    return text
                real = resolve_undecodable_name(tail, by_alias[head])
                if real is None:
                    return text
                changes.setdefault(text, f"{head}.{real}")
                return f"{head}.{real}"
        real = resolve_undecodable_name(text, every)
        if real is None:
            return text
        changes.setdefault(text, real)
        return real

    def fix_list(values):
        return [fix(v) for v in values] if isinstance(values, list) else values

    def fix_deep(value):
        if isinstance(value, dict):
            return {fix(k): fix_deep(v) for k, v in value.items()}
        if isinstance(value, list):
            return [fix_deep(v) for v in value]
        return fix(value)

    for step in steps:
        if "columns" in step:
            step["columns"] = fix_list(step["columns"])
        if isinstance(step.get("columns_role"), dict):
            step["columns_role"] = {fix(k): v for k, v in step["columns_role"].items()}
        if "params" in step:
            step["params"] = fix_deep(step["params"])
        for entry in step.get("produces") or []:
            if isinstance(entry, dict) and "sources" in entry:
                entry["sources"] = fix_list(entry["sources"])
    for table in plan.get("tables") or []:
        if isinstance(table, dict) and "columns_involved" in table:
            table["columns_involved"] = fix_list(table["columns_involved"])

    return plan, list(changes.items())
