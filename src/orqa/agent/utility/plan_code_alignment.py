"""Deterministic plan<->code column-coverage check.

Companion to ``difficulty_estimator.py``: a code-computed fact fed into the
code judge as extra context, not a replacement for its own reading of the
code. It answers exactly one narrow question a regex CAN answer reliably —
does every column the plan says this query depends on (``tables[].
columns_involved``, decided once at planning time and never re-derived at
generation time — see ``structured_outputs.Table.reason``'s docstring)
actually appear as a literal column reference somewhere in the generated
code? — and leaves everything else (silent filter bias, wrong literal
values, unjustified operations) to the judge, which has the reasoning a
regex doesn't.

This exists because the code judge's own "name the phrase that justifies an
operation" check only verifies code -> question grounding; it has no
symmetric check that every column the PLAN already committed to actually
made it into the code. A plan step can be silently dropped in code
generation (or in a later correction round) without the code becoming
implausible on its face — see the audit findings this was written for:
a `distinct businesses` aggregate promised in the plan and never computed,
an `is_air_quality` flag derived and then never used. Both are invisible to
"does this code look reasonable" and cheap to catch mechanically instead.

Deliberately a WARNING signal, not a hard gate: pandas code can legitimately
reference a column only through a renamed variable, an earlier `.rename()`,
or a positional/attribute access that never re-quotes the original name, so
a flagged column is a prompt for the judge to look closer, not proof of a
defect. False negatives (a real drop the regex doesn't catch, e.g. a plan
column mentioned in a comment but not code) are expected and fine — the
judge's own reasoning is still the backstop.
"""

import re
from typing import Iterable, Sequence

# A column name is only checked when it looks like a genuine identifier (not
# noise a looser match would flag on) — otherwise a very short/generic name
# common to many columns invites false positives that would just teach the
# judge to ignore this signal.
_MIN_COLUMN_LENGTH = 3


def _quoted_literal_present(column: str, code: str) -> bool:
    """True if ``column`` appears as a quoted string literal anywhere in
    ``code`` — how pandas source is written for `df['col']`, `.agg(x=('col',
    'mean'))`, `groupby(['col'])`, `rename(columns={'col': ...})`, and every
    other real column reference observed across this pipeline's generated
    code. Deliberately NOT anchored to a specific call shape (subscript,
    kwarg, dict key, ...) — the earlier, shape-specific version of this idea
    missed dict-literal aggregations like `.agg({'col': 'count'})` entirely;
    matching the quoted literal anywhere sidesteps that without needing to
    enumerate every pandas call shape that can hold a column name.
    """
    pattern = r"['\"]" + re.escape(column) + r"['\"]"
    return re.search(pattern, code) is not None


def missing_plan_columns(columns_involved: Iterable[str], code: str) -> list[str]:
    """Columns the plan declared necessary that never appear (quoted) in the
    generated code, in declaration order, deduplicated. Empty when every
    declared column shows up somewhere in the code, or when nothing was
    declared.
    """
    seen: set[str] = set()
    missing: list[str] = []
    for column in columns_involved:
        if not column or len(column) < _MIN_COLUMN_LENGTH or column in seen:
            continue
        seen.add(column)
        if not _quoted_literal_present(column, code):
            missing.append(column)
    return missing


def alignment_warning(tables: Sequence[dict], code: str) -> str:
    """Human-readable warning block for the code-judge payload, or ``""``
    when nothing is flagged. ``tables`` is the query's own ``tables`` list
    (each entry a dict with ``name`` and ``columns_involved``, exactly the
    shape already attached to every executed query) — the union of every
    table's ``columns_involved`` is checked against ``code`` as one pool
    rather than per-table, since a column can be consumed after a join/union
    without being re-scoped to its origin table.
    """
    all_columns: list[str] = []
    for table in tables or []:
        if isinstance(table, dict):
            all_columns.extend(table.get("columns_involved") or [])

    missing = missing_plan_columns(all_columns, code or "")
    if not missing:
        return ""

    return (
        "\n\n### DETERMINISTIC PRE-CHECK (informational, not a verdict)\n"
        "The plan's `tables[].columns_involved` names the following column(s) "
        "as necessary to answer the question, but none of them appear as a "
        "quoted literal anywhere in the code below: "
        f"{', '.join(repr(c) for c in missing)}. This does not automatically "
        "mean a requirement was dropped — the code may reference one through "
        "a rename, a positional access, or a derived variable — but treat it "
        "as a specific thing to verify in Check 1 rather than assuming the "
        "column made it into the implementation."
    )
