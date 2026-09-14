"""Column-provenance resolution for structured query plans.

A plan may legitimately reference a column that exists in no raw table — an
``aggregate`` result, a ``derive`` output, a ``rank`` position. Before plan
steps carried a ``produces`` declaration there was no way to *say* so, and
:meth:`QueryPlanner.validate_plan` fell back on a blunt heuristic: once any
step could have derived something, stop rejecting unknown columns. That
disabled column checking for the rest of the plan, so every hallucinated name
after the first derivation was silently accepted.

This module replaces that guess with exact resolution. Every column a step
names must resolve to either a raw schema column or a derived column some
earlier step declared, whose own sources transitively root in raw columns.
The only thing that can now fail is a name no step declared and no table
contains — which is the definition of a hallucinated column.

Two public entry points:

* :func:`resolve_plan_columns` — walks the steps in order and returns every
  :class:`Violation` it finds (all of them, not just the first).
* :func:`compose_feedback` — turns those violations into the retry message
  the planner model reads. See that function for the context-economy rules.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "SOURCE_FREE_OPERATIONS",
    "Violation",
    "resolve_plan_columns",
    "compose_feedback",
    "plan_derived_columns",
]

# Operations that genuinely read no source column, so an empty `sources` list
# is legitimate rather than an omission: a row count, a positional rank, a
# literal constant. Every other operation must name what it was computed
# from — forcing a bogus source onto a `count(*)` would teach the model to
# fabricate lineage, which is the exact failure this module exists to stop.
#
# Defined here rather than beside the plan models so that this module, which
# is pure resolution logic, stays free of a pydantic dependency and can be
# exercised on its own. `prompting.models` imports it back for the field
# description.
SOURCE_FREE_OPERATIONS = frozenset({
    "count", "size", "row_number", "rank", "literal", "constant",
})

# A raw column root: (table alias, column name).
Root = Tuple[str, str]

# Max "did you mean" names per violation. Three is enough to be useful and
# short enough that the clause never dominates the line.
_MAX_SUGGESTIONS = 3

# difflib's default cutoff. Deliberately tighter than
# QueryValidator._suggest_columns' 0.3: that one feeds a correction loop that
# benefits from a wide net, whereas a "did you mean" on a plan-validation line
# is noise unless the match is genuinely close.
_SUGGESTION_CUTOFF = 0.6


def _suggest(name: str, available: Iterable[str]) -> List[str]:
    """Up to :data:`_MAX_SUGGESTIONS` close matches for *name*, best first.

    Returns ``[]`` rather than a weak guess when nothing is close — the
    feedback composer omits the clause entirely in that case, instead of
    printing an unhelpful "did you mean: <nothing like it>".
    """
    pool = sorted({c for c in available if c})
    if not name or not pool:
        return []
    return difflib.get_close_matches(
        name, pool, n=_MAX_SUGGESTIONS, cutoff=_SUGGESTION_CUTOFF
    )


@dataclass
class Violation:
    """One column problem found while resolving a plan's columns.

    ``kind`` selects the heading and fix text (see :data:`_KINDS`); ``column``
    is what the composer dedupes on, so the same bad name referenced by five
    steps collapses to one line rather than five.
    """

    kind: str
    step_order: int
    step_op: str
    column: str
    detail: str = ""  # per-violation specifics appended to the entry line
    suggestions: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Kind:
    """Presentation for one violation kind: a heading and the shared fix."""

    heading: str
    fixes: Tuple[str, ...]


# Every kind :func:`resolve_plan_columns` can emit, in the order they appear
# in the feedback. Ordering is by how actionable the fix is — a typo the model
# can correct outright comes before a structural problem needing a rewrite —
# so that when the global cap truncates, what survives is what is easiest to
# act on.
#
# The fix text is per-KIND, not per-violation: within a kind the fix is
# identical, so one line covers every column listed under it. That grouping is
# where most of the message's compression comes from.
_KINDS: Dict[str, _Kind] = {
    "unknown_column": _Kind(
        heading="UNKNOWN COLUMNS — not in the tables you were given",
        fixes=("use the real column name.",),
    ),
    "undeclared_column": _Kind(
        heading="UNDECLARED COLUMNS — in no table, and no step declares them",
        fixes=(
            # {column}/{source} are filled from the first column actually
            # listed under this heading, so the example is directly
            # copy-pasteable rather than something to translate first.
            'if the step computes it, declare it there, e.g. "produces": '
            '[{{"name": "{column}", "operation": "sum", "sources": '
            '["{source}"]}}]',
            "if you meant a column that already exists, use its real name.",
        ),
    ),
    "bad_source": _Kind(
        heading="BAD SOURCES — a declared column is computed from something "
                "that does not exist",
        fixes=("correct the source column, or drop this derived column.",),
    ),
    "ambiguous_source": _Kind(
        heading="AMBIGUOUS SOURCES — the same column name is in several tables",
        fixes=("qualify it with the table it comes from, e.g. Table_0.column.",),
    ),
    "missing_sources": _Kind(
        heading="MISSING SOURCES — a declared column says nothing about where "
                "it came from",
        fixes=(
            "list the column(s) it is computed from in `sources`. Leave "
            "`sources` empty only for a count-like operation ("
            + ", ".join(sorted(SOURCE_FREE_OPERATIONS))
            + ").",
        ),
    ),
    "name_collides_raw": _Kind(
        heading="NAME ALREADY TAKEN — a declared column has the name of a real "
                "column",
        fixes=(
            "pick a different name; to change a real column in place use a "
            "`clean` step with a `cast` action instead of declaring it here.",
        ),
    ),
    "declared_twice": _Kind(
        heading="DECLARED TWICE — one column cannot have two origins",
        fixes=(
            "declare it once, in the step that creates it. Every later step "
            "may then reference it by name freely — do not re-declare it.",
        ),
    ),
    "forward_reference": _Kind(
        heading="USED BEFORE IT EXISTS — referenced by a step that runs before "
                "the step declaring it",
        fixes=("move the declaring step earlier, or reorder the steps.",),
    ),
    "dropped_column": _Kind(
        heading="ALREADY DROPPED — removed by an earlier `clean` step",
        fixes=(
            "remove that `drop_column` action, or stop using the column "
            "afterwards.",
        ),
    ),
}


def _entry_fields(entry) -> Tuple[str, str, List[str]]:
    """``(name, operation, sources)`` from a ``produces`` entry.

    Accepts a :class:`~orqa.agent.prompting.models.DerivedColumn` or the plain
    dict a plan carries once it has been dumped, so the resolver works on both
    a live plan and a persisted one.
    """
    if isinstance(entry, dict):
        name = entry.get("name") or ""
        operation = entry.get("operation") or ""
        sources = entry.get("sources") or []
    else:
        name = getattr(entry, "name", "") or ""
        operation = getattr(entry, "operation", "") or ""
        sources = getattr(entry, "sources", None) or []
    if isinstance(sources, str):  # tolerate a bare string for a single source
        sources = [sources]
    return (
        name.strip(),
        operation.strip(),
        [s.strip() for s in sources if isinstance(s, str) and s.strip()],
    )


def _split_source(source: str, known_aliases: Set[str]) -> Tuple[Optional[str], str]:
    """Split ``"Table_0.amount"`` into ``("Table_0", "amount")``.

    The split is EXACT, not heuristic: aliases are the machine-generated
    ``Table_N`` vocabulary, so a source counts as qualified only when the text
    before its first ``.`` is a known alias. A column whose own name contains
    a dot can therefore never be misread as a qualified reference.
    """
    if "." in source:
        head, tail = source.split(".", 1)
        if head in known_aliases and tail:
            return head, tail
    return None, source


def _produced_names(steps: Sequence) -> Dict[str, int]:
    """Map every name any step declares to the order of its FIRST declaration.

    Pre-scanned before the walk so a reference to a name declared *later* can
    be reported as a reordering problem rather than a hallucination — a
    different problem with a different fix.
    """
    declared: Dict[str, int] = {}
    for step in steps:
        for entry in (getattr(step, "produces", None) or []):
            name, _, _ = _entry_fields(entry)
            if name and name not in declared:
                declared[name] = getattr(step, "order", 0)
    return declared


def plan_derived_columns(plan) -> Set[str]:
    """Every column name a plan's steps declare in ``produces``.

    The single source of truth for "this column is created, not read", used by
    the planner to build its registry and threaded downstream so
    :meth:`QueryValidator.prefilter_dataframes` can recognise a derived column
    by exact lookup instead of pattern-matching the generated code.
    """
    steps = plan.get("steps") if isinstance(plan, dict) else getattr(plan, "steps", None)
    names: Set[str] = set()
    for step in (steps or []):
        produces = (
            step.get("produces") if isinstance(step, dict)
            else getattr(step, "produces", None)
        ) or []
        for entry in produces:
            name, _, _ = _entry_fields(entry)
            if name:
                names.add(name)
    return names


def resolve_plan_columns(
    steps: Sequence,
    raw_columns_by_alias: Dict[str, Set[str]],
    known_aliases: Optional[Set[str]] = None,
    declared_output_columns: Optional[Callable[[object], Set[str]]] = None,
    clean_dropped_columns: Optional[Callable[[object], Set[str]]] = None,
) -> Tuple[List[Violation], Dict[str, Set[Root]]]:
    """Resolve every column a plan references against its real origin.

    Walks ``steps`` in order maintaining a registry of declared derived
    columns mapped to their transitive raw ``(alias, column)`` roots. A
    column resolves if it is a raw column of a table the step names, or a
    derived column an earlier step declared. Anything else is a violation.

    ALL violations are collected rather than raising on the first: the planner
    gets exactly one retry before falling back to a degraded free-text plan,
    so surfacing one problem per round-trip would burn that retry on a plan
    with two mistakes.

    Args:
        steps: The plan's steps, in any order (``order`` is read off each).
        raw_columns_by_alias: alias -> real schema columns. Copied, not
            mutated — the caller's schema is left intact.
        known_aliases: Aliases that may qualify a source. Defaults to the keys
            of *raw_columns_by_alias*.
        declared_output_columns: Optional best-effort harvest of output names
            from a step's free-form ``params`` (see
            ``QueryPlanner._declared_output_columns``). Kept as a fallback so
            a plan that declares an output only the old way still validates.
        clean_dropped_columns: Optional harvest of a ``clean`` step's
            ``drop_column`` targets (see
            ``QueryPlanner._clean_step_dropped_columns``).

    Returns:
        ``(violations, derived_by_name)`` — the latter mapping each declared
        derived column to the raw roots it ultimately reads from.
    """
    aliases = set(known_aliases) if known_aliases is not None else set(raw_columns_by_alias)
    # Working copy: `clean` steps legitimately remove columns from scope, but
    # the caller's schema must survive the walk unchanged. Derived names are
    # NEVER written in here — keeping raw and derived in separate namespaces
    # is what makes a hallucinated name distinguishable from a real one a step
    # later, which the old single-set approach could not do.
    raw: Dict[str, Set[str]] = {a: set(c) for a, c in (raw_columns_by_alias or {}).items()}
    every_raw: Set[str] = set().union(*raw.values()) if raw else set()

    violations: List[Violation] = []
    derived_by_name: Dict[str, Set[Root]] = {}
    declared_at = _produced_names(steps)
    # column -> order of the `clean` step that removed it (raw columns and any
    # derived column whose lineage rooted on one).
    dropped_at: Dict[str, int] = {}

    ordered = sorted(steps, key=lambda s: getattr(s, "order", 0))

    for step in ordered:
        order = getattr(step, "order", 0)
        op = getattr(step, "op", "") or ""
        step_tables = list(getattr(step, "tables", None) or [])

        # Raw columns this step may read. A step naming no tables acts on the
        # accumulated pipeline result, so every raw column is in scope for it.
        if step_tables:
            raw_in_scope: Set[str] = set()
            for table in step_tables:
                raw_in_scope |= raw.get(table, set())
        else:
            raw_in_scope = set().union(*raw.values()) if raw else set()

        # ── this step's own declarations ──────────────────────────────────
        own_names: Set[str] = set()
        for entry in (getattr(step, "produces", None) or []):
            name, operation, sources = _entry_fields(entry)
            if not name:
                continue

            if name in raw_in_scope:
                violations.append(Violation(
                    kind="name_collides_raw", step_order=order, step_op=op,
                    column=name,
                    detail=f"already a real column of {_owners_text(name, raw)}",
                ))
            elif name in derived_by_name:
                violations.append(Violation(
                    kind="declared_twice", step_order=order, step_op=op,
                    column=name,
                    detail=f"first declared in step {declared_at.get(name, '?')}",
                ))

            roots: Set[Root] = set()
            if not sources and operation.lower() not in SOURCE_FREE_OPERATIONS:
                violations.append(Violation(
                    kind="missing_sources", step_order=order, step_op=op,
                    column=name,
                    detail=f"operation {operation!r} reads at least one column",
                ))
            for source in sources:
                resolved, problem = _resolve_source(
                    source, aliases, raw, derived_by_name
                )
                if problem is not None:
                    problem.step_order, problem.step_op = order, op
                    problem.detail = (
                        f"source of {name!r}: {problem.detail}"
                        if problem.detail else f"source of {name!r}"
                    )
                    violations.append(problem)
                else:
                    roots |= resolved

            # Register even when the entry was flagged above: a rejected
            # declaration must not cascade into a second wave of "undeclared
            # column" violations for every later step that uses it. One root
            # cause, one message.
            derived_by_name.setdefault(name, set()).update(roots)
            own_names.add(name)

        # ── columns this step reads ───────────────────────────────────────
        params_fallback: Set[str] = set()
        if declared_output_columns is not None:
            try:
                params_fallback = set(declared_output_columns(step) or set())
            except Exception:  # free-form params: a bad shape is not fatal
                params_fallback = set()

        # An output declared only the OLD way, in free-form `params`
        # (output_column / new_column / aggregations keys), is a created
        # column like any other — register it so LATER steps resolve it too,
        # not just this one. Its root set is empty: `params` records that a
        # column is produced but never says from what, which is precisely the
        # gap `produces` exists to close.
        for name in params_fallback:
            derived_by_name.setdefault(name, set())

        allowed = raw_in_scope | set(derived_by_name) | own_names | params_fallback

        referenced = list(getattr(step, "columns", None) or [])
        # `correlate`/`limit`/`rank` name columns in params rather than in
        # `columns`, so they are invisible to the loop below unless collected
        # here too.
        if op in ("correlate", "limit", "rank"):
            params = getattr(step, "params", None) or {}
            if isinstance(params, dict):
                referenced += [c for c in (params.get("by") or []) if isinstance(c, str)]
                referenced += [
                    c for c in (params.get("group_by") or []) if isinstance(c, str)
                ]

        seen_here: Set[str] = set()
        for column in referenced:
            if not column or column in allowed or column in seen_here:
                continue
            seen_here.add(column)
            violations.append(
                _reference_violation(
                    column, order, op, raw_in_scope, declared_at, dropped_at
                )
            )

        # ── columns this step removes ─────────────────────────────────────
        dropped: Set[str] = set()
        if clean_dropped_columns is not None:
            try:
                dropped = set(clean_dropped_columns(step) or set())
            except Exception:
                dropped = set()
        for column in dropped:
            for table in (step_tables or list(raw)):
                raw.get(table, set()).discard(column)
            every_raw.discard(column)
            dropped_at[column] = order
            # A derived column rooted on a dropped column is no longer
            # computable — invalidate it too, so a later reference reports the
            # real cause (the drop) instead of a puzzling "undeclared column".
            for name, roots in list(derived_by_name.items()):
                if any(root_col == column for _, root_col in roots):
                    del derived_by_name[name]
                    dropped_at[name] = order

    return violations, derived_by_name


def _owners_text(column: str, raw: Dict[str, Set[str]]) -> str:
    """Human-readable list of the aliases holding *column*."""
    owners = sorted(a for a, cols in raw.items() if column in cols)
    return " and ".join(owners) if owners else "a table you were given"


def _resolve_source(
    source: str,
    aliases: Set[str],
    raw: Dict[str, Set[str]],
    derived_by_name: Dict[str, Set[Root]],
) -> Tuple[Set[Root], Optional[Violation]]:
    """Resolve one ``produces`` source to the raw roots it reads from.

    Resolution is NAME-FIRST: a source naming an already-declared derived
    column resolves as derived and any alias prefix is discarded unread —
    absent, correct, or wrong all behave identically. That is deliberate. A
    derived column's roots may span several tables (a ratio of a sum over one
    table and a count over another), so there is no single correct alias for
    it, and demanding one would force exactly the fabrication this check
    exists to prevent.

    Name-first is unambiguous only because a name can be a raw column or a
    derived column but never both — enforced by the ``name_collides_raw``
    check in :func:`resolve_plan_columns`. That check is not merely
    anti-shadowing hygiene; it is the precondition that makes this function
    sound. Do not remove it.
    """
    alias, column = _split_source(source, aliases)

    if column in derived_by_name:
        return set(derived_by_name[column]), None

    if alias is not None:
        if column in raw.get(alias, set()):
            return {(alias, column)}, None
        return set(), Violation(
            kind="bad_source", step_order=0, step_op="", column=source,
            detail=f"{column!r} is not in {alias}",
            suggestions=_suggest(column, raw.get(alias, set())),
        )

    owners = sorted(a for a, cols in raw.items() if column in cols)
    if len(owners) == 1:
        return {(owners[0], column)}, None
    if len(owners) > 1:
        return set(), Violation(
            kind="ambiguous_source", step_order=0, step_op="", column=source,
            detail=f"{column!r} is in {' and '.join(owners)}",
        )

    every = set().union(*raw.values()) if raw else set()
    return set(), Violation(
        kind="bad_source", step_order=0, step_op="", column=source,
        detail=f"{column!r} is in no table and no earlier step declares it",
        suggestions=_suggest(column, every | set(derived_by_name)),
    )


def _reference_violation(
    column: str,
    order: int,
    op: str,
    raw_in_scope: Set[str],
    declared_at: Dict[str, int],
    dropped_at: Dict[str, int],
) -> Violation:
    """Classify an unresolvable column reference into the kind whose fix fits.

    The distinction is what makes the feedback actionable: telling the model
    to "declare it in `produces`" when it simply misspelled an existing column
    sends it the wrong way, and offering spelling suggestions for a column it
    genuinely means to compute sends it the other wrong way.
    """
    if column in dropped_at:
        return Violation(
            kind="dropped_column", step_order=order, step_op=op, column=column,
            detail=f"dropped by the `clean` step {dropped_at[column]}",
        )
    if column in declared_at and declared_at[column] > order:
        return Violation(
            kind="forward_reference", step_order=order, step_op=op, column=column,
            detail=f"declared later, in step {declared_at[column]}",
        )
    suggestions = _suggest(column, raw_in_scope)
    if suggestions:
        return Violation(
            kind="unknown_column", step_order=order, step_op=op, column=column,
            suggestions=suggestions,
        )
    return Violation(
        kind="undeclared_column", step_order=order, step_op=op, column=column,
    )


# Caps on the composed message. The planner's retry prompt re-sends the whole
# (cached) planning prompt, so this block is the delta — but a plan with forty
# broken references is structurally wrong, and enumerating all forty teaches
# the model nothing the first few do not. Past the cap it needs the pattern,
# not the inventory.
_MAX_PER_KIND = 4
_MAX_TOTAL = 10


def compose_feedback(
    violations: Sequence[Violation], example_source: str = "Table_0.column"
) -> str:
    """The retry message the planner model reads. ``""`` when nothing is wrong.

    Batched violations plus a single retry mean this text is the only thing
    standing between a fixable plan and the degraded free-text fallback, so it
    is built to five rules:

    1. **The governing rule leads.** A bare list of symptoms gets fixed one
       column at a time, often by inventing a different bad column; one
       sentence of rule reframes all of them at once.
    2. **Every line pairs a problem with an action.** No diagnosis without a
       fix.
    3. **Grouped by kind, not by step** — within a kind the fix is identical,
       so one fix line serves every column under it.
    4. **Nothing already in the prompt is repeated.** No schemas, no
       re-teaching of the `produces` format, no restating the plan; all of it
       sits in the cached prefix above. The lone exception is one inline
       `produces` example, which is the only fix needing syntax the model has
       not yet written — and it is built from the plan's own column and a real
       alias (*example_source*) so it can be pasted rather than translated.
    5. **Deduped and capped** — see :data:`_MAX_PER_KIND` / :data:`_MAX_TOTAL`.
    """
    if not violations:
        return ""

    # (kind, column) -> first occurrence, plus how many further steps repeat it
    grouped: Dict[str, Dict[str, Tuple[Violation, int]]] = {}
    for violation in violations:
        bucket = grouped.setdefault(violation.kind, {})
        existing = bucket.get(violation.column)
        if existing is None:
            bucket[violation.column] = (violation, 0)
        else:
            first, extra = existing
            bucket[violation.column] = (first, extra + 1)

    total_problems = sum(len(b) for b in grouped.values())
    lines = [
        f"### PLAN VALIDATION FAILED — {total_problems} "
        f"problem{'s' if total_problems != 1 else ''}, fix all of them",
        "Every column a step names must exist in a table you were given, or be "
        "declared in an earlier step's `produces`.",
    ]

    shown = 0
    skipped_kinds = 0
    for kind, meta in _KINDS.items():
        bucket = grouped.get(kind)
        if not bucket:
            continue
        if shown >= _MAX_TOTAL:
            skipped_kinds += len(bucket)
            continue

        entries = sorted(bucket.values(), key=lambda pair: pair[0].step_order)
        room = min(_MAX_PER_KIND, _MAX_TOTAL - shown)
        visible, hidden = entries[:room], len(entries) - room

        lines.append("")
        lines.append(f"{meta.heading}:")
        for violation, extra_steps in visible:
            lines.append("  " + _entry_line(violation, extra_steps))
        if hidden > 0:
            lines.append(f"  …and {hidden} more of the same kind")
        for fix in meta.fixes:
            lines.append(f"  → {_fill(fix, visible[0][0], example_source)}")
        shown += len(visible)

    if skipped_kinds:
        lines.append("")
        lines.append(f"…and {skipped_kinds} more problem(s) of other kinds.")

    lines.append("")
    lines.append(
        "Fix exactly these problems. Keep the question, the steps, and "
        "everything else unchanged."
    )
    return "\n".join(lines)


def _fill(fix: str, violation: Violation, example_source: str) -> str:
    """Fill a fix template with this heading's first real column.

    Templates without placeholders pass through untouched, and a malformed one
    degrades to its literal text rather than raising — the feedback path must
    never be the thing that breaks validation.
    """
    try:
        return fix.format(column=violation.column, source=example_source)
    except (KeyError, IndexError, ValueError):
        return fix


def _entry_line(violation: Violation, extra_steps: int) -> str:
    """One violation as ``step N (op) "column" — detail — did you mean: ...``."""
    parts = [
        f'step {violation.step_order} ({violation.step_op}) "{violation.column}"'
    ]
    if violation.detail:
        parts.append(violation.detail)
    if violation.suggestions:
        parts.append("did you mean: " + ", ".join(violation.suggestions))
    line = " — ".join(parts)
    if extra_steps:
        line += f"  (+{extra_steps} more step{'s' if extra_steps != 1 else ''})"
    return line
