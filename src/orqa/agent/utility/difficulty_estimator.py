"""Deterministic DIFFICULTY-tier estimator for structured query plans.

Companion to ``keyword_searchability.py``: a code-computed fact fed into the
plan judge / planner loop rather than an LLM vote, for exactly the parts of
the DIFFICULTY rubric (see ``conf/prompts/query_planner.md``) that are pure
functions of a plan's own ``steps`` — step counts, op-type diversity,
group-by cardinality, aggregation counts, and whether tables were combined
by extending one growing chain or by merging two independently-built
branches. An LLM never needs to "count" any of this; it only needs to judge
the two things a counter genuinely can't: whether two flag/bucket steps
preserve GENUINELY DIFFERENT messy-data patterns, and whether a bucket
step's branching reflects real domain judgment rather than an arbitrary
split. Both stay judge-voted (``plan_judge.md`` Check 5) — this module
computes everything else and reports it as a given fact.

Tier assignment is MAX-of-triggers (mirrors the rubric's own nested-OR
shape), not a summed point score: an earlier additive-weights draft
double-counted a single grouped multi-aggregation across four separate
bonus terms and pushed a plainly-medium plan into "hard". Each trigger
below independently claims a tier; the plan's tier is the highest any
trigger reaches. This also fixes a real contradiction in the prose rubric
this replaces, which said "table count alone is NOT a structural qualifier"
in one line and "three or more tables combined into one linear composition"
(a pure table-count rule) as a medium trigger a few lines later — the
DEEP-MERGE check below (independent branches, not raw table count) is what
that trigger actually meant, and now a single wide join/union step
combining any number of already-raw tables never costs more than one step
regardless of how many aliases its `tables` list carries.

The thresholds are named constants, not embedded magic numbers, so they can
be recalibrated against a labelled corpus without touching the trigger
logic itself.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

_TIER_ORDER = ("easy", "medium", "hard")

# --- Calibration knobs -----------------------------------------------------
# Starting values mirror the qualitative rubric this replaces; recalibrate
# against a labelled corpus of already-judge-approved plans if the mix of
# tiers a batch produces drifts from the intended ~1/3 split.
_EASY_STEP_MAX = 2          # structural steps (non-clean, non-plain-derive)
_MEDIUM_STEP_MIN = 3
_HARD_STEP_MIN = 4
_HARD_OP_DIVERSITY_MIN = 3   # distinct op types among the counted steps
_EASY_AGG_MAX = 1            # aggregate/correlate steps
_MEDIUM_AGG_MIN = 2
_HARD_GROUP_KEY_MIN = 2      # group-by keys on the widest `group` step
_MEDIUM_DQ_SIGNAL_MIN = 2    # flag/bucket derive steps


def _op(step: Any) -> str:
    return getattr(step, "op", None) or (step.get("op") if isinstance(step, dict) else None)


def _tables(step: Any) -> list[str]:
    val = getattr(step, "tables", None)
    if val is None and isinstance(step, dict):
        val = step.get("tables")
    return list(val or [])


def _columns(step: Any) -> list[str]:
    val = getattr(step, "columns", None)
    if val is None and isinstance(step, dict):
        val = step.get("columns")
    return list(val or [])


def _params(step: Any) -> dict:
    val = getattr(step, "params", None)
    if val is None and isinstance(step, dict):
        val = step.get("params")
    return dict(val or {})


def _actions(step: Any) -> list[dict]:
    actions = _params(step).get("actions") or []
    return [a for a in actions if isinstance(a, dict)]


def _dq_technique(step: Any) -> Optional[str]:
    """``"bucket"``, ``"flag"``, or ``None`` for a non-DQ-preserving step.

    A step counts as ``"bucket"`` if ANY of its actions use that technique
    (bucket is the stronger signal), else ``"flag"`` if any action does,
    else ``None``. Only meaningful for a `derive` step.
    """
    if _op(step) != "derive":
        return None
    techniques = {a.get("technique") for a in _actions(step)}
    if "bucket" in techniques:
        return "bucket"
    if "flag" in techniques:
        return "flag"
    return None


def _is_plain_derive(step: Any) -> bool:
    """A `derive` step that does NOT preserve a flag/bucket DQ pattern —
    ordinary computed-column prep, never analytical complexity on its own
    (see the DATA QUALITY / DIFFICULTY sections of query_planner.md)."""
    return _op(step) == "derive" and _dq_technique(step) is None


def _counts_structurally(step: Any) -> bool:
    """Whether a step is ever eligible to count toward EITHER axis's step
    total — everything except `clean` (bookkeeping) and a plain `derive`
    (routine prep, just like `clean`)."""
    return _op(step) != "clean" and not _is_plain_derive(step)


def _bucket_output_columns(step: Any) -> list[str]:
    return [
        a.get("output_column")
        for a in _actions(step)
        if a.get("technique") == "bucket" and a.get("output_column")
    ]


class _DisjointSet:
    """Union-Find over table aliases, tracking each set's table COUNT.

    Used to tell "extend one growing chain by one more raw table" (cheap —
    the merged-in side is a fresh singleton) apart from "merge two
    branches that were EACH already built up from their own prior
    join/union" (the genuinely expensive case the HARD tier's "independent
    branches" trigger means) — never by how many aliases a single step's
    `tables` list happens to name at once.
    """

    def __init__(self, aliases: list[str]):
        self._parent = {a: a for a in aliases}
        self._size = {a: 1 for a in aliases}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._size[x] = 1
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def size(self, x: str) -> int:
        return self._size[self.find(x)]

    def union_many(self, xs: list[str]) -> int:
        """Union every alias in ``xs`` into one set.

        Returns the number of sides that were ALREADY multi-table (size >=
        2) before this merge — 2+ means this single step merged two
        independently-built branches, the HARD "independent branches"
        trigger. 0 or 1 means at most one side was a genuine prior branch
        (a raw table being folded into an existing chain, or the very
        first join of the plan) — never itself the expensive case.
        """
        roots = {self.find(x) for x in xs}
        if len(roots) <= 1:
            return 0
        deep_sides = sum(1 for r in roots if self._size[r] >= 2)
        total = sum(self._size[r] for r in roots)
        new_root = next(iter(roots))
        for r in roots:
            self._parent[r] = new_root
        self._size[new_root] = total
        return deep_sides


@dataclass
class DifficultyEstimate:
    """Deterministic facts about one plan's structural/DQ complexity.

    ``structural_tier``/``data_engineering_tier`` are each computed by
    MAX-of-triggers (see module docstring). ``tier`` is the max of the two,
    matching the rubric's "a plan only needs to satisfy ONE axis" rule.

    ``dq_hard_pending_judge`` is True when the DETERMINISTIC half of the
    hard-tier data-engineering trigger holds (a bucket step feeds a further
    group/aggregate) but the plan's actual hard-tier credit still depends
    on the judge confirming the bucket's internal logic is genuinely
    compound (3+ real categories, not an arbitrary split) — that
    confirmation is not something this module can compute, since bucket
    categories only exist as free-text `rule` prose, never a structured
    field. Treated as sufficient for the pre-judge gate (see
    ``QueryPlanner``); Check 5 in ``plan_judge.md`` makes the final call.
    """

    tier: str
    structural_tier: str
    data_engineering_tier: str
    structural_step_count: int
    op_diversity: int
    agg_correlate_count: int
    group_key_count: int
    group_multi_agg: bool
    deep_merge_events: int
    dq_signal_count: int
    dq_hard_pending_judge: bool
    explanation: str = ""


def _tier_reached(triggers: list[str]) -> str:
    if not triggers:
        return "easy"
    return max(triggers, key=_TIER_ORDER.index)


def estimate_plan_tier(plan: Any) -> DifficultyEstimate:
    """Compute the deterministic DIFFICULTY tier for ``plan``.

    ``plan`` is any object (or dict) exposing ``.steps`` / ``.tables`` the
    way ``SQLQueryPlan``/``PandasQueryPlan`` do — never imported here
    directly to avoid a circular import with ``QueryPlanner``.
    """
    steps = list(getattr(plan, "steps", None) or (plan.get("steps") if isinstance(plan, dict) else []) or [])
    table_entries = getattr(plan, "tables", None) or (plan.get("tables") if isinstance(plan, dict) else []) or []
    aliases = [getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None) for t in table_entries]
    aliases = [a for a in aliases if a]

    counted_steps = [s for s in steps if _counts_structurally(s)]
    structural_step_count = len(counted_steps)
    op_diversity = len({_op(s) for s in counted_steps})
    agg_correlate_count = sum(1 for s in counted_steps if _op(s) in ("aggregate", "correlate"))
    group_steps = [s for s in steps if _op(s) == "group"]
    group_key_count = max((len(_columns(s)) for s in group_steps), default=0)
    group_multi_agg = bool(group_steps) and agg_correlate_count >= 2

    dsu = _DisjointSet(aliases)
    deep_merge_events = 0
    for step in steps:
        if _op(step) not in ("join", "union"):
            continue
        deep_sides = dsu.union_many(_tables(step))
        if deep_sides >= 2:
            deep_merge_events += 1

    dq_signal_count = sum(1 for s in steps if _dq_technique(s) is not None)
    bucket_outputs: set[str] = set()
    for s in steps:
        if _dq_technique(s) == "bucket":
            bucket_outputs.update(_bucket_output_columns(s))
    dq_bucket_feeds_aggregate = bool(bucket_outputs) and any(
        _op(s) in ("group", "aggregate") and bucket_outputs & set(_columns(s))
        for s in steps
    )

    structural_triggers = []
    if deep_merge_events >= 1:
        structural_triggers.append("hard")
    if structural_step_count >= _HARD_STEP_MIN and op_diversity >= _HARD_OP_DIVERSITY_MIN:
        structural_triggers.append("hard")
    if group_key_count >= _HARD_GROUP_KEY_MIN and group_multi_agg:
        structural_triggers.append("hard")
    if group_multi_agg:
        structural_triggers.append("medium")
    if structural_step_count >= _MEDIUM_STEP_MIN:
        structural_triggers.append("medium")
    if agg_correlate_count >= _MEDIUM_AGG_MIN:
        structural_triggers.append("medium")
    structural_tier = _tier_reached(structural_triggers)

    dq_triggers = []
    if dq_bucket_feeds_aggregate:
        dq_triggers.append("hard")
    if dq_signal_count >= _MEDIUM_DQ_SIGNAL_MIN:
        dq_triggers.append("medium")
    data_engineering_tier = _tier_reached(dq_triggers)

    tier = max([structural_tier, data_engineering_tier], key=_TIER_ORDER.index)

    explanation = (
        f"structural={structural_tier} (steps={structural_step_count}, "
        f"op_types={op_diversity}, agg/correlate={agg_correlate_count}, "
        f"group_keys={group_key_count}, group_multi_agg={group_multi_agg}, "
        f"independent_branch_merges={deep_merge_events}); "
        f"data_engineering={data_engineering_tier} (flag/bucket_steps="
        f"{dq_signal_count}, bucket_feeds_aggregate={dq_bucket_feeds_aggregate})"
    )

    return DifficultyEstimate(
        tier=tier,
        structural_tier=structural_tier,
        data_engineering_tier=data_engineering_tier,
        structural_step_count=structural_step_count,
        op_diversity=op_diversity,
        agg_correlate_count=agg_correlate_count,
        group_key_count=group_key_count,
        group_multi_agg=group_multi_agg,
        deep_merge_events=deep_merge_events,
        dq_signal_count=dq_signal_count,
        dq_hard_pending_judge=dq_bucket_feeds_aggregate,
        explanation=explanation,
    )


def build_reconciliation_feedback(estimate: DifficultyEstimate, declared_tier: str) -> str:
    """Numeric, actionable feedback for a planner correction round.

    Named per-trigger gaps rather than a vague "make it harder/easier" —
    the planner gets told exactly which countable thing to change.
    """
    if estimate.tier == declared_tier:
        return ""

    order = _TIER_ORDER.index
    if order(estimate.tier) < order(declared_tier):
        gaps = []
        if declared_tier in ("medium", "hard"):
            if estimate.structural_step_count < _MEDIUM_STEP_MIN:
                gaps.append(
                    f"add chained steps (currently {estimate.structural_step_count} "
                    f"non-clean, non-plain-derive steps; medium needs "
                    f"{_MEDIUM_STEP_MIN}+)"
                )
            if not estimate.group_multi_agg:
                gaps.append(
                    "add a `group` step feeding 2+ aggregations, or a second "
                    "`aggregate`/`correlate` step"
                )
            if estimate.dq_signal_count < _MEDIUM_DQ_SIGNAL_MIN:
                gaps.append(
                    "add a second `derive` step using technique flag/bucket "
                    "that preserves a genuinely DIFFERENT messy-data pattern "
                    "than any existing one"
                )
        if declared_tier == "hard":
            if estimate.deep_merge_events < 1:
                gaps.append(
                    "combine two INDEPENDENTLY pre-built branches (each "
                    "already the result of its own join/union) with a "
                    "further join/union, rather than folding one more raw "
                    "table into a single growing chain"
                )
            if not (estimate.structural_step_count >= _HARD_STEP_MIN and estimate.op_diversity >= _HARD_OP_DIVERSITY_MIN):
                gaps.append(
                    f"chain {_HARD_STEP_MIN}+ steps spanning {_HARD_OP_DIVERSITY_MIN}+ "
                    f"distinct op types (currently {estimate.structural_step_count} "
                    f"steps across {estimate.op_diversity} op types)"
                )
            if not (estimate.group_key_count >= _HARD_GROUP_KEY_MIN and estimate.group_multi_agg):
                gaps.append(
                    f"key a `group` step on {_HARD_GROUP_KEY_MIN}+ columns while "
                    "feeding 2+ aggregations"
                )
        return (
            f"Computed tier is `{estimate.tier}` but this plan is slotted "
            f"`{declared_tier}` — it's TOO EASY for that slot. Add STEPS (not "
            f"a difficulty relabel) via any ONE of: " + "; or ".join(gaps) + "."
        )
    else:
        return (
            f"Computed tier is `{estimate.tier}` but this plan is slotted "
            f"`{declared_tier}` — it's TOO HARD for that slot "
            f"({estimate.explanation}). Simplify the steps (drop a join, a "
            "chained step, or a derive-preserve step) until BOTH axes sit "
            f"at or below `{declared_tier}`, without changing the "
            "`difficulty` label itself."
        )
