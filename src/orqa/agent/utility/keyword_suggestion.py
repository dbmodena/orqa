"""Deterministic keyword suggestion for a table group's retrievability.

The plan judge's keyword-searchability check (see
``orqa.agent.utility.keyword_searchability``) verifies a plan's
``question_keywords`` AFTER the planner has already guessed them — a
rejected plan only learns it guessed wrong, then guesses again, burning an
LLM correction round each time (see ``StatementOrchestrator._judge_plans``).

This module inverts that: given the exact table group a plan will use,
BEFORE any plan is drafted, it computes a keyword set EMPIRICALLY VERIFIED
against the real reverse index to surface every one of those tables within
a target top-K — so the planner can be handed working vocabulary to build
the question around from the start, instead of discovering it by trial and
error after rejection. When no such set exists even after an exhaustive
search, that is a sound (not merely heuristic) proof the table group cannot
be jointly retrieved by ANY natural-language question within that top-K:
BM25 scoring is a deterministic function of literal term overlap with a
document's own indexed text, so a query term absent from a table's own
title/tags/columns/publisher contributes exactly zero to that table's
score no matter how a question phrases it — there is no synonym an LLM
could invent that isn't already covered by exhaustively trying the table's
own real vocabulary.

The search is a greedy local search whose fitness function is the real
``index.search()`` call — not an IDF/heuristic approximation of it — so it
is correct by construction for whichever backend (``DatasetIndex`` or
``ESDatasetIndex``) is configured, and naturally accounts for BM25's
cross-term interactions (a term that helps one target can simultaneously
boost an unrelated competitor high enough to still push a target out of the
window; a heuristic score-per-term estimate would miss that). Fitness is
rank-aware, not a binary "in top-K or not": a term that moves a table from
rank 40 to rank 12 (still outside a top-6 window) is real progress a purely
binary signal would discard as "no improvement" and could stall the search
before a second term stacks on top of it to actually cross the line.
"""

from typing import Any, Optional

from orqa.benchmark.index import tokenize, _record_field_texts

# Generic connective words that are almost never what makes a dataset
# distinguishable (unlike the domain-specific terms in a title/tag) — kept
# out of the candidate pool so the greedy search isn't wasting evaluation
# budget on additions that are essentially guaranteed to be useless noise
# every table shares with half the corpus.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "by", "with", "from", "as", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "data", "dataset",
})

# How far past top_k to look when ranking a still-missing table, so the
# greedy search can see incremental progress (rank 40 -> rank 12) even
# while a table is still outside the window it ultimately needs to reach.
# Not itself a pass/fail bound — only ``top_k`` decides that.
_RANK_CEILING_MULTIPLIER = 8
_RANK_CEILING_FLOOR = 60


def _terms_from_text(*texts: str) -> list[str]:
    seen: list[str] = []
    seen_set: set[str] = set()
    for text in texts:
        for tok in tokenize(text):
            if tok in _STOPWORDS or len(tok) < 2 or tok in seen_set:
                continue
            seen_set.add(tok)
            seen.append(tok)
    return seen


def _tiered_pools(record: dict) -> tuple[list[str], list[str]]:
    """(primary, fallback) candidate terms for one table.

    Primary = title + tags + publisher. Title/tags are the two
    highest-weighted fields (see ``index.FIELD_WEIGHTS``); publisher is
    added alongside them despite its lower weight (1.5, still well above
    ``description``'s 1.0) because it shares their shape — short, concise
    text (an agency name/abbreviation), not noisy prose — and is exactly
    the kind of term a real portal user searches by ("DOT crash data",
    "DOE enrollment"). Fallback = columns only, tried when the primary
    pool can't get every table into top_k on its own: column names/labels
    can carry a genuinely distinguishing term, but the field is the
    noisiest of the four (every column's name+label+description
    concatenated, prone to generic tokens like "date"/"id"/"count") and
    the least natural for a lay question to organically echo, so it's a
    second resort rather than searched from the start. ``description``
    (the lowest-weighted field) is never used: at that weight it is
    rarely what actually swings a ranking, and is the least natural
    source for a lay question to echo anyway.
    """
    fields = _record_field_texts(record)
    primary = _terms_from_text(fields["title"], fields["tags"], fields["publisher"])
    fallback = [t for t in _terms_from_text(fields["columns"]) if t not in primary]
    return primary, fallback


def suggest_retrievable_keywords(
    tables: list[dict],
    index: Optional[Any],
    top_k: int,
    max_iterations: int = 15,
) -> dict:
    """Greedily find a keyword set that surfaces every table within top_k.

    Args:
        tables: ``[{"alias": str, "resource_id": str}, ...]`` for the
            group a plan needs — same shape
            ``check_keyword_searchability`` takes.
        index: A ``DatasetIndex``/``ESDatasetIndex`` (or ``None`` when
            unavailable for this portal — no-ops to an empty, unachieved
            result rather than raising, same degradation convention as
            ``check_keyword_searchability``).
        top_k: The target window every table must land inside.
        max_iterations: Cap on greedy add/remove steps PER TIER (primary,
            then fallback if the primary pool alone doesn't achieve it) —
            so a table group with no good shared vocabulary fails fast
            rather than hammering the index indefinitely.

    Returns:
        ``{"keywords": [str, ...], "achieved": bool, "hit_count": int,
        "missing_tables": [alias, ...], "ranks": {alias: int},
        "iterations_used": int, "used_fallback_fields": bool}``.
        ``achieved`` is True only when EVERY table landed in top_k;
        otherwise ``keywords`` is still the best set the search found,
        useful as a starting point even when incomplete. ``ranks`` is each
        table's final 1-indexed rank (or a large sentinel if not found even
        within a wide ceiling search) — a table can be ``achieved`` while
        sitting right at the top_k boundary (rank == top_k, the most
        fragile possible margin) rather than comfortably inside it; the
        search prefers a lower (safer) rank as a tie-break once hit_count
        is equal, but ``ranks`` lets the caller see the actual margin
        achieved rather than only the pass/fail bit. ``used_fallback_fields``
        tells the caller whether the primary title/tags/publisher
        vocabulary alone was enough, or whether column names were needed
        to close the gap.
    """
    if index is None or not tables:
        return {
            "keywords": [],
            "achieved": False,
            "hit_count": 0,
            "missing_tables": [t["alias"] for t in tables],
            "iterations_used": 0,
            "used_fallback_fields": False,
        }

    primary_pools: dict[str, list[str]] = {}
    fallback_pools: dict[str, list[str]] = {}
    for t in tables:
        record = index.get(t["resource_id"])
        primary_pools[t["alias"]], fallback_pools[t["alias"]] = (
            _tiered_pools(record) if record else ([], [])
        )

    ceiling = max(top_k * _RANK_CEILING_MULTIPLIER, _RANK_CEILING_FLOOR)

    def evaluate(keywords: set[str]) -> tuple[int, dict[str, int]]:
        """(hit_count within top_k, {alias: rank within ceiling or ceiling+1})."""
        if not keywords:
            return 0, {t["alias"]: ceiling + 1 for t in tables}
        results = index.search(list(keywords), top_k=ceiling)
        order = {r.resource_id: i + 1 for i, r in enumerate(results)}
        ranks = {t["alias"]: order.get(t["resource_id"], ceiling + 1) for t in tables}
        hit_count = sum(1 for rank in ranks.values() if rank <= top_k)
        return hit_count, ranks

    def fitness(hit_count: int, ranks: dict[str, int]) -> tuple[int, int, int]:
        # Primary: how many tables are actually in the window. Secondary
        # (breaks ties/plateaus while any table is still outside): NEGATIVE
        # rank-sum of tables still outside it, so moving a buried table from
        # rank 40 to rank 12 counts as progress even before it crosses
        # top_k. Tertiary (breaks ties once hit_count is equal): NEGATIVE
        # rank-sum of tables ALREADY inside the window — without this, a
        # table that just barely squeaks in at rank top_k (the single most
        # fragile possible spot: a slightly different top_k, or the corpus
        # gaining one more competing dataset over time, and it falls back
        # out) scores IDENTICALLY to one sitting comfortably at rank 1, so
        # the search would never prefer the safer margin. This costs no
        # extra index.search() calls — it only changes which of the SAME
        # round's already-evaluated candidates wins a tie.
        missing_regret = sum(rank for alias, rank in ranks.items() if rank > top_k)
        hit_regret = sum(rank for alias, rank in ranks.items() if rank <= top_k)
        return (hit_count, -missing_regret, -hit_regret)

    selected: set[str] = set()
    current_hit_count, current_ranks = evaluate(selected)
    current_fitness = fitness(current_hit_count, current_ranks)
    iterations_used = 0
    used_fallback_fields = False

    for pool_map in (primary_pools, fallback_pools):
        if pool_map is fallback_pools:
            if current_hit_count == len(tables):
                break
            used_fallback_fields = True

        for _ in range(max_iterations):
            if current_hit_count == len(tables):
                break
            iterations_used += 1

            missing_aliases = [
                t["alias"] for t in tables
                if current_ranks[t["alias"]] > top_k
            ]
            untried = [
                term
                for alias in missing_aliases
                for term in pool_map[alias]
                if term not in selected
            ]

            best_term: Optional[str] = None
            best_state: Optional[tuple[int, dict[str, int]]] = None
            best_fit = current_fitness
            for term in untried:
                trial_hit_count, trial_ranks = evaluate(selected | {term})
                trial_fit = fitness(trial_hit_count, trial_ranks)
                if trial_fit > best_fit:
                    best_term, best_state, best_fit = term, (trial_hit_count, trial_ranks), trial_fit

            if best_term is not None:
                selected = selected | {best_term}
                current_hit_count, current_ranks = best_state
                current_fitness = best_fit
                continue

            # No addition improved fitness — try dropping a currently
            # selected term, in case it's a common word boosting a
            # competitor at least as much as it boosts a target (the
            # failure mode a pure "more keywords never hurts" assumption
            # misses).
            best_removed: Optional[str] = None
            for term in list(selected):
                trial_hit_count, trial_ranks = evaluate(selected - {term})
                trial_fit = fitness(trial_hit_count, trial_ranks)
                if trial_fit > current_fitness:
                    best_removed = term
                    best_state, best_fit = (trial_hit_count, trial_ranks), trial_fit
                    break

            if best_removed is not None:
                selected = selected - {best_removed}
                current_hit_count, current_ranks = best_state
                current_fitness = best_fit
                continue

            # Neither move improves anything within this tier — converged
            # (or stuck); stop this tier's loop and fall through to the
            # fallback tier (if this was the primary one) or return.
            break

    return {
        "keywords": sorted(selected),
        "achieved": current_hit_count == len(tables),
        "hit_count": current_hit_count,
        "missing_tables": [
            t["alias"] for t in tables if current_ranks[t["alias"]] > top_k
        ],
        # Final rank of every table (1-indexed; ceiling+1 if not found even
        # within the wide ceiling search) — lets a caller distinguish a
        # comfortable margin (rank 1-2 of a top_k=6 window) from a fragile
        # one (rank 6 of 6), which "achieved: True" alone can't tell apart.
        "ranks": dict(current_ranks),
        "iterations_used": iterations_used,
        "used_fallback_fields": used_fallback_fields,
    }
