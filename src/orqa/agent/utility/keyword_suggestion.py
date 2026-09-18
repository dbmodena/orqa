"""Deterministic keyword suggestion for a table group's retrievability.

The plan judge's retrievability check (see
``orqa.agent.utility.retrievability_gate``) verifies a plan's QUESTION
AFTER the planner has already written it — a rejected plan only learns it
guessed wrong, then guesses again, burning an LLM correction round each
time (see ``StatementOrchestrator._judge_plans``).

This module inverts that: given the exact table group a plan will use,
BEFORE any plan is drafted, it searches the tables' real indexed vocabulary
— title, resource name, tags, publisher AND the tables' own column schema,
all competing in the same search rather than columns being held back as a
rescue tier — and verifies every trial against the same reverse index the
judge will use.
That index may be lexical BM25 or the configured lexical/semantic hybrid;
either way, an achieved suggestion is correct by construction for the active
backend and target top-K. A miss only means the bounded candidate search did
not find a working combination — with semantic retrieval, synonyms outside
the literal metadata vocabulary may still work, so a miss is not a proof
that every possible natural-language phrasing is unretrievable.

The search is a greedy local search whose fitness function is the real
``index.search()`` call, not an approximation. Fitness is rank-aware, not a
binary "in top-K or not": a term that moves a table from rank 40 to rank 12
(still outside a top-6 window) is progress a binary signal would discard.
Hybrid indexes can expose ``search_many``; additions/removals are then query-
embedded in a batch so semantic optimization does not make one provider call
per candidate term.

The result is BUDGETED, not minimal: at most four keywords per table (see
``_MAX_KEYWORDS_PER_TABLE``), because the planner has to weave every one of
them into a question in natural prose. Within that budget the search never
trims a term for being merely redundant, and a longer set is usually the
safer one — it tends to rank at 1 rather than at the top_k boundary the
climb stops on.

The budget is why the search works from both ends. The greedy climb grows a
set one term at a time and naturally stays small, but it needs ONE single
term to be a strict improvement before it can move at all, which a purely
conjunctive table never offers: one of hundreds of same-titled resources,
told apart only by a date plus a qualifier plus a publisher, each term
individually shared with every sibling. For those, a fallback pass starts
from the tables' entire vocabulary — far past the budget, but informative
about which terms carry the group — ranks its terms by what their absence
costs, and cuts back to the budget's worth of the most load-bearing ones.

A THIRD tier, ``_exhaustive_rescue``, escalates only when both of the above
still fall short (including when the full-vocabulary seed itself misses —
the one case the fallback above has nothing left to try). Neither the climb
nor the fallback is exhaustive: the climb can get stuck short of the true
optimum (it only ever takes a move that is a STRICT improvement on the
previous one), and the fallback sweeps budget-sized prefixes of exactly ONE
ordering, not every combination. The rescue instead scores every candidate
term standalone, keeps only the highest-signal ``_EXHAUSTIVE_POOL_SIZE`` of
them, and evaluates literally every combination of that reduced pool up to
the keyword budget — genuinely exhaustive over that pool, bounded by
construction regardless of how large the raw vocabulary is. Each stage is
ONE batched ``search_many`` round-trip (hundreds to low thousands of query
texts embedded together), not one call per combination, which is what keeps
it affordable enough to run as a rare escalation.
"""

import itertools
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

# Schema bookkeeping tokens that almost every table's header carries
# ("id", "index", pandas' "Unnamed: 0", ...). Like _STOPWORDS above they
# are never what distinguishes ONE dataset from the rest of the corpus,
# and unlike a real column name they read as a machine artifact rather
# than a word the planner can weave into a natural-language question.
_COLUMN_NOISE = frozenset({
    "id", "ids", "idx", "index", "key", "col", "cols", "column", "columns",
    "field", "fields", "unnamed", "nan", "null", "none", "row", "rows",
    "record", "records", "number", "num", "no",
})

# Per-table cap on how many column-derived terms enter the candidate pool.
# Column vocabulary is open-ended in a way title/tags vocabulary is not (a
# wide table with per-column descriptions can contribute hundreds of
# terms), and every extra candidate is a real index.search() — or, on a
# hybrid index, a real query embedding — in EVERY greedy round. The cap
# keeps columns first-class without letting one 80-column table decide the
# cost of the whole search.
_MAX_COLUMN_TERMS = 25

# Hard cap on the returned keyword count, per table in the group. The
# result is handed to the planner as an anchor every plan's question must
# weave into natural prose (see QueryPlanner._render_retrievable_keywords),
# and that is only writable while the list stays short: a question can
# carry five terms about a table, not twenty. The search is free to be
# non-minimal WITHIN this budget — nothing trims a merely redundant term —
# but never to exceed it, so a group of N tables returns at most 5N terms.
_MAX_KEYWORDS_PER_TABLE = 5

# How far past top_k to look when ranking a still-missing table, so the
# greedy search can see incremental progress (rank 40 -> rank 12) even
# while a table is still outside the window it ultimately needs to reach.
# Not itself a pass/fail bound — only ``top_k`` decides that.
#
# The floor is deliberately far wider than any plausible top_k, because
# every rank past the ceiling collapses to ONE sentinel value: too narrow a
# ceiling doesn't just lose precision, it erases the gradient the forward
# search climbs. On a portal full of near-duplicates (the UK CKAN corpus
# has 1426 resources sharing the title "Organogram of Staff Roles &
# Salaries", tags empty, no published column schema) every single term is
# shared with hundreds of siblings, so the best first step sits at rank
# ~184 — real, usable progress that a 60-wide ceiling reported as "not
# found", identically to a term at rank 3000. The search then saw a
# perfectly flat landscape, could not take a first step, and gave up after
# one round with no keywords at all.
#
# A deeper look is close to free on the backends this runs against: the
# built-in BM25 index scores and sorts the whole matching set regardless of
# top_k (only the result-construction loop truncates), and the hybrid index
# already pulls the ENTIRE corpus from its lexical half before fusing, so
# top_k only decides where the fused list is cut.
_RANK_CEILING_MULTIPLIER = 8
_RANK_CEILING_FLOOR = 500

# ``_exhaustive_rescue``'s own bounds (see the module docstring's third
# tier). Independent of ``max_iterations``/the budget above — those govern
# the greedy climb's round count and the RESULT size respectively, these
# govern how big a combinatorial search the rescue is allowed to run.
#
# Only the ``_EXHAUSTIVE_POOL_SIZE`` highest-standalone-signal terms are
# even considered — combinations grow combinatorially in pool size, so this
# is what keeps the rescue bounded regardless of how large the raw
# candidate vocabulary is (a wide table's column pool alone can be
# ``_MAX_COLUMN_TERMS`` long, well past what's combinatorially feasible).
_EXHAUSTIVE_POOL_SIZE = 12

# Hard circuit breaker on total combinations evaluated, independent of pool
# size/budget math — so a large multi-table budget (max_keywords_per_table
# * len(tables)) can never turn "exhaustive" into "unbounded". Combinations
# are tried smallest-size-first (see _exhaustive_rescue), so hitting this
# cap means larger sizes are simply never reached rather than the rescue
# failing outright.
_EXHAUSTIVE_MAX_TRIALS = 4000


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


def _candidate_pools(
    record: dict,
    actual_columns: Optional[list[str]] = None,
    link_columns: Optional[list[str]] = None,
    max_column_terms: int = _MAX_COLUMN_TERMS,
) -> tuple[list[str], list[str], list[str]]:
    """Return one table's (metadata, link-column, other-column) terms.

    All three pools feed the SAME greedy search — the split is a stable
    ordering used to break exact fitness ties (see the candidate ordering
    in :func:`suggest_retrievable_keywords`), not a priority ladder that
    withholds a pool until another is exhausted.

    ``link_columns`` are the columns this table is JOINED or UNIONED on
    (the group's ``involved_cols``). They come first because a term drawn
    from them is the one candidate that can help every table at once: the
    group shares ONE combined search, so an ordinary term lifts its own
    table while pushing the others' competitors around, whereas a join key
    or a union's shared column is by construction vocabulary the whole
    group has in common. They are also what the question is actually about
    ("...by borough", "...per year"), so anchoring on them reads naturally
    in the prose the planner has to write.

    ``actual_columns`` comes from the already-loaded DataFrame and is
    listed before the portal's own column metadata: it is the schema the
    generated query will really run against, and CKAN records frequently
    publish no column schema at all even though the downloaded table has
    one.
    """
    fields = _record_field_texts(record)
    metadata = _terms_from_text(
        fields["title"], fields["resource_name"], fields["tags"], fields["publisher"]
    )
    # Column NAMES and labels ahead of column DESCRIPTIONS: a name is the
    # identifying handle a question gets phrased around ("borough",
    # "casualty"), while a description is prose long enough to swamp the
    # pool — and the cap below — with incidental terms.
    names, labels, descriptions = [], [], []
    for col in record.get("columns") or []:
        for key, bucket in (
            ("name", names), ("label", labels), ("description", descriptions)
        ):
            value = col.get(key)
            if value:
                bucket.append(str(value))
    link = [
        term
        for term in _terms_from_text(
            " ".join(str(column) for column in link_columns or [])
        )
        if term not in _COLUMN_NOISE
    ]
    columns = [
        term
        for term in _terms_from_text(
            " ".join(str(column) for column in actual_columns or []),
            " ".join(names),
            " ".join(labels),
            " ".join(descriptions),
        )
        if term not in metadata and term not in link and term not in _COLUMN_NOISE
    ][:max_column_terms]
    return metadata, link, columns


def _exhaustive_rescue(
    n_targets: int,
    all_terms: list[str],
    keyword_budget: int,
    evaluate_many,
    fitness,
    current_hit_count: int,
    current_ranks: dict[str, int],
    current_fitness: tuple[int, int, int],
) -> tuple[Optional[set[str]], int, dict[str, int], tuple[int, int, int], int]:
    """Last-resort escalation — see the module docstring's third tier.

    Two batched stages, run ONLY when the greedy climb and the seed-and-cut
    fallback (the caller's own earlier attempts) both still fall short:

    1. Score every candidate term STANDALONE (one batched round) and keep
       only the ``_EXHAUSTIVE_POOL_SIZE`` highest-signal ones — bounds the
       combinatorics regardless of how large ``all_terms`` is.
    2. Evaluate literally every combination of that reduced pool, smallest
       size first, up to ``keyword_budget`` — genuinely exhaustive over the
       pool, not a local search — stopping before ``_EXHAUSTIVE_MAX_TRIALS``
       total combinations (a circuit breaker; for a large multi-table
       budget this can mean larger sizes are never reached, so — like the
       greedy climb and the fallback before it — a miss here still isn't a
       proof nothing within budget would have worked, only that this bounded
       search didn't find it).

    Returns ``(best_combo_or_None, hit_count, ranks, fitness,
    iterations_used)``. ``best_combo_or_None`` is ``None`` when nothing beat
    ``current_fitness``, so the caller's existing result stands unchanged.
    """
    if not all_terms:
        return None, current_hit_count, current_ranks, current_fitness, 0

    iterations_used = 1
    standalone_states = evaluate_many([{term} for term in all_terms])
    ranked = sorted(
        zip(all_terms, standalone_states),
        key=lambda pair: fitness(*pair[1]),
        reverse=True,
    )
    pool = [term for term, _ in ranked[:_EXHAUSTIVE_POOL_SIZE]]

    best_combo: Optional[set[str]] = None
    best_hit_count, best_ranks, best_fit = current_hit_count, current_ranks, current_fitness

    trials_used = 0
    for size in range(1, min(keyword_budget, len(pool)) + 1):
        combos = list(itertools.combinations(pool, size))
        if trials_used + len(combos) > _EXHAUSTIVE_MAX_TRIALS:
            break
        trials_used += len(combos)
        states = evaluate_many([set(combo) for combo in combos])
        iterations_used += 1
        for combo, (hit_count, ranks) in zip(combos, states):
            fit = fitness(hit_count, ranks)
            if fit > best_fit:
                best_combo, best_hit_count, best_ranks, best_fit = set(combo), hit_count, ranks, fit
        if best_hit_count == n_targets:
            break

    return best_combo, best_hit_count, best_ranks, best_fit, iterations_used


def suggest_retrievable_keywords(
    tables: list[dict],
    index: Optional[Any],
    top_k: int,
    max_iterations: int = 20,
    max_column_terms: int = _MAX_COLUMN_TERMS,
    max_keywords_per_table: int = _MAX_KEYWORDS_PER_TABLE,
) -> dict:
    """Greedily find a keyword set that surfaces every table within top_k.

    Args:
        tables: ``[{"alias": str, "resource_id": str, "columns": [str, ...]},
            ...]`` for the group a plan needs. ``columns`` is optional and
            carries the loaded table's real column names — a first-class
            source of candidate keywords alongside title/tags/publisher (the
            index weights the ``columns`` field at 2.0, just under tags), and
            the only schema vocabulary available at all when portal metadata
            omits its column list, as CKAN records frequently do.
            ``link_columns`` (optional) names the columns the group is
            JOINED or UNIONED on — the caller's ``involved_cols`` for this
            alias. Their vocabulary is tried ahead of everything else and
            wins exact fitness ties, and on a multi-table group it also
            seeds the search (see the warm start below). When no caller
            supplies it, the column names the tables literally share stand
            in for it.
        index: A ``DatasetIndex``/``ESDatasetIndex`` (or ``None`` when
            unavailable for this portal — no-ops to an empty, unachieved
            result rather than raising, same degradation convention as
            ``check_keyword_searchability``).
        top_k: The target window every table must land inside.
        max_iterations: Cap on the greedy climb's add/remove steps — so a
            table group with no good shared vocabulary fails fast rather
            than hammering the index indefinitely. The climb normally
            exits earlier, either on achieving top_k for every table or on
            converging (no single add or removal improves fitness), and is
            skipped altogether when the one-search seed check already
            achieved it.
        max_column_terms: Per-table cap on column-derived candidates (see
            ``_MAX_COLUMN_TERMS``), bounding the per-round trial cost that
            a very wide table would otherwise impose.
        max_keywords_per_table: Hard cap on the RESULT, multiplied by the
            number of tables — a two-table group may return at most twice
            this many keywords. See ``_MAX_KEYWORDS_PER_TABLE``.

    Returns:
        ``{"keywords": [str, ...], "achieved": bool, "hit_count": int,
        "missing_tables": [alias, ...], "ranks": {alias: int},
        "iterations_used": int, "column_keywords": [str, ...]}``.
        ``achieved`` is True only when EVERY table landed in top_k;
        otherwise ``keywords`` is still the best set the search found,
        useful as a starting point even when incomplete. ``ranks`` is each
        table's final 1-indexed rank (or a large sentinel if not found even
        within a wide ceiling search) — a table can be ``achieved`` while
        sitting right at the top_k boundary (rank == top_k, the most
        fragile possible margin) rather than comfortably inside it; the
        search prefers a lower (safer) rank as a tie-break once hit_count
        is equal, but ``ranks`` lets the caller see the actual margin
        achieved rather than only the pass/fail bit. The set is verified
        and BUDGETED, not minimal: it never exceeds
        ``max_keywords_per_table * len(tables)``, but nothing trims a term
        inside that budget just for being redundant. ``column_keywords``
        is the subset of ``keywords`` that no table's title/tags/publisher
        could have supplied — i.e. what the column schema contributed to
        the winning combination.
    """
    if index is None or not tables:
        return {
            "keywords": [],
            "achieved": False,
            "hit_count": 0,
            "missing_tables": [t["alias"] for t in tables],
            "iterations_used": 0,
            "column_keywords": [],
        }

    # What the group joins/unions on, as the caller's match constraint
    # names it (``involved_cols``). There is deliberately NO fallback that
    # guesses this from the column names the tables happen to share: on
    # this corpus that set is the ENTIRE narrower schema for 79 of 105
    # real groups — a union shares all its columns by definition — so
    # "prioritize the link columns" would silently become "promote every
    # column above the title", which measurably retrieves worse. Absent a
    # caller-supplied constraint, the pools below simply carry no link
    # tier and the search behaves as it did before.
    link_by_alias = {
        t["alias"]: [str(c) for c in (t.get("link_columns") or [])]
        for t in tables
    }

    metadata_pools: dict[str, list[str]] = {}
    link_pools: dict[str, list[str]] = {}
    column_pools: dict[str, list[str]] = {}
    for t in tables:
        record = index.get(t["resource_id"])
        (
            metadata_pools[t["alias"]],
            link_pools[t["alias"]],
            column_pools[t["alias"]],
        ) = _candidate_pools(
            record or {},
            list(t.get("columns") or []),
            link_by_alias[t["alias"]],
            max_column_terms,
        )

    # Vocabulary ONLY a column could have supplied: a term can sit in one
    # table's column pool and another table's title, so this is a
    # corpus-wide difference rather than a per-table one — it exists purely
    # to report, after the fact, what the schema contributed.
    column_only = {
        term
        for pools in (link_pools, column_pools)
        for pool in pools.values()
        for term in pool
    } - {
        term for pool in metadata_pools.values() for term in pool
    }

    ceiling = max(top_k * _RANK_CEILING_MULTIPLIER, _RANK_CEILING_FLOOR)

    def states_from_results(
        result_batches: list[list[Any]],
    ) -> list[tuple[int, dict[str, int]]]:
        states = []
        for results in result_batches:
            order = {r.resource_id: i + 1 for i, r in enumerate(results)}
            ranks = {
                t["alias"]: order.get(t["resource_id"], ceiling + 1)
                for t in tables
            }
            hit_count = sum(1 for rank in ranks.values() if rank <= top_k)
            states.append((hit_count, ranks))
        return states

    def evaluate_many(
        keyword_sets: list[set[str]],
    ) -> list[tuple[int, dict[str, int]]]:
        """Evaluate trials together when the index can batch query embeddings."""
        if not keyword_sets:
            return []
        empty_state = (0, {t["alias"]: ceiling + 1 for t in tables})
        states: list[Optional[tuple[int, dict[str, int]]]] = [None] * len(keyword_sets)
        nonempty_positions = [i for i, keywords in enumerate(keyword_sets) if keywords]
        if nonempty_positions:
            queries = [sorted(keyword_sets[i]) for i in nonempty_positions]
            if hasattr(index, "search_many"):
                batches = index.search_many(queries, top_k=ceiling)
            else:
                batches = [index.search(query, top_k=ceiling) for query in queries]
            for position, state in zip(nonempty_positions, states_from_results(batches)):
                states[position] = state
        return [state if state is not None else empty_state for state in states]

    def evaluate(keywords: set[str]) -> tuple[int, dict[str, int]]:
        """(hit_count within top_k, {alias: rank within ceiling or ceiling+1})."""
        return evaluate_many([keywords])[0]

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

    keyword_budget = max(1, max_keywords_per_table * len(tables))

    # The group's join/union vocabulary, offered to the climb's first
    # round as a single compound opening move (see below). A multi-table
    # group gets ONE combined search, so a term describing only one table
    # spends the shared budget pushing the other tables' competitors
    # around, while a join key or a union's shared columns are by
    # construction vocabulary the whole group has in common — and the
    # climb, judging one term at a time against whichever table is
    # currently missing, would rarely assemble them by itself.
    link_start = list(dict.fromkeys(
        term for t in tables for term in link_pools[t["alias"]]
    ))[:keyword_budget] if len(tables) > 1 else []

    for _ in range(max_iterations):
        if current_hit_count == len(tables):
            break
        iterations_used += 1

        missing_aliases = [
            t["alias"] for t in tables
            if current_ranks[t["alias"]] > top_k
        ]
        # Everything the still-missing tables can offer, metadata AND
        # column vocabulary, evaluated together in ONE batched round.
        # Columns are deliberately not a rescue tier that only opens once
        # title/tags/publisher has given up: a column name is often the
        # only token the whole group shares (three tables about different
        # things all carrying "borough"), and the term that WINS a round
        # here is decided by the real index, not by which field it came
        # from. Metadata terms are merely laid out first, across all
        # missing aliases, so that the strictly-greater comparison below
        # keeps the metadata candidate when two are EXACTLY as good — the
        # index weights title/tags above columns (see FIELD_WEIGHTS), and
        # a title term reads more naturally in the question the planner
        # then has to weave it into.
        # At the budget the climb can no longer grow, but it can still
        # SWAP: the removal sweep below drops a term that is carrying its
        # weight badly, which frees a slot for a better one next round.
        untried = list(dict.fromkeys(
            term
            for pools in (link_pools, metadata_pools, column_pools)
            for alias in missing_aliases
            for term in pools[alias]
            if term not in selected
        )) if len(selected) < keyword_budget else []

        # This round's candidate moves. Normally one term each; on the
        # opening round the join/union vocabulary is offered as ONE move
        # as well, so it is judged as the group-wide anchor it is rather
        # than term by term. It competes on the same fitness as every
        # single term, and is listed LAST precisely so it does NOT take
        # ties: it spends several of the budget's slots at once, so it
        # has to actually rank the group better than the best single term
        # to be worth them. Without that, a join on columns the search
        # did not need ("comments", "dateofsampling") would tie the
        # one-term answer and win, burning three slots and putting three
        # more words into the question for nothing.
        additions: list[set[str]] = [{term} for term in untried]
        if link_start and not selected:
            additions.append(set(link_start))

        best_addition: Optional[set[str]] = None
        best_state: Optional[tuple[int, dict[str, int]]] = None
        best_fit = current_fitness
        trial_states = evaluate_many(
            [selected | addition for addition in additions]
        )
        for addition, (trial_hit_count, trial_ranks) in zip(additions, trial_states):
            trial_fit = fitness(trial_hit_count, trial_ranks)
            if trial_fit > best_fit:
                best_addition = addition
                best_state = (trial_hit_count, trial_ranks)
                best_fit = trial_fit

        if best_addition is not None:
            selected = selected | best_addition
            current_hit_count, current_ranks = best_state
            current_fitness = best_fit
            continue

        # No addition improved fitness — try dropping a currently
        # selected term, in case it's a common word boosting a
        # competitor at least as much as it boosts a target.
        best_removed: Optional[str] = None
        removal_terms = sorted(selected)
        removal_states = evaluate_many(
            [selected - {term} for term in removal_terms]
        )
        for term, (trial_hit_count, trial_ranks) in zip(
            removal_terms, removal_states
        ):
            trial_fit = fitness(trial_hit_count, trial_ranks)
            if trial_fit > best_fit:
                best_removed = term
                best_state = (trial_hit_count, trial_ranks)
                best_fit = trial_fit

        if best_removed is not None:
            selected = selected - {best_removed}
            current_hit_count, current_ranks = best_state
            current_fitness = best_fit
            continue

        # Neither an addition nor a removal improves anything, over the
        # metadata AND column vocabulary of every still-missing table —
        # converged (or stuck at a local optimum); stop rather than keep
        # paying for index calls that go nowhere.
        break

    # ── Seed and cut ──────────────────────────────────────────────────
    # The climb grows from the empty set, so it needs ONE single term to
    # be a strict improvement before it can move at all. A table whose
    # identity is purely conjunctive has no such term: one of hundreds of
    # same-titled resources, told apart only by a date plus a qualifier
    # plus a publisher, each individually shared with every sibling. For
    # those, start from the opposite end — the tables' entire vocabulary,
    # a query too long to ever return but informative about which terms
    # carry the group — and cut it down to the budget.
    #
    # Runs only when the climb fell short, so the ordinary path pays
    # nothing, and stops after a SINGLE search when even the full
    # vocabulary misses: there is then nothing to cut down to.
    if current_hit_count < len(tables):
        seed = list(dict.fromkeys(
            term
            for pools in (link_pools, metadata_pools, column_pools)
            for t in tables
            for term in pools[t["alias"]]
        ))
        if seed:
            seed_hit_count, seed_ranks = evaluate(set(seed))
            iterations_used += 1

            if seed_hit_count == len(tables):
                # Rank every term by what its ABSENCE costs (one batched
                # round of leave-one-out trials), most load-bearing
                # first, then evaluate the budget's worth of prefixes of
                # that order and keep the best. Prefixes are swept
                # exhaustively rather than bisected because adding a term
                # can worsen a ranking as easily as improve it — "this
                # prefix achieves" is not monotone in prefix length, so a
                # binary search over it could converge on nothing.
                loss_states = evaluate_many([set(seed) - {term} for term in seed])
                iterations_used += 1
                loss = {
                    term: fitness(hit_count, ranks)
                    for term, (hit_count, ranks) in zip(seed, loss_states)
                }
                # Ascending: the term whose removal scores WORST is doing
                # the most work, so it comes first.
                ordered = sorted(seed, key=lambda term: loss[term])

                prefix_states = evaluate_many([
                    set(ordered[:n])
                    for n in range(1, min(keyword_budget, len(ordered)) + 1)
                ])
                iterations_used += 1
                # Strictly-greater keeps the SHORTEST prefix among equals,
                # since prefixes are evaluated shortest-first. A full-
                # vocabulary seed that achieves does NOT guarantee any
                # prefix within budget does; when none beats the climb's
                # result, that result stands.
                for n, (trial_hit_count, trial_ranks) in enumerate(prefix_states, 1):
                    trial_fit = fitness(trial_hit_count, trial_ranks)
                    if trial_fit > current_fitness:
                        selected = set(ordered[:n])
                        current_hit_count, current_ranks = trial_hit_count, trial_ranks
                        current_fitness = trial_fit

            # ── Exhaustive rescue (module docstring's third tier) ──────
            # Escalates only when the climb AND the fallback above both
            # still fall short — including when the full-vocabulary seed
            # itself missed (seed_hit_count < len(tables)), the one case
            # the fallback has nothing left to try for, since a smaller,
            # more focused combination can rank better than the noisy
            # full-vocabulary query on a semantic/hybrid backend even
            # though it's a subset of it.
            if current_hit_count < len(tables):
                rescued, r_hit_count, r_ranks, r_fitness, r_iterations = _exhaustive_rescue(
                    len(tables), seed, keyword_budget, evaluate_many, fitness,
                    current_hit_count, current_ranks, current_fitness,
                )
                iterations_used += r_iterations
                if rescued is not None:
                    selected = rescued
                    current_hit_count, current_ranks, current_fitness = (
                        r_hit_count, r_ranks, r_fitness
                    )

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
        # Which of the winning keywords came from column vocabulary alone
        # (see ``column_only``) — lets the caller see whether the schema
        # was what made this group retrievable, without re-deriving it.
        "column_keywords": sorted(term for term in selected if term in column_only),
    }
