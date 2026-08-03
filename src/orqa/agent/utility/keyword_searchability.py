"""Deterministic keyword-searchability check for the plan judge panel.

The plan judge's ``question_approval`` layer already asks an LLM to GUESS
whether a question's vocabulary would surface the right table(s) in a
reverse index (see ``conf/prompts/plan_judge.md`` Check 1) — but never
actually runs that search. This module does: given the plan's
``question_keywords`` and the actual tables it uses, query the portal's
real reverse index (``orqa.benchmark.index.DatasetIndex`` or
``orqa.benchmark.es_index.ESDatasetIndex`` — both expose the same
``search(keywords, top_k)`` contract) and check whether every table comes
back in the top-K results.

No-ops to an automatic pass when the index isn't available (not yet built,
Elasticsearch unreachable) or the plan has no keywords to check, so this
layer degrades the same way an unconfigured ``JudgePanel`` does — never
blocking a run on retrieval infrastructure being up.
"""

from typing import Any, Optional


def check_keyword_searchability(
    question_keywords: list[str],
    tables: list[dict],
    index: Optional[Any],
    top_k: int = 10,
) -> dict:
    """Whether keyword-searching ``question_keywords`` would surface every table.

    Args:
        question_keywords: The plan's retrieval keywords for its question.
        tables: ``[{"alias": str, "resource_id": str, "keywords": [str, ...]},
            ...]`` for every table THIS plan uses. ``keywords`` (the table's
            own analysis keywords, optional) is never used for matching —
            only as a last-resort fallback in the rejection feedback below,
            when the record can't be looked up in the index directly (see
            the ``record`` fallback note there for why it's second choice).
        index: A ``DatasetIndex``/``ESDatasetIndex`` (or ``None`` when no
            index is available for this portal).
        top_k: How many search results to check the tables against.

    Returns:
        ``{"approved": bool, "missing_tables": [alias, ...], "feedback": str}``.
    """
    if index is None or not question_keywords or not tables:
        return {"approved": True, "missing_tables": [], "feedback": ""}

    results = index.search(question_keywords, top_k=top_k)
    retrieved_ids = {r.resource_id for r in results}

    missing = [t for t in tables if t["resource_id"] not in retrieved_ids]
    if not missing:
        return {"approved": True, "missing_tables": [], "feedback": ""}

    missing_aliases = [t["alias"] for t in missing]
    ok_aliases = [t["alias"] for t in tables if t["resource_id"] in retrieved_ids]

    # The REAL indexed text for each missed table — its actual title/tags,
    # pulled straight from the reverse index's own record — NOT the LLM
    # table-analysis "keywords" field. A prior run showed those two can
    # diverge in ways that silently defeat retrieval even at a "perfect"
    # keyword match: a table analysed with the keyword "VisionZero" (one
    # camelCase word) is for-real indexed under the tag "vision zero" (two
    # words) — BM25 tokenizes each into different tokens, so neither side
    # ever matches the other, and every correction round just re-guesses
    # synonyms of the WRONG word split with no way to discover the real one.
    # Falls back to the analysis keywords only when the record itself can't
    # be looked up (e.g. removed from the portal since analysis ran).
    vocab_bits = []
    for t in missing:
        record = None
        try:
            record = index.get(t["resource_id"])
        except Exception:
            pass
        if record:
            bit = f"{t['alias']}'s real indexed title: \"{record.get('title', '')}\""
            tags = record.get("tags") or []
            if tags:
                bit += f", tags: {', '.join(tags)}"
            vocab_bits.append(bit)
        elif t.get("keywords"):
            vocab_bits.append(
                f"{t['alias']}'s analysed keywords (record lookup failed, so "
                f"exact indexed wording is unavailable): {', '.join(t['keywords'])}"
            )
    vocab_text = (" " + "; ".join(vocab_bits) + ".") if vocab_bits else ""

    # What ACTUALLY surfaced instead, straight from the index — further
    # ground truth on the real wording (title casing, abbreviations, ...)
    # that wins under this ranking, for the planner to pattern-match against
    # directly rather than guess blind.
    top_titles = [r.title for r in results[:5] if getattr(r, "title", None)]
    competing_text = (
        f" The top {len(top_titles)} result(s) retrieved instead: "
        + "; ".join(f'"{t}"' for t in top_titles) + "."
    ) if top_titles else ""

    # A multi-table plan runs exactly ONE combined search — this same
    # question_keywords list, checked against this same top-K window (K
    # itself scales with table count, see the caller's adaptive top_k) —
    # never one search per table. That means fixing the missed table(s)
    # below is a BALANCING problem, not an isolated one: a revision that
    # over-fits their vocabulary at the expense of the keywords currently
    # carrying the table(s) that DO already retrieve can fix one and break
    # another. Spelled out explicitly whenever there's more than one table
    # in play, since nothing else here would tell the planner this search
    # is shared rather than per-table.
    scope_text = ""
    if len(tables) > 1:
        scope_text = (
            f" This plan uses {len(tables)} tables sharing ONE combined "
            f"search, so its question_keywords must surface ALL of them "
            f"within this same top {top_k} results — "
        )
        scope_text += (
            f"{', '.join(ok_aliases)} already retrieve(s) fine with the "
            f"current keywords, so don't lose that while fixing "
            f"{', '.join(missing_aliases)}."
            if ok_aliases else
            "balance vocabulary across every table below rather than "
            "optimizing only one at the expense of the others."
        )

    return {
        "approved": False,
        "missing_tables": missing_aliases,
        "feedback": (
            f"Searching the reverse index with this question's keywords "
            f"({', '.join(question_keywords)}) does not surface "
            f"{', '.join(missing_aliases)} within the top {top_k} results."
            f"{vocab_text}{competing_text}{scope_text}"
        ),
    }
