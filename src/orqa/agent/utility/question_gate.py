"""Deterministic gate for a stage-1 question — no LLM, no plan.

A question written by the question agent (see ``orqa.agent.agents.
QuestionStage``) is checked here, cheapest check first, before any judge is
paid for:

1. **leak** — a raw column identifier pasted into the prose.
2. **retrieval** — the retriever panel's majority vote (lexical gate,
   semantic, hybrid RRF) over the question, via
   ``retrievability_gate.check_question_retrievability``.

The verified retrieval anchor (see ``orqa.agent.utility.keyword_suggestion``)
is handed to the question writer as vocabulary to build the question around,
and to the retrieval check as the lexical gate's first query: a question that
holds a table's WHOLE anchor (e.g. ``[belfast, ni]``) passes the lexical vote,
because that exact query already surfaced the table (see
``retrievability_gate._question_terms_vote``). The gate does not REQUIRE the
anchor: a question that words the table differently still passes when its own
words find it.

Anchor terms are compared on ``benchmark.index.tokenize`` tokens — the same
tokenizer the index applies — so "present in the question" means exactly
"would be a query term for the index".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ...benchmark.index import tokenize
from ...benchmark.retrieval_panel import RetrieverPanel
from .retrievability_gate import RetrievalContract, check_question_retrievability


def _normalized(term: str) -> str:
    return " ".join(tokenize(term))


def covered_terms(question: str, anchors: dict[str, list[str]]) -> list[str]:
    """The anchor terms that ARE in the question, deduplicated, in anchor
    order — the deterministic ``question_keywords`` of a stage-1 question
    (never an LLM-authored list that could drift from the prose). A term is
    present when EVERY one of its tokens is among the question's tokens (the
    index is bag-of-words, so adjacency is not required)."""
    question_tokens = set(tokenize(question))
    seen: set[str] = set()
    out: list[str] = []
    for terms in anchors.values():
        for term in terms:
            tokens = tokenize(term)
            key = _normalized(term)
            if not tokens or key in seen:
                continue
            if all(token in question_tokens for token in tokens):
                seen.add(key)
                out.append(term)
    return out


@dataclass
class QuestionGateResult:
    approved: bool
    # None when approved; else which check stopped it: "leak" | "retrieval".
    # Checks run in that order and the first miss short-circuits.
    failed_at: Optional[str] = None
    feedback: str = ""
    retrieval: Optional[dict] = None


_RETRIEVAL_FIX = (
    "Reword the QUESTION so each missed table's real vocabulary appears "
    "naturally in its own prose, in the exact wording given (a topic worded "
    "differently, e.g. one merged word instead of the indexed multi-word "
    "term, will not match even if it means the same thing). Keep it a "
    "single plain-language question, the way an average user would ask it: "
    "never paste a column header, code or identifier — say the idea in "
    "everyday words."
)


def gate_question(
    question: str,
    *,
    anchors: Optional[dict[str, list[str]]] = None,
    contract: Optional[RetrievalContract] = None,
    panel: Optional[RetrieverPanel] = None,
    per_table_top_k: Optional[int] = None,
    leak_check: Optional[Callable[[str], Optional[str]]] = None,
) -> QuestionGateResult:
    """Run the deterministic checks on ``question``, cheapest first.

    ``anchors``: ``{alias: [term, ...]}`` — the verified vocabulary the
    question was written from (per table for a multi-table group); used to
    make a retrieval miss's feedback name concrete words.
    ``per_table_top_k``: see ``check_question_retrievability``.
    ``leak_check``: ``question -> reason | None`` (kept a parameter so this
    module stays free of the agents package).

    With no contract / panel the retrieval check is a no-op, the same
    degradation convention as the retrievability gate itself.
    """
    if not (question or "").strip():
        return QuestionGateResult(False, "leak", "The question is empty.")

    if leak_check is not None:
        reason = leak_check(question)
        if reason:
            return QuestionGateResult(
                False,
                "leak",
                f"The question {reason}. Write it in plain everyday words "
                "with no column identifiers.",
            )

    retrieval = check_question_retrievability(
        question, contract, panel, anchors or None, per_table_top_k
    )
    if not retrieval["approved"]:
        return QuestionGateResult(
            False,
            "retrieval",
            f"{retrieval['feedback']}\n{_RETRIEVAL_FIX}".strip(),
            retrieval=retrieval,
        )
    return QuestionGateResult(True, retrieval=retrieval)
