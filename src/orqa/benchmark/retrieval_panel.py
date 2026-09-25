"""The retriever panel behind the retrievability contract.

A question is only *retrievable* if it would actually be found — not by one
search method's opinion, but by AGREEMENT across independently-built
retrievers: lexical BM25 over the raw question text, a dense/semantic
ranking of the embedded question, and BM25 over keywords an independent LLM
call extracts from the question (the same call the benchmark solver itself
makes — see ``orqa.agent.agents.BenchmarkSolver.BenchmarkSolverAgent.
generate_keywords``). See ``orqa.agent.utility.retrievability_gate`` for how
the panel's vote becomes a pass/fail gate.

Ranks are those of the exact gold table: a retriever finds a table when the
table itself — not merely another file of its dataset — is in the window.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Optional, Sequence

# How far into a ranking a retriever is searched before giving up. Wider
# than any plausible top_k so a rank just outside the pass/fail window is
# still a real number for diagnostics, not a flat "not found" sentinel.
logger = logging.getLogger(__name__)

RANK_CEILING = 200

# Every retriever a panel can hold, in the order they vote and are reported.
#   lexical_question — BM25 over the raw question text (additive)
#   dense_question   — embedding search over the question
#   llm_keywords     — the solver's own LLM keyword extraction, searched
#                      lexically (AND-matched, portal-restricted, when the
#                      panel is given a portal view as its ``keyword_index``)
#   hybrid_rrf       — Reciprocal Rank Fusion of llm_keywords + dense_question,
#                      the fusion LakeGen's Solr hybrid mode uses
RETRIEVER_NAMES = ("lexical_question", "dense_question", "llm_keywords", "hybrid_rrf")


def _absent_rank(ranking: Sequence[str]) -> int:
    """The rank reported for a gold id that is NOT in ``ranking``: strictly
    beyond any window a caller can ask for. ``len(ranking) + 1`` — what this
    used to be — sits INSIDE the window whenever a ranking is shorter than
    ``top_k`` (a lexical AND search that matches only a handful of records,
    an RRF list, a tiny corpus), so a table the retriever never returned
    would pass."""
    return max(len(ranking), RANK_CEILING) + 1


def _unpack_keyword_result(result: Any) -> tuple[list[str], dict]:
    """Normalize a keyword extractor's return into ``(keywords, usage)``.

    Accepts either a plain ``[str, ...]`` (or ``{"keywords": [...]}``) or the
    ``(payload, usage)`` tuple ``BenchmarkSolverAgent.generate_keywords``
    returns, so a caller may pass that method directly.
    """
    usage: dict = {}
    payload = result
    if isinstance(result, tuple) and len(result) == 2:
        payload, usage = result
    if isinstance(payload, dict):
        keywords = list(payload.get("keywords") or [])
    else:
        keywords = list(payload or [])
    return keywords, usage or {}


def make_llm_keyword_extractor(config_path) -> Callable[[str], Any]:
    """A ``keyword_extractor`` bound to a fresh ``BenchmarkSolverAgent`` —
    the solver's OWN keyword-generation call (``conf/prompts/
    benchmark_search_keywords.md``), reused here so the retrievability gate
    checks against the exact vocabulary a real solver would search with.

    Imported lazily to avoid a module-load-time dependency between
    ``orqa.benchmark`` and ``orqa.agent`` — mirrors ``orqa.benchmark.index.
    load_index``'s lazy hybrid-retrieval import.
    """
    from ..agent.agents.BenchmarkSolver import BenchmarkSolverAgent

    return BenchmarkSolverAgent(config_path).generate_keywords


class RetrieverPanel:
    """Builds and queries whichever retrievers are actually available.

    Args:
        lexical_index: A ``DatasetIndex``/``ESDatasetIndex``-compatible
            backend (``search(keywords, top_k)`` -> ``[SearchResult, ...]``).
            When ``hybrid_index`` is given, its own ``lexical_index`` is used
            instead, so ``lexical_question`` always searches plain BM25 even
            when the caller's main index is the hybrid fusion.
        hybrid_index: A ``HybridDatasetIndex`` exposing ``semantic_search_many``,
            or ``None`` to skip the ``dense_question`` retriever.
        keyword_extractor: ``question -> keywords`` (or ``(payload, usage)``,
            see ``_unpack_keyword_result``), or ``None`` to skip
            ``llm_keywords``. See ``make_llm_keyword_extractor``.
        retrievers: Which retrievers VOTE (names from ``RETRIEVER_NAMES``);
            ``None`` — the default — is every retriever that is available.
            A named retriever that is unavailable (no hybrid index, no
            keyword extractor) is skipped; if that leaves none, every
            available one votes instead (with a warning) rather than a gate
            that can approve nothing.
        keyword_index: The index ``llm_keywords`` searches. Defaults to the
            lexical index; pass a ``PortalIndexView`` to make that retriever
            answer the way the portal's AND-matching keyword search does.
        universe: Resource ids the target search engine actually holds. Every
            retriever's ranking is restricted to them, so a retriever that
            ranks a wider collection (the local embedding cache covers the
            whole normalized metadata; a Solr core may hold only part of it)
            competes on the same field the real portal does. ``None`` keeps
            every ranked id.
        rrf_k / rrf_depth: ``hybrid_rrf``'s RRF constant (LakeGen uses 60) and
            how many results each side contributes before fusing (LakeGen
            fetches top 20).
    """

    def __init__(
        self,
        lexical_index: Any,
        hybrid_index: Optional[Any] = None,
        keyword_extractor: Optional[Callable[[str], Any]] = None,
        retrievers: Optional[Sequence[str]] = None,
        keyword_index: Optional[Any] = None,
        rrf_k: int = 60,
        rrf_depth: int = 20,
        universe: Optional[Iterable[str]] = None,
    ):
        self.lexical_index = getattr(hybrid_index, "lexical_index", None) or lexical_index
        self.keyword_index = keyword_index if keyword_index is not None else self.lexical_index
        self.hybrid_index = hybrid_index
        self.keyword_extractor = keyword_extractor
        if rrf_k < 1 or rrf_depth < 1:
            raise ValueError("rrf_k and rrf_depth must be >= 1")
        self.rrf_k = int(rrf_k)
        self.rrf_depth = int(rrf_depth)
        self.universe = frozenset(universe) if universe is not None else None
        if retrievers is not None:
            unknown = set(retrievers) - set(RETRIEVER_NAMES)
            if unknown:
                raise ValueError(
                    f"unknown retriever(s) {sorted(unknown)}; valid: {list(RETRIEVER_NAMES)}"
                )
        self._requested = tuple(retrievers) if retrievers else None
        self._keyword_cache: dict[str, list[str]] = {}
        self._usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _available(self) -> list[str]:
        names = ["lexical_question"]
        if self.hybrid_index is not None:
            names.append("dense_question")
        if self.keyword_extractor is not None:
            names.append("llm_keywords")
        if self.hybrid_index is not None and self.keyword_extractor is not None:
            names.append("hybrid_rrf")
        return names

    @property
    def retriever_names(self) -> list[str]:
        """The retrievers that vote: available ones, narrowed to the
        requested ``retrievers`` when given."""
        available = self._available()
        if self._requested is None:
            # Default: what has always voted. ``hybrid_rrf`` is opt-in — adding
            # it silently would shift every existing min_agreement decision.
            return [n for n in available if n != "hybrid_rrf"]
        chosen = [n for n in available if n in self._requested]
        if not chosen:
            logger.warning(
                "None of the requested retrievers %s is available (have %s); "
                "voting with every available retriever instead.",
                list(self._requested), available,
            )
            return [n for n in available if n != "hybrid_rrf"]
        return chosen

    def pop_usage(self) -> dict:
        """Token usage accumulated by ``llm_keywords`` calls since the last
        call, reset to zero. Callers fold this into their own run-level
        token accounting (see ``StatementOrchestrator._accumulate_tokens``).
        """
        usage = dict(self._usage_total)
        self._usage_total = {key: 0 for key in usage}
        return usage

    def _keywords_for(self, question: str) -> list[str]:
        if question in self._keyword_cache:
            return self._keyword_cache[question]
        if self.keyword_extractor is None:
            return []
        keywords, usage = _unpack_keyword_result(self.keyword_extractor(question))
        for key in self._usage_total:
            self._usage_total[key] += usage.get(key, 0)
        self._keyword_cache[question] = keywords
        return keywords

    def rank(self, question: str) -> dict[str, list[str]]:
        """``{retriever_name: [resource_id, ...]}``, each ranked up to
        ``RANK_CEILING``. Only the retrievers this panel was actually built
        with appear."""
        active = self.retriever_names
        # hybrid_rrf fuses the other two, so it needs their rankings even when
        # they do not vote themselves.
        fuse = "hybrid_rrf" in active
        computed: dict[str, list[str]] = {}

        def keep(results) -> list[str]:
            ids = [r.resource_id for r in results]
            if self.universe is None:
                return ids
            return [rid for rid in ids if rid in self.universe]

        if "lexical_question" in active:
            if question:
                lexical_results = self.lexical_index.search(question, top_k=RANK_CEILING)
            else:
                lexical_results = []
            computed["lexical_question"] = keep(lexical_results)

        if self.hybrid_index is not None and ("dense_question" in active or fuse):
            if question:
                semantic_batches = self.hybrid_index.semantic_search_many(
                    [question], top_k=RANK_CEILING
                )
                semantic_results = semantic_batches[0] if semantic_batches else []
            else:
                semantic_results = []
            computed["dense_question"] = keep(semantic_results)

        if self.keyword_extractor is not None and ("llm_keywords" in active or fuse):
            keywords = self._keywords_for(question) if question else []
            keyword_results = (
                self.keyword_index.search(keywords, top_k=RANK_CEILING) if keywords else []
            )
            computed["llm_keywords"] = keep(keyword_results)

        if fuse:
            computed["hybrid_rrf"] = self._rrf(
                [computed["llm_keywords"], computed["dense_question"]]
            )

        return {name: computed[name] for name in active}

    def _rrf(self, rankings: Sequence[Sequence[str]]) -> list[str]:
        """Reciprocal Rank Fusion: each ranking contributes ``1 / (k + rank)``
        for its top ``rrf_depth`` results; a result absent from a ranking
        gets nothing from it, so ranking well on EITHER side is enough.
        Ties break on the first ranking's order, then id — deterministic."""
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, resource_id in enumerate(ranking[: self.rrf_depth], start=1):
                scores[resource_id] = scores.get(resource_id, 0.0) + 1.0 / (self.rrf_k + rank)
        first = {rid: i for i, rid in enumerate(rankings[0])} if rankings else {}
        return sorted(scores, key=lambda r: (-scores[r], first.get(r, len(first)), r))

    def ranks(self, ranking: Sequence[str], gold_ids: Iterable[str]) -> dict[str, int]:
        """1-indexed rank of each gold id in ``ranking``. A gold id that never
        appears gets the sentinel ``_absent_rank`` — beyond ``RANK_CEILING``
        however short the ranking is."""
        order = {resource_id: i for i, resource_id in enumerate(ranking, start=1)}
        ceiling = _absent_rank(ranking)
        return {gold_id: order.get(gold_id, ceiling) for gold_id in gold_ids}

    def vote(
        self,
        question: str,
        gold_ids: Sequence[str],
        top_k: int,
        min_agreement: Optional[int] = None,
        rankings: Optional[dict[str, list[str]]] = None,
        extra: Optional[dict[str, dict]] = None,
    ) -> dict:
        """Rank ``question`` on every available retriever and vote.

        A retriever "passes" when EVERY gold id's rank is <= ``top_k``.
        ``approved`` is true when at least ``min_agreement`` retrievers pass
        — or, when fewer than ``min_agreement`` retrievers are available at
        all, when ALL of them pass.

        ``rankings``: a precomputed :meth:`rank` result for this same
        ``question``. Lets a caller that votes SEVERAL gold sets against one
        question (e.g. one vote per table) pay for the retrievers — an
        embedding call and an LLM keyword extraction — once, not per vote.

        ``extra``: verdicts of voters that are not a single ranking of the
        question — ``{name: {"pass": bool, "ranks": {...}, ...}}`` — each
        counted as ONE vote next to
        the ranked retrievers (see ``retrievability_gate._question_terms_vote``,
        whose one verdict aggregates a keyword check per table).

        Returns ``{"per_retriever": {name: {"pass", "ranks"}}, "passes": int, "approved": bool,
        "llm_keywords": [str, ...]}``.
        """
        if rankings is None:
            rankings = self.rank(question)
        per_retriever: dict[str, dict] = {}
        passes = 0
        for name, ranking in rankings.items():
            ranks = self.ranks(ranking, gold_ids)
            # An EMPTY ranking finds nothing, so it can never pass: a missing
            # gold table is ranked ``len(ranking) + 1``, which for an empty
            # ranking is 1 — inside a top_k=1 window. That is what a
            # keyword extractor returning no keywords, or a question sharing
            # no term with the index, would otherwise silently approve.
            ok = (
                bool(ranking)
                and bool(ranks)
                and all(rank <= top_k for rank in ranks.values())
            )
            if ok:
                passes += 1
            per_retriever[name] = {"pass": ok, "ranks": ranks}

        for name, verdict in (extra or {}).items():
            per_retriever[name] = verdict
            if verdict["pass"]:
                passes += 1

        n_available = len(rankings) + len(extra or {})
        required = min(min_agreement, n_available) if min_agreement is not None else n_available
        approved = n_available > 0 and passes >= max(required, 1)

        return {
            "per_retriever": per_retriever,
            "passes": passes,
            "approved": approved,
            "llm_keywords": list(self._keyword_cache.get(question, [])),
        }
