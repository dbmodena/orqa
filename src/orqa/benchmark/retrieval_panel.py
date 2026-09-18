"""The retriever panel behind the two-level retrievability contract.

A question is only *retrievable* if it would actually be found — not by one
search method's opinion, but by AGREEMENT across independently-built
retrievers: lexical BM25 over the raw question text, a dense/semantic
ranking of the embedded question, and BM25 over keywords an independent LLM
call extracts from the question (the same call the benchmark solver itself
makes — see ``orqa.agent.agents.BenchmarkSolver.BenchmarkSolverAgent.
generate_keywords``). See ``orqa.agent.utility.retrievability_gate`` for how
the panel's vote becomes a pass/fail gate.

Ranks are collapsed to FAMILY ranks (see ``orqa.benchmark.families.
FamilyIndex``) for the pass/fail decision — Level A only asks whether the
right CKAN dataset was reached, never the specific file — while per-resource
ranks are also returned for diagnostics.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Sequence

from .families import FamilyIndex

# How far into a ranking a retriever is searched before giving up. Wider
# than any plausible top_k so a rank just outside the pass/fail window is
# still a real number for diagnostics, not a flat "not found" sentinel.
RANK_CEILING = 200


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
        family_index: Collapses per-resource ranks to per-family ranks (see
            ``orqa.benchmark.families.FamilyIndex``); ``None`` treats every
            resource as its own singleton family.
    """

    def __init__(
        self,
        lexical_index: Any,
        hybrid_index: Optional[Any] = None,
        keyword_extractor: Optional[Callable[[str], Any]] = None,
        family_index: Optional[FamilyIndex] = None,
    ):
        self.lexical_index = getattr(hybrid_index, "lexical_index", None) or lexical_index
        self.hybrid_index = hybrid_index
        self.keyword_extractor = keyword_extractor
        self.family_index = family_index
        self._keyword_cache: dict[str, list[str]] = {}
        self._usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @property
    def retriever_names(self) -> list[str]:
        names = ["lexical_question"]
        if self.hybrid_index is not None:
            names.append("dense_question")
        if self.keyword_extractor is not None:
            names.append("llm_keywords")
        return names

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
        rankings: dict[str, list[str]] = {}
        if question:
            lexical_results = self.lexical_index.search(question, top_k=RANK_CEILING)
        else:
            lexical_results = []
        rankings["lexical_question"] = [r.resource_id for r in lexical_results]

        if self.hybrid_index is not None:
            if question:
                semantic_batches = self.hybrid_index.semantic_search_many(
                    [question], top_k=RANK_CEILING
                )
                semantic_results = semantic_batches[0] if semantic_batches else []
            else:
                semantic_results = []
            rankings["dense_question"] = [r.resource_id for r in semantic_results]

        if self.keyword_extractor is not None:
            keywords = self._keywords_for(question) if question else []
            keyword_results = (
                self.lexical_index.search(keywords, top_k=RANK_CEILING) if keywords else []
            )
            rankings["llm_keywords"] = [r.resource_id for r in keyword_results]

        return rankings

    def _family_of(self, resource_id: str) -> str:
        return self.family_index.family_id(resource_id) if self.family_index else resource_id

    def family_ranks(self, ranking: Sequence[str], gold_ids: Iterable[str]) -> dict[str, int]:
        """1-indexed rank of each gold id's FAMILY — the best (lowest) rank
        among any of its family's members appearing in ``ranking``. A gold
        id whose family never appears gets the sentinel ``len(ranking) + 1``.
        """
        best_family_rank: dict[str, int] = {}
        for i, resource_id in enumerate(ranking, start=1):
            family = self._family_of(resource_id)
            if family not in best_family_rank:
                best_family_rank[family] = i
        ceiling = len(ranking) + 1
        return {
            gold_id: best_family_rank.get(self._family_of(gold_id), ceiling)
            for gold_id in gold_ids
        }

    def resource_ranks(self, ranking: Sequence[str], gold_ids: Iterable[str]) -> dict[str, int]:
        """1-indexed rank of each gold id itself (not its family)."""
        order = {resource_id: i for i, resource_id in enumerate(ranking, start=1)}
        ceiling = len(ranking) + 1
        return {gold_id: order.get(gold_id, ceiling) for gold_id in gold_ids}

    def vote(
        self,
        question: str,
        gold_ids: Sequence[str],
        top_k: int,
        min_agreement: Optional[int] = None,
    ) -> dict:
        """Rank ``question`` on every available retriever and vote.

        A retriever "passes" when EVERY gold id's family rank is <= ``top_k``.
        ``approved`` is true when at least ``min_agreement`` retrievers pass
        — or, when fewer than ``min_agreement`` retrievers are available at
        all, when ALL of them pass (see the module docstring's Level A).

        Returns ``{"per_retriever": {name: {"pass", "family_ranks",
        "resource_ranks"}}, "passes": int, "approved": bool,
        "llm_keywords": [str, ...]}``.
        """
        rankings = self.rank(question)
        per_retriever: dict[str, dict] = {}
        passes = 0
        for name, ranking in rankings.items():
            family_ranks = self.family_ranks(ranking, gold_ids)
            resource_ranks = self.resource_ranks(ranking, gold_ids)
            ok = bool(family_ranks) and all(rank <= top_k for rank in family_ranks.values())
            if ok:
                passes += 1
            per_retriever[name] = {
                "pass": ok,
                "family_ranks": family_ranks,
                "resource_ranks": resource_ranks,
            }

        n_available = len(rankings)
        required = min(min_agreement, n_available) if min_agreement is not None else n_available
        approved = n_available > 0 and passes >= max(required, 1)

        return {
            "per_retriever": per_retriever,
            "passes": passes,
            "approved": approved,
            "llm_keywords": list(self._keyword_cache.get(question, [])),
        }
