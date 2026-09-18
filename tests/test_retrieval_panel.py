import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orqa.benchmark.families import FamilyIndex
from orqa.benchmark.retrieval_panel import RetrieverPanel


@dataclass
class FakeResult:
    resource_id: str


class FakeLexicalIndex:
    """A tiny stand-in for DatasetIndex: each query returns a FIXED ranking
    (independent of the actual keywords), configured per test."""

    def __init__(self, ranking: list[str], records: dict[str, dict] | None = None):
        self._ranking = ranking
        self._records = records or {}

    def search(self, keywords, top_k: int = 10, only_available: bool = False):
        return [FakeResult(rid) for rid in self._ranking[:top_k]]

    def get(self, resource_id: str):
        return self._records.get(resource_id)


class FakeHybridIndex:
    def __init__(self, lexical_index, semantic_ranking: list[str]):
        self.lexical_index = lexical_index
        self._semantic_ranking = semantic_ranking

    def semantic_search_many(self, texts: list[str], top_k: int = 10):
        return [[FakeResult(rid) for rid in self._semantic_ranking[:top_k]] for _ in texts]


def fake_keyword_extractor(keywords: list[str], usage: dict | None = None):
    def _extract(question: str):
        return {"keywords": keywords}, (usage or {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    return _extract


class TestRetrieverPanelRanking(unittest.TestCase):
    def test_lexical_only_ranking(self):
        lexical = FakeLexicalIndex(["a", "b", "c"])
        panel = RetrieverPanel(lexical)
        self.assertEqual(panel.retriever_names, ["lexical_question"])
        rankings = panel.rank("some question")
        self.assertEqual(rankings["lexical_question"], ["a", "b", "c"])

    def test_all_three_retrievers_available(self):
        lexical = FakeLexicalIndex(["a", "b", "c"])
        hybrid = FakeHybridIndex(lexical, ["b", "a", "c"])
        panel = RetrieverPanel(
            lexical, hybrid_index=hybrid, keyword_extractor=fake_keyword_extractor(["x"])
        )
        self.assertEqual(
            panel.retriever_names, ["lexical_question", "dense_question", "llm_keywords"]
        )
        rankings = panel.rank("question")
        self.assertEqual(rankings["dense_question"], ["b", "a", "c"])
        self.assertEqual(rankings["llm_keywords"], ["a", "b", "c"])

    def test_hybrid_lexical_index_preferred(self):
        base_lexical = FakeLexicalIndex(["x"])
        hybrid_lexical = FakeLexicalIndex(["a", "b"])
        hybrid = FakeHybridIndex(hybrid_lexical, ["a"])
        panel = RetrieverPanel(base_lexical, hybrid_index=hybrid)
        # lexical_index passed directly should be IGNORED in favor of
        # hybrid_index.lexical_index.
        self.assertIs(panel.lexical_index, hybrid_lexical)

    def test_keyword_cache_hits_once(self):
        calls = []

        def counting_extractor(question):
            calls.append(question)
            return {"keywords": ["a"]}, {"total_tokens": 5}

        lexical = FakeLexicalIndex(["a"])
        panel = RetrieverPanel(lexical, keyword_extractor=counting_extractor)
        panel.rank("same question")
        panel.rank("same question")
        self.assertEqual(len(calls), 1)
        self.assertEqual(panel.pop_usage()["total_tokens"], 5)
        # Popped once — accumulator resets.
        panel.rank("same question")
        self.assertEqual(panel.pop_usage()["total_tokens"], 0)


class TestRetrieverPanelRanks(unittest.TestCase):
    def setUp(self):
        records = [
            {"dataset_id": "fam1", "resource_id": "gold"},
            {"dataset_id": "fam1", "resource_id": "sibling"},
            {"dataset_id": "fam2", "resource_id": "other"},
        ]
        self.family_index = FamilyIndex(records, Path("/nonexistent"))

    def test_family_rank_collapses_to_best_member(self):
        lexical = FakeLexicalIndex(["other", "sibling", "gold"])
        panel = RetrieverPanel(lexical, family_index=self.family_index)
        ranking = panel.rank("q")["lexical_question"]
        ranks = panel.family_ranks(ranking, ["gold"])
        # "sibling" (same family as "gold") ranks 2nd, ahead of "gold" itself
        # at 3rd — family rank should be the BEST (2), not gold's own (3).
        self.assertEqual(ranks["gold"], 2)

    def test_resource_rank_is_exact(self):
        lexical = FakeLexicalIndex(["other", "sibling", "gold"])
        panel = RetrieverPanel(lexical, family_index=self.family_index)
        ranking = panel.rank("q")["lexical_question"]
        ranks = panel.resource_ranks(ranking, ["gold"])
        self.assertEqual(ranks["gold"], 3)

    def test_missing_gets_ceiling_sentinel(self):
        lexical = FakeLexicalIndex(["other"])
        panel = RetrieverPanel(lexical, family_index=self.family_index)
        ranking = panel.rank("q")["lexical_question"]
        ranks = panel.family_ranks(ranking, ["gold"])
        self.assertEqual(ranks["gold"], len(ranking) + 1)

    def test_no_family_index_treats_resource_as_own_family(self):
        lexical = FakeLexicalIndex(["sibling", "gold"])
        panel = RetrieverPanel(lexical)  # no family_index
        ranking = panel.rank("q")["lexical_question"]
        ranks = panel.family_ranks(ranking, ["gold"])
        self.assertEqual(ranks["gold"], 2)  # not collapsed to sibling's rank 1


class TestRetrieverPanelVote(unittest.TestCase):
    def setUp(self):
        records = [
            {"dataset_id": "fam1", "resource_id": "t0"},
            {"dataset_id": "fam2", "resource_id": "t1"},
        ]
        self.family_index = FamilyIndex(records, Path("/nonexistent"))

    def test_two_of_three_pass(self):
        # lexical_question and llm_keywords both search the SAME lexical
        # index (the fake ignores its query argument), so both rank
        # [t0, t1] within top_k=2 and pass; dense_question's semantic
        # ranking is deliberately unrelated and fails.
        lexical = FakeLexicalIndex(["t0", "t1"])
        hybrid = FakeHybridIndex(lexical, ["other", "other2"])
        panel = RetrieverPanel(
            lexical,
            hybrid_index=hybrid,
            keyword_extractor=fake_keyword_extractor(["x"]),
            family_index=self.family_index,
        )
        result = panel.vote("q", ["t0", "t1"], top_k=2, min_agreement=2)
        self.assertEqual(result["passes"], 2)
        self.assertFalse(result["per_retriever"]["dense_question"]["pass"])
        self.assertTrue(result["approved"])

    def test_fewer_retrievers_than_min_agreement_requires_all(self):
        lexical = FakeLexicalIndex(["t0", "t1"])
        panel = RetrieverPanel(lexical, family_index=self.family_index)  # only 1 retriever
        result = panel.vote("q", ["t0", "t1"], top_k=2, min_agreement=2)
        self.assertEqual(len(result["per_retriever"]), 1)
        self.assertTrue(result["approved"])  # the 1 available retriever passed

    def test_single_retriever_failing_rejects(self):
        lexical = FakeLexicalIndex(["other", "other2", "other3"])
        panel = RetrieverPanel(lexical, family_index=self.family_index)
        result = panel.vote("q", ["t0", "t1"], top_k=2, min_agreement=2)
        self.assertFalse(result["approved"])
        self.assertEqual(result["passes"], 0)

    def test_llm_keywords_recorded_after_vote(self):
        lexical = FakeLexicalIndex(["t0", "t1"])
        panel = RetrieverPanel(
            lexical, keyword_extractor=fake_keyword_extractor(["alpha", "beta"]),
            family_index=self.family_index,
        )
        result = panel.vote("q", ["t0", "t1"], top_k=2, min_agreement=1)
        self.assertEqual(result["llm_keywords"], ["alpha", "beta"])


if __name__ == "__main__":
    unittest.main()
