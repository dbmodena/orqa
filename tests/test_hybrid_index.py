import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from orqa.benchmark.hybrid_index import HybridDatasetIndex
from orqa.benchmark.index import SearchResult


class FakeLexicalIndex:
    datasets_path = Path("/nonexistent")
    datasets_format = "csv"
    source = "test"

    def dataset_filepath(self, resource_id: str) -> Path:
        return Path(f"/nonexistent/{resource_id}.csv")

    def get(self, resource_id: str):
        return {"dataset_id": resource_id, "title": resource_id}


def _result(resource_id: str, score: float) -> SearchResult:
    return SearchResult(
        resource_id=resource_id,
        dataset_id=resource_id,
        title=resource_id,
        publisher=None,
        tags=[],
        score=score,
        matched_terms=[],
        csv_path=f"/nonexistent/{resource_id}.csv",
        csv_exists=False,
        dataset_url=None,
    )


def _build_index(**kwargs) -> HybridDatasetIndex:
    # 3 embedded resources, distinct unit-ish vectors so cosine similarity
    # differences are meaningful (only their RELATIVE order matters below).
    embedding_ids = ["a", "b", "c"]
    embedding_vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32
    )
    return HybridDatasetIndex(
        FakeLexicalIndex(), embedding_ids, embedding_vectors, embedding_client=None,
        **kwargs,
    )


class TestFusionValidation(unittest.TestCase):
    def test_rejects_unknown_fusion_method(self):
        with self.assertRaises(ValueError):
            _build_index(fusion_method="borda_count")

    def test_rejects_non_positive_rrf_k(self):
        with self.assertRaises(ValueError):
            _build_index(fusion_method="rrf", rrf_k=0)

    def test_defaults_to_weighted_score(self):
        index = _build_index()
        self.assertEqual(index.fusion_method, "weighted_score")
        self.assertEqual(index.rrf_k, 60)


class TestFuseWeightedScore(unittest.TestCase):
    """Locks in the ORIGINAL fusion behavior is unchanged by default."""

    def test_weighted_average_of_normalized_scores(self):
        index = _build_index(lexical_weight=0.5, semantic_weight=0.5)
        lexical_results = [_result("a", 10.0), _result("b", 5.0)]
        # cosine values for a, b, c respectively — "a" best on semantic too.
        semantic_values = np.array([1.0, -1.0, 0.0])
        fused = index._fuse(lexical_results, semantic_values, top_k=10, only_available=False)
        by_id = {r.resource_id: r for r in fused}
        # a: lexical=10/10=1.0, semantic=(1+1)/2=1.0 -> hybrid=0.5*1+0.5*1=1.0
        self.assertAlmostEqual(by_id["a"].score, 1.0, places=6)
        # b: lexical=5/10=0.5, semantic=(-1+1)/2=0.0 -> hybrid=0.5*0.5+0.5*0=0.25
        self.assertAlmostEqual(by_id["b"].score, 0.25, places=6)
        self.assertEqual(fused[0].resource_id, "a")


class TestFuseRRF(unittest.TestCase):
    def test_score_is_exactly_the_sum_of_each_sides_reciprocal_rank(self):
        # Deliberately NOT a monotone case (best on one side isn't best
        # overall) — proves the score is the literal RRF formula per side,
        # not some other heuristic that happens to agree on easy cases.
        # "b" is #1 on lexical but WORST (rank 3 of 3) on semantic; "a" is
        # #2 on lexical but #1 on semantic — consistently good on both
        # beats being #1 on only one, exactly as 1/(k+2)+1/(k+1) >
        # 1/(k+1)+1/(k+3) works out numerically.
        index = _build_index(fusion_method="rrf", rrf_k=60)
        lexical_results = [_result("b", 10.0), _result("a", 1.0)]
        # embedding_ids order is [a, b, c]; normalized (cosine+1)/2 order
        # is a=0.6, c=0.5, b=0.05 -> semantic ranks a=1, c=2, b=3.
        semantic_values = np.array([0.2, -0.9, 0.0])  # a, b, c

        fused = index._fuse(lexical_results, semantic_values, top_k=10, only_available=False)
        by_id = {r.resource_id: r for r in fused}

        k = index.rrf_k
        expected_b = 1.0 / (k + 1) + 1.0 / (k + 3)  # lexical rank 1, semantic rank 3
        expected_a = 1.0 / (k + 2) + 1.0 / (k + 1)  # lexical rank 2, semantic rank 1
        self.assertAlmostEqual(by_id["b"].score, expected_b, places=9)
        self.assertAlmostEqual(by_id["a"].score, expected_a, places=9)
        self.assertGreater(by_id["a"].score, by_id["b"].score)
        self.assertEqual(fused[0].resource_id, "a")

    def test_absent_from_one_side_gets_no_cross_penalty(self):
        # "z" appears in the lexical results only (never embedded at all —
        # not one of the index's embedding_ids). Its RRF score should be
        # EXACTLY the lexical term alone, no penalty for the missing side.
        index = _build_index(fusion_method="rrf", rrf_k=60)
        lexical_results = [_result("z", 10.0)]
        semantic_values = np.array([0.1, 0.1, 0.1])  # a, b, c — irrelevant to "z"

        fused = index._fuse(lexical_results, semantic_values, top_k=10, only_available=False)
        by_id = {r.resource_id: r for r in fused}
        self.assertAlmostEqual(by_id["z"].score, 1.0 / (index.rrf_k + 1), places=9)

    def test_ignores_lexical_semantic_weights(self):
        # fusion_method="rrf" must not be silently modulated by
        # lexical_weight/semantic_weight — those are weighted_score-only.
        common_kwargs = dict(fusion_method="rrf", rrf_k=60)
        index_a = _build_index(lexical_weight=0.9, semantic_weight=0.1, **common_kwargs)
        index_b = _build_index(lexical_weight=0.1, semantic_weight=0.9, **common_kwargs)
        lexical_results = [_result("a", 10.0), _result("b", 5.0)]
        semantic_values = np.array([0.5, -0.5, 0.0])

        fused_a = index_a._fuse(lexical_results, semantic_values, top_k=10, only_available=False)
        fused_b = index_b._fuse(lexical_results, semantic_values, top_k=10, only_available=False)
        self.assertEqual(
            [(r.resource_id, r.score) for r in fused_a],
            [(r.resource_id, r.score) for r in fused_b],
        )


if __name__ == "__main__":
    unittest.main()
