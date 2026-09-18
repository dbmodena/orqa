import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orqa.agent.utility.keyword_suggestion import suggest_retrievable_keywords


@dataclass
class FakeResult:
    resource_id: str


class FakeCombinatorialIndex:
    """An index whose ranking is looked up by the EXACT query keyword set
    (as a frozenset), not computed from any real scoring model — lets a
    test define an adversarial landscape where a greedy hill-climb (only
    ever a single add/remove move from its current state) provably cannot
    reach the answer, while an exhaustive search over combinations can."""

    def __init__(self, ranking_by_keys: dict, records: dict, default_ranking=None):
        self._ranking_by_keys = ranking_by_keys
        self._records = records
        self._default = default_ranking or []

    def search(self, keywords, top_k: int = 10, only_available: bool = False):
        key = frozenset(keywords) if not isinstance(keywords, str) else frozenset(keywords.split())
        ranking = self._ranking_by_keys.get(key, self._default)
        return [FakeResult(rid) for rid in ranking[:top_k]]

    def get(self, resource_id: str):
        return self._records.get(resource_id)


def _fs(*terms):
    return frozenset(terms)


class TestExhaustiveRescue(unittest.TestCase):
    """See keyword_suggestion.py's module docstring, third tier — escalates
    only when the greedy climb AND the seed-and-cut fallback both fall
    short of top_k."""

    def setUp(self):
        # Every OTHER combination not listed here defaults to "T not found"
        # (competitors only) — i.e. any addition beyond the crafted trap is
        # a fitness regression, which is what strands the greedy climb.
        self.rankings = {
            _fs(): ["c1", "c2", "c3"],
            # The single best FIRST move (rank 3) — better than any other
            # lone term (rank 5 each) — so round 1 of the greedy climb
            # commits to it.
            _fs("zeta"): ["c1", "c2", "T"],
            _fs("alpha"): ["c1", "c2", "c3", "c4", "T"],
            _fs("beta"): ["c1", "c2", "c3", "c4", "T"],
            _fs("gamma"): ["c1", "c2", "c3", "c4", "T"],
            # Once "zeta" is selected, every single addition makes T
            # disappear entirely — no further greedy move is possible, and
            # removing "zeta" (back to the empty query) is worse too, so
            # the climb converges at {"zeta"}, rank 3, never achieving
            # top_k=1.
            _fs("zeta", "alpha"): ["c1", "c2", "c3"],
            _fs("zeta", "beta"): ["c1", "c2", "c3"],
            _fs("zeta", "gamma"): ["c1", "c2", "c3"],
            # The full-vocabulary seed ALSO misses, so the existing
            # seed-and-cut fallback (which only sweeps prefixes when the
            # full seed itself achieves) has nothing to offer either.
            _fs("zeta", "alpha", "beta", "gamma"): ["c1", "c2", "c3"],
            # The actual answer: a pair the greedy climb never tries,
            # because it never backtracks past its first committed move.
            _fs("alpha", "beta"): ["T"],
        }
        self.records = {"T": {"title": "zeta alpha beta gamma"}}
        self.index = FakeCombinatorialIndex(self.rankings, self.records)
        self.tables = [{"alias": "Table_0", "resource_id": "T", "columns": []}]

    def test_rescue_finds_the_combination_greedy_climb_cannot(self):
        result = suggest_retrievable_keywords(self.tables, self.index, top_k=1)
        self.assertTrue(result["achieved"])
        self.assertEqual(result["ranks"]["Table_0"], 1)
        self.assertEqual(set(result["keywords"]), {"alpha", "beta"})

    def test_no_rescue_needed_reports_achieved_without_it(self):
        # Sanity check the fixture/harness itself: when the greedy climb's
        # very first move already reaches top_k, the result is achieved
        # without ever touching the adversarial trap above.
        rankings = {_fs(): ["c1"], _fs("zeta"): ["T"]}
        index = FakeCombinatorialIndex(rankings, self.records)
        result = suggest_retrievable_keywords(self.tables, index, top_k=1)
        self.assertTrue(result["achieved"])
        self.assertEqual(result["keywords"], ["zeta"])

    def test_genuinely_unreachable_table_stays_unachieved(self):
        # No crafted answer exists anywhere in the rankings — the rescue
        # must not fabricate a false positive.
        rankings = {_fs(): ["c1", "c2", "c3"]}
        index = FakeCombinatorialIndex(rankings, self.records, default_ranking=["c1", "c2", "c3"])
        result = suggest_retrievable_keywords(self.tables, index, top_k=1)
        self.assertFalse(result["achieved"])


if __name__ == "__main__":
    unittest.main()
