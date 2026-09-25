import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orqa.agent.utility.retrievability_gate import (
    RetrievalContract,
    TableContract,
    build_contract,
    check_question_retrievability,
)
from orqa.benchmark.retrieval_panel import RetrieverPanel


@dataclass
class FakeResult:
    resource_id: str


class FakeLexicalIndex:
    def __init__(self, ranking: list[str], records: dict[str, dict] | None = None):
        self._ranking = ranking
        self._records = records or {}

    def search(self, keywords, top_k: int = 10, only_available: bool = False):
        return [FakeResult(rid) for rid in self._ranking[:top_k]]

    def get(self, resource_id: str):
        return self._records.get(resource_id)


class TestCheckQuestionRetrievability(unittest.TestCase):
    def _panel(self, ranking):
        return RetrieverPanel(FakeLexicalIndex(ranking))

    def test_no_panel_or_contract_auto_passes(self):
        result = check_question_retrievability("some question", None, None)
        self.assertTrue(result["approved"])
        self.assertEqual(result["missing_tables"], [])

    def test_empty_question_auto_passes(self):
        contract = RetrievalContract(top_k=2, min_agreement=1, tables=[
            TableContract("Table_0", "gold"),
        ])
        panel = self._panel(["gold"])
        result = check_question_retrievability("", contract, panel)
        self.assertTrue(result["approved"])

    def test_plain_text_with_pinned_only_keywords_now_fails(self):
        # Regression test for the bug this whole feature fixes: a question
        # whose PROSE never mentions the table, even if some separately
        # -maintained keyword list would have. The gate now searches the
        # QUESTION TEXT itself.
        contract = RetrievalContract(top_k=2, min_agreement=1, tables=[
            TableContract("Table_0", "gold"),
        ])
        panel = self._panel(["unrelated1", "unrelated2"])  # gold not retrieved at all
        result = check_question_retrievability(
            "How many widgets were sold last year", contract, panel
        )
        self.assertFalse(result["approved"])
        self.assertIn("Table_0", result["missing_tables"])
        self.assertIn("Retrievability", result["feedback"])

    def test_level_a_miss_names_unused_anchor_keywords(self):
        # The anchor was computed for this table group but never made it
        # into the question's prose — the feedback should say so and name
        # the exact missing terms, not just report ranks.
        contract = RetrievalContract(top_k=2, min_agreement=1, tables=[
            TableContract("Table_0", "gold"),
        ])
        panel = self._panel(["unrelated1", "unrelated2"])  # gold not retrieved at all
        result = check_question_retrievability(
            "How many widgets were sold last year",
            contract,
            panel,
            ["organogram", "staff"],
        )
        self.assertFalse(result["approved"])
        self.assertIn(
            "missing from the question: organogram, staff", result["feedback"]
        )

    def test_level_a_miss_flags_anchor_present_but_insufficient(self):
        # The anchor IS already in the question's prose, so telling the
        # planner to "add these terms" again would be useless — the
        # feedback should say the anchor alone isn't enough instead.
        contract = RetrievalContract(top_k=2, min_agreement=1, tables=[
            TableContract("Table_0", "gold"),
        ])
        panel = self._panel(["unrelated1", "unrelated2"])  # gold not retrieved at all
        result = check_question_retrievability(
            "How many staff work at this organogram",
            contract,
            panel,
            ["organogram"],
        )
        self.assertFalse(result["approved"])
        self.assertIn("already present in the question", result["feedback"])
        self.assertIn("still misses the required rank", result["feedback"])

    def test_a_question_that_finds_the_table_is_approved(self):
        contract = RetrievalContract(top_k=2, min_agreement=1, tables=[
            TableContract("Table_0", "gold"),
        ])
        result = check_question_retrievability(
            "How many staff worked at this organisation in 2021",
            contract,
            self._panel(["gold", "other"]),
        )
        self.assertTrue(result["approved"])
        self.assertEqual(result["missing_tables"], [])
        self.assertNotIn("missing_facets", result)

    def test_only_the_exact_table_counts_not_a_file_of_its_dataset(self):
        # "sibling" shares the gold table's dataset but is a different file:
        # finding it is not finding the table.
        contract = RetrievalContract(top_k=2, min_agreement=1, tables=[
            TableContract("Table_0", "gold"),
        ])
        result = check_question_retrievability(
            "How many staff", contract, self._panel(["sibling", "other"])
        )
        self.assertFalse(result["approved"])
        self.assertEqual(result["missing_tables"], ["Table_0"])


class TestBuildContract(unittest.TestCase):
    TABLES = [
        {"alias": "Table_0", "resource_id": "t0"},
        {"alias": "Table_1", "resource_id": "t1"},
    ]

    def test_top_k_formula(self):
        contract = build_contract(
            self.TABLES, top_k_per_table=10, max_top_k=15, min_agreement=2,
        )
        # min(15, 10*2) = 15
        self.assertEqual(contract.top_k, 15)
        self.assertEqual([t.alias for t in contract.tables], ["Table_0", "Table_1"])
        self.assertEqual(contract.gold_ids, ["t0", "t1"])

    def test_multi_table_top_k_capped_by_max(self):
        contract = build_contract(
            self.TABLES, top_k_per_table=10, max_top_k=8, min_agreement=1,
        )
        # min(8, 10*2) = 8
        self.assertEqual(contract.top_k, 8)

    def test_single_table_defaults_to_rank_1(self):
        # A single-table plan targets literal rank 1 by default, NOT the
        # top_k_per_table/max_top_k window — see build_contract's docstring.
        contract = build_contract(
            self.TABLES[:1], top_k_per_table=10, max_top_k=20, min_agreement=1,
        )
        self.assertEqual(contract.top_k, 1)

    def test_single_table_top_k_override(self):
        contract = build_contract(
            self.TABLES[:1], top_k_per_table=10, max_top_k=20, min_agreement=1,
            single_table_top_k=3,
        )
        self.assertEqual(contract.top_k, 3)

    def test_the_contract_carries_no_family_or_facet_data(self):
        contract = build_contract(
            self.TABLES[:1], top_k_per_table=10, max_top_k=20, min_agreement=1,
        )
        self.assertEqual(
            contract.to_dict(),
            {"contract_version": 2, "top_k": 1, "min_agreement": 1, "tables": {"t0": "Table_0"}},
        )


if __name__ == "__main__":
    unittest.main()
