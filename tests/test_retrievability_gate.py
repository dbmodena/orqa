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
from orqa.benchmark.families import FamilyIndex
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
    def setUp(self):
        records = [
            {"dataset_id": "fam1", "resource_id": "gold", "resource_name": "Organogram 2021"},
            {"dataset_id": "fam1", "resource_id": "sib", "resource_name": "Organogram 2020"},
        ]
        self.family_index = FamilyIndex(records, Path("/nonexistent"))
        self.records_by_id = {r["resource_id"]: r for r in records}
        self.record_lookup = self.records_by_id.get

    def _panel(self, ranking):
        lexical = FakeLexicalIndex(ranking, self.records_by_id)
        return RetrieverPanel(lexical, family_index=self.family_index)

    def test_no_panel_or_contract_auto_passes(self):
        result = check_question_retrievability("some question", None, None)
        self.assertTrue(result["approved"])
        self.assertEqual(result["missing_tables"], [])

    def test_empty_question_auto_passes(self):
        contract = RetrievalContract(top_k=2, min_agreement=1, tables=[
            TableContract("Table_0", "gold", "fam1", 2, facets=[], residual_siblings=[]),
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
            TableContract("Table_0", "gold", "fam1", 2, facets=[], residual_siblings=[]),
        ])
        panel = self._panel(["unrelated1", "unrelated2"])  # gold not retrieved at all
        result = check_question_retrievability(
            "How many widgets were sold last year", contract, panel
        )
        self.assertFalse(result["approved"])
        self.assertIn("Table_0", result["missing_tables"])
        self.assertIn("Level A", result["feedback"])

    def test_level_a_miss_names_unused_anchor_keywords(self):
        # The anchor was computed for this table group but never made it
        # into the question's prose — the feedback should say so and name
        # the exact missing terms, not just report ranks.
        contract = RetrievalContract(top_k=2, min_agreement=1, tables=[
            TableContract("Table_0", "gold", "fam1", 2, facets=[], residual_siblings=[]),
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
            TableContract("Table_0", "gold", "fam1", 2, facets=[], residual_siblings=[]),
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

    def test_missing_facet_gives_feedback_naming_it(self):
        from orqa.benchmark.families import Facet

        contract = RetrievalContract(top_k=2, min_agreement=1, tables=[
            TableContract(
                "Table_0", "gold", "fam1", 2,
                facets=[Facet(kind="temporal", label="period: 2021", period=None)],
                residual_siblings=[],
            ),
        ])
        # Give the temporal facet a real Period so missing_facets can compare it.
        from orqa.benchmark.families import Period
        from datetime import date
        contract.tables[0].facets[0] = Facet(
            kind="temporal", label="period: 2021",
            period=Period(date(2021, 1, 1), date(2021, 12, 31)),
        )
        panel = self._panel(["gold", "sib"])  # Level A passes fine
        result = check_question_retrievability(
            "How many staff work at this organisation", contract, panel
        )
        self.assertFalse(result["approved"])
        self.assertTrue(result["missing_facets"])
        self.assertEqual(result["missing_facets"][0]["table"], "Table_0")
        self.assertIn("Level B", result["feedback"])
        self.assertIn("period: 2021", result["feedback"])

    def test_plan_with_facets_and_two_of_three_retrievers_is_approved(self):
        from orqa.benchmark.families import Facet, Period
        from datetime import date

        contract = RetrievalContract(top_k=2, min_agreement=1, tables=[
            TableContract(
                "Table_0", "gold", "fam1", 2,
                facets=[Facet(
                    kind="temporal", label="period: 2021",
                    period=Period(date(2021, 1, 1), date(2021, 12, 31)),
                )],
                residual_siblings=[],
            ),
        ])
        panel = self._panel(["gold", "sib"])
        result = check_question_retrievability(
            "How many staff worked at this organisation in 2021", contract, panel
        )
        self.assertTrue(result["approved"])
        self.assertEqual(result["missing_tables"], [])
        self.assertEqual(result["missing_facets"], [])


class TestBuildContract(unittest.TestCase):
    def test_top_k_formula(self):
        records = {
            "t0": {"resource_id": "t0", "dataset_id": "d0"},
            "t1": {"resource_id": "t1", "dataset_id": "d1"},
        }
        family_index = FamilyIndex(list(records.values()), Path("/nonexistent"))
        contract = build_contract(
            [{"alias": "Table_0", "resource_id": "t0"}, {"alias": "Table_1", "resource_id": "t1"}],
            family_index,
            records.get,
            top_k_per_table=10,
            max_top_k=15,
            min_agreement=2,
            max_residual_siblings=5,
        )
        # min(15, 10*2) = 15
        self.assertEqual(contract.top_k, 15)
        self.assertEqual(len(contract.tables), 2)

    def test_multi_table_top_k_capped_by_max(self):
        records = {
            "t0": {"resource_id": "t0", "dataset_id": "d0"},
            "t1": {"resource_id": "t1", "dataset_id": "d1"},
        }
        family_index = FamilyIndex(list(records.values()), Path("/nonexistent"))
        contract = build_contract(
            [{"alias": "Table_0", "resource_id": "t0"}, {"alias": "Table_1", "resource_id": "t1"}],
            family_index, records.get,
            top_k_per_table=10, max_top_k=8, min_agreement=1, max_residual_siblings=5,
        )
        # min(8, 10*2) = 8
        self.assertEqual(contract.top_k, 8)

    def test_single_table_defaults_to_rank_1(self):
        # A single-table plan targets literal rank 1 by default, NOT the
        # top_k_per_table/max_top_k window — see build_contract's docstring.
        records = {"t0": {"resource_id": "t0", "dataset_id": "d0"}}
        family_index = FamilyIndex(list(records.values()), Path("/nonexistent"))
        contract = build_contract(
            [{"alias": "Table_0", "resource_id": "t0"}],
            family_index, records.get,
            top_k_per_table=10, max_top_k=20, min_agreement=1, max_residual_siblings=5,
        )
        self.assertEqual(contract.top_k, 1)

    def test_single_table_top_k_override(self):
        records = {"t0": {"resource_id": "t0", "dataset_id": "d0"}}
        family_index = FamilyIndex(list(records.values()), Path("/nonexistent"))
        contract = build_contract(
            [{"alias": "Table_0", "resource_id": "t0"}],
            family_index, records.get,
            top_k_per_table=10, max_top_k=20, min_agreement=1, max_residual_siblings=5,
            single_table_top_k=3,
        )
        self.assertEqual(contract.top_k, 3)

    def test_residual_siblings_populated_without_scope_loader(self):
        records = {
            "gold": {"resource_id": "gold", "dataset_id": "fam", "resource_name": "CSV"},
            "sib": {"resource_id": "sib", "dataset_id": "fam", "resource_name": "CSV"},
        }
        family_index = FamilyIndex(list(records.values()), Path("/nonexistent"))
        contract = build_contract(
            [{"alias": "Table_0", "resource_id": "gold"}],
            family_index, records.get,
            top_k_per_table=10, max_top_k=10, min_agreement=1, max_residual_siblings=5,
        )
        self.assertEqual(contract.tables[0].residual_siblings, ["sib"])

    def test_scope_loader_resolves_residual_siblings(self):
        records = {
            "gold": {"resource_id": "gold", "dataset_id": "fam", "resource_name": "CSV"},
            "sib": {"resource_id": "sib", "dataset_id": "fam", "resource_name": "CSV"},
        }
        family_index = FamilyIndex(list(records.values()), Path("/nonexistent"))

        def scope_loader(resource_id, columns):
            scopes = {"gold": {"region": "London"}, "sib": {"region": "Leeds"}}
            return scopes.get(resource_id)

        contract = build_contract(
            [{"alias": "Table_0", "resource_id": "gold"}],
            family_index, records.get,
            top_k_per_table=10, max_top_k=10, min_agreement=1, max_residual_siblings=5,
            scope_loader=scope_loader,
        )
        self.assertEqual(contract.tables[0].residual_siblings, [])
        self.assertTrue(any(f.kind == "scope" for f in contract.tables[0].facets))


if __name__ == "__main__":
    unittest.main()
