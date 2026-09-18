import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orqa.benchmark.families import (
    FamilyIndex,
    Period,
    distinguishing_facets,
    missing_facets,
    temporal_facts,
)


class TestTemporalFacts(unittest.TestCase):
    def test_iso_date(self):
        facts = temporal_facts("Published on 2022-09-30 for review.")
        self.assertIn(Period(date(2022, 9, 30), date(2022, 9, 30), "2022-09-30"), facts)

    def test_day_month_year_matches_iso(self):
        facts = temporal_facts("Snapshot as of 30 September 2022")
        self.assertTrue(any(p.start == p.end == date(2022, 9, 30) for p in facts))
        # "30 September 2022" and "2022-09-30" normalize to the SAME span.
        iso_facts = temporal_facts("2022-09-30")
        self.assertEqual(
            {(p.start, p.end) for p in facts if p.span_days == 0},
            {(p.start, p.end) for p in iso_facts},
        )

    def test_year_range(self):
        facts = temporal_facts("Staff numbers 2018 to 2019")
        self.assertIn(Period(date(2018, 1, 1), date(2019, 12, 31), ""), facts)

    def test_fiscal_year_short_form(self):
        facts = temporal_facts("Organogram 2021/22")
        self.assertIn(Period(date(2021, 1, 1), date(2022, 12, 31), ""), facts)

    def test_fiscal_year_hyphen_form(self):
        facts = temporal_facts("Organogram 2021-22")
        self.assertIn(Period(date(2021, 1, 1), date(2022, 12, 31), ""), facts)

    def test_quarter(self):
        facts = temporal_facts("Q1 2022 report")
        self.assertIn(Period(date(2022, 1, 1), date(2022, 3, 31), ""), facts)

    def test_month_range_equals_quarter(self):
        facts = temporal_facts("January to March 2022 figures")
        self.assertIn(Period(date(2022, 1, 1), date(2022, 3, 31), ""), facts)

    def test_week_range(self):
        facts = temporal_facts("week 36-39 2020")
        self.assertEqual(len(facts), 1)
        period = next(iter(facts))
        self.assertEqual(period.start, date.fromisocalendar(2020, 36, 1))
        self.assertEqual(period.end, date.fromisocalendar(2020, 39, 7))

    def test_bare_year(self):
        facts = temporal_facts("Staff roles 2021")
        self.assertIn(Period(date(2021, 1, 1), date(2021, 12, 31), "2021"), facts)

    def test_no_double_count_inside_specific_date(self):
        # "31 December 2021" must not ALSO yield a separate "December 2021"
        # or bare "2021" fact from the same span.
        facts = temporal_facts("Organogram as of 31 December 2021")
        day_facts = [p for p in facts if p.span_days == 0]
        self.assertEqual(len(day_facts), 1)
        self.assertEqual(day_facts[0].start, date(2021, 12, 31))
        # No OTHER fact should overlap that same mention.
        self.assertEqual(len(facts), 1)

    def test_month_range_not_double_counted(self):
        facts = temporal_facts("January to March 2022")
        # Must be exactly the one range fact, not also a bare "March 2022"
        # or bare "2022" pulled from inside it.
        self.assertEqual(len(facts), 1)

    def test_empty_text(self):
        self.assertEqual(temporal_facts(""), set())

    def test_invalid_date_falls_back_to_month(self):
        # "30 February" isn't a real calendar date — must not raise, and
        # degrades gracefully to the still-valid "February 2022" month fact.
        facts = temporal_facts("30 February 2022")
        self.assertEqual(facts, {Period(date(2022, 2, 1), date(2022, 2, 28))})


class TestDistinguishingFacets(unittest.TestCase):
    def test_temporal_facet_separates_siblings(self):
        gold = {"resource_name": "Organogram 31 December 2021", "resource_description": ""}
        siblings = {
            "sib1": {"resource_name": "Organogram 30 June 2021", "resource_description": ""},
            "sib2": {"resource_name": "Organogram 31 March 2022", "resource_description": ""},
        }
        result = distinguishing_facets(gold, siblings)
        self.assertEqual(result["residual_siblings"], [])
        self.assertTrue(any(f.kind == "temporal" for f in result["facets"]))

    def test_coarsest_granularity_preferred(self):
        # Siblings differ only by YEAR, so the year alone should suffice —
        # the full day-level date should not be required.
        gold = {"resource_name": "Organogram 31 December 2021", "resource_description": ""}
        siblings = {
            "sib1": {"resource_name": "Organogram 31 December 2020", "resource_description": ""},
        }
        result = distinguishing_facets(gold, siblings)
        temporal = [f for f in result["facets"] if f.kind == "temporal"]
        self.assertTrue(temporal)
        # The winning facet should be the YEAR span (365 days), not the
        # single day (0 days) — coarsest that still separates.
        self.assertEqual(temporal[0].period.start, date(2021, 1, 1))
        self.assertEqual(temporal[0].period.end, date(2021, 12, 31))

    def test_word_facet_junior_senior(self):
        gold = {"resource_name": "Organogram (Junior) 2021", "resource_description": ""}
        siblings = {
            "sib1": {"resource_name": "Organogram (Senior) 2021", "resource_description": ""},
        }
        result = distinguishing_facets(gold, siblings)
        self.assertEqual(result["residual_siblings"], [])
        labels = [f.label for f in result["facets"]]
        self.assertTrue(any("junior" in label.lower() for label in labels))

    def test_organogram_like_family_junior_senior_times_dates(self):
        gold = {"resource_name": "Organogram Junior 31 December 2021", "resource_description": ""}
        siblings = {
            "senior_same_date": {
                "resource_name": "Organogram Senior 31 December 2021", "resource_description": ""
            },
            "junior_other_date": {
                "resource_name": "Organogram Junior 30 June 2021", "resource_description": ""
            },
            "senior_other_date": {
                "resource_name": "Organogram Senior 30 June 2020", "resource_description": ""
            },
        }
        result = distinguishing_facets(gold, siblings)
        self.assertEqual(result["residual_siblings"], [])

    def test_no_metadata_signal_falls_back_to_scope(self):
        gold = {"resource_name": "CSV", "resource_description": ""}
        siblings = {
            "sib1": {"resource_name": "CSV", "resource_description": ""},
        }
        # Metadata alone can't separate them.
        no_scope = distinguishing_facets(gold, siblings)
        self.assertEqual(no_scope["residual_siblings"], ["sib1"])

        result = distinguishing_facets(
            gold,
            siblings,
            gold_scope={"region": "London"},
            sibling_scopes={"sib1": {"region": "Manchester"}},
        )
        self.assertEqual(result["residual_siblings"], [])
        self.assertTrue(any(f.kind == "scope" for f in result["facets"]))

    def test_residual_when_truly_indistinguishable(self):
        gold = {"resource_name": "CSV", "resource_description": ""}
        siblings = {"sib1": {"resource_name": "CSV", "resource_description": ""}}
        result = distinguishing_facets(gold, siblings)
        self.assertEqual(result["residual_siblings"], ["sib1"])
        self.assertEqual(result["facets"], [])

    def test_no_siblings_no_facets_needed(self):
        gold = {"resource_name": "Organogram 2021", "resource_description": ""}
        result = distinguishing_facets(gold, {})
        self.assertEqual(result["residual_siblings"], [])
        self.assertEqual(result["facets"], [])


class TestMissingFacets(unittest.TestCase):
    def test_temporal_satisfied_by_exact_match(self):
        gold = {"resource_name": "Organogram 31 December 2021", "resource_description": ""}
        siblings = {"sib1": {"resource_name": "Organogram 31 December 2020", "resource_description": ""}}
        facets = distinguishing_facets(gold, siblings)["facets"]
        missing = missing_facets("Show me staff roles in 2021", facets)
        self.assertEqual(missing, [])

    def test_temporal_satisfied_by_more_specific_date(self):
        gold = {"resource_name": "Organogram 2021", "resource_description": ""}
        siblings = {"sib1": {"resource_name": "Organogram 2020", "resource_description": ""}}
        facets = distinguishing_facets(gold, siblings)["facets"]
        # facet is the whole year 2021; a specific date WITHIN it satisfies.
        missing = missing_facets("Staff roles as of 30 September 2021", facets)
        self.assertEqual(missing, [])

    def test_temporal_not_satisfied_by_vaguer_period(self):
        gold = {"resource_name": "Organogram 31 December 2021", "resource_description": ""}
        siblings = {"sib1": {"resource_name": "Organogram 30 June 2021", "resource_description": ""}}
        facets = distinguishing_facets(gold, siblings)["facets"]
        # facet requires the specific day; stating only the year is too vague.
        missing = missing_facets("Staff roles in 2021", facets)
        self.assertTrue(missing)

    def test_word_facet_with_plural_stripping(self):
        gold = {"resource_name": "Organogram (Junior)", "resource_description": ""}
        siblings = {"sib1": {"resource_name": "Organogram (Senior)", "resource_description": ""}}
        facets = distinguishing_facets(gold, siblings)["facets"]
        missing = missing_facets("How many juniors are there", facets)
        self.assertEqual(missing, [])

    def test_word_facet_missing(self):
        gold = {"resource_name": "Organogram (Junior)", "resource_description": ""}
        siblings = {"sib1": {"resource_name": "Organogram (Senior)", "resource_description": ""}}
        facets = distinguishing_facets(gold, siblings)["facets"]
        missing = missing_facets("How many staff are there", facets)
        self.assertTrue(missing)

    def test_no_facets_nothing_missing(self):
        self.assertEqual(missing_facets("any question", []), [])


class TestFamilyIndex(unittest.TestCase):
    def test_family_grouping(self):
        records = [
            {"dataset_id": "organogram", "resource_id": "r1"},
            {"dataset_id": "organogram", "resource_id": "r2"},
            {"dataset_id": "other", "resource_id": "r3"},
        ]
        index = FamilyIndex(records, Path("/nonexistent"))
        self.assertEqual(index.family_id("r1"), "organogram")
        self.assertEqual(sorted(index.siblings("r1")), ["r2"])
        self.assertEqual(index.family_size("r1"), 2)
        self.assertEqual(index.siblings("r3"), [])
        self.assertEqual(index.family_size("r3"), 1)

    def test_no_dataset_id_is_singleton_family(self):
        records = [{"resource_id": "r1"}]
        index = FamilyIndex(records, Path("/nonexistent"))
        self.assertEqual(index.family_id("r1"), "r1")
        self.assertEqual(index.siblings("r1"), [])

    def test_unknown_resource_is_its_own_family(self):
        index = FamilyIndex([], Path("/nonexistent"))
        self.assertEqual(index.family_id("unknown"), "unknown")
        self.assertEqual(index.siblings("unknown"), [])


if __name__ == "__main__":
    unittest.main()
