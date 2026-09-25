import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orqa.benchmark.families import FamilyIndex
from orqa.normalize_metadata import _identifying_resource_name


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


class TestIdentifyingResourceName(unittest.TestCase):
    """``resource_name`` is what this module builds its word facets from, and
    what the index weights as highly as the title. On data.gov.uk an
    ArcGIS-harvested resource is named after its format, so without this rule
    5,508 of 27,390 UK records would carry "CSV" into both.
    """

    TITLE = "Regions (December 2024) Boundaries EN BFC"

    def test_a_name_that_is_only_a_format_is_dropped(self):
        self.assertIsNone(_identifying_resource_name("CSV", self.TITLE))
        self.assertIsNone(_identifying_resource_name("CSV Download", self.TITLE))
        self.assertIsNone(_identifying_resource_name("Download the data file", self.TITLE))

    def test_a_name_repeating_the_title_is_dropped(self):
        self.assertIsNone(_identifying_resource_name(self.TITLE, self.TITLE))

    def test_a_name_of_its_own_is_kept(self):
        self.assertEqual(
            _identifying_resource_name("2021-12-31 Organogram (Junior)", self.TITLE),
            "2021-12-31 Organogram (Junior)",
        )

    def test_a_trailing_file_extension_is_stripped(self):
        self.assertEqual(
            _identifying_resource_name("2022 NI Water Results.csv", self.TITLE),
            "2022 NI Water Results",
        )

    def test_a_format_token_inside_a_real_name_is_removed(self):
        self.assertEqual(
            _identifying_resource_name("Organogram - Senior CSV data", self.TITLE),
            "Organogram - Senior data",
        )
        self.assertEqual(
            _identifying_resource_name("Trees (CSV)", self.TITLE), "Trees"
        )

    def test_a_format_glued_to_another_word_survives(self):
        """Splitting inside a token would mangle real words."""
        self.assertEqual(
            _identifying_resource_name("December 2018CSV", self.TITLE),
            "December 2018CSV",
        )

    def test_cleaning_that_reveals_the_title_drops_the_name(self):
        self.assertIsNone(_identifying_resource_name(f"{self.TITLE}.csv", self.TITLE))

    def test_missing_and_blank_names(self):
        self.assertIsNone(_identifying_resource_name(None, self.TITLE))
        self.assertIsNone(_identifying_resource_name("   ", self.TITLE))


if __name__ == "__main__":
    unittest.main()
