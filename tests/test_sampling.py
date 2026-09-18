import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orqa.statement_generation import (
    _cap_matches,
    _family_stratified_order,
    _sample_single_table_datasets,
)


def _family_of_prefix(stem: str) -> str:
    """Test helper: family = everything before the last '_' run, mimicking
    a CKAN dataset id shared by several files ("organogram_2021",
    "organogram_2022", ... all in family "organogram")."""
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


class TestFamilyStratifiedOrder(unittest.TestCase):
    def test_no_family_exceeds_cap(self):
        files = [Path(f"organogram_{i}.csv") for i in range(20)] + [
            Path(f"other_{i}.csv") for i in range(3)
        ]
        order = _family_stratified_order(files, _family_of_prefix, seed=42, max_groups_per_family=2)
        counts: dict[str, int] = {}
        for f in order:
            fam = _family_of_prefix(f.stem)
            counts[fam] = counts.get(fam, 0) + 1
        self.assertTrue(all(c <= 2 for c in counts.values()))
        # organogram (20 files) capped at 2, other (3 files) capped at 2.
        self.assertEqual(counts.get("organogram"), 2)
        self.assertEqual(counts.get("other"), 2)

    def test_every_family_gets_a_turn_before_second_round(self):
        files = [Path("a_1.csv"), Path("a_2.csv"), Path("b_1.csv")]
        order = _family_stratified_order(files, _family_of_prefix, seed=1, max_groups_per_family=2)
        # "b" (only 1 file) must appear before "a"'s SECOND file, since
        # round-robin gives every family a turn each round.
        families_in_order = [_family_of_prefix(f.stem) for f in order]
        first_a_index = families_in_order.index("a")
        second_a_index = families_in_order.index("a", first_a_index + 1)
        self.assertIn("b", families_in_order[: second_a_index + 1])

    def test_deterministic_under_seed(self):
        files = [Path(f"fam{i % 5}_{i}.csv") for i in range(30)]
        order1 = _family_stratified_order(files, _family_of_prefix, seed=7, max_groups_per_family=3)
        order2 = _family_stratified_order(files, _family_of_prefix, seed=7, max_groups_per_family=3)
        self.assertEqual(order1, order2)

    def test_different_seed_can_differ(self):
        files = [Path(f"fam{i % 5}_{i}.csv") for i in range(30)]
        order1 = _family_stratified_order(files, _family_of_prefix, seed=1, max_groups_per_family=3)
        order2 = _family_stratified_order(files, _family_of_prefix, seed=2, max_groups_per_family=3)
        self.assertNotEqual(order1, order2)

    def test_covers_every_file_when_uncapped_enough(self):
        files = [Path(f"fam{i}_x.csv") for i in range(10)]  # each its own family
        order = _family_stratified_order(files, _family_of_prefix, seed=0, max_groups_per_family=5)
        self.assertEqual(set(order), set(files))


class TestSampleSingleTableDatasets(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.datasets_path = Path(self._tmpdir.name)
        # 10 files in family "organogram", 5 standalone singleton families.
        for i in range(10):
            (self.datasets_path / f"organogram_{i}.csv").write_text("a,b\n1,2\n")
        for i in range(5):
            (self.datasets_path / f"standalone{i}.csv").write_text("a,b\n1,2\n")

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_stratified_sampling_respects_cap(self):
        sampled = _sample_single_table_datasets(
            self.datasets_path, count=6, seed=0,
            family_of=_family_of_prefix, max_groups_per_family=1,
        )
        families = {_family_of_prefix(p.stem) for p in sampled}
        # With a cap of 1 per family and 6 families total (organogram +
        # 5 standalones), asking for 6 should draw at most 1 from organogram.
        organogram_count = sum(1 for p in sampled if _family_of_prefix(p.stem) == "organogram")
        self.assertLessEqual(organogram_count, 1)
        self.assertEqual(len(sampled), 6)

    def test_stratified_sampling_deterministic(self):
        sampled1 = _sample_single_table_datasets(
            self.datasets_path, count=6, seed=3,
            family_of=_family_of_prefix, max_groups_per_family=2,
        )
        sampled2 = _sample_single_table_datasets(
            self.datasets_path, count=6, seed=3,
            family_of=_family_of_prefix, max_groups_per_family=2,
        )
        self.assertEqual(sampled1, sampled2)

    def test_without_family_of_falls_back_to_plain_sampling(self):
        sampled = _sample_single_table_datasets(self.datasets_path, count=4, seed=0)
        self.assertEqual(len(sampled), 4)

    def test_stratified_sampling_with_shape_filter(self):
        sampled = _sample_single_table_datasets(
            self.datasets_path, count=6, seed=0, limit_to_n_columns=10,
            family_of=_family_of_prefix, max_groups_per_family=1,
        )
        organogram_count = sum(1 for p in sampled if _family_of_prefix(p.stem) == "organogram")
        self.assertLessEqual(organogram_count, 1)


class TestCapMatches(unittest.TestCase):
    def _match(self, *stems: str) -> dict:
        return {"aliases": {f"Table_{i}": stem for i, stem in enumerate(stems)}}

    def test_family_cap_across_all_tables_of_a_match(self):
        matches = (
            [self._match(f"organogram_{i}") for i in range(10)]
            + [self._match(f"standalone{i}") for i in range(5)]
        )
        capped = _cap_matches(
            matches, count=6, seed=0,
            family_of=_family_of_prefix, max_groups_per_family=1,
        )
        organogram_count = sum(
            1 for m in capped
            if any(_family_of_prefix(s) == "organogram" for s in m["aliases"].values())
        )
        self.assertLessEqual(organogram_count, 1)
        self.assertEqual(len(capped), 6)

    def test_multi_table_match_counts_against_every_family_it_touches(self):
        # A match joining one "organogram" table and one "standalone" table
        # consumes a slot from BOTH families.
        matches = [
            self._match("organogram_1", "standalone0"),
            self._match("organogram_2"),
            self._match("standalone1"),
        ]
        capped = _cap_matches(
            matches, count=2, seed=0,
            family_of=_family_of_prefix, max_groups_per_family=1,
        )
        self.assertEqual(len(capped), 2)
        # Only one match may include "organogram" family, and only one may
        # include "standalone" family — the joining match uses up both.
        organogram_matches = [
            m for m in capped
            if any(_family_of_prefix(s) == "organogram" for s in m["aliases"].values())
        ]
        standalone_matches = [
            m for m in capped
            if any(_family_of_prefix(s) == "standalone" for s in m["aliases"].values())
        ]
        self.assertLessEqual(len(organogram_matches), 1)
        self.assertLessEqual(len(standalone_matches), 1)

    def test_deterministic_under_seed(self):
        matches = [self._match(f"fam{i % 4}_{i}") for i in range(20)]
        capped1 = _cap_matches(matches, count=8, seed=5, family_of=_family_of_prefix, max_groups_per_family=2)
        capped2 = _cap_matches(matches, count=8, seed=5, family_of=_family_of_prefix, max_groups_per_family=2)
        self.assertEqual(capped1, capped2)

    def test_none_count_returns_unchanged(self):
        matches = [self._match(f"fam{i}") for i in range(5)]
        result = _cap_matches(matches, count=None, seed=0)
        self.assertEqual(result, matches)

    def test_count_at_or_above_total_returns_unchanged(self):
        matches = [self._match(f"fam{i}") for i in range(5)]
        result = _cap_matches(matches, count=10, seed=0)
        self.assertEqual(result, matches)

    def test_without_family_of_falls_back_to_plain_sample(self):
        matches = [self._match(f"fam{i}") for i in range(10)]
        result = _cap_matches(matches, count=4, seed=0)
        self.assertEqual(len(result), 4)

    def test_tight_cap_fills_out_from_leftover(self):
        # 2 families, cap 1 each -> only 2 matches strictly satisfy the cap,
        # but count=4 is requested; the remainder is filled from leftover.
        matches = [self._match("fam_a_1"), self._match("fam_a_2"),
                   self._match("fam_b_1"), self._match("fam_b_2")]
        result = _cap_matches(matches, count=4, seed=0, family_of=_family_of_prefix, max_groups_per_family=1)
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()
