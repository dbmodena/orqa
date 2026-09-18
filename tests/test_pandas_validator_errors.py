import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from orqa.agent.validators.PandasValidator import PandasValidator


class TestNormalizePandasError(unittest.TestCase):
    def setUp(self):
        self.validator = PandasValidator([], [], {})

    def _raise_str_accessor_on_numeric(self):
        series = pd.Series([585559.83, 100.0])
        series.str.replace(",", "", regex=False)

    def test_str_accessor_on_numeric_column_gets_a_specific_message(self):
        try:
            self._raise_str_accessor_on_numeric()
        except AttributeError as e:
            normalized = self.validator._normalize_pandas_error(e)
        else:
            self.fail("pandas did not raise — test fixture is stale")

        self.assertIsInstance(normalized, AttributeError)
        message = str(normalized)
        self.assertIn("NUMERIC", message)
        self.assertIn("dtypes", message)
        # Must NOT fall through to the generic, unhelpful fallback.
        self.assertNotIn("Bad attribute — check method name and object type", message)

    def test_true_missing_str_attribute_keeps_its_own_message(self):
        # A genuinely different case (e.g. .str called where there's no
        # such accessor at all) must still route to the OTHER branch, not
        # get swallowed by the new numeric-dtype one.
        try:
            (1.0).str
        except AttributeError as e:
            normalized = self.validator._normalize_pandas_error(e)
        else:
            self.fail("float.str did not raise — test fixture is stale")

        message = str(normalized)
        self.assertIn("cast first", message)
        self.assertNotIn("NUMERIC", message)

    def test_unrecognized_attribute_error_still_falls_back(self):
        normalized = self.validator._normalize_pandas_error(
            AttributeError("'Foo' object has no idea what you mean")
        )
        self.assertIn("Bad attribute", str(normalized))


if __name__ == "__main__":
    unittest.main()
