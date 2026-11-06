import sys
import unittest
from unittest import mock
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
        self.assertIn("Column Statistics", message)
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
        # A float has no .str because it is a SCALAR (an aggregate's result).
        self.assertIn("SCALAR", message)
        self.assertNotIn("NUMERIC", message)

    def test_tz_mismatch_hint_converts_the_aware_side_to_utc(self):
        aware = pd.Series(pd.to_datetime(["2024-03-10 02:30"]).tz_localize("Europe/Rome"))
        # Both ways pandas words it: a subtraction and a comparison.
        for operation in (lambda: aware - pd.Timestamp("2024-03-10"), lambda: aware < pd.Timestamp("2024-03-10")):
            try:
                operation()
            except TypeError as e:
                normalized = self.validator._normalize_pandas_error(e)
            else:
                self.fail("pandas did not raise — test fixture is stale")
            message = str(normalized)
            self.assertIn(".dt.tz_convert('UTC').dt.tz_localize(None)", message)
            self.assertNotIn("Dtype mismatch", message)
        # The advised fix gives the UTC clock: 02:30 in Rome is 01:30 UTC.
        fixed = aware.dt.tz_convert("UTC").dt.tz_localize(None)
        self.assertEqual(fixed.iloc[0], pd.Timestamp("2024-03-10 01:30"))

    def test_unrecognized_attribute_error_still_falls_back(self):
        normalized = self.validator._normalize_pandas_error(
            AttributeError("'Foo' object has no idea what you mean")
        )
        self.assertIn("Bad attribute", str(normalized))


T = pd.DataFrame({
    "Area": ["Cambridge", "Fenland", "Fenland", "Peterborough"],
    "Age (years)": [1, 2, 3, 4],
    "2018": ["1,050", "2,300", "3,400", "4,500"],
    "IssueDate": ["04/03/2019", "12/23/2013", "01/18/2017", "02/06/2025"],
    "code": ["DD1", "15", "16", "17"],
    "stamp": ["2020-01-01T10:00:00+00:00"] * 4,
    "note": ["rue", None, "rue de la", None],
    "empty": [None, None, None, None],
})
CODES = pd.DataFrame({"Area": [1, 2], "v": [10, 20]})


def validator(*frames, timeout=60):
    frames = frames or (T,)
    names = [f"Table_{i}" for i in range(len(frames))]
    return PandasValidator(list(frames), names, {n: n for n in names}, timeout=timeout)


def feedback(code, *frames):
    """The normalized error of running ``code`` in-process (no sandbox)."""
    v = validator(*frames)
    frames = frames or (T,)
    try:
        v._preprocess_query(code)
        v._run_query(code, list(frames), v.table_names)
    except Exception as e:  # noqa: BLE001
        return str(e)
    raise AssertionError(f"ran without error: {code}")


def result_of(code, *frames):
    v = validator(*frames)
    frames = frames or (T,)
    v._preprocess_query(code)
    return v._run_query(code, list(frames), v.table_names)


class ValidCodeIsNotRewritten(unittest.TestCase):
    """The code runs AS WRITTEN with the builtins every executor provides."""

    def test_literals_with_apostrophes_hashes_and_backticks(self):
        self.assertEqual(len(result_of("result = Table_0[Table_0['note'] == \"rue de la\"]")), 1)
        self.assertEqual(result_of("result = Table_0['code'].str.replace('#', '', regex=False).tolist()")[0], "DD1")
        self.assertEqual(len(result_of("result = Table_0.query('`Age (years)` > 2')")), 2)
        self.assertIn("double", result_of("result = Table_0.eval('double = `Age (years)` * 2')").columns)

    def test_builtins_and_numpy_import(self):
        self.assertEqual(result_of("result = round(float(Table_0['Age (years)'].mean()), 1)"), 2.5)
        self.assertEqual(result_of("result = sorted(Table_0['Area'].unique().tolist())[0]"), "Cambridge")
        self.assertTrue(result_of("result = any(isinstance(c, str) for c in Table_0.columns) and all(Table_0['Age (years)'] > 0)"))
        self.assertEqual(result_of("try:\n    x = Table_0['nope']\nexcept KeyError:\n    x = None\nresult = len(Table_0)"), 4)
        frame = result_of("import numpy as np\nresult = Table_0.assign(r=np.where(Table_0['Age (years)'] > 2, 'hi', 'lo'))")
        self.assertEqual(frame["r"].tolist(), ["lo", "lo", "hi", "hi"])

    def test_names_containing_print_or_eval_are_not_banned_calls(self):
        self.assertEqual(result_of("footprint = Table_0.assign(footprint=Table_0['Age (years)'] * 2)\nresult = int(footprint['footprint'].sum())"), 20)
        self.assertEqual(result_of("x = int(Table_0['Age (years)'].sum())\nprint(x)\nresult = x"), 10)

    def test_the_benchmark_policy_is_enforced_with_its_reason(self):
        self.assertIn("only pandas and numpy", feedback("import re\nresult = Table_0[Table_0['Area'].str.contains(re.escape('a'))]"))
        self.assertIn("`open` is not allowed", feedback("f = open('x')\nresult = Table_0"))
        self.assertIn("`.to_csv` is not allowed", feedback("Table_0.to_csv('x.csv')\nresult = Table_0"))
        self.assertIn("dunder", feedback("result = Table_0.__class__"))

    def test_the_result_must_be_a_value(self):
        self.assertIn("print(...), which is not a value", feedback("x = Table_0.head()\nprint(x)"))
        self.assertIn("final value is None", feedback("x = Table_0.head()\nx.sort_values('Area', inplace=True)"))
        text = feedback("total = 0\nfor c in ['Age (years)']:\n    total += Table_0[c].sum()")
        self.assertIn("did not resolve to a value", text)
        self.assertNotIn("Value error in the operation", text)  # the validator's own message is not re-wrapped


class ErrorsPointAtTheColumnStatisticsRecipe(unittest.TestCase):
    def test_casts_on_number_text(self):
        text = feedback("result = Table_0['2018'].astype(float).sum()")
        self.assertIn("thousands comma", text)
        self.assertIn("pd.to_numeric(s.str.strip().str.replace(',', '', regex=False), errors='coerce')", text)
        text = feedback("result = Table_0['code'].astype(int).max()")
        self.assertIn("`DD1` has a non-numeric affix", text)
        self.assertIn("`not numbers`", text)
        self.assertIn("TEXT column was compared (>)", feedback("result = Table_0[Table_0['2018'] > 100]"))
        self.assertIn("Arithmetic ('-') on a TEXT column", feedback("result = Table_0['IssueDate'] - Table_0['IssueDate']"))
        self.assertIn("`mean` ran over a TEXT", feedback("result = Table_0.groupby('Area')[['2018']].mean()"))

    def test_dates(self):
        text = feedback("result = pd.to_datetime(Table_0['IssueDate'], format='%Y-%m-%d')")
        self.assertIn("`%Y-%m-%d` is not how the values are written", text)
        self.assertIn("`date text` fact", text)
        self.assertIn(".dt used on a column that is not parsed yet", feedback("result = Table_0['IssueDate'].dt.year"))
        text = feedback("d = pd.to_datetime(Table_0['stamp'], utc=True)\nresult = Table_0[d >= pd.Timestamp('2010-01-01')]")
        self.assertIn(".dt.tz_convert('UTC').dt.tz_localize(None)", text)

    def test_masks_and_conditions(self):
        self.assertIn("na=False", feedback("result = Table_0[Table_0['note'].str.contains('rue')]"))
        self.assertIn("`&`, `|`, `~`", feedback("result = Table_0[Table_0['Age (years)'] > 1 and Table_0['Age (years)'] < 4]"))

    def test_merge_key_dtype_mismatch_is_not_reported_as_an_absent_column(self):
        text = feedback("result = Table_0.merge(Table_1, on='Area')", T, CODES)
        self.assertIn("DIFFERENT dtypes", text)
        self.assertIn("object vs int64", text)
        self.assertNotIn("absent from one or both", text)

    def test_key_errors_say_where_the_column_went(self):
        text = feedback("result = Table_0.groupby('Area')[['Age (years)']].sum()['Area']")
        self.assertIn("it is the INDEX", text)
        text = feedback("result = Table_0.set_index('Area').loc['Nowhere Town']")
        self.assertIn("ROW LABEL", text)
        text = feedback("result = Table_0.drop(columns=['nope'])")
        self.assertIn("'nope' not found", text)
        self.assertIn("Closest real columns", text)

    def test_empty_frames_are_named_as_such(self):
        self.assertIn("EMPTY", feedback("result = Table_0[Table_0['Area'] == 'nowhere'].iloc[0]"))
        self.assertIn("EMPTY", feedback("result = Table_0[Table_0['Area'] == 'nowhere'].set_index('Area')['Age (years)'].idxmax()"))
        text = feedback("result = Table_0['Age (years)'].max()[0]")
        self.assertIn("SCALAR", text)
        self.assertNotIn("EMPTY", text)
        # A numpy scalar's .round(2) is valid code: it only "failed" because
        # the size check indexed the () shape of every numpy scalar result.
        self.assertEqual(result_of("result = Table_0['Age (years)'].max().round(2)"), 4)

    def test_names_and_attributes(self):
        self.assertIn("import numpy as np", feedback("result = Table_0['Age (years)'].replace(0, np.nan)"))
        self.assertIn("Table_0", feedback("result = df['Age (years)'].sum()"))
        self.assertIn("backticks", feedback("result = Table_0.query('Age (years) > 1')"))
        self.assertIn("`s.str.lower(...)`", feedback("result = Table_0[Table_0['Area'].str.strip().lower() == 'x']"))
        self.assertIn("aggregate it first", feedback("result = Table_0.groupby('Area').total"))
        self.assertIn("inplace=True", feedback("result = Table_0.sort_values('Area', inplace=True).head()"))

    def test_pandas_specific_exceptions(self):
        self.assertIn("named aggregation", feedback("result = Table_0.groupby('Area').agg({'Age (years)': {'t': 'sum'}})"))
        self.assertIn("duplicates='drop'", feedback("result = pd.qcut(Table_0['Age (years)'].clip(upper=1), 4)"))
        self.assertIn(".rename(columns=", feedback("x = Table_0[['Area']]\nx.columns = ['a', 'b']\nresult = x"))
        self.assertIn("suffixes", feedback("result = Table_0.join(Table_0)"))
        self.assertIn("pivot_table", feedback("result = Table_0.pivot(index='Area', columns='stamp', values='Age (years)')"))
        self.assertIn("Int64", feedback("result = Table_0['empty'].astype(int)"))  # None in an object column
        self.assertIn("Int64", feedback("result = Table_0['Age (years)'].where(Table_0['Age (years)'] > 2).astype(int)"))  # NaN


class SandboxAndResultChecks(unittest.TestCase):
    """Through validate_queries: the sandbox and the checks on the result."""

    def errors(self, code, expect_type="", *frames, timeout=60):
        v = validator(*frames, timeout=timeout)
        q = {"id": 1, "question": "How many?", "code": code, "expected_result_type": expect_type,
             "tables": [{"name": n, "columns_involved": []} for n in v.table_names]}
        ok, _, good, errors = v.validate_queries({"queries": [q]})
        return (errors[0] if errors else None), ok

    def test_results_larger_than_the_pipe_buffer_come_back(self):
        # 5000 rows of text: well over the ~64 KB pipe buffer that used to
        # deadlock the sandbox until the timeout.
        big = pd.DataFrame({"k": [f"value-{i:06d}" for i in range(5000)], "n": range(5000)})
        text, ok = self.errors("result = Table_0[Table_0['n'] >= 0]", "table", big, timeout=20)
        self.assertTrue(ok, text)

    def test_large_results_are_shrunk_but_counted(self):
        big = pd.DataFrame({"n": range(30_000)})
        text, ok = self.errors("result = Table_0", "number", big, timeout=20)
        self.assertIn("30000 row(s)", text)

    def test_raw_row_dumps_and_join_explosions_keep_their_diagnosis(self):
        big = pd.DataFrame({"n": range(30)})
        with mock.patch.object(PandasValidator, "MAX_RESULT_ROWS", 10):
            text, ok = self.errors("result = Table_0", "table", big, timeout=20)
        self.assertIn("raw rows", text)
        self.assertNotIn("Operation exceeded the memory limit", text)
        keys = pd.DataFrame({"k": [1] * 2000})
        with mock.patch("orqa.agent.validators.PandasValidator._EXPLOSION_MIN_ROWS", 1000):
            text, ok = self.errors("result = Table_0.merge(Table_1, on='k')", "table", keys, keys, timeout=20)
        self.assertIn("join key is not unique", text)
        self.assertNotIn("Operation exceeded the memory limit", text)

    def test_all_null_inf_and_none_results(self):
        text, _ = self.errors("result = Table_0.assign(d=pd.to_datetime(Table_0['IssueDate'], format='%Y-%m-%d', errors='coerce')).groupby('Area')['d'].min()")
        self.assertIn("RESULT IS ALL NULL", text)
        text, ok = self.errors("result = Table_0[['Area', 'empty']]")
        self.assertTrue(ok, text)  # null in the source table is not a failed cast
        text, _ = self.errors("result = Table_0.assign(r=Table_0['Age (years)'] / 0)")
        self.assertIn("UNDEFINED VALUES", text)
        text, _ = self.errors("x = Table_0.head()\nx.sort_values('Area', inplace=True)")
        self.assertIn("final value is None", text)

    def test_a_runaway_loop_is_stopped_with_the_timeout_message(self):
        text, ok = self.errors("while True:\n    pass\nresult = 1", "number", T, timeout=2)
        self.assertFalse(ok)
        self.assertIn("ran for more than 2s", text)


if __name__ == "__main__":
    unittest.main()
