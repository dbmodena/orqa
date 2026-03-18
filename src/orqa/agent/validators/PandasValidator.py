import pandas as pd
import polars as pl
from io import StringIO
import sys
import threading
import functools
from typing import List, Dict, Tuple, Any
from .QueryValidator import QueryValidator
import re

UNAUTHORIZED_COMMANDS = [
    'read_csv', 'read_excel', 'read_json', 'read_parquet', 'print',
    'read_sql', 'read_table', 'read_html', 'read_pickle',
    'to_csv', 'to_excel', 'to_json', 'to_parquet', 'to_pickle',
    'open(', 'os.', 'sys.', 'exec(', 'eval('
]

PANDAS_CARTESIAN_PATTERNS = [
    (
        r'\.merge\s*\([^)]*how\s*=\s*[\'"]cross[\'"]\s*[^)]*\)',
        "Explicit cross merge (how='cross')"
    ),
    (
        r'\.merge\s*\(\s*\w+\s*\)',
        "merge() called with no join keys — may fall back to cross product"
    ),
    (
        r'\.assign\s*\([^)]*=\s*[\'"]?\d+[\'"]?\s*\)[^.]*\.merge\s*\(',
        "Dummy key merge pattern detected (assign constant key before merge)"
    ),
    (
        r'\bpd\.merge\s*\([^)]*\)',
        "pd.merge() — verify join keys are explicitly specified"
    ),
    (
        r'\.join\s*\(\s*\w+\s*(?:,\s*how\s*=\s*[\'"](?:left|right|outer|inner)[\'"])?\s*\)',
        "join() called without explicit 'on' key"
    ),
]

PANDAS_DANGEROUS_OPS = {
    "merge": {
        "type": "method", "check": "result",
        "error": (
            "Merge produced {{rows:,}} rows ({{mb:.1f}}MB), exceeding the limit.\n"
            "Causes: non-unique join key or low-cardinality column causing row multiplication.\n"
            "Suggestions:\n"
            "  1. Verify join keys are unique in at least one table\n"
            "  2. Add more specific join conditions\n"
            "  3. Pre-aggregate before merging"
        ),
    },
    "join": {
        "type": "method", "check": "result",
        "error": (
            "Join produced {{rows:,}} rows ({{mb:.1f}}MB), exceeding the limit.\n"
            "Verify join keys and consider pre-filtering tables."
        ),
    },
    "corr": {
        "type": "method", "check": "input",
        "error": (
            "DataFrame passed to .corr() is {{mb:.1f}}MB ({{rows:,}} rows), exceeding the limit.\n"
            "Pre-filter or aggregate before computing correlation."
        ),
    },
    "concat": {
        "type": "function", "check": "result",
        "error": (
            "pd.concat produced {{rows:,}} rows ({{mb:.1f}}MB), exceeding the limit.\n"
            "Consider concatenating in smaller batches or pre-filtering."
        ),
    },
}

MEMORY_LIMITS = {"max_rows": 500, "max_mb": 500}


def _check_dataframe(df: pd.DataFrame, error_template: str, op_name: str) -> None:
    rows = len(df)
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    if rows > MEMORY_LIMITS["max_rows"] or mb > MEMORY_LIMITS["max_mb"]:
        raise MemoryError(
            f"Operation '{op_name}' exceeded memory limits:\n"
            + error_template.format(rows=rows, mb=mb)
        )


def patch_pandas() -> callable:
    originals = {}

    for op_name, config in PANDAS_DANGEROUS_OPS.items():
        error_template = config["error"]
        check = config["check"]

        if config["type"] == "method":
            original = getattr(pd.DataFrame, op_name)
            originals[("method", op_name)] = original

            def make_wrapper(orig, tmpl, name, chk):
                @functools.wraps(orig)
                def wrapper(self_df, *args, **kwargs):
                    if chk == "input":
                        _check_dataframe(self_df, tmpl, name)
                    result = orig(self_df, *args, **kwargs)
                    if chk == "result" and isinstance(result, pd.DataFrame):
                        _check_dataframe(result, tmpl, name)
                    return result
                return wrapper

            setattr(pd.DataFrame, op_name, make_wrapper(original, error_template, op_name, check))

        elif config["type"] == "function":
            original = getattr(pd, op_name)
            originals[("function", op_name)] = original

            def make_func_wrapper(orig, tmpl, name, chk):
                @functools.wraps(orig)
                def wrapper(*args, **kwargs):
                    result = orig(*args, **kwargs)
                    if chk == "result" and isinstance(result, pd.DataFrame):
                        _check_dataframe(result, tmpl, name)
                    return result
                return wrapper

            setattr(pd, op_name, make_func_wrapper(original, error_template, op_name, check))

    def restore():
        for (op_type, op_name), original in originals.items():
            if op_type == "method":
                setattr(pd.DataFrame, op_name, original)
            elif op_type == "function":
                setattr(pd, op_name, original)

    return restore


def run_with_timeout(func, args=(), timeout=30):
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func(*args)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(
            f"Query exceeded the {timeout}s time limit and was aborted.\n"
            "This usually means a Cartesian product or missing join condition.\n"
            "Simplify the query or add more specific join/filter conditions."
        )
    if exception[0] is not None:
        raise exception[0]
    return result[0]


class PandasValidator(QueryValidator):
    """Validator for pandas/polars queries."""

    def _preprocess_query(self, query: str) -> str:
        warnings = self.check_pandas_cartesian(query)
        if warnings:
            raise ValueError(
                "Potentially dangerous query — possible Cartesian product detected:\n"
                + "\n".join(f"  - {w}" for w in warnings)
                + "\nAvoid how='cross', dummy key merges, or merge() without explicit join keys."
            )
        if re.search(r"'[^']+'\s*\.str\.", query) and not re.search(r"\[[^']*'[^']+'\s*\]\s*\.str\.", query):
            raise ValueError(
                "Invalid use of string accessor: '.str.lower()' was called on a string literal.\n\n"
                "The `.str` accessor can only be used on pandas Series (i.e., DataFrame columns), not on plain Python strings.\n\n"
                "How to fix it:\n"
                "  ✓ Correct:  Table_0['key'].str.lower()\n"
                "  ✗ Incorrect: 'key'.str.lower()\n\n"
                "Tip:\n"
                "  - Use DataFrame column references (Table_X['column_name']) when applying string operations.\n"
                "  - This commonly occurs in merge keys. For example:\n"
                "      df1.merge(df2,\n"
                "          left_on=df1['key'].str.lower(),\n"
                "          right_on=df2['key'].str.lower()\n"
                "      )"
            )
        return self.clean_pandas(query)

    def _execute_query(self, query: str) -> Any:
        restore = patch_pandas()
        try:
            return run_with_timeout(self._run_query, args=(query,), timeout=30)
        finally:
            restore()

    def check_pandas_cartesian(self, query: str) -> list[str]:
        warnings = []
        for pattern, message in PANDAS_CARTESIAN_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE | re.DOTALL):
                warnings.append(message)
        return warnings

    def _run_query(self, query: str) -> Any:
        safe_builtins = {
            'abs': abs, 'min': min, 'max': max, 'sum': sum,
            'len': len, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter,
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
            'True': True, 'False': False, 'None': None,
        }

        namespace = {
            'pd': pd,
            'pl': pl,
            '__builtins__': safe_builtins
        }

        for df, name in zip(self.dataframes, self.table_names):
            namespace[name] = df

        local_namespace = namespace.copy()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            statements = [s.strip() for s in query.split(';') if s.strip()]
            if not statements:
                return None

            for stmt in statements[:-1]:
                exec(compile(stmt, '<string>', 'exec'), local_namespace)

            last = statements[-1]
            try:
                return eval(last, local_namespace)
            except SyntaxError:
                exec(compile(last, '<string>', 'exec'), local_namespace)
                for val in reversed(list(local_namespace.values())):
                    if isinstance(val, (pd.DataFrame, pd.Series)):
                        return val
                return None
        except Exception as e:
            raise self._normalize_pandas_error(e)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _build_empty_result_feedback(self) -> str:
        return "\n".join([
            "The query returned an empty DataFrame (0 rows).",
            "This usually means a merge or filter condition matched nothing due to case or type mismatches.",
            "Suggestions:",
            "  1. String join keys: apply .str.lower() on BOTH sides before merging:",
            "       df1.merge(df2, left_on=df1['key'].str.lower(), right_on=df2['key'].str.lower())",
            "     or use .assign() before chaining:",
            "       df1.assign(_key=df1['key'].str.lower()).merge(",
            "           df2.assign(_key=df2['key'].str.lower()), on='_key', ...)",
            "  2. Numeric columns stored as strings: cast with .astype(float) before filtering or aggregating.",
            "  3. Boolean filters: check that the condition is not accidentally excluding all rows",
            "     (e.g. a date range or category value that does not exist in the data).",
            "Rewrite the query applying these normalizations where relevant.",
        ])

    def _get_language_name(self) -> str:
        return 'Python'

    def clean_pandas(self, query_code: str) -> str:
        if not query_code:
            return ''

        query = query_code.replace('`', "'")
        _query_calls: list = []

        def _stash(m: re.Match) -> str:
            _query_calls.append(m.group(0))
            return f'__QUERYCALL_{len(_query_calls) - 1}__'

        query = re.sub(r'\.query\((["\']).*?\1\)', _stash, query, flags=re.DOTALL)
        query = query.replace('"', "'")

        def _restore(m: re.Match) -> str:
            return _query_calls[int(m.group(1))]

        query = re.sub(r'__QUERYCALL_(\d+)__', _restore, query)
        query = re.sub(r'\\\s+\n', '\\\n', query)
        query = re.sub(r'\\\n\s*', ' ', query)

        lines = query.split(';')
        cleaned = [line.strip() for line in lines if 'import' not in line.lower()]
        cleaned = [re.sub(r'#.*$', '', line).strip() for line in cleaned]
        cleaned = [line for line in cleaned if not any(cmd in line for cmd in UNAUTHORIZED_COMMANDS)]
        cleaned = [line for line in cleaned if line]
        return '; '.join(cleaned).strip()

    def _normalize_pandas_error(self, e: Exception) -> Exception:
        msg = str(e).lower()
        if isinstance(e, KeyError):
            raw = str(e).strip('"').strip("'")
            if "not in index" in raw.lower():
                return KeyError(
                    f"{raw}\n\nColumn selection error:\n"
                    f"  - The column you are trying to access does not exist in the DataFrame\n"
                    f"  - This often happens after a merge when duplicate columns are suffixed\n\n"
                    f"Suggestions:\n"
                    f"  1. Check column names using df.columns\n"
                    f"  2. If the column existed in both tables, use the suffixed version:\n"
                )
            key_desc = raw if "not found" in raw.lower() else f"Columns not found: {raw}"
            return KeyError(
                f"{key_desc}\n\nPossible causes:\n"
                f"  1. Column does not exist — check spelling and case\n"
                f"  2. Incorrectly suffixed column\n"
                f"  3. Column exists in BOTH tables and must be suffixed\n"
            )

        if "merge" in msg and any(x in msg for x in ["key", "on", "left_on", "right_on"]):
            return KeyError(
                f"{str(e)}\n\nMerge key error:\n"
                f"  - Ensure both join columns exist\n"
                f"  - Ensure consistent casing (use .str.lower() on BOTH sides)\n"
            )

        if "groupby" in msg and "not found" in msg:
            return KeyError(f"{str(e)}\n\nGroupBy column error:\n  - The grouping column does not exist\n")

        if any(x in msg for x in ["could not convert", "dtype", "unsupported operand"]):
            return TypeError(
                f"{str(e)}\n\nType error:\n"
                f"  - A column may have incorrect dtype (e.g., string instead of numeric)\n"
                f"  - Try using .astype(float)\n"
            )

        if "empty" in msg:
            return ValueError(self._build_empty_result_feedback())

        if isinstance(e, (SyntaxError, IndentationError, TabError)):
            return SyntaxError(
                f'Invalid Python syntax in query.\nError: {str(e)}\n'
                f'Tip: Ensure multiple statements are separated by newlines (\\n)'
            )

        return e