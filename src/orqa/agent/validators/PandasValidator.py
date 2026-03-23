import pandas as pd
import polars as pl
import functools
from io import StringIO
import sys
from typing import Any
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
    # Dummy key: flag .assign(_key=<integer>) — the underscore-prefixed constant
    # key is the canonical cartesian product trick. Legitimate .assign() calls
    # use descriptive names and real values (e.g. assign(year=2024), assign(_c=col.str.lower())).
    (
        r'\.assign\s*\(\s*_\w+\s*=\s*\d+\s*\)',
        "Dummy key merge detected — assigning a constant integer to a _-prefixed key "
        "then joining on it produces a Cartesian product.\n"
        "Every row in both tables shares the same key value, so every row matches every other row.\n"
        "Use a real shared column as the join key instead."
    ),
    (
        r'\bpd\.merge\s*\([^)]*\)',
        "pd.merge() — verify join keys are explicitly specified"
    ),
    (
        r'\.join\s*\(\s*\w+\s*(?:,\s*how\s*=\s*[\'"](?:left|right|outer|inner)[\'"])?\s*\)',
        "join() called without explicit 'on' key"
    ),
    # Stale Series: only flag when left_on/right_on=Table_X[...] appears AFTER a prior
    # .merge() in the same chain — i.e. the pattern .merge(...).merge(...left_on=Table_X[)
    (
        r'\.merge\s*\([^)]*\)[^;]*\.merge\s*\([^)]*(?:left_on|right_on)\s*=\s*\w+\[',
        "Stale Series used as left_on/right_on key in a chained merge.\n"
        "After the first .merge() the original table index no longer matches the result,\n"
        "so Table_X['col'] as a key silently misaligns rows and drops columns.\n\n"
        "The correct pattern is to normalise ALL join keys with .assign() BEFORE the chain:\n"
        "  ✓ (Table_A.assign(_key=Table_A['col'].str.lower())\n"
        "      .merge(Table_B.assign(_key=Table_B['col'].str.lower()), on='_key')\n"
        "      .merge(Table_C.assign(_key=Table_C['col'].str.lower()), on='_key'))\n\n"
        "  ✗ Table_A.merge(Table_B, on='col').merge(Table_C, left_on=Table_A['col'].str.lower())\n\n"
        "Do NOT use Table_X['col'] as left_on/right_on after a prior merge has already\n"
        "changed the index. Use string column names (on='col') or .assign() instead."
    ),
]

# Operations to intercept with memory guards inside the sandbox.
PANDAS_DANGEROUS_OPS = {
    "merge": {
        "type": "method", "check": "result",
        "error": (
            "Merge produced {rows:,} rows ({mb:.1f}MB), exceeding the limit.\n"
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
            "Join produced {rows:,} rows ({mb:.1f}MB), exceeding the limit.\n"
            "Verify join keys and consider pre-filtering tables."
        ),
    },
    "corr": {
        "type": "method", "check": "input",
        "error": (
            "DataFrame passed to .corr() is {mb:.1f}MB ({rows:,} rows), exceeding the limit.\n"
            "Pre-filter or aggregate before computing correlation."
        ),
    },
    "concat": {
        "type": "function", "check": "result",
        "error": (
            "pd.concat produced {rows:,} rows ({mb:.1f}MB), exceeding the limit.\n"
            "Consider concatenating in smaller batches or pre-filtering."
        ),
    },
}


def _check_dataframe(df: pd.DataFrame, error_template: str, op_name: str, max_rows: int, max_mb: float) -> None:
    rows = len(df)
    mb   = df.memory_usage(deep=True).sum() / (1024 ** 2)
    if rows > max_rows or mb > max_mb:
        raise MemoryError(
            f"Operation '{op_name}' exceeded memory limits:\n"
            + error_template.format(rows=rows, mb=mb)
        )


def _patch_pandas(max_rows: int, max_mb: float):
    """
    Monkey-patches dangerous pandas operations with memory guards.
    Returns a restore callable — always call it in a finally block.
    """
    originals = {}

    for op_name, config in PANDAS_DANGEROUS_OPS.items():
        tmpl  = config["error"]
        check = config["check"]

        if config["type"] == "method":
            original = getattr(pd.DataFrame, op_name)
            originals[("method", op_name)] = original

            def make_method_wrapper(orig, t, name, chk):
                @functools.wraps(orig)
                def wrapper(self_df, *args, **kwargs):
                    if chk == "input":
                        _check_dataframe(self_df, t, name, max_rows, max_mb)
                    result = orig(self_df, *args, **kwargs)
                    if chk == "result" and isinstance(result, pd.DataFrame):
                        _check_dataframe(result, t, name, max_rows, max_mb)
                    return result
                return wrapper

            setattr(pd.DataFrame, op_name, make_method_wrapper(original, tmpl, op_name, check))

        elif config["type"] == "function":
            original = getattr(pd, op_name)
            originals[("function", op_name)] = original

            def make_func_wrapper(orig, t, name, chk):
                @functools.wraps(orig)
                def wrapper(*args, **kwargs):
                    result = orig(*args, **kwargs)
                    if chk == "result" and isinstance(result, pd.DataFrame):
                        _check_dataframe(result, t, name, max_rows, max_mb)
                    return result
                return wrapper

            setattr(pd, op_name, make_func_wrapper(original, tmpl, op_name, check))

    def restore():
        for (op_type, op_name), original in originals.items():
            if op_type == "method":
                setattr(pd.DataFrame, op_name, original)
            elif op_type == "function":
                setattr(pd, op_name, original)

    return restore


class PandasValidator(QueryValidator):
    """Validator for pandas/polars queries."""

    MAX_RESULT_ROWS = 5_000_000

    # ------------------------------------------------------------------
    # Execution — sandbox wraps _run_query via base _execute_query,
    # so we only need to patch pandas inside _run_query itself.
    # ------------------------------------------------------------------
    def _run_query(self, query: str,dataframes:list) -> Any:
        restore = _patch_pandas(max_rows=self.MAX_RESULT_ROWS, max_mb=self.mem_limit_mb)
        try:
            return self._exec(query,dataframes)
        finally:
            restore()

    def _exec(self, query: str,dataframes:list) -> Any:
        """Inner execution: eval/exec the cleaned query string."""
        safe_builtins = {
            'abs': abs, 'min': min, 'max': max, 'sum': sum,
            'len': len, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter,
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
            'True': True, 'False': False, 'None': None,
        }
        local_ns = {'pd': pd, 'pl': pl, '__builtins__': safe_builtins}
        for df, name in zip(dataframes, self.table_names):
            local_ns[name] = df

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            statements = [s.strip() for s in query.split(';') if s.strip()]
            if not statements:
                return None

            for stmt in statements[:-1]:
                exec(compile(stmt, '<string>', 'exec'), local_ns)

            last = statements[-1]
            try:
                return eval(last, local_ns)
            except SyntaxError:
                exec(compile(last, '<string>', 'exec'), local_ns)
                for val in reversed(list(local_ns.values())):
                    if isinstance(val, (pd.DataFrame, pd.Series)):
                        return val
                return None

        except Exception as e:
            raise self._normalize_pandas_error(e)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------
    def _preprocess_query(self, query: str) -> str:
        warnings = self._check_pandas_cartesian(query)
        if warnings:
            raise ValueError(
                "Potentially dangerous query — possible Cartesian product detected:\n"
                + "\n".join(f"  - {w}" for w in warnings)
                + "\nAvoid how='cross', dummy key merges, or merge() without explicit join keys."
            )
        # Check each '.str.' occurrence individually — a query mixing valid
        # Table_X['col'].str. and invalid 'literal'.str. must still be caught.
        for m in re.finditer(r"'([^']+)'\s*\.str\.", query):
            preceding = query[:m.start()].rstrip()
            if not preceding.endswith('['):
                col = m.group(1)
                raise ValueError(
                    f"Invalid use of string accessor: \'{col}\'.str. was called on a string literal.\n\n"
                    "The `.str` accessor can only be used on a DataFrame column (a pandas Series), "
                    "not on a plain Python string.\n\n"
                    f"If \'{col}\' is a column name after a merge (e.g. a suffixed column), "
                    "do NOT call .str.lower() on the string name itself.\n"
                    "Instead normalise the keys BEFORE merging using .assign():\n\n"
                    f"  \u2713 Table_A.assign(_key=Table_A[\'{col}\'].str.lower())\n"
                    f"           .merge(Table_B.assign(_key=Table_B[\'{col}\'].str.lower()), on=\'_key\')\n"
                    f"  \u2717 .merge(Table_B, left_on=\'{col}\'.str.lower())\n\n"
                    "Never apply .str.lower() to a quoted column name string — "
                    "always use DataFrame['column_name'].str.lower()."
                )

        cleaned = self._clean_pandas(query)
        self._check_stripped_query(query, cleaned)
        return cleaned

    def _check_stripped_query(self, original: str, cleaned: str) -> None:
        """
        Raise a clear error when clean_pandas strips the query down to nothing
        or to a trivial expression that cannot produce a DataFrame result.
        Diagnoses *why* it was stripped so the LLM gets actionable feedback.
        """
        if cleaned:
            return

        # Split on both newlines and semicolons for per-statement analysis
        lines = [l.strip() for l in re.split(r'[;\n]', original) if l.strip()]

        # Identify which categories of lines were removed
        has_imports  = any('import' in l.lower() for l in lines)
        has_banned   = any(any(cmd in l for cmd in UNAUTHORIZED_COMMANDS) for l in lines)
        # A line is comment-only if stripping everything after # leaves nothing
        has_comments = all(re.sub(r'#.*$', '', l).strip() == '' for l in lines) if lines else False

        reasons = []
        if has_imports:
            reasons.append(
                "  - import statements are not allowed and were removed.\n"
                "    All required libraries (pd, pl) are pre-imported automatically.\n"
                "    If you used 'import time' or similar to implement a delay, "
                "remove it entirely — delays have no place in a data query."
            )
        if has_banned:
            banned_found = [
                cmd for cmd in UNAUTHORIZED_COMMANDS
                if any(cmd in l for l in lines)
            ]
            reasons.append(
                f"  - the following unauthorized commands were removed: {', '.join(banned_found)}\n"
                "    Do not use file I/O, system access, or exec/eval in queries."
            )
        if has_comments:
            reasons.append(
                "  - the query contained only comments and no executable code."
            )
        if not reasons:
            reasons.append(
                "  - all lines were removed during cleaning (unknown reason).\n"
                "    Ensure the query contains valid pandas/polars expressions."
            )

        raise ValueError(
            "The query was entirely removed during cleaning and cannot be executed.\n\n"
            "Reasons:\n" + "\n".join(reasons) + "\n\n"
            "Rewrite the query using only pandas/polars DataFrame operations on the "
            "available table variables: " + ", ".join(self.table_names) + "."
        )

    def _check_pandas_cartesian(self, query: str) -> list[str]:
        return [
            msg for pattern, msg in PANDAS_CARTESIAN_PATTERNS
            if re.search(pattern, query, re.IGNORECASE | re.DOTALL)
        ]

    def _clean_pandas(self, query_code: str) -> str:
        if not query_code:
            return ''

        query = query_code.replace('`', "'")
        stash: list = []

        def _stash(m: re.Match) -> str:
            stash.append(m.group(0))
            return f'__QUERYCALL_{len(stash) - 1}__'

        query = re.sub(r'\.query\((["\']).*?\1\)', _stash, query, flags=re.DOTALL)
        query = query.replace('"', "'")
        query = re.sub(r'__QUERYCALL_(\d+)__', lambda m: stash[int(m.group(1))], query)
        query = re.sub(r'\\\s+\n', '\\\n', query)
        query = re.sub(r'\\\n\s*', ' ', query)

        # Split on newlines first to isolate import/comment lines from valid expressions.
        # A standalone 'import time' on its own line is removed; a chained merge
        # expression that spans multiple lines is rejoined before semicolon-splitting.
        newline_parts = query.splitlines()
        # Remove import-only lines and comment-only lines individually
        newline_parts = [l for l in newline_parts if 'import' not in l.lower()]
        newline_parts = [re.sub(r'#.*$', '', l).strip() for l in newline_parts]
        newline_parts = [l for l in newline_parts if l]
        # Rejoin and split on semicolons for multi-statement separation
        rejoined = ' '.join(newline_parts)
        statements = [s.strip() for s in rejoined.split(';')]
        cleaned = [s for s in statements if not any(cmd in s for cmd in UNAUTHORIZED_COMMANDS)]
        cleaned = [s for s in cleaned if s]
        return '; '.join(cleaned).strip()

    # ------------------------------------------------------------------
    # Error normalisation
    # ------------------------------------------------------------------
    def _normalize_pandas_error(self, e: Exception) -> Exception:
        msg = str(e).lower()

        if isinstance(e, KeyError):
            raw = str(e).strip('"').strip("'")
            if "not in index" in raw.lower():
                return KeyError(
                    f"{raw}\n\nColumn selection error:\n"
                    "  - The column does not exist in the DataFrame.\n"
                    "  - This often happens after a merge when duplicate columns are suffixed.\n"
                    "  - It can also happen when a stale Series is used as a merge key:\n"
                    "    After a merge, the original table index no longer matches the result.\n"
                    "    Using Table_X['col'] directly as left_on/right_on in a chained merge\n"
                    "    silently misaligns rows, causing columns to disappear.\n\n"
                    "Suggestions:\n"
                    "  1. Check column names after merge using df.columns\n"
                    "  2. If the column existed in both tables, use the suffixed version\n"
                    "  3. Never use Table_X['col'] as a merge key after a prior merge —\n"
                    "     use .assign() to bake keys in before the chain:\n"
                    "       ✓ Table_1.assign(_key=Table_1['col'].str.lower())\n"
                    "                .merge(Table_2.assign(_key=Table_2['col'].str.lower()), on='_key')\n"
                    "       ✗ .merge(Table_2, left_on=Table_1['col'].str.lower())"
                )
            key_desc = raw if "not found" in raw.lower() else f"Columns not found: {raw}"
            return KeyError(
                f"{key_desc}\n\nPossible causes:\n"
                "  1. Column does not exist — check spelling and case\n"
                "  2. Incorrectly suffixed column after merge\n"
                "  3. Column exists in BOTH tables and must be suffixed\n"
            )

        if "duplicate columns" in msg and "suffixes" in msg:
            dupes = re.findall(r"'([^']+)'", str(e))
            dupes_str = ", ".join(dupes[:4]) if dupes else "see error above"
            return ValueError(
                f"{str(e)}\n\n"
                "Duplicate column suffix error — the same suffix is applied across "
                "multiple merges, creating columns with identical names.\n\n"
                f"Duplicate columns detected: {dupes_str}\n\n"
                "How to fix it:\n"
                "  1. Use unique suffixes at each merge step:\n"
                "       .merge(Table_1, suffixes=(\'_t0\', \'_t1\'))\n"
                "       .merge(Table_2, suffixes=(\'\', \'_t2\'))\n"
                "       .merge(Table_3, suffixes=(\'\', \'_t3\'))\n\n"
                "  2. Select only needed columns before each merge to avoid collisions:\n"
                "       df[[\'join_key\', \'col_a\']].merge("
                "Table_2[[\'join_key\', \'col_b\']], on=\'join_key\')\n\n"
                "  3. Drop conflicting columns before the next merge:\n"
                "       .drop(columns=[\'program_type_lower\'])\n\n"
                "Tip: in 3+ table chains, select only the columns needed for the final "
                "result at each step to prevent suffix collisions."
            )

        if "merge" in msg and any(x in msg for x in ["key", "on", "left_on", "right_on"]):
            return KeyError(
                f"{str(e)}\n\nMerge key error:\n"
                "  - Ensure both join columns exist\n"
                "  - Ensure consistent casing (use .str.lower() on BOTH sides)\n"
            )

        if "groupby" in msg and "not found" in msg:
            return KeyError(f"{str(e)}\n\nGroupBy column error:\n  - The grouping column does not exist\n")

        if any(x in msg for x in ["could not convert", "dtype", "unsupported operand"]):
            return TypeError(
                f"{str(e)}\n\nType error:\n"
                "  - A column may have incorrect dtype (e.g., string instead of numeric)\n"
                "  - Try using .astype(float)\n"
            )

        if "empty" in msg:
            return ValueError(self._build_empty_result_feedback())

        if isinstance(e, (SyntaxError, IndentationError, TabError)):
            return SyntaxError(
                f"Invalid Python syntax in query.\nError: {str(e)}\n"
                "Tip: Ensure multiple statements are separated by newlines (\\n)"
            )

        return e

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------
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
            "  3. Boolean filters: check that the condition is not accidentally excluding all rows.",
            "Rewrite the query applying these normalizations where relevant.",
        ])

    def _get_language_name(self) -> str:
        return 'Python'