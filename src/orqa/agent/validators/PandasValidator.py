import pandas as pd
import polars as pl
import functools
from collections import defaultdict
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
    def _run_query(self, query: str, dataframes: list, table_names: list) -> Any:
        restore = _patch_pandas(max_rows=self.MAX_RESULT_ROWS, max_mb=self.mem_limit_mb)
        try:
            return self._exec(query, dataframes, table_names)
        finally:
            restore()

    def _exec(self, query: str, dataframes: list, table_names: list) -> Any:
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
        # Use the per-query table_names so the name→DataFrame pairing is correct
        # regardless of the order tables were declared in the query's tables field.
        for df, name in zip(dataframes, table_names):
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
                raise ValueError(
                    "The last statement did not produce a DataFrame or Series result.\n"
                    "The query must end with an expression that evaluates to a DataFrame.\n"
                    "Suggestions:\n"
                    "  1. End with the variable that holds the final result: e.g. 'result_df'\n"
                    "  2. Do not end with an assignment like 'df = ...' — end with 'df' on its own line\n"
                    f"  Available tables: {', '.join(self.table_names)}\n"
                )

        except Exception as e:
            raise self._normalize_pandas_error(e)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    # ------------------------------------------------------------------
    # Connectivity check — detects disjoint sub-queries
    # ------------------------------------------------------------------
    def _check_query_connectivity(self, query: str) -> None:
        """
        Verify that all tables referenced in the query are connected into a
        single result through merge / join / concat operations.

        Strategy
        --------
        1. Split the query into statements (separated by ; or newlines).
        2. For each statement, *fully resolve* the expression to the set of
           source tables it transitively combines — handling chained .merge()
           and pd.concat([]) recursively.
        3. When a statement is an assignment (var = expr), record the resolved
           set under the variable name so later statements can reference it.
        4. Build an adjacency graph: two source tables are "connected" when
           they co-appear in any resolved intermediate or final set.
        5. BFS over the graph — if any referenced table is unreachable, the
           query is disjoint and we raise a descriptive ValueError.
        """
        table_names_set = set(self.table_names)

        # var_tables: known name → frozenset of source tables it "carries"
        # Seeded with the raw table names themselves.
        var_tables: dict[str, frozenset] = {t: frozenset({t}) for t in table_names_set}

        # ----------------------------------------------------------------
        # Helpers
        # ----------------------------------------------------------------
        def tables_of(expr: str) -> frozenset:
            """Union of all table-sets for every known name found in expr."""
            result: set = set()
            for name, ts in var_tables.items():
                if re.search(rf'\b{re.escape(name)}\b', expr):
                    result |= ts
            return frozenset(result)

        def first_positional_arg(arg_str: str) -> str:
            """Substring up to the first top-level comma."""
            depth = 0
            for i, ch in enumerate(arg_str):
                if ch in ('(', '[', '{'): depth += 1
                elif ch in (')', ']', '}'): depth -= 1
                elif ch == ',' and depth == 0: return arg_str[:i]
            return arg_str

        def resolve(expr: str) -> frozenset:
            """
            Fully resolve expr to the frozenset of source tables it combines.
            Handles chained .merge() and pd/pl.concat([]) recursively so that
            intermediate variable references are expanded before accumulation.
            """
            accumulated: frozenset = tables_of(expr)

            # Every .merge(<rhs>, ...) — accumulate the rhs table set
            for m in re.finditer(r'\.merge\s*\(', expr):
                start = m.end()
                depth, i = 1, start
                while i < len(expr) and depth:
                    if expr[i] == '(': depth += 1
                    elif expr[i] == ')': depth -= 1
                    i += 1
                rhs_raw = first_positional_arg(expr[start:i - 1])
                accumulated = accumulated | resolve(rhs_raw)

            # Every pd/pl.concat([...])
            for m in re.finditer(r'(?:pd|pl)\.concat\s*\(\s*\[', expr):
                start = m.end()
                depth, i = 1, start
                while i < len(expr) and depth:
                    if expr[i] in ('(', '['): depth += 1
                    elif expr[i] in (')', ']'): depth -= 1
                    i += 1
                accumulated = accumulated | resolve(expr[start:i - 1])

            return accumulated

        def extract_assignment(stmt: str):
            """Return (lhs, rhs) for simple assignments, else (None, stmt)."""
            m = re.match(r'^\s*([A-Za-z_]\w*)\s*=(?!=)\s*(.+)', stmt, re.DOTALL)
            if m:
                return m.group(1), m.group(2).strip()
            return None, stmt

        # ----------------------------------------------------------------
        # Walk statements
        # ----------------------------------------------------------------
        flat = re.sub(r'\\\s*\n\s*', ' ', query)
        statements = [s.strip() for s in re.split(r'[;\n]', flat) if s.strip()]

        # Collect every resolved set (intermediate variables AND bare expressions)
        all_sets: list[frozenset] = []

        for stmt in statements:
            lhs, rhs_expr = extract_assignment(stmt)
            resolved = resolve(rhs_expr)
            if lhs:
                var_tables[lhs] = resolved   # expose to subsequent statements
            all_sets.append(resolved)

        # ----------------------------------------------------------------
        # Which source tables does the query actually reference?
        # ----------------------------------------------------------------
        referenced = frozenset(
            t for t in table_names_set
            if re.search(rf'\b{re.escape(t)}\b', query)
        )
        if len(referenced) < 2:
            return  # single table or empty — nothing to check

        # ----------------------------------------------------------------
        # Build adjacency from all_sets
        # ----------------------------------------------------------------
        adjacency: dict[str, set] = defaultdict(set)
        for ts in all_sets:
            nodes = list(ts & referenced)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    adjacency[nodes[i]].add(nodes[j])
                    adjacency[nodes[j]].add(nodes[i])

        # ----------------------------------------------------------------
        # BFS reachability
        # ----------------------------------------------------------------
        start = next(iter(referenced))
        visited: set = {start}
        queue = [start]
        while queue:
            node = queue.pop()
            for nb in adjacency.get(node, set()):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        if referenced <= visited:
            return  # ✓ fully connected

        # ----------------------------------------------------------------
        # Identify islands and raise
        # ----------------------------------------------------------------
        remaining = set(referenced)
        islands: list[frozenset] = []
        while remaining:
            root = next(iter(remaining))
            island: set = {root}
            q = [root]
            while q:
                node = q.pop()
                for nb in adjacency.get(node, set()):
                    if nb not in island:
                        island.add(nb)
                        q.append(nb)
            islands.append(frozenset(island))
            remaining -= island

        island_descriptions = "\n".join(
            f"  Group {i + 1}: {', '.join(sorted(island))}"
            for i, island in enumerate(islands)
        )
        raise ValueError(
            f"Disjoint query detected — the query produces {len(islands)} independent "
            f"sub-results that are never combined:\n\n"
            f"{island_descriptions}\n\n"
            "Every table must be connected to the others through at least one "
            "merge(), join(), or concat() that links them into a single result.\n\n"
            "Common causes:\n"
            "  1. Two separate merge chains that are never joined together\n"
            "  2. A concat() that includes some tables but not all\n"
            "  3. A subquery assigned to a variable that is never merged back in\n\n"
            "Fix: ensure the final expression combines ALL table groups, for example:\n"
            "  ✓ group1_result.merge(group2_result, on='shared_key')\n"
            "  ✓ pd.concat([group1_result, group2_result])\n"
            "  ✗ group1_result  # group2 tables were computed but never joined in"
        )

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

        #self._check_query_connectivity(query)
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
        raw = str(e).strip('"').strip("'")

        # ------------------------------------------------------------------
        # KeyError — missing column / label
        # ------------------------------------------------------------------
        if isinstance(e, KeyError):
            if "not in index" in raw.lower() or "not in index" in msg:
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
                "  3. Column exists in BOTH tables and must be suffixed (_x/_y)\n"
            )

        # ------------------------------------------------------------------
        # IndexError — iloc/positional out of bounds
        # ------------------------------------------------------------------
        if isinstance(e, IndexError):
            return IndexError(
                f"{raw}\n\nPositional index out of bounds:\n"
                "  - .iloc[n] references a row or column position that does not exist.\n"
                "  - The DataFrame may have fewer rows than expected after filtering/merging.\n"
                "Suggestions:\n"
                "  1. Use .loc[] with label-based access instead of .iloc[] where possible\n"
                "  2. Check DataFrame length with len(df) before positional access\n"
                "  3. Avoid hardcoding row positions — use filtering conditions instead\n"
            )

        # ------------------------------------------------------------------
        # AttributeError — .str on non-string, chained None, bad method name
        # ------------------------------------------------------------------
        if isinstance(e, AttributeError):
            if "nonetype" in msg and "has no attribute" in msg:
                attr = re.search(r"has no attribute '([^']+)'", str(e))
                attr_name = attr.group(1) if attr else "unknown"
                return AttributeError(
                    f"{raw}\n\nNoneType attribute error — an intermediate result was None:\n"
                    f"  - A method in the chain returned None before .{attr_name} was called.\n"
                    "  - Common cause: calling a method that modifies in-place (e.g. .sort_values(inplace=True))\n"
                    "    and then chaining further operations on its return value.\n"
                    "Suggestions:\n"
                    "  1. Never use inplace=True in a chain — use df = df.sort_values(...) separately\n"
                    "  2. Break the chain into steps and check each result is not None\n"
                )
            if "str" in msg and "has no attribute" in msg:
                return AttributeError(
                    f"{raw}\n\n.str accessor error:\n"
                    "  - .str was used on a column that is not of string dtype.\n"
                    "  - Cast the column first: df['col'].astype(str).str.lower()\n"
                    "  - Or check the column dtype with df.dtypes before applying .str\n"
                )
            if "series" in msg and "has no attribute" in msg:
                attr = re.search(r"has no attribute '([^']+)'", str(e))
                attr_name = attr.group(1) if attr else "unknown"
                return AttributeError(
                    f"{raw}\n\nSeries has no attribute '{attr_name}':\n"
                    "  - This method does not exist on a pandas Series.\n"
                    "  - If you expected a DataFrame here, the preceding operation may have\n"
                    "    collapsed it to a Series (e.g. selecting a single column).\n"
                )
            return AttributeError(
                f"{raw}\n\nAttribute error — a method or property does not exist:\n"
                "  - Check spelling of the method name\n"
                "  - Verify the object type at this point in the chain is what you expect\n"
            )

        # ------------------------------------------------------------------
        # NameError — undefined variable (bare function call, typo in table name)
        # ------------------------------------------------------------------
        if isinstance(e, NameError):
            name_match = re.search(r"name '([^']+)' is not defined", str(e))
            name = name_match.group(1) if name_match else "unknown"
            if name == "merge":
                return NameError(
                    "NameError: 'merge' is not defined.\n\n"
                    "merge() must be called as a DataFrame method, not a bare function:\n"
                    "  ✓ Table_A.merge(Table_B, on='key')\n"
                    "  ✗ merge(Table_A, Table_B, on='key')\n"
                )
            if any(name == t for t in self.table_names):
                return NameError(
                    f"NameError: table '{name}' is not defined in this scope.\n\n"
                    f"Available tables: {', '.join(self.table_names)}\n"
                    "  - Check for a typo in the table name\n"
                    "  - Table names are case-sensitive\n"
                )
            return NameError(
                f"NameError: '{name}' is not defined.\n\n"
                f"Available names in scope: pd, pl, {', '.join(self.table_names)}\n"
                "  - Only the pre-imported libraries (pd, pl) and the listed table variables are available\n"
                "  - Do not use import statements or reference external variables\n"
            )

        # ------------------------------------------------------------------
        # MemoryError — from _patch_pandas guards
        # ------------------------------------------------------------------
        if isinstance(e, MemoryError):
            return MemoryError(
                f"{raw}\n\n"
                "The operation exceeded the memory limit.\n"
                "Suggestions:\n"
                "  1. Pre-filter rows before the expensive operation\n"
                "  2. Select only the columns you need before merging\n"
                "  3. Aggregate earlier in the pipeline to reduce row count\n"
            )

        # ------------------------------------------------------------------
        # ZeroDivisionError
        # ------------------------------------------------------------------
        if isinstance(e, ZeroDivisionError):
            return ZeroDivisionError(
                f"{raw}\n\nDivision by zero:\n"
                "  - A denominator column contains zero or the filtered result is empty.\n"
                "Suggestions:\n"
                "  1. Guard with: df['col'].replace(0, float('nan')) before dividing\n"
                "  2. Filter out zero-denominator rows first\n"
            )

        # ------------------------------------------------------------------
        # ValueError — grouped by specificity, avoiding the 'empty' false-positive
        # ------------------------------------------------------------------
        if isinstance(e, ValueError):
            # MergeError / on= column absent from one side
            if "you are trying to merge on" in msg or ("merge" in msg and "key" in msg and "not in" in msg):
                cols = re.findall(r"'([^']+)'", str(e))
                col_hint = f" ({', '.join(cols[:2])})" if cols else ""
                return ValueError(
                    f"{raw}\n\nMerge key error{col_hint}:\n"
                    "  - The join column does not exist in one (or both) of the tables.\n"
                    "Suggestions:\n"
                    "  1. Verify the column exists in BOTH tables before merging\n"
                    "  2. Use left_on= / right_on= if the column has different names in each table\n"
                    "  3. Check for typos — column names are case-sensitive\n"
                )

            if "duplicate columns" in msg and "suffixes" in msg:
                dupes = re.findall(r"'([^']+)'", str(e))
                dupes_str = ", ".join(dupes[:4]) if dupes else "see error above"
                return ValueError(
                    f"{raw}\n\n"
                    "Duplicate column suffix error — the same suffix produces duplicate names.\n\n"
                    f"Duplicate columns detected: {dupes_str}\n\n"
                    "How to fix it:\n"
                    "  1. Use unique suffixes at each merge step:\n"
                    "       .merge(Table_1, suffixes=('_t0', '_t1'))\n"
                    "       .merge(Table_2, suffixes=('', '_t2'))\n\n"
                    "  2. Select only needed columns before each merge to avoid collisions:\n"
                    "       df[['join_key', 'col_a']].merge(Table_2[['join_key', 'col_b']], on='join_key')\n\n"
                    "  3. Drop conflicting columns before the next merge with .drop(columns=[...])\n\n"
                    "Tip: in 3+ table chains, select only the columns needed at each step."
                )

            if "no objects to concatenate" in msg:
                return ValueError(
                    f"{raw}\n\npd.concat received an empty list:\n"
                    "  - All DataFrames passed to concat were filtered to zero rows, or the list itself is empty.\n"
                    "Suggestions:\n"
                    "  1. Check each DataFrame for rows before concatenating\n"
                    "  2. Verify filter conditions are not excluding all data\n"
                )

            if "must pass list-like" in msg or "no numeric data to plot" in msg:
                return ValueError(
                    f"{raw}\n\nagg/apply input error:\n"
                    "  - The column passed to agg() or apply() is not the expected type.\n"
                    "  - This often means a string column was passed where numeric was expected.\n"
                    "Suggestions:\n"
                    "  1. Cast with .astype(float) before aggregating\n"
                    "  2. Check that groupby() selected the right columns\n"
                )

            if "cannot reindex" in msg or "reindex" in msg and "not unique" in msg:
                return ValueError(
                    f"{raw}\n\nReindex error — duplicate index values:\n"
                    "  - The DataFrame index contains duplicates, which makes reindexing ambiguous.\n"
                    "Suggestions:\n"
                    "  1. Call .reset_index(drop=True) after merges or groupby operations\n"
                    "  2. Use .drop_duplicates() if duplicate rows are not expected\n"
                )

            if "out of bounds" in msg and ("datetime" in msg or "date" in msg):
                return ValueError(
                    f"{raw}\n\nDatetime out of bounds:\n"
                    "  - A date value falls outside pandas' representable range (approx 1677–2262).\n"
                    "Suggestions:\n"
                    "  1. Filter extreme dates before conversion\n"
                    "  2. Use pd.to_datetime(..., errors='coerce') to turn invalid dates into NaT\n"
                )

            # Generic ValueError fallthrough — don't trigger empty-result feedback
            # just because the word 'empty' appears in an unrelated message
            if "empty" in msg and any(x in msg for x in [
                "no objects", "nothing to", "no columns", "no rows", "result is empty"
            ]):
                return ValueError(self._build_empty_result_feedback())

            return e

        # ------------------------------------------------------------------
        # TypeError — operand / dtype mismatches
        # ------------------------------------------------------------------
        if isinstance(e, TypeError):
            if "unsupported operand" in msg:
                ops = re.search(r"unsupported operand type\(s\) for (.+?): '([^']+)' and '([^']+)'", str(e))
                if ops:
                    op, t1, t2 = ops.group(1), ops.group(2), ops.group(3)
                    return TypeError(
                        f"{raw}\n\nType mismatch for operator '{op}':\n"
                        f"  - Cannot apply '{op}' between '{t1}' and '{t2}'.\n"
                        "  - One column is likely stored as a string instead of a number.\n"
                        "Suggestions:\n"
                        "  1. Cast with .astype(float) before arithmetic\n"
                        "  2. Use pd.to_numeric(df['col'], errors='coerce') for mixed columns\n"
                    )
            if "could not convert" in msg or "invalid literal" in msg:
                return TypeError(
                    f"{raw}\n\nType conversion error:\n"
                    "  - A column contains non-numeric values that cannot be cast.\n"
                    "Suggestions:\n"
                    "  1. Use pd.to_numeric(df['col'], errors='coerce') to coerce bad values to NaN\n"
                    "  2. Inspect unique values with df['col'].unique() before casting\n"
                )
            if "dtype" in msg:
                return TypeError(
                    f"{raw}\n\nDtype mismatch:\n"
                    "  - An operation was applied to a column with an incompatible dtype.\n"
                    "  - Check column dtypes with df.dtypes and cast as needed before operating.\n"
                )
            return TypeError(f"{raw}\n\nType error — check column dtypes and operation compatibility.\n")

        # ------------------------------------------------------------------
        # SyntaxError / IndentationError / TabError
        # ------------------------------------------------------------------
        if isinstance(e, (SyntaxError, IndentationError, TabError)):
            return SyntaxError(
                f"Invalid Python syntax in query.\nError: {raw}\n"
                "Tip: Ensure multiple statements are separated by semicolons (;) or newlines\n"
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