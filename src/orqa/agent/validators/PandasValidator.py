import ast
import builtins
import numpy as np
import pandas as pd
import polars as pl
import functools
from collections import defaultdict
from io import StringIO
import sys
from typing import Any
from .QueryValidator import QueryValidator
from ...utils import summarize_large_value
import re

UNAUTHORIZED_COMMANDS = [
    'read_csv', 'read_excel', 'read_json', 'read_parquet', 'print',
    'read_sql', 'read_table', 'read_html', 'read_pickle',
    'to_csv', 'to_excel', 'to_json', 'to_parquet', 'to_pickle',
    'open(', 'os.', 'sys.', 'exec(', 'eval('
]

# Modules generated code is allowed to `import` directly inside the sandbox
# (see _check_imports / _exec). Mirrors the data-analysis stack the
# sandbox already pre-injects (pd/pl) plus the safe stdlib modules.
# Anything outside this set (os, sys, subprocess, socket, shutil, ...)
# raises ImportError instead of silently being stripped.
ALLOWED_IMPORT_MODULES = {
    "pandas", "polars", "numpy",
    "math", "statistics", "decimal", "fractions", "random",
    "datetime", "collections", "itertools", "functools", "operator",
    "re", "json", "string", "warnings", "typing",
}


def _check_imports(tree: ast.AST) -> None:
    """Enforce ALLOWED_IMPORT_MODULES on the generated code's own import
    statements, statically.

    Enforcement must NOT happen at runtime via a restricted ``__import__`` in
    the sandbox builtins: C/Cython library code that lazily imports a module
    (e.g. pandas' ``Timestamp.strftime`` does ``import time``) resolves
    ``__import__`` from the nearest *Python* frame — which is the generated
    code's frame — so a runtime hook rejects imports the generated code never
    wrote. Checking the AST scopes the policy to what the code actually says.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            # Relative imports (level > 0) have no resolvable top-level
            # module inside exec'd code — reject them outright.
            names = [node.module] if node.module and not node.level else [""]
        else:
            continue
        for name in names:
            if name.split('.')[0] not in ALLOWED_IMPORT_MODULES:
                raise ImportError(
                    f"Import of {name!r} is not allowed in generated code. "
                    f"Allowed modules: {', '.join(sorted(ALLOWED_IMPORT_MODULES))}"
                )


PANDAS_CARTESIAN_PATTERNS = [
    # 1. Explicit cross merge — always a cartesian product.
    (
        r'\.merge\s*\([^)]*how\s*=\s*[\'"]cross[\'"]\s*[^)]*\)',
        "Explicit cross merge (how='cross')"
    ),
    # 2. Dummy constant key — assigning an integer literal to a _-prefixed column before merge
    #    produces a full cartesian product.  Legitimate key normalisation uses expressions,
    #    not bare integers: .assign(_key=df['col'].str.lower()) is fine.
    (
        r'\.assign\s*\(\s*_\w+\s*=\s*\d+\s*\)',
        "Dummy key merge: assigning a constant integer to a _-prefixed key produces a Cartesian "
        "product. Use a real shared column instead."
    ),
    # 3. pd.merge() with two bare table-name arguments and no keyword join keys.
    #    pd.merge(df1, df2) with no on=/left_on=/right_on= will use ALL common columns as keys
    #    which is almost never intentional and can silently produce wrong results.
    #    NOTE: pd.merge(df1, df2, on='key') does NOT match because the third token breaks \w+\s*\).
    (
        r'\bpd\.merge\s*\(\s*\w+\s*,\s*\w+\s*\)',
        "pd.merge() called with two bare table names and no explicit join key — "
        "add on=, left_on=, or right_on=."
    ),
    # 4. Stale Series reference as left_on/right_on.
    #    After a chained merge the original DataFrame's index is no longer aligned, so
    #    Table_X['col'] passed as left_on/right_on silently misaligns rows.
    #    The previous pattern used [^)]* which cannot handle nested parens (e.g. suffixes=('_a','_b'))
    #    and was therefore both fragile and prone to false positives on valid chained merges.
    #    This simpler pattern catches the actual anti-pattern directly wherever it appears.
    (
        r'(?:left_on|right_on)\s*=\s*[A-Za-z_]\w*\s*\[',
        "Stale Series as left_on/right_on — passing Table_X['col'] misaligns rows after a prior merge.\n"
        "  ✓ Table_A.assign(_key=Table_A['col'].str.lower()).merge(Table_B.assign(_key=...), on='_key')\n"
        "  ✗ .merge(Table_C, left_on=Table_A['col'].str.lower())"
    ),
    # REMOVED — bare single-arg .merge(df): .merge(other) is valid pandas and uses shared columns.
    # REMOVED — .join() without on=: index-based joins (.join(other)) are perfectly valid pandas.
]

PANDAS_DANGEROUS_OPS = {
    "merge": {
        "type": "method", "check": "result",
        "error": "Merge produced {rows:,} rows ({mb:.1f}MB), limit exceeded.\nCauses: non-unique join key or low-cardinality column.\nFix: ensure key is unique in one table, add conditions, or pre-aggregate.",
    },
    "join": {
        "type": "method", "check": "result",
        "error": "Join produced {rows:,} rows ({mb:.1f}MB), limit exceeded.\nVerify join keys and consider pre-filtering.",
    },
    "corr": {
        "type": "method", "check": "input",
        "error": "DataFrame passed to .corr() is {mb:.1f}MB ({rows:,} rows), limit exceeded.\nPre-filter or aggregate first.",
    },
    "concat": {
        "type": "function", "check": "result",
        "error": "pd.concat produced {rows:,} rows ({mb:.1f}MB), limit exceeded.\nConcatenate in smaller batches or pre-filter.",
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

    def _empty_result_is_error(self, ordered_names: list) -> bool:
        """
        FIX: For multi-table queries, sample data frequently has no overlapping join keys,
        so a logically correct merge or concat will return 0 rows on the sample subset.
        Rejecting these as errors forces the LLM into spurious correction cycles that burn
        tokens without fixing anything real.

        Strategy:
          - Single-table: always reject empty — a filter/transform on one table that returns
            nothing is almost certainly wrong.
          - Multi-table: allow empty — the query structure is validated (connectivity, table
            usage, field coverage) but we do not penalise zero rows from sample-data key mismatches.
        """
        return len(ordered_names) == 1

    def _run_query(self, query: str, dataframes: list, table_names: list) -> Any:
        restore = _patch_pandas(max_rows=self.MAX_RESULT_ROWS, max_mb=self.mem_limit_mb)
        try:
            return self._exec(query, dataframes, table_names)
        finally:
            restore()

    def _exec(self, query: str, dataframes: list, table_names: list) -> Any:
        safe_builtins = {
            'abs': abs, 'min': min, 'max': max, 'sum': sum,
            'len': len, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter,
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
            'True': True, 'False': False, 'None': None,
            # Real __import__: import policy is enforced statically by
            # _check_imports before execution. A restricted hook here breaks
            # library-internal lazy imports (see _check_imports docstring).
            '__import__': builtins.__import__,
        }
        local_ns = {
            'pd': pd, 'pl': pl, '__builtins__': safe_builtins,
        }
        for df, name in zip(dataframes, table_names):
            local_ns[name] = df

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            # Execute the query as ordinary multi-line Python — the same
            # execution model as the judge-time executor
            # (QueryExecutor._inject_result + exec) and the benchmark executor
            # (_run_pandas), so code that passes validation can never fail
            # downstream on syntax. The old statement-at-a-time model
            # (split on ';', compile each) accepted flattened compound
            # statements ("a = 1; if x: b = 2") that a whole-source parse
            # rejects, which surfaced as judge-time
            # "invalid syntax (<string>, line 1)" failures.
            tree = ast.parse(query)
            if not tree.body:
                return None
            _check_imports(tree)

            last = tree.body[-1]
            if isinstance(last, ast.Expr):
                # Run everything before the last expression, then eval it as
                # the query's result.
                if tree.body[:-1]:
                    head = ast.Module(body=tree.body[:-1], type_ignores=[])
                    exec(compile(head, '<string>', 'exec'), local_ns)
                result = eval(
                    compile(ast.Expression(body=last.value), '<string>', 'eval'),
                    local_ns,
                )
            else:
                exec(compile(tree, '<string>', 'exec'), local_ns)
                # Resolve the value the same way the judge-time executor does
                # (QueryExecutor._assignment_base_name): take the base name of
                # the last statement's assignment target — bare (`x = ...`) or
                # in-place (`x[...] = ...` / `x.col = ...`) — rather than
                # scanning the namespace for any DataFrame/Series, so static
                # validation and judge-time execution agree on what a query's
                # "result" is.
                target_name = self._assignment_base_name(last)
                result = local_ns.get(target_name) if target_name else None
                if result is None:
                    raise ValueError(
                        "Last statement did not resolve to a value.\n"
                        "End with a bare variable name (not an assignment), or "
                        "with `<name> = ...` / `<name>[...] = ...` so that name "
                        "can be resolved.\n"
                        f"Available tables: {', '.join(table_names)}"
                    )

            # FIX: empty-result check removed from _exec.
            # validate_queries() in the base class is the single authoritative place for
            # this check, and it now gates the rejection through _empty_result_is_error()
            # so multi-table queries are not penalised for sample-data key mismatches.
            # Having the check here too caused double-firing and also bypassed the gate.

            return result

        except Exception as e:
            raise self._normalize_pandas_error(e, source=query)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    @staticmethod
    def _assignment_base_name(node: ast.AST) -> "str | None":
        """Return the base variable name an assignment ultimately targets.

        Unwraps Subscript/Attribute chains (``result["col"] = ...`` -> "result",
        ``df.loc[...] = ...`` -> "df") so both a bare rewrite and an in-place
        mutation are recognized by name. Returns ``None`` for non-assignment
        nodes. Mirrors ``QueryExecutor._assignment_base_name`` so static
        validation and judge-time execution resolve a query's output the same
        way.
        """
        def _base(t: ast.AST) -> ast.AST:
            while isinstance(t, (ast.Subscript, ast.Attribute)):
                t = t.value
            return t

        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            return None

        for t in targets:
            base = _base(t)
            if isinstance(base, ast.Name):
                return base.id
        return None

    # ------------------------------------------------------------------
    # Connectivity check
    # ------------------------------------------------------------------
    def _check_query_connectivity(self, query: str) -> None:
        table_names_set = set(self.table_names)
        var_tables: dict[str, frozenset] = {t: frozenset({t}) for t in table_names_set}

        def tables_of(expr: str) -> frozenset:
            result: set = set()
            for name, ts in var_tables.items():
                if re.search(rf'\b{re.escape(name)}\b', expr):
                    result |= ts
            return frozenset(result)

        def first_positional_arg(arg_str: str) -> str:
            depth = 0
            for i, ch in enumerate(arg_str):
                if ch in ('(', '[', '{'): depth += 1
                elif ch in (')', ']', '}'): depth -= 1
                elif ch == ',' and depth == 0: return arg_str[:i]
            return arg_str

        def resolve(expr: str) -> frozenset:
            accumulated: frozenset = tables_of(expr)
            for m in re.finditer(r'\.merge\s*\(', expr):
                start = m.end()
                depth, i = 1, start
                while i < len(expr) and depth:
                    if expr[i] == '(': depth += 1
                    elif expr[i] == ')': depth -= 1
                    i += 1
                rhs_raw = first_positional_arg(expr[start:i - 1])
                accumulated = accumulated | resolve(rhs_raw)
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
            m = re.match(r'^\s*([A-Za-z_]\w*)\s*=(?!=)\s*(.+)', stmt, re.DOTALL)
            if m:
                return m.group(1), m.group(2).strip()
            return None, stmt

        flat = re.sub(r'\\\s*\n\s*', ' ', query)
        statements = [s.strip() for s in re.split(r'[;\n]', flat) if s.strip()]
        all_sets: list[frozenset] = []

        for stmt in statements:
            lhs, rhs_expr = extract_assignment(stmt)
            resolved = resolve(rhs_expr)
            if lhs:
                var_tables[lhs] = resolved
            all_sets.append(resolved)

        referenced = frozenset(
            t for t in table_names_set
            if re.search(rf'\b{re.escape(t)}\b', query)
        )
        if len(referenced) < 2:
            return

        adjacency: dict[str, set] = defaultdict(set)
        for ts in all_sets:
            nodes = list(ts & referenced)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    adjacency[nodes[i]].add(nodes[j])
                    adjacency[nodes[j]].add(nodes[i])

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
            return

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
            f"Disjoint query — {len(islands)} independent sub-results never combined:\n"
            f"{island_descriptions}\n\n"
            "All tables must be linked via merge(), join(), or concat().\n"
            "  ✓ group1.merge(group2, on='key') / pd.concat([group1, group2])\n"
            "  ✗ group1  # group2 computed but never joined"
        )

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------
    def _preprocess_query(self, query: str) -> str:
        warnings = self._check_pandas_cartesian(query)
        if warnings:
            raise ValueError(
                "Dangerous query — possible Cartesian product:\n"
                + "\n".join(f"  - {w}" for w in warnings)
                + "\nAvoid how='cross', dummy key merges, or merge() without explicit join keys."
            )

        for m in re.finditer(r"'([^']+)'\s*\.str\.", query):
            preceding = query[:m.start()].rstrip()
            if not preceding.endswith('['):
                col = m.group(1)
                raise ValueError(
                    f"'.str' on string literal '{col}' — use it on a Series, not a quoted name.\n"
                    f"  ✓ Table_A.assign(_key=Table_A['{col}'].str.lower()).merge(..., on='_key')\n"
                    f"  ✗ .merge(Table_B, left_on='{col}'.str.lower())"
                )

        cleaned = self._clean_pandas(query)
        self._check_stripped_query(query, cleaned)
        return cleaned

    def _check_stripped_query(self, original: str, cleaned: str) -> None:
        if cleaned:
            return

        lines = [l.strip() for l in re.split(r'[;\n]', original) if l.strip()]
        has_banned   = any(any(cmd in l for cmd in UNAUTHORIZED_COMMANDS) for l in lines)
        has_comments = all(re.sub(r'#.*$', '', l).strip() == '' for l in lines) if lines else False

        reasons = []
        if has_banned:
            banned_found = [cmd for cmd in UNAUTHORIZED_COMMANDS if any(cmd in l for l in lines)]
            reasons.append(f"  - unauthorized commands removed: {', '.join(banned_found)}")
        if has_comments:
            reasons.append("  - query contained only comments, no executable code")
        if not reasons:
            reasons.append("  - all lines removed during cleaning (check for valid pandas/polars expressions)")

        raise ValueError(
            "Query entirely removed during cleaning — cannot execute.\n"
            + "\n".join(reasons) + "\n"
            f"Rewrite using only DataFrame operations on: {', '.join(self.table_names)}"
        )

    def _check_pandas_cartesian(self, query: str) -> list[str]:
        return [
            msg for pattern, msg in PANDAS_CARTESIAN_PATTERNS
            if re.search(pattern, query, re.IGNORECASE | re.DOTALL)
        ]

    def _clean_pandas(self, query_code: str) -> str:
        """Normalise generated code while PRESERVING its line structure.

        Code must stay ordinary multi-line Python end-to-end: the validator
        (_exec), the judge-time executor (QueryExecutor) and the benchmark
        executor all exec the whole source, and QueryValidator writes the
        cleaned code back into ``query["code"]`` — so what is saved in the
        final JSON is byte-identical to what was validated. The old
        ';'-flattening destroyed any code with control flow (``if x:; y``)
        and made per-statement validation disagree with whole-source
        execution downstream.
        """
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

        # Strip comments but keep each line's indentation intact.
        lines = [re.sub(r'#.*$', '', l).rstrip() for l in query.splitlines()]
        lines = [l for l in lines if l.strip()]

        # Imports are kept (not stripped): the static AST check
        # (see _check_imports / _exec) enforces ALLOWED_IMPORT_MODULES
        # before execution instead, so generated code can
        # `import numpy as np` and similar directly rather than relying
        # solely on pre-injected names.
        cleaned_lines = [
            l for l in lines
            if not any(cmd in l for cmd in UNAUTHORIZED_COMMANDS)
        ]
        cleaned = '\n'.join(cleaned_lines)

        if cleaned_lines != lines:
            # Dropping an unauthorized line can break structure (e.g. empty an
            # `if` body). If the filtered code no longer parses while the
            # unfiltered code does, removal isn't safe — reject with feedback
            # instead of saving silently-mutilated code.
            try:
                ast.parse(cleaned)
            except SyntaxError:
                try:
                    ast.parse('\n'.join(lines))
                except SyntaxError:
                    return cleaned  # was already broken — let _exec report it
                dropped = [
                    l.strip() for l in lines
                    if any(cmd in l for cmd in UNAUTHORIZED_COMMANDS)
                ]
                raise ValueError(
                    "Unauthorized command(s) inside a control-flow block "
                    "cannot be removed safely:\n"
                    + "\n".join(f"  - {d}" for d in dropped[:3])
                    + "\nRewrite the query without: "
                    + ", ".join(UNAUTHORIZED_COMMANDS)
                )
        return cleaned

    # ------------------------------------------------------------------
    # Error normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def _syntax_error_snippet(
        source: str, lineno: "int | None", offset: "int | None", context: int = 2
    ) -> str:
        """Numbered source lines around a SyntaxError, with a caret at the
        offending column — so the correction LLM can find the exact spot
        instead of counting lines inside a JSON-escaped code string."""
        if not source or not lineno:
            return ""
        lines = source.splitlines()
        if not (1 <= lineno <= len(lines)):
            return ""

        start = max(1, lineno - context)
        end = min(len(lines), lineno + context)
        rendered = []
        for i in range(start, end + 1):
            marker = ">>" if i == lineno else "  "
            prefix = f"{marker} {i:>4} | "
            rendered.append(f"{prefix}{lines[i - 1]}")
            if i == lineno and offset:
                rendered.append(" " * (len(prefix) + offset - 1) + "^")
        return "\n".join(rendered)

    def _per_table_column_suggestions(self, missing_col: str, max_per_table: int = 4) -> str:
        """Closest real column names for ``missing_col``, grouped BY TABLE.

        Pooling every DataFrame's columns into one flat list (which this used
        to do) can suggest a name that exists only in a DIFFERENT table than
        the one the code was indexing — sending the correction loop to a
        column it still cannot use there. Grouping keeps every suggestion
        attached to the table that actually has it. Returns "" when nothing
        is close enough to suggest.
        """
        if not missing_col:
            return ""
        lines: list[str] = []
        for name, df in zip(self.table_names, self.dataframes):
            columns = getattr(df, "columns", None)
            if columns is None:
                continue
            close = self._suggest_columns(
                missing_col, [str(c) for c in columns], max_suggestions=max_per_table
            )
            if close:
                lines.append(f"  {name}: {', '.join(close)}")
        if not lines:
            return ""
        return "\nClosest real columns, by table:\n" + "\n".join(lines)

    def _normalize_pandas_error(self, e: Exception, source: str = "") -> Exception:
        msg = str(e).lower()
        # A conversion/cast failure (pd.to_numeric, .astype(...), ...) embeds
        # the FULL offending cell value verbatim in str(e) — unbounded, so a
        # huge cell (e.g. a WKT geometry blob) would otherwise blow up the
        # correction prompt below. `raw_full` keeps the untouched message for
        # internal parsing (e.g. the KeyError column-name extraction, which
        # never deals with data-sized values); every `{raw}` interpolated
        # into a message actually sent to the LLM uses the shielded copy.
        raw_full = str(e).strip('"').strip("'")
        raw = summarize_large_value(raw_full)

        if isinstance(e, KeyError):
            col_name = raw_full.strip("'").strip('"')
            suggestion_msg = self._per_table_column_suggestions(col_name)

            if "not in index" in raw.lower() or "not in index" in msg:
                return KeyError(
                    f"{raw}\nColumn not found — missing, or renamed/suffixed by an earlier "
                    f"step.{suggestion_msg}\n"
                    "Fix: reference the post-merge name (_x/_y), or pre-normalise into a key column:\n"
                    "  ✓ Table_A.assign(_key=Table_A['col'].str.lower()).merge(..., on='_key')\n"
                    "  ✗ .merge(Table_B, left_on=Table_A['col'].str.lower())"
                )
            return KeyError(
                f"{raw}\nColumn not found in the table it was read from.{suggestion_msg}\n"
                "Fix: use one of the real column names above, verbatim (they are case- "
                "and whitespace-sensitive); after a merge use the suffixed name (_x/_y). "
                "Do not invent a column that is not listed."
            )

        if isinstance(e, IndexError):
            return IndexError(
                f"{raw}\n.iloc out of bounds — DataFrame may have fewer rows than expected.\n"
                "Use .loc[] for label access, or filter instead of hardcoding positions."
            )

        if isinstance(e, AttributeError):
            if "nonetype" in msg and "has no attribute" in msg:
                attr = re.search(r"has no attribute '([^']+)'", str(e))
                attr_name = attr.group(1) if attr else "unknown"
                return AttributeError(
                    f"{raw}\nChain returned None before .{attr_name} — likely inplace=True on a prior step.\n"
                    "Remove inplace=True and assign explicitly: df = df.sort_values(...)"
                )
            if "str" in msg and "has no attribute" in msg:
                return AttributeError(
                    f"{raw}\n.str on non-string column — cast first: df['col'].astype(str).str.lower()"
                )
            if "series" in msg and "has no attribute" in msg:
                attr = re.search(r"has no attribute '([^']+)'", str(e))
                attr_name = attr.group(1) if attr else "unknown"
                return AttributeError(
                    f"{raw}\nSeries has no attribute '{attr_name}' — preceding op may have collapsed DataFrame to Series."
                )
            if "dataframe" in msg and "has no attribute" in msg:
                attr = re.search(r"has no attribute '([^']+)'", str(e))
                attr_name = attr.group(1) if attr else "unknown"
                return AttributeError(
                    f"{raw}\nDataFrame has no attribute '{attr_name}' — preceding op likely already returned a DataFrame, not a Series.\n"
                    "Common cause: .agg(name=(col, func), ...) called without .groupby() first — named aggregation\n"
                    "only collapses to a single row via groupby; on a plain DataFrame it does not produce a Series,\n"
                    "so a trailing .to_frame() (or other Series-only method) fails."
                )
            return AttributeError(f"{raw}\nBad attribute — check method name and object type at this step.")

        if isinstance(e, NameError):
            name_match = re.search(r"name '([^']+)' is not defined", str(e))
            name = name_match.group(1) if name_match else "unknown"
            if name == "merge":
                return NameError(
                    "merge() must be a DataFrame method, not a bare function.\n"
                    "  ✓ Table_A.merge(Table_B, on='key')  ✗ merge(Table_A, Table_B, on='key')"
                )
            return NameError(
                f"'{name}' is not defined.\n"
                f"Available: pd, pl, {', '.join(self.table_names)}\n"
                "Only pre-imported libraries and listed table variables are in scope."
            )

        if isinstance(e, MemoryError):
            return MemoryError(
                f"{raw}\nOperation exceeded memory limit.\n"
                "Fix: pre-filter rows, select only needed columns, or aggregate before the operation."
            )

        if isinstance(e, ZeroDivisionError):
            return ZeroDivisionError(
                f"{raw}\nDivision by zero — guard with df['col'].replace(0, float('nan')) or filter zero rows first."
            )

        if isinstance(e, ValueError):
            if "you are trying to merge on" in msg or ("merge" in msg and "key" in msg and "not in" in msg):
                cols = re.findall(r"'([^']+)'", str(e))
                col_hint = f" ({', '.join(cols[:2])})" if cols else ""
                return ValueError(
                    f"{raw}\nMerge key error{col_hint} — column absent from one or both tables.\n"
                    "Use left_on=/right_on= if column names differ; check case and spelling."
                )

            if "duplicate columns" in msg and "suffixes" in msg:
                dupes = re.findall(r"'([^']+)'", str(e))
                dupes_str = ", ".join(dupes[:4]) if dupes else "see error"
                return ValueError(
                    f"{raw}\nDuplicate column names after merge: {dupes_str}\n"
                    "Fix: use unique suffixes per step, select only needed columns before merging, "
                    "or .drop(columns=[...]) conflicting columns first."
                )

            if "no objects to concatenate" in msg:
                return ValueError(
                    f"{raw}\npd.concat received empty list — all DataFrames may be empty due to over-filtering."
                )

            if "must pass list-like" in msg or "no numeric data to plot" in msg:
                return ValueError(
                    f"{raw}\nagg/apply got wrong type — cast with .astype(float) or check groupby column selection."
                )

            if "cannot reindex" in msg or ("reindex" in msg and "not unique" in msg):
                return ValueError(
                    f"{raw}\nDuplicate index — call .reset_index(drop=True) after merges or groupby."
                )

            if "out of bounds" in msg and ("datetime" in msg or "date" in msg):
                return ValueError(
                    f"{raw}\nDatetime out of bounds (valid range ~1677–2262).\n"
                    "Use pd.to_datetime(..., errors='coerce') to convert bad values to NaT."
                )

            if "empty" in msg and any(x in msg for x in [
                "no objects", "nothing to", "no columns", "no rows", "result is empty"
            ]):
                return ValueError(self._build_empty_result_feedback())

            if "grouper" in msg and "not 1-dimensional" in msg:
                return ValueError(
                    f"{raw}\nGroup key resolved to several columns — the name is duplicated "
                    "in the frame (usually both sides of a merge kept it).\n"
                    "Fix: group on the suffixed name (_x/_y), or drop/rename the duplicate "
                    "right after the merge that introduced it."
                )

            if "already exists" in msg and ("cannot insert" in msg or "reset_index" in msg):
                return ValueError(
                    f"{raw}\n.reset_index() would overwrite an existing column of the same name.\n"
                    "Fix: .reset_index(drop=True), or rename the index before resetting."
                )

            if "length of values" in msg and "length of index" in msg:
                return ValueError(
                    f"{raw}\nAssigned a sequence whose length differs from the frame's row count "
                    "— usually a value computed from a DIFFERENT (filtered or grouped) frame.\n"
                    "Fix: compute the value on the same frame you assign it to, or align via "
                    ".merge()/.map() on a key instead of positional assignment."
                )

            if "tz-naive" in msg or "tz-aware" in msg:
                return ValueError(
                    f"{raw}\nComparing timezone-aware and timezone-naive datetimes.\n"
                    "Fix: normalise both sides — pd.to_datetime(col, utc=True), or "
                    ".dt.tz_localize(None) on the aware side."
                )

            if "agg function failed" in msg or "could not convert string" in msg:
                return ValueError(
                    f"{raw}\nAggregation hit non-numeric values in a column it had to sum/average.\n"
                    "Fix: pd.to_numeric(df['col'], errors='coerce') before aggregating — a raw "
                    ".astype(float) raises on the first stray token (see the clean step and "
                    "Column Statistics' bad_token_counts)."
                )

            return ValueError(
                f"{raw}\nValue error.\nFix: check the dtypes and the exact arguments of the "
                "operation on the line above — most often a column holding text where a "
                "number is required."
            )

        if isinstance(e, TypeError):
            if "unsupported operand" in msg:
                ops = re.search(r"unsupported operand type\(s\) for (.+?): '([^']+)' and '([^']+)'", str(e))
                if ops:
                    op, t1, t2 = ops.group(1), ops.group(2), ops.group(3)
                    return TypeError(
                        f"{raw}\nType mismatch for '{op}': '{t1}' vs '{t2}' — cast with .astype(float) or pd.to_numeric()."
                    )
            if "could not convert" in msg or "invalid literal" in msg:
                return TypeError(
                    f"{raw}\nConversion error — non-numeric values in column. Use pd.to_numeric(df['col'], errors='coerce')."
                )
            if "dtype" in msg:
                return TypeError(f"{raw}\nDtype mismatch — check df.dtypes and cast before operating.")
            return TypeError(f"{raw}\nType error — check column dtypes and operation compatibility.")

        if isinstance(e, (SyntaxError, IndentationError, TabError)):
            lineno = getattr(e, "lineno", None)
            offset = getattr(e, "offset", None)
            detail = getattr(e, "msg", None) or raw
            location = f" at line {lineno}" if lineno else ""
            snippet = self._syntax_error_snippet(source, lineno, offset)
            parts = [f"Syntax error{location}: {detail}"]
            if snippet:
                parts.append(f"Offending code:\n{snippet}")
            parts.append(
                "Write ordinary multi-line Python: one statement per line, "
                "standard indentation for control-flow blocks. Do NOT join "
                "statements with semicolons."
            )
            return SyntaxError("\n".join(parts))

        # Fallback: wrap unknown exception types with type name and ≤200 chars
        wrapped_msg = self._wrap_unknown_exception(e)
        return type(e)(f"{wrapped_msg}\nUnexpected error — review the operation and input data.")

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------
    # Shared tail for every empty-result variant. A genuinely correct query over
    # data that simply has no matching rows is a PLAN problem, not a code one —
    # the pipeline routes a plan-compliant empty result to a plan-level retry
    # (see StatementOrchestrator._retry_empty_results), so the correction model
    # must not "fix" emptiness by loosening a filter the question actually
    # requires. Saying so here is what keeps it from silently answering a
    # different question than the one it was given.
    _EMPTY_RESULT_TAIL = (
        "Only widen a predicate the question does not actually require. If every "
        "filter is faithful to the question and the data genuinely has no matching "
        "rows, keep the query as it is and say so in `motivation` — do not broaden "
        "the scope to force rows."
    )

    def _build_empty_result_feedback(self, query: str = "") -> str:
        q = query.lower()
        if "concat" in q:
            head = (
                "Empty result — at least one pd.concat input is empty, so the union has "
                "no rows.\n"
                "Fix, in order: (1) check each branch separately — one filter is "
                "excluding everything; (2) normalise string comparisons with "
                "df['col'].str.strip().str.lower() == value.lower(); (3) cast numeric "
                "text with pd.to_numeric(df['col'], errors='coerce') before comparing."
            )
        elif len(self.table_names) == 1:
            head = (
                "Empty result — the filter chain matched no rows.\n"
                "Fix, in order: (1) drop the predicates one at a time to find which one "
                "empties the frame; (2) normalise string comparisons with "
                "df['col'].str.strip().str.lower() == value.lower() (case/whitespace "
                "mismatch silently drops everything); (3) cast numeric text with "
                "pd.to_numeric(df['col'], errors='coerce'); (4) check the date range "
                "against the column's real min/max in Column Statistics."
            )
        else:
            head = (
                "Empty result — the merge produced no matching rows, or a filter after "
                "it removed them all.\n"
                "Fix, in order: (1) switch the merge to how='left' to see which side "
                "fails to match; (2) normalise the join keys on BOTH sides before "
                "merging — t1.assign(_k=t1['key'].str.strip().str.lower()).merge("
                "t2.assign(_k=t2['key'].str.strip().str.lower()), on='_k'); (3) confirm "
                "the two key columns hold the same kind of value (Column Statistics' "
                "top_values shows both); (4) then drop post-merge filters one at a time."
            )
        return f"{head}\n{self._EMPTY_RESULT_TAIL}"

    def _get_language_name(self) -> str:
        return 'Python'