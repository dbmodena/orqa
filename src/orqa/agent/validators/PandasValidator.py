"""Static + sandboxed validation of generated pandas code, with feedback the
correction model can act on.

Three executors run the same code, and the validator must agree with all of
them, or code that passes here fails downstream (or the reverse):

* this sandbox (``_exec``);
* the judge-time executor (``orqa.queries.query_execution.QueryExecutor``):
  ``exec`` with full builtins and the tables as ``Table_i`` variables;
* the benchmark harness (LakeGen's reference execution): the same, with
  only pandas and numpy importable and the names / attributes in
  ``FORBIDDEN_NAMES`` / ``FORBIDDEN_ATTRIBUTES`` rejected statically.

So the sandbox runs the code AS WRITTEN (no quote, comment or line
rewriting: the validated text is what gets saved as the reference) with
full builtins minus the forbidden names, and enforces the benchmark's
static policy up front with a message that says why.
"""

import ast
import builtins
import functools
import re
import sys
from collections import defaultdict
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd

from .QueryValidator import QueryValidator
from ..utility.undecodable_names import escaped_literal, is_undecodable, undecodable_matches
from ...utils import summarize_large_value

# --- The benchmark's static policy (mirrors LakeGen reference_execution) ---
ALLOWED_IMPORT_MODULES = {"pandas", "numpy"}
FORBIDDEN_NAMES = {
    "__import__", "breakpoint", "compile", "eval", "exec", "globals", "help",
    "input", "locals", "open", "os", "pathlib", "shutil", "subprocess", "sys",
}
FORBIDDEN_ATTRIBUTES = {
    "read_csv", "read_excel", "read_feather", "read_html", "read_json",
    "read_orc", "read_parquet", "read_pickle", "read_sql",
    "to_clipboard", "to_csv", "to_excel", "to_feather", "to_hdf", "to_json",
    "to_orc", "to_parquet", "to_pickle", "to_sql",
}
# Builtins the sandbox withholds: the forbidden names that are builtins
# (``__import__`` stays — import statements need it; writing it in the code
# is what the static check rejects) plus the interpreter exits.
_BUILTINS_WITHHELD = (FORBIDDEN_NAMES - {"__import__"}) | {"exit", "quit"}

_POLICY_WHY = (
    "the benchmark re-executes this code with only pandas and numpy "
    "importable and no file or system access"
)


def _check_code_policy(tree: ast.AST) -> None:
    """Reject, before execution, what the benchmark harness rejects:
    imports beyond pandas/numpy, forbidden names, I/O and dunder attributes.
    Static (an AST walk), never a runtime hook: C/Cython library code that
    lazily imports a module resolves ``__import__`` from the generated
    code's frame, so a runtime hook would reject imports the code never
    wrote."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module and not node.level else [""]
            else:
                names = [alias.name for alias in node.names]
            for name in names:
                if name.split(".")[0] not in ALLOWED_IMPORT_MODULES:
                    raise ImportError(
                        f"`import {name or '.'}` is not allowed: {_POLICY_WHY} "
                        "(`import numpy as np` is fine). Rewrite without it: pandas "
                        "string methods take regex=True for pattern work, "
                        "pd.to_datetime / pd.Timestamp / pd.Timedelta replace datetime, "
                        "numpy covers the maths."
                    )
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(
                f"`{node.id}` is not allowed in generated code: {_POLICY_WHY}. "
                "Compute the answer from the table variables only, with pandas and numpy."
            )
        elif isinstance(node, ast.Attribute) and (
            node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRIBUTES
        ):
            raise ValueError(
                f"`.{node.attr}` is not allowed: {_POLICY_WHY}. The tables are already "
                "loaded as the Table_i variables; end the code with the result value "
                "instead of reading or writing files, and never touch dunder attributes."
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
    (
        r'(?:left_on|right_on)\s*=\s*[A-Za-z_]\w*\s*\[',
        "Stale Series as left_on/right_on — passing Table_X['col'] misaligns rows after a prior merge.\n"
        "  ✓ Table_A.assign(_key=Table_A['col'].str.lower()).merge(Table_B.assign(_key=...), on='_key')\n"
        "  ✗ .merge(Table_C, left_on=Table_A['col'].str.lower())"
    ),
]

# A join/concat/corr whose output is this many times larger than its largest
# input is a many-to-many blow-up (a cartesian product, a non-unique key on
# both sides): the ratio, not the absolute size, tells a correct left join
# on a 64M-row table from an explosion.
_EXPLOSION_RATIO = 5
_EXPLOSION_MIN_ROWS = 1_000_000

PANDAS_DANGEROUS_OPS = {
    "merge": {
        "type": "method", "check": "result",
        "error": "merge produced {rows:,} rows from inputs of at most {inputs:,} rows ({mb:.0f} MB).\n"
                 "Cause: the join key is not unique on either side (every left row matched many right rows), or the keys are a cartesian product.\n"
                 "Fix: make the key unique on one side (deduplicate or pre-aggregate it per key), join on the full key the plan declares, or filter before joining.",
    },
    "join": {
        "type": "method", "check": "result",
        "error": "join produced {rows:,} rows from inputs of at most {inputs:,} rows ({mb:.0f} MB).\n"
                 "Cause: the index is not unique on either side. Fix: deduplicate or pre-aggregate one side per key, or use merge(on=...) with the full key.",
    },
    "corr": {
        "type": "method", "check": "input",
        "error": "corr() over {rows:,} rows x {cols} columns ({mb:.0f} MB) exceeds the memory limit.\n"
                 "Fix: select the two or three numeric columns to correlate (`df[['a', 'b']].corr()`) instead of the whole frame.",
    },
    "concat": {
        "type": "function", "check": "result",
        "error": "pd.concat produced {rows:,} rows ({mb:.0f} MB), over the limit.\n"
                 "Fix: concatenate the filtered or aggregated frames, not the raw tables.",
    },
}


def _check_dataframe(df: pd.DataFrame, error_template: str, op_name: str, max_mb: float,
                     inputs: int = 0) -> None:
    rows = len(df)
    exploded = inputs and rows > _EXPLOSION_MIN_ROWS and rows > inputs * _EXPLOSION_RATIO
    if not exploded and rows <= _EXPLOSION_MIN_ROWS:
        return  # small outputs are never worth a deep memory scan
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    if exploded or mb > max_mb:
        err = MemoryError(
            f"Operation '{op_name}' exceeded the limits:\n"
            + error_template.format(rows=rows, mb=mb, inputs=inputs, cols=df.shape[1])
        )
        err.orqa_feedback = True  # already a diagnosis: not re-wrapped by _normalize_pandas_error
        raise err


def _patch_pandas(max_mb: float):
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
                        _check_dataframe(self_df, t, name, max_mb, inputs=len(self_df))
                    result = orig(self_df, *args, **kwargs)
                    if chk == "result" and isinstance(result, pd.DataFrame):
                        other = args[0] if args else kwargs.get("right", kwargs.get("other"))
                        inputs = max(len(self_df), len(other) if hasattr(other, "__len__") else 0)
                        _check_dataframe(result, t, name, max_mb, inputs=inputs)
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
                        _check_dataframe(result, t, name, max_mb)
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


# ---------------------------------------------------------------------------
# Hints shared by several error branches. They point the correction model at
# the Column Statistics fact that says how the value is stored and at the
# recipe the generation prompt teaches for it.
# ---------------------------------------------------------------------------

NUMBER_TEXT_HINT = (
    "The column holds number TEXT. Cast it with the recipe of its `number text` fact in "
    "Column Statistics: strip the shown affix, remove the shown thousands separator, replace "
    "the shown decimal mark with '.', then `pd.to_numeric(..., errors='coerce')` — e.g. "
    "`thousands comma`: `pd.to_numeric(s.str.strip().str.replace(',', '', regex=False), errors='coerce')`; "
    "`decimal comma`: `.str.replace(',', '.', regex=False)` first. Never `.astype(float)` / "
    "`int(...)` / `.astype(int)`: they raise on the first stray value. Values listed under "
    "`not numbers` / `missing tokens` become NaN, which is intended."
)
DATE_TEXT_HINT = (
    "The column holds date TEXT. Parse it with the exact format its `date text` fact in Column "
    "Statistics shows, before any `.dt` access, date comparison or subtraction: "
    "`pd.to_datetime(s.str.strip(), format='<format>', errors='coerce')` (ISO: `format='ISO8601'`; "
    "with an `offsets` fact follow the offsets recipe: `utc=True` then `.dt.tz_localize(None)`). "
    "Values listed under `not dates` become NaT, which is intended. Compare parsed dates with "
    "`pd.Timestamp('YYYY-MM-DD')`, never with a string."
)
EMPTY_FRAME_HINT = (
    "The frame is EMPTY at this point: an earlier filter matched no rows (a value spelled "
    "differently from the stored one — check the exact spelling, case and whitespace in Column "
    "Statistics — a text column compared as a number, or a date range outside the column's "
    "parsed min..max). Fix the filter; only then take a row/argmax from the result."
)
QUERY_STRING_HINT = (
    "Inside `.query()` / `.eval()` strings: wrap a column name holding spaces, punctuation or "
    "starting with a digit in backticks (`` `Age (years)` ``, `` `2018` ``), quote string "
    "literals, and reference Python variables with @name. Plain boolean indexing "
    "(`df[df['col'] == value]`) is the safer form."
)

_STR_METHODS = {
    "lower", "upper", "strip", "lstrip", "rstrip", "replace", "contains", "startswith",
    "endswith", "split", "len", "title", "capitalize", "zfill", "extract", "findall",
    "match", "fullmatch", "slice", "pad", "isnumeric", "isdigit", "isalpha", "get",
    "join", "cat", "count", "find", "normalize", "removeprefix", "removesuffix",
}
_DT_ATTRIBUTES = {
    "year", "month", "day", "hour", "minute", "second", "date", "time", "dayofweek",
    "weekday", "day_name", "month_name", "quarter", "days", "strftime", "to_period",
    "floor", "ceil", "round", "normalize", "dayofyear", "week", "isocalendar", "total_seconds",
}
_RESULT_ROWS_KEPT = 10_000


def _is_tz_mismatch(msg: str) -> bool:
    """A zoned and a naive datetime met (``msg`` lowercased): "cannot subtract
    tz-naive and tz-aware ...", or pandas 2.3's "invalid comparison between
    dtype=datetime64[ns, <zone>] and timestamp"."""
    return "tz-naive" in msg or "tz-aware" in msg or (
        "invalid comparison" in msg and re.search(r"datetime64\[\w+, ", msg) is not None
    )


def _tz_mismatch_message(raw: str) -> str:
    return (
        f"{raw}\nComparing timezone-aware and timezone-naive datetimes.\n"
        "Fix: put both sides on naive UTC (Column Statistics dates are UTC): "
        "on the aware side .dt.tz_convert('UTC').dt.tz_localize(None) — a bare "
        ".dt.tz_localize(None) keeps the local clock, not UTC; parse date text "
        "that carries offsets with the recipe for its `offsets` fact."
    )


def _feedback(message: str) -> ValueError:
    """A ValueError the validator raises itself: already feedback, so
    ``_normalize_pandas_error`` passes it through untouched."""
    err = ValueError(message)
    err.orqa_feedback = True
    return err


def _is_print_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"


def _value_hint(value: str) -> str:
    """What a stored value that failed a numeric cast tells about the column."""
    v = value.strip()
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", v):
        return f"`{value}` carries a thousands comma"
    if re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+(,\d+)?", v) or re.fullmatch(r"[-+]?\d+,\d+", v):
        return f"`{value}` uses a decimal comma (and/or a thousands dot)"
    if re.fullmatch(r"[-+]?\d{1,3}([  ']\d{3})+([.,]\d+)?", v):
        return f"`{value}` carries a thousands space/apostrophe"
    if re.search(r"\d", v) and re.search(r"[^\d.,\s+-]", v):
        return f"`{value}` has a non-numeric affix or unit around the digits"
    if re.search(r"\d", v):
        return f"`{value}` is not a plain number"
    return (
        f"`{value}` is not a number at all: such rows are listed under `not numbers` / "
        "`missing tokens` in Column Statistics and must become NaN (errors='coerce'), "
        "not break the cast"
    )


class PandasValidator(QueryValidator):
    """Validator for pandas queries."""

    def _empty_result_is_error(self, ordered_names: list) -> bool:
        """Single-table: an empty result means a filter matched nothing.
        Multi-table: allowed — the structure is validated (table usage,
        field coverage) but zero rows from keys that genuinely do not match
        is a plan-level outcome, not a code bug."""
        return len(ordered_names) == 1

    def _run_query(self, query: str, dataframes: list, table_names: list) -> Any:
        restore = _patch_pandas(max_mb=self.mem_limit_mb)
        try:
            return self._exec(query, dataframes, table_names)
        finally:
            restore()

    def _exec(self, query: str, dataframes: list, table_names: list) -> Any:
        # Full builtins minus the forbidden names: what the judge-time
        # executor and the benchmark harness give the same code.
        safe_builtins = {k: v for k, v in vars(builtins).items() if k not in _BUILTINS_WITHHELD}
        # `np` is not pre-bound, as at judge time: the code imports numpy
        # itself (`import numpy as np`), which every executor accepts.
        local_ns = {'pd': pd, '__builtins__': safe_builtins}
        for df, name in zip(dataframes, table_names):
            local_ns[name] = df

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            # Ordinary multi-line Python, the same execution model as the
            # judge-time executor (QueryExecutor._inject_result + exec) and
            # the benchmark executor: the whole source is parsed, everything
            # but the last statement is exec'd, and the last statement gives
            # the result (a bare expression, or the name it assigns).
            tree = ast.parse(query)
            if not tree.body:
                return None
            _check_code_policy(tree)

            last = tree.body[-1]
            if isinstance(last, ast.Expr):
                if _is_print_call(last.value):
                    raise _feedback(
                        "The last statement is print(...), which is not a value: the "
                        "result of the code is its last expression or assignment. End "
                        "with the value itself (`result = ...` or a bare `result`) and "
                        "drop the print."
                    )
                if tree.body[:-1]:
                    head = ast.Module(body=tree.body[:-1], type_ignores=[])
                    exec(compile(head, '<string>', 'exec'), local_ns)
                result = eval(
                    compile(ast.Expression(body=last.value), '<string>', 'eval'),
                    local_ns,
                )
            else:
                exec(compile(tree, '<string>', 'exec'), local_ns)
                target_name = self._assignment_base_name(last)
                if not target_name:
                    raise _feedback(
                        "Last statement did not resolve to a value.\n"
                        "End with a bare variable name (not a loop, an if block or a "
                        "function definition), or with `<name> = ...` / `<name>[...] = ...` "
                        "so that name can be resolved.\n"
                        f"Available tables: {', '.join(table_names)}"
                    )
                result = local_ns.get(target_name)
            if result is None:
                raise _feedback(
                    "The code's final value is None, not a result. The last statement "
                    "evaluated to nothing: a method called with inplace=True (returns "
                    "None — assign its return value without inplace instead), a print(...), "
                    "or a variable never assigned. End with the DataFrame, Series or scalar "
                    "that answers the question."
                )
            return self._shrink_result(result)

        except Exception as e:
            raise self._normalize_pandas_error(e, source=query)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _shrink_result(self, result: Any) -> Any:
        """What crosses back from the sandbox: the size check on the full
        result, the all-null columns noted on it, then at most
        ``_RESULT_ROWS_KEPT`` rows (a 46M-row frame pickled through the
        queue took minutes and said nothing more than its head)."""
        try:
            self._check_result_size(result)
        except MemoryError as err:
            err.orqa_feedback = True
            raise
        if isinstance(result, (pd.DataFrame, pd.Series)):
            total = len(result)
            null_columns = self._all_null_columns(result)
            if total > _RESULT_ROWS_KEPT:
                result = result.head(_RESULT_ROWS_KEPT)
            try:
                result.attrs["orqa_total_rows"] = total
                result.attrs["orqa_null_columns"] = null_columns
            except Exception:  # a subclass without attrs: the head is still fine
                pass
        return result

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
    # Connectivity check (not wired in: kept for a future opt-in)
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
            for m in re.finditer(r'pd\.concat\s*\(\s*\[', expr):
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
    # Pre-processing: static checks only; the code is never rewritten
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

        try:
            tree = ast.parse(query)
        except SyntaxError as e:
            raise self._normalize_pandas_error(e, source=query) from None
        if not tree.body:
            raise ValueError(
                "The code holds no executable statement (only comments or blank lines).\n"
                f"Write the pandas code that answers the question on: {', '.join(self.table_names)}"
            )
        _check_code_policy(tree)
        return query

    def _check_pandas_cartesian(self, query: str) -> list[str]:
        return [
            msg for pattern, msg in PANDAS_CARTESIAN_PATTERNS
            if re.search(pattern, query, re.IGNORECASE | re.DOTALL)
        ]

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

    def _tables_holding(self, column: str) -> list[str]:
        """The table aliases whose frame has a column named exactly ``column``."""
        return [
            name for name, df in zip(self.table_names, self.dataframes)
            if column in [str(c) for c in getattr(df, "columns", [])]
        ]

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
                # A header holding U+FFFD is shown as the literal to type.
                shown = (escaped_literal(c) if is_undecodable(c) else c for c in close)
                lines.append(f"  {name}: {', '.join(shown)}")
        if not lines:
            return ""
        return "\nClosest real columns, by table:\n" + "\n".join(lines)

    @staticmethod
    def _keyerror_names(raw: str) -> list[str]:
        """The label(s) a pandas KeyError is about, from its message forms:
        ``'col'``, ``"['a', 'b'] not in index"``, ``"['x'] not found in axis"``,
        ``"None of [Index(['a', 'b'], dtype='object')] are in the [columns]"``,
        ``"Column 'x' does not exist!"``."""
        m = re.search(r"\[Index\(\[(.*?)\](?:,|\))", raw, re.S) or re.search(
            r"^\[(.*?)\] not (?:in index|found in axis)", raw, re.S
        )
        if m:
            names = re.findall(r"'((?:[^'\\]|\\.)*)'", m.group(1)) or re.findall(r'"([^"]*)"', m.group(1))
            if names:
                return names
        m = re.search(r"Column '(.+?)' does not exist", raw)
        if m:
            return [m.group(1)]
        return [raw.strip("'\"")]

    def _key_error(self, raw_full: str, raw: str, msg: str, source: str) -> KeyError:
        names = self._keyerror_names(raw_full)
        shown = ", ".join(f"'{n}'" for n in names[:4])
        held = {n: self._tables_holding(n) for n in names}
        if names and all(held[n] for n in names):
            where = "; ".join(f"'{n}' is a column of {', '.join(held[n])}" for n in names[:4])
            return KeyError(
                f"{raw}\nThe column exists ({where}) but not in the frame indexed at this "
                "step. After groupby(...) / set_index(...) it is the INDEX, not a column: "
                "add .reset_index() (or groupby(..., as_index=False)) before indexing it; "
                "after a merge it may carry a suffix (_x/_y) or have been dropped by a "
                "column selection; after .agg(...) the result holds the aggregate names only."
            )
        every = [str(c) for df in self.dataframes for c in getattr(df, "columns", [])]
        damaged = {n: undecodable_matches(n, every) for n in names[:4]}
        damaged = {n: m for n, m in damaged.items() if m}
        if damaged:
            # "Total £" for the header "Total �": the model typed the
            # symbol it thinks the undecodable byte was, and retypes it when
            # shown the real name raw.
            real = "; ".join(
                f"'{n}' is {' or '.join(escaped_literal(r) for r in m)}" for n, m in damaged.items()
            )
            example = escaped_literal(next(iter(damaged.values()))[0])
            return KeyError(
                f"{raw}\nColumn {shown} not found: the real header holds U+FFFD, a character the "
                f"source file could not decode, and it is part of the name ({real}). Write the "
                f"name exactly so, with the ASCII escape \\ufffd in a normal (not raw) string, "
                f"e.g. df[{example}] — never the symbol you think was meant."
            )
        suggestion_msg = "".join(self._per_table_column_suggestions(n) for n in names[:3])
        if not suggestion_msg and re.search(r"\.(loc|at|xs)\s*[\[(]", source or ""):
            return KeyError(
                f"{raw}\nNo column is named {shown}: this KeyError is about a ROW LABEL / "
                "stored VALUE looked up with .loc/.at, which is absent. Values must be "
                "spelled exactly as stored (see the quoted values in Column Statistics — "
                "case, accents, whitespace); filter with a boolean mask "
                "(`df[df['col'] == value]`) rather than indexing a label that may not exist."
            )
        if "not in index" in msg or "not found in axis" in msg or "none of" in msg:
            return KeyError(
                f"{raw}\nColumn(s) {shown} not found — missing, misspelled, or renamed/suffixed "
                f"by an earlier step.{suggestion_msg}\n"
                "Fix: use the real column names, verbatim (case- and whitespace-sensitive); "
                "after a merge use the suffixed name (_x/_y); never invent a column."
            )
        return KeyError(
            f"{raw}\nColumn {shown} not found in the frame it was read from.{suggestion_msg}\n"
            "Fix: use one of the real column names above, verbatim (they are case- "
            "and whitespace-sensitive); after a merge use the suffixed name (_x/_y). "
            "Do not invent a column that is not listed."
        )

    def _attribute_error(self, raw: str, msg: str, e: Exception) -> AttributeError:
        attr_m = re.search(r"has no attribute '([^']+)'", str(e))
        attr = attr_m.group(1) if attr_m else ""
        if "nonetype" in msg and "has no attribute" in msg:
            return AttributeError(
                f"{raw}\nChain returned None before .{attr or 'unknown'} — likely inplace=True "
                "on a prior step (every inplace method returns None).\n"
                "Remove inplace=True and assign explicitly: df = df.sort_values(...)"
            )
        if "can only use .dt accessor" in msg:
            return AttributeError(f"{raw}\n.dt used on a column that is not parsed yet. {DATE_TEXT_HINT}")
        if "str accessor" in msg and "string values" in msg:
            return AttributeError(
                f"{raw}\n.str accessor used on a column pandas already parsed as "
                "NUMERIC, not text — .str only works on object/string-dtype "
                "columns, and a numeric column has no string form to clean.\n"
                "Fix: check the column's type in Column Statistics (`(int)`, `(float)`) "
                "instead of assuming it's still text. If it's already numeric, "
                "drop the .str step entirely — the thousands-separator/formatting "
                "was handled at load time. If it's genuinely object-dtype "
                "elsewhere but this column loaded differently, cast explicitly: "
                "df['col'].astype(str).str.replace(',', '', regex=False)."
            )
        if ("'series' object" in msg or "'index' object" in msg) and attr in _STR_METHODS:
            return AttributeError(
                f"{raw}\nString methods on a Series go through the .str accessor: "
                f"`s.str.{attr}(...)`, chained as `s.str.strip().str.lower()` — a bare "
                f"`.{attr}()` exists only on a single Python str."
            )
        if ("'series' object" in msg or "'index' object" in msg) and attr in _DT_ATTRIBUTES:
            return AttributeError(
                f"{raw}\n`.{attr}` is a datetime property: reach it through `.dt.{attr}` on a "
                f"PARSED datetime column. {DATE_TEXT_HINT}"
            )
        if "groupby' object has no attribute" in msg:
            return AttributeError(
                f"{raw}\nA GroupBy object has no attribute '{attr}': aggregate it first — "
                "`.size()`, `.agg(name=('col', 'sum'))`, `['col'].sum()` / `.mean()` — or, "
                f"if '{attr}' is a column, select it as `.groupby(...)['{attr}']`. Then "
                "`.reset_index()` to get the keys back as columns."
            )
        if "'str' object has no attribute" in msg or re.search(r"'(float|int|numpy\.\w+|timestamp)' object has no attribute", msg):
            return AttributeError(
                f"{raw}\nThe value is already a SCALAR here (a single str/number/Timestamp "
                "after .iloc[...], .max(), .sum(), .idxmax(), ...), not a Series or "
                f"DataFrame — '{attr}' is a Series/DataFrame method. Call it one step "
                "earlier, on the column or frame, or finish with the scalar as the result."
            )
        if "str" in msg and "has no attribute" in msg and attr == "str":
            return AttributeError(
                f"{raw}\n.str on a non-text object — select a column first "
                "(df['col'].str.lower()); a DataFrame has no .str."
            )
        if "'series' object" in msg and "has no attribute" in msg:
            return AttributeError(
                f"{raw}\nSeries has no attribute '{attr}' — a preceding operation already "
                "collapsed the DataFrame to a Series (a single-column selection, an "
                "aggregation); apply DataFrame methods before that step, or use "
                "`.to_frame()` / `.reset_index()` to get a DataFrame back."
            )
        if "'dataframe' object" in msg and "has no attribute" in msg:
            held = self._tables_holding(attr)
            if held:
                return AttributeError(
                    f"{raw}\n'{attr}' is a column (of {', '.join(held)}) — read it with "
                    f"`df['{attr}']`, not attribute access (names with spaces, punctuation "
                    "or digits never work as attributes), and only on a frame that still "
                    "holds it (not after an aggregation that dropped it)."
                )
            if attr in _STR_METHODS:
                return AttributeError(
                    f"{raw}\n`.{attr}` is a string method: select the text column first — "
                    f"`df['col'].str.{attr}(...)` — a DataFrame has no .str."
                )
            return AttributeError(
                f"{raw}\nDataFrame has no attribute '{attr}' — preceding op likely already returned a DataFrame, not a Series.\n"
                "Common cause: .agg(name=(col, func), ...) called without .groupby() first — named aggregation\n"
                "only collapses to a single row via groupby; on a plain DataFrame it does not produce a Series,\n"
                "so a trailing .to_frame() (or other Series-only method) fails."
            )
        if "module 'pandas' has no attribute" in msg or "module 'numpy' has no attribute" in msg:
            return AttributeError(
                f"{raw}\n'{attr}' is not a function of that library — check the exact pandas / "
                "numpy name (pd.to_numeric, pd.to_datetime, pd.Timestamp, np.where, np.select, "
                "np.isfinite, ...)."
            )
        return AttributeError(f"{raw}\nBad attribute — check method name and object type at this step.")

    def _name_error(self, e: Exception, source: str) -> NameError:
        name_match = re.search(r"name '([^']+)' is not defined", str(e))
        name = name_match.group(1) if name_match else "unknown"
        tables = ", ".join(self.table_names)
        if name == "merge":
            return NameError(
                "merge() must be a DataFrame method, not a bare function.\n"
                "  ✓ Table_A.merge(Table_B, on='key')  ✗ merge(Table_A, Table_B, on='key')"
            )
        if name in ("np", "numpy"):
            return NameError(
                "'np' is not defined: numpy is not pre-bound. Add `import numpy as np` as the "
                "first line (pandas and numpy are the only importable modules)."
            )
        if type(e).__name__ == "UndefinedVariableError" or re.search(r"\.(query|eval)\s*\(", source or ""):
            return NameError(
                f"'{name}' is not defined inside a .query()/.eval() string. {QUERY_STRING_HINT}"
            )
        if name in ("df", "data", "table", "dataframe", "frame"):
            return NameError(
                f"'{name}' is not defined: there is no generic frame. The tables are the "
                f"variables {tables} (use them directly); pd is pandas; add "
                "`import numpy as np` for numpy."
            )
        return NameError(
            f"'{name}' is not defined.\n"
            f"In scope: pd, the tables {tables}, Python builtins, and anything the code "
            "assigns on an earlier line (define it first, in the same code — nothing "
            "carries over from elsewhere). numpy needs `import numpy as np`."
        )

    def _value_error(self, raw: str, msg: str, e: Exception, source: str) -> ValueError:
        full = str(e)
        if "you are trying to merge on" in msg:
            types = re.search(r"merge on (\S+) and (\S+) columns for key '([^']+)'", full)
            left, right, key = types.groups() if types else ("?", "?", "the key")
            return ValueError(
                f"{raw}\nThe join key '{key}' has DIFFERENT dtypes on the two sides ({left} vs "
                f"{right}): the column is absent from neither table. Cast both sides to the same "
                "type before merging — a code stored as text on one side: `.astype(str)` on the "
                "numeric side, or the number-text recipe (`pd.to_numeric(..., errors='coerce')`) "
                "on the text side, as Column Statistics shows the two columns are stored; strip "
                "whitespace and leading zeros only if both sides agree on them."
            )
        if "merge" in msg and "key" in msg and "not in" in msg:
            cols = re.findall(r"'([^']+)'", full)
            col_hint = f" ({', '.join(cols[:2])})" if cols else ""
            return ValueError(
                f"{raw}\nMerge key error{col_hint} — column absent from one or both tables.\n"
                "Use left_on=/right_on= if column names differ; check case and spelling."
            )
        if "duplicate columns" in msg and "suffixes" in msg:
            dupes = re.findall(r"'([^']+)'", full)
            dupes_str = ", ".join(dupes[:4]) if dupes else "see error"
            return ValueError(
                f"{raw}\nDuplicate column names after merge: {dupes_str}\n"
                "Fix: use unique suffixes per step, select only needed columns before merging, "
                "or .drop(columns=[...]) conflicting columns first."
            )
        if "columns overlap but no suffix" in msg:
            return ValueError(
                f"{raw}\n.join() found the same column names on both sides. Use "
                "`.merge(other, on=key, suffixes=('', '_other'))` (or `.join(other, "
                "lsuffix=..., rsuffix=...)`), and select only the columns you need from "
                "each side before joining."
            )
        if "no objects to concatenate" in msg:
            return ValueError(
                f"{raw}\npd.concat received an empty list — every branch was empty or the "
                "list comprehension produced nothing. Check each branch's filter separately."
            )
        if "invalid literal for int()" in msg or "could not convert string to float" in msg:
            value = re.search(r": '(.*)'$", full, re.S)
            shown = _value_hint(value.group(1)) if value else "a stored value is not a plain number"
            via = "a .corr()/aggregation" if re.search(r"\.(corr|mean|sum|median|std)\s*\(", source or "") else "a cast"
            return ValueError(f"{raw}\n{via} hit text: {shown}. {NUMBER_TEXT_HINT}")
        if "doesn't match format" in msg or "unconverted data remains" in msg or "unknown datetime string format" in msg \
                or "unable to parse" in msg or "not likely a datetime" in msg or "out of bounds nanosecond" in msg:
            fmt = re.search(r'format "([^"]+)"', full)
            tried = f" (`{fmt.group(1)}` is not how the values are written)" if fmt else ""
            return ValueError(f"{raw}\nThe date parse failed{tried}. {DATE_TEXT_HINT}")
        if "cannot mask with non-boolean array containing na" in msg:
            return ValueError(
                f"{raw}\nThe boolean mask holds NaN where the column is null (e.g. "
                "`.str.contains(...)` on a column with nulls, or `== value` on a `bool with "
                "nulls` column). Fix: `.str.contains(..., na=False)`, or `.fillna(False)` / "
                "`.eq(True)` on the mask, so null rows are excluded rather than undefined."
            )
        if "truth value of a" in msg and "ambiguous" in msg:
            return ValueError(
                f"{raw}\nA Series was used where Python needs ONE boolean: `and` / `or` / `not` "
                "between conditions, or `if series:`. Combine conditions element-wise with "
                "`&`, `|`, `~` and parentheses around each — `df[(df['a'] > 1) & (df['b'] < 5)]`; "
                "in an `if`, reduce with `.any()` / `.all()` / `.empty`."
            )
        if "nested renamer is not supported" in msg:
            return ValueError(
                f"{raw}\nNested dict aggregation (`{{'col': {{'name': 'sum'}}}}`) is not supported. "
                "Use named aggregation: `.agg(name=('col', 'sum'), other=('col2', 'mean'))`."
            )
        if "argmax of an empty" in msg or "argmin of an empty" in msg or "zero-size array" in msg \
                or ("empty" in msg and "sequence" in msg):
            return ValueError(f"{raw}\n{EMPTY_FRAME_HINT}")
        if "length mismatch" in msg and "new values" in msg:
            return ValueError(
                f"{raw}\nThe list assigned to .columns has a different length than the frame's "
                "columns. Rename with a mapping instead: `.rename(columns={{'old': 'new'}})`, "
                "or select the columns first so the count matches."
            )
        if "bin edges must be unique" in msg:
            return ValueError(
                f"{raw}\npd.qcut found repeated quantile edges (many identical values). Pass "
                "`duplicates='drop'` (fewer bins), or use pd.cut with explicit bounds from the "
                "column's range / quartiles in Column Statistics."
            )
        if "index contains duplicate entries, cannot reshape" in msg:
            return ValueError(
                f"{raw}\n.pivot() needs one row per (index, columns) pair and the data has "
                "repeats. Use `pd.pivot_table(..., aggfunc='sum')` (or 'mean'/'first') to "
                "aggregate the repeats, or groupby the pair first."
            )
        if "must pass list-like" in msg or "no numeric data to plot" in msg:
            return ValueError(
                f"{raw}\nagg/apply got wrong type — cast with pd.to_numeric or check groupby column selection."
            )
        if "cannot reindex" in msg or ("reindex" in msg and "not unique" in msg):
            return ValueError(
                f"{raw}\nDuplicate index — call .reset_index(drop=True) after merges or groupby."
            )
        if "out of bounds" in msg and ("datetime" in msg or "date" in msg):
            return ValueError(
                f"{raw}\nDatetime out of bounds (valid range ~1677–2262): a placeholder year such "
                "as 1900/9999 or a wrong format. Parse with errors='coerce' (bad values become "
                "NaT) using the format from the `date text` fact."
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
        if "cannot set a dataframe with multiple columns" in msg or "columns must be same length as key" in msg:
            return ValueError(
                f"{raw}\nA multi-column frame was assigned to a single column. Assign one Series "
                "per column, or select the one column you meant (`df[['a']]` is a DataFrame, "
                "`df['a']` a Series)."
            )
        if _is_tz_mismatch(msg):
            return ValueError(_tz_mismatch_message(raw))
        if "agg function failed" in msg:
            return self._aggregation_on_text(raw, full)
        if re.search(r"\.(query|eval)\s*\(", source or "") and (
            "not a supported function" in msg or "invalid syntax" in msg or "unexpected" in msg or "only named functions" in msg
        ):
            return ValueError(f"{raw}\n{QUERY_STRING_HINT}")
        return ValueError(
            f"{raw}\nValue error in the operation above. Most often: a column still holding TEXT "
            "where a number or date is needed — cast it with the Column Statistics recipe "
            "(`number text` → pd.to_numeric(..., errors='coerce') after removing the shown "
            "separators; `date text` → pd.to_datetime(..., format=<shown>, errors='coerce')) — "
            "or an empty frame after a filter that matched nothing."
        )

    @staticmethod
    def _pandas_named_error(name: str, raw: str, msg: str) -> "Exception | None":
        """pandas.errors classes that derive from Exception, not ValueError, and
        the ValueError subclasses whose message needs its own hint."""
        if name == "SpecificationError" or "nested renamer is not supported" in msg:
            return ValueError(
                f"{raw}\nNested dict aggregation (`{{'col': {{'name': 'sum'}}}}`) is not supported. "
                "Use named aggregation: `.agg(name=('col', 'sum'), other=('col2', 'mean'))`."
            )
        if name == "IntCastingNaNError" or "cannot convert non-finite values" in msg:
            return ValueError(
                f"{raw}\n.astype(int) on a column holding NaN (nulls, or values the cast could not "
                "read). Keep it float, use `.astype('Int64')` (nullable integer), or `.fillna(0)` / "
                "drop the null rows first when the question allows."
            )
        if name == "IndexingError" or "unalignable boolean series" in msg:
            return ValueError(
                f"{raw}\nThe boolean mask was built from a DIFFERENT frame (another index) than "
                "the one it filters. Build the mask from the same frame you index "
                "(`df[df['col'] > 1]`), or align with `.loc[mask.index]` / `.reset_index(drop=True)` on both."
            )
        if name == "InvalidIndexError" or "reindexing only valid with uniquely valued index" in msg:
            return ValueError(
                f"{raw}\nThe index holds duplicate labels, so alignment is ambiguous. "
                "`.reset_index(drop=True)` after the merge/concat/filter that produced it, or "
                "aggregate to one row per label first."
            )
        if name == "DataError" or "no numeric types to aggregate" in msg:
            return ValueError(
                f"{raw}\nThe aggregation found no numeric column: the columns are text. Cast "
                f"the measure first. {NUMBER_TEXT_HINT}"
            )
        if name == "DuplicateLabelError":
            return ValueError(f"{raw}\nDuplicate column labels: rename or drop the duplicate before this step.")
        return None

    @staticmethod
    def _aggregation_on_text(raw: str, full: str) -> ValueError:
        how = re.search(r"how->(\w+)", full)
        verb = how.group(1) if how else "the aggregation"
        return ValueError(
            f"{raw}\n`{verb}` ran over a TEXT (object) column. Aggregate only the numeric columns "
            "— select them explicitly (`df[['a', 'b']]`, `['col'].mean()`) — and cast a column "
            f"stored as number text first. {NUMBER_TEXT_HINT}"
        )

    def _type_error(self, raw: str, msg: str, e: Exception, source: str) -> TypeError:
        full = str(e)
        if _is_tz_mismatch(msg):
            return TypeError(_tz_mismatch_message(raw))
        if "agg function failed" in msg:
            return TypeError(str(self._aggregation_on_text(raw, full)))
        cmp = re.search(r"'([<>=!]+)' not supported between instances of '(\w+)' and '(\w+)'", full)
        if cmp:
            op, t1, t2 = cmp.groups()
            if "str" in (t1, t2) and ({t1, t2} & {"int", "float"}):
                return TypeError(
                    f"{raw}\nA TEXT column was compared ({op}) with a number. {NUMBER_TEXT_HINT} "
                    "(A `date text` column is parsed with pd.to_datetime and its shown format "
                    "instead, then compared with pd.Timestamp.)"
                )
            if "str" in (t1, t2) and "Timestamp" in (t1, t2):
                return TypeError(
                    f"{raw}\nA parsed date was compared with a string. Compare with "
                    "`pd.Timestamp('YYYY-MM-DD')`, and parse any date-text column with its "
                    "shown format before comparing."
                )
            return TypeError(
                f"{raw}\n'{op}' between a {t1} and a {t2}: the two sides are stored as different "
                "kinds of value. Cast the column to the kind the comparison needs (Column "
                "Statistics shows how it is stored) before comparing."
            )
        ops = re.search(r"unsupported operand type\(s\) for (.+?): '([^']+)' and '([^']+)'", full)
        concat = "can only concatenate str" in msg
        if ops or concat:
            op, t1, t2 = ops.groups() if ops else ("+", "str", "?")
            if "str" in (t1, t2):
                return TypeError(
                    f"{raw}\nArithmetic ('{op}') on a TEXT column. For numbers: {NUMBER_TEXT_HINT} "
                    "For a difference of two date columns: parse both with pd.to_datetime and "
                    "the format their `date text` facts show (errors='coerce'), then subtract "
                    "and take `.dt.days`."
                )
            if "NoneType" in (t1, t2):
                return TypeError(
                    f"{raw}\nOne operand is None: a method called with inplace=True, or a lookup "
                    "that found nothing. Assign the method's return value instead."
                )
            return TypeError(
                f"{raw}\nType mismatch for '{op}': '{t1}' vs '{t2}' — cast the column to the "
                "numeric type with pd.to_numeric(..., errors='coerce') (per its Column "
                "Statistics fact) before the arithmetic."
            )
        if "int() argument must be" in msg and "nonetype" in msg:
            return TypeError(
                f"{raw}\n.astype(int) / int() on a column holding None (nulls). Keep it float, use "
                "`.astype('Int64')` (nullable integer), or `.fillna(0)` / drop the null rows first "
                "when the question allows."
            )
        if "datetime64 type does not support" in msg:
            return TypeError(
                f"{raw}\nDates cannot be summed or averaged. Use min/max/count on the parsed "
                "dates, or convert a difference to days first (`(a - b).dt.days`)."
            )
        if "could not convert" in msg or "invalid literal" in msg:
            return TypeError(f"{raw}\nConversion hit text. {NUMBER_TEXT_HINT}")
        if "unhashable type" in msg:
            return TypeError(
                f"{raw}\nAn unhashable object (a Series, list or dict) was used as a key or set "
                "member: pass column NAMES to groupby/merge/isin (`.isin([...])` takes a list "
                "of values, `groupby('col')` a name), not Series objects."
            )
        if "object is not callable" in msg:
            return TypeError(
                f"{raw}\nSomething that is not a function was called: `df('col')` instead of "
                "`df['col']`, or a column whose name shadows a method (`df.count` is the "
                "method, `df['count']` the column)."
            )
        if "object of type 'nonetype' has no len" in msg or "'nonetype' object is not" in msg:
            return TypeError(
                f"{raw}\nA value is None here — an inplace=True call or a failed lookup. Assign "
                "the method's return value (no inplace) and check the step before."
            )
        if "dtype" in msg:
            return TypeError(
                f"{raw}\nDtype mismatch — the operation needs a different kind of value than "
                "the column holds; cast it per its Column Statistics fact (number text / date text)."
            )
        return TypeError(
            f"{raw}\nType error — the operation received a kind of value it does not accept; "
            "check the column's stored type in Column Statistics and cast it first."
        )

    def _normalize_pandas_error(self, e: Exception, source: str = "") -> Exception:
        if getattr(e, "orqa_feedback", False):
            return e
        msg = str(e).lower()
        # A conversion/cast failure (pd.to_numeric, .astype(...), ...) embeds
        # the FULL offending cell value verbatim in str(e) — unbounded, so a
        # huge cell (e.g. a WKT geometry blob) would otherwise blow up the
        # correction prompt below. `raw_full` keeps the untouched message for
        # internal parsing; every `{raw}` interpolated into a message actually
        # sent to the LLM uses the shielded copy.
        raw_full = str(e).strip('"').strip("'")
        raw = summarize_large_value(raw_full)

        if isinstance(e, KeyError):
            return self._key_error(raw_full, raw, msg, source)

        if isinstance(e, IndexError):
            if "positional indexer" in msg or "out of bounds for axis" in msg or "index out of bounds" in msg:
                return IndexError(
                    f"{raw}\nPositional access past the end: the frame is shorter than the "
                    f"position — usually EMPTY. {EMPTY_FRAME_HINT} Prefer `.head(1)` / "
                    "`.iloc[:1]`, which never raise, or `.idxmax()` for the row of the maximum."
                )
            if "invalid index to scalar variable" in msg:
                return IndexError(
                    f"{raw}\nA SCALAR was indexed like a sequence: `.sum()`, `.max()`, "
                    "`.mean()`, `.count()`, `.iloc[0]` on a Series, ... return one value — use it "
                    "as it is, without `[0]`."
                )
            # (A numpy scalar's `.round(2)` is valid: the "fails this way" hint
            # once given here came from the validator's own size check, which
            # indexed the () shape of every numpy scalar result.)
            return IndexError(
                f"{raw}\nIndex past the end of a tuple/list: the sequence is empty or shorter than "
                "the position."
            )

        if isinstance(e, AttributeError):
            return self._attribute_error(raw, msg, e)

        if isinstance(e, NameError):
            return self._name_error(e, source)

        if isinstance(e, ImportError):
            return e  # already the policy message

        if isinstance(e, MemoryError):
            return MemoryError(
                f"{raw}\nOperation exceeded the memory limit.\n"
                "Fix: pre-filter rows, select only needed columns, or aggregate before the operation."
            )

        if isinstance(e, ZeroDivisionError):
            return ZeroDivisionError(
                f"{raw}\nDivision by zero — guard with df['col'].replace(0, float('nan')) or filter zero rows first."
            )

        if isinstance(e, OverflowError):
            return OverflowError(
                f"{raw}\nNumeric overflow: convert to float64 (`.astype('float64')`) or reduce "
                "the magnitude (a sum of yyyymmdd date numbers, an id column summed) — "
                "check the column is a quantity, not an identifier (Column Statistics)."
            )

        named = self._pandas_named_error(type(e).__name__, raw, msg)
        if named is not None:
            return named

        if isinstance(e, ValueError):
            return self._value_error(raw, msg, e, source)

        if isinstance(e, TypeError):
            return self._type_error(raw, msg, e, source)

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
                "standard indentation for control-flow blocks, matching quotes and "
                "brackets (a value holding an apostrophe goes in double quotes). "
                "Do NOT join statements with semicolons."
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
                "empties the frame; (2) compare values exactly as Column Statistics shows "
                "them stored (case, accents, whitespace — normalise both sides with "
                "df['col'].str.strip().str.lower() == value.lower() when the question allows); "
                "(3) a text column compared with a number or date: cast it first with its "
                "number text / date text recipe; (4) check the date range against the "
                "column's parsed min..max in Column Statistics."
            )
        else:
            head = (
                "Empty result — the merge produced no matching rows, or a filter after "
                "it removed them all.\n"
                "Fix, in order: (1) switch the merge to how='left' to see which side "
                "fails to match; (2) normalise the join keys on BOTH sides before "
                "merging — t1.assign(_k=t1['key'].str.strip().str.lower()).merge("
                "t2.assign(_k=t2['key'].str.strip().str.lower()), on='_k'); (3) confirm "
                "the two key columns hold the same kind of value (compare both columns' "
                "values and shapes in Column Statistics); (4) then drop post-merge filters one at a time."
            )
        return f"{head}\n{self._EMPTY_RESULT_TAIL}"

    def _get_language_name(self) -> str:
        return 'Python'
