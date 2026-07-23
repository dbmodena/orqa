import pandas as pd
from pathlib import Path
from typing import Any, Optional
from .. import utils
import textwrap
import ast
import logging

logger = logging.getLogger(__name__)

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


class QueryExecutor:
    """
    Loads the CSV tables referenced in a generated_queries.json entry and
    executes SQL or Pandas queries against them.

    For SQL queries the tables are registered in an in-memory DuckDB connection
    under the names Table_0, Table_1, … so the generated SQL works as-is.

    For Pandas queries the code is executed with exec() inside a namespace that
    exposes the same Table_0 / Table_1 / … variables as DataFrames.
    The Pandas code is expected to assign its final result to a variable called
    `result`.

    NOTE: Tables are passed to DuckDB / pandas *with all their columns intact*.
    The `columns_involved` field on each query table describes which columns the
    query *references* (for joins, filters, selects), not which columns to
    expose to the executor.  Pre-filtering to only those columns would break any
    query that selects additional columns, and was the root cause of
    non-deterministic results across repeated executions.
    """

    def __init__(self, datasets_path: Path, bad_tokens: list = []):
        self.datasets_path = Path(datasets_path)
        self.bad_tokens = bad_tokens

    def execute(self, entry: dict, query: dict, query_kind: str) -> pd.DataFrame | None:
        """
        Execute a single query dict (as found inside entry["data"]["queries"])
        using the table mapping defined in entry["tables"].

        Loads the tables from disk on every call. Callers that execute many
        queries against the same tables (e.g. the validator-to-judge loop)
        should load once via ``load_tables`` and call ``execute_prepared``
        instead, to avoid re-reading the same CSVs from disk repeatedly.

        Returns a pandas DataFrame with the query result, or None on failure.
        """
        tables_map: dict[str, str] = entry.get("tables", {})
        # Load full DataFrames — do NOT pre-filter columns.
        # columns_involved describes query intent, not which columns to expose.
        dataframes = self.load_tables(tables_map)
        return self.execute_prepared(query, query_kind, dataframes)

    def execute_prepared(
        self, query: dict, query_kind: str, dataframes: dict[str, pd.DataFrame]
    ) -> pd.DataFrame | None:
        """
        Execute a single query dict against already-loaded ``dataframes``
        (as returned by ``load_tables``), skipping the disk read.

        Callers are responsible for making sure ``dataframes`` still carries
        every column the query might reference — the same "all columns
        intact" invariant documented on the class applies here. Row-level
        sampling of ``dataframes`` (e.g. to bound an ML-skill query's runtime)
        is safe to do before calling this method, but column pre-filtering is
        not.
        """
        code: str = query.get("code", "")

        if query_kind.lower() == "sql":
            return self._execute_sql(code, dataframes)
        elif query_kind.lower() in ("pandas", "python"):
            return self._execute_pandas(code, dataframes)
        else:
            raise ValueError(f"Unknown query_type: '{query_kind}'")

    def load_tables(self, tables_map: dict[str, str]) -> dict[str, pd.DataFrame]:
        dataframes: dict[str, pd.DataFrame] = {}
        for alias, dataset_id in tables_map.items():
            csv_path = self.datasets_path / f"{dataset_id}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"CSV file not found for table '{alias}': {csv_path}"
                )
            df = utils.pd_read_dataset(
                csv_path,
                opts={
                    "csv":     {"na_values": self.bad_tokens, "low_memory": False},
                    "parquet": {"na_values": self.bad_tokens, "low_memory": False},
                },
            )
            df = df.dropna()
            dataframes[alias] = df
            logger.debug("%s: %s", alias, df.columns)
        return dataframes

    def _execute_sql(self, sql: str, dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not HAS_DUCKDB:
            raise ImportError(
                "duckdb is required for SQL execution. "
                "Install it with: pip install duckdb"
            )
        con = duckdb.connect(database=":memory:")
        for alias, df in dataframes.items():
            con.register(alias, df)
        try:
            result = con.execute(sql).fetchdf()
        finally:
            con.close()
        return result

    def _execute_pandas(self, code: str, dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
        code = self._inject_result(code)
        namespace: dict[str, Any] = {"pd": pd, **dataframes}
        exec(code, namespace)
        result = namespace.get("result")
        if result is None:
            raise RuntimeError(
                "Pandas query did not assign a 'result' variable. "
                "Make sure the generated code ends with `result = …`."
                f"Code:\n{code}\nNamespace:\n{namespace}"
            )
        if isinstance(result, pd.DataFrame):
            #print(result)
            return result
        elif isinstance(result, pd.Series):
            #print(result.to_frame())
            return result.to_frame()
        elif isinstance(result, (int, float, str, bool)):
            #print( pd.DataFrame([{"result": result}]))
            return pd.DataFrame([{"result": result}])
        elif isinstance(result, list):
            #print(pd.DataFrame(result))
            return pd.DataFrame(result)
        else:
            #print(pd.DataFrame([result]))
            return pd.DataFrame([result])

    @staticmethod
    def _assignment_base_name(node: ast.AST) -> Optional[str]:
        """Return the base variable name an assignment ultimately targets.

        Unwraps Subscript/Attribute chains (``result["col"] = ...`` -> "result",
        ``df.loc[...] = ...`` -> "df") so both a bare rewrite and an in-place
        mutation are recognized by name — this is how every skill markdown's
        documented pattern ends (``result = df.copy();
        result["predicted_value"] = ...``). Returns ``None`` for non-assignment
        nodes.
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

    def _inject_result(self, code: str) -> str:
        """
        Normalise `code` so a `result` variable is bound after exec'ing it.

        Three cases, in order:
          1. The last statement already assigns `result` (bare or in-place,
             e.g. ``result["col"] = ...``) -> unchanged.
          2. The last statement is a bare expression -> rewritten as
             ``result = <expr>``.
          3. The last statement assigns some OTHER name (``output = ...``,
             ``output["col"] = ...``) -> that name is aliased to `result` via
             an appended ``result = <name>`` line, so the code doesn't have to
             literally use the name "result" at all.
        Anything else (no assignment, no expression) is rejected.
        """
        tree = ast.parse(textwrap.dedent(code))

        if not tree.body:
            raise ValueError("Code block is empty.")

        last = tree.body[-1]
        target_name = self._assignment_base_name(last)

        if target_name == "result":
            return code

        if isinstance(last, ast.Expr):
            lines = code.splitlines()
            expr_lines = lines[last.lineno - 1 : last.end_lineno]
            expr_lines[0] = "result = " + expr_lines[0]
            lines[last.lineno - 1 : last.end_lineno] = expr_lines
            return "\n".join(lines)

        if target_name is not None:
            return f"{code}\nresult = {target_name}"

        raise RuntimeError(
            "Pandas query's last statement neither assigns a variable nor is "
            "a plain expression that can be captured as the result. "
            "End with `<name> = ...`, `<name>[...] = ...`, or a bare "
            "expression."
        )
