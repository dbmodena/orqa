import pandas as pd
from pathlib import Path
from typing import Any
from .. import utils
import textwrap
import ast

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
    """
    def __init__(self, datasets_path: Path,bad_tokens:list=[]):
        self.datasets_path = Path(datasets_path)
        self.bad_tokens = bad_tokens

    def execute(self, entry: dict, query: dict) -> pd.DataFrame | None:
        """
        Execute a single query dict (as found inside entry["data"]["queries"])
        using the table mapping defined in entry["tables"].

        Returns a pandas DataFrame with the query result, or None on failure.
        """
        tables_map: dict[str, str] = entry.get("tables", {})
        query_type: str = query.get("query_type", "sql").lower()
        code: str = query.get("code", "")

        dataframes = self._load_tables(tables_map)

        if query_type == "sql":
            return self._execute_sql(code, dataframes)
        elif query_type in ("pandas", "python"):
            return self._execute_pandas(code, dataframes)
        else:
            raise ValueError(f"Unknown query_type: '{query_type}'")

    def _load_tables(self, tables_map: dict[str, str]) -> dict[str, pd.DataFrame]:
        dataframes: dict[str, pd.DataFrame] = {}
        for alias, dataset_id in tables_map.items():
            csv_path = self.datasets_path / f"{dataset_id}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"CSV file not found for table '{alias}': {csv_path}"
                )
            df = utils.remove_bad_tokens(pd.read_csv(csv_path, low_memory=False),self.bad_tokens)
            dataframes[alias] = df
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
            )
        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame(result)
        return result
    

    
    def _inject_result(self, code: str) -> str:
        """
        If the last statement in `code` is a bare expression (not already
        assigned to `result`), rewrite it as `result = <expr>`.
        """
        tree = ast.parse(textwrap.dedent(code))

        if not tree.body:
            raise ValueError("Code block is empty.")

        last = tree.body[-1]

        # Already assigns result — nothing to do
        if (
            isinstance(last, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "result"
                for t in last.targets
            )
        ) or (
            isinstance(last, (ast.AnnAssign, ast.AugAssign))
            and isinstance(getattr(last, "target", None), ast.Name)
            and last.target.id == "result"
        ):
            return code

        # Last statement is a bare expression — turn it into result = <expr>
        if isinstance(last, ast.Expr):
            lines = code.splitlines()
            # Grab the source lines that belong to the last node
            expr_lines = lines[last.lineno - 1 : last.end_lineno]
            expr_lines[0] = "result = " + expr_lines[0]
            lines[last.lineno - 1 : last.end_lineno] = expr_lines
            return "\n".join(lines)

        raise RuntimeError(
            "Pandas query did not assign a 'result' variable and the last "
            "statement is not a plain expression that can be auto-wrapped. "
            "Make sure the generated code ends with `result = …`."
        )