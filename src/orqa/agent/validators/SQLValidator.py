import duckdb
import pandas as pd
import polars as pl
from typing import List, Dict, Tuple, Any
from .QueryValidator import QueryValidator
import re
# Matches any single-quoted SQL string literal, including '' escapes inside.
_LITERAL_RE = re.compile(r"'(?:''|[^'])*'")

UNAUTHORIZED_SQL_COMMANDS = [
    # Memory/config manipulation
    'set memory_limit', 'set threads', 'set max_memory',
    'set worker_threads', 'pragma', 'set enable',

    # File I/O (all valid and dangerous in DuckDB)
    'copy ', 'export database', 'import database',
    'read_csv', 'read_parquet', 'read_json', 'read_csv_auto',
    'attach ', 'detach ',

    # Extension loading
    'load ', 'install ',

    # DDL - schema modification
    'create table', 'create view', 'create index',
    'drop table', 'drop view', 'drop index',
    'alter table', 'truncate',

    # DML - data modification
    'insert into', 'update ', 'delete from',

    # System access
    'information_schema', 'pg_', 'shell(', 'system(',
    'call dbms', 'exec ',
]
SQL_CARTESIAN_PATTERNS = [
    # FROM with multiple tables and no JOIN keyword
    (
        r'\bFROM\b[^;]+,[^;]+\bWHERE\b',
        "Implicit cross join via comma-separated tables in FROM clause"
    ),
    # Explicit CROSS JOIN
    (
        r'\bCROSS\s+JOIN\b',
        "Explicit CROSS JOIN detected"
    ),
    # JOIN with no ON or USING clause (e.g. JOIN table WHERE ...)
    (
        r'\bJOIN\s+\w+\s+(?!AS\s+\w+\s*)(?!ON\b)(?!USING\b)\b(?:WHERE|GROUP|ORDER|LIMIT|LEFT|RIGHT|INNER|OUTER|;|$)',
        "JOIN without ON or USING clause"
    ),
    # FROM with multiple comma-separated tables and no WHERE at all
    (
        r'\bFROM\s+\w+\s*,\s*\w+(?:\s*,\s*\w+)*\s*(?:GROUP|ORDER|LIMIT|;|$)',
        "Multiple tables in FROM with no WHERE/JOIN condition"
    ),
    # USING or ON clause that always evaluates to true: ON 1=1
    (
        r'\bON\s+1\s*=\s*1\b',
        "Always-true JOIN condition (ON 1=1)"
    ),
]

class SQLValidator(QueryValidator):
    """Validator for SQL queries."""

    def _preprocess_query(self, query: str) -> str:
        self._check_banned_sql_commands(query)
        query = self._strip_set_statements(query)
        warnings = self.check_sql_cartesian(query)
        if warnings:
            raise ValueError(
                "Potentially dangerous query — possible Cartesian product detected:\n"
                + "\n".join(f"  - {w}" for w in warnings)
                + "\nRewrite using explicit JOIN ... ON conditions."
            )
        query_code = self.remove_redundant_aliases(query)
        query_code = self.fix_aggregates(query_code)
        return query_code.replace("`", '"')
    
    def _check_banned_sql_commands(self, query: str) -> None:
        query_lower = query.lower()
        found = [cmd for cmd in UNAUTHORIZED_SQL_COMMANDS if cmd in query_lower]
        if found:
            raise ValueError(
                f"Query contains unauthorized SQL commands: {', '.join(found)}\n"
                "These commands are not permitted for security and stability reasons."
            )
    def check_sql_cartesian(self,query: str) -> list[str]:
        warnings = []
        for pattern, message in SQL_CARTESIAN_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE | re.DOTALL):
                warnings.append(message)
        return warnings
    
    def _strip_set_statements(self, query: str) -> str:
        """Remove any SET configuration statements from the query."""
        # Matches SET ... ; at the start or anywhere in the query
        return re.sub(r'\bSET\s+\w+\s*=\s*[^;]+;?\s*', '', query, flags=re.IGNORECASE).strip()

    # ------------------------------------------------------------------
    # fix_aggregates
    # ------------------------------------------------------------------
    def fix_aggregates(self, query: str) -> str:
        """
        Wraps bare aggregate arguments in TRY_CAST(... AS DOUBLE) so that
        text-typed amount/price columns don't cause type errors.

        Single-quoted string literals are masked before the regex walk and
        restored afterwards, so occurrences like WHERE label = 'SUM(x)'
        are never touched.
        """
        # 1. Stash all string literals and replace with safe placeholders.
        literals: list[str] = []

        def _stash(m: re.Match) -> str:
            literals.append(m.group(0))
            return f"__LIT_{len(literals) - 1}__"

        masked = _LITERAL_RE.sub(_stash, query)

        ## 2. Run the transformation on literal-free text.
        #masked = self._fix_agg_core(masked)

        ## 3. Restore the original literals.
        def _restore(m: re.Match) -> str:
            return literals[int(m.group(1))]

        return re.sub(r"__LIT_(\d+)__", _restore, masked)

    def _fix_agg_core(self, query: str) -> str:
        """Core aggregate-wrapping logic; operates on literal-free text."""
        result = []
        i = 0
        agg_pattern = re.compile(r'\b(SUM|AVG|MIN|MAX|COUNT)\s*\(', re.IGNORECASE)

        while i < len(query):
            match = agg_pattern.search(query, i)
            if not match:
                result.append(query[i:])
                break

            result.append(query[i:match.start()])
            func = match.group(1)

            # Walk forward to find the real matching closing parenthesis.
            depth = 0
            j = match.end() - 1   # position of opening '('
            while j < len(query):
                if query[j] == '(':
                    depth += 1
                elif query[j] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1

            content = query[match.end():j].strip()

            # COUNT(*) — never wrap the wildcard.
            if content == '*':
                result.append(f'{func}(*)')

            # Already wrapped in TRY_CAST / CAST — do not double-wrap.
            elif re.match(r'(TRY_CAST|CAST)\s*\(', content, re.IGNORECASE):
                result.append(f'{func}({content})')

            # DISTINCT — keep keyword outside TRY_CAST.
            elif distinct_match := re.match(
                r'(DISTINCT\s+)(.*)', content, re.IGNORECASE | re.DOTALL
            ):
                distinct_kw = distinct_match.group(1)
                expr = distinct_match.group(2).strip()
                if expr == '*':
                    result.append(f'{func}({distinct_kw}*)')
                else:
                    result.append(f'{func}({distinct_kw}TRY_CAST({expr} AS DOUBLE))')

            else:
                result.append(f'{func}(TRY_CAST({content} AS DOUBLE))')

            i = j + 1

        return ''.join(result)

    # ------------------------------------------------------------------
    # remove_redundant_aliases
    # ------------------------------------------------------------------
    def remove_redundant_aliases(self, query: str) -> str:
        """
        Removes self-aliases like  FROM table AS table  or  JOIN table AS table.
        The word-boundary on JOIN also covers LEFT JOIN, INNER JOIN, etc.
        """
        pattern = re.compile(
            r'\b(FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s+\2\b',
            flags=re.IGNORECASE
        )
        return pattern.sub(r'\1 \2', query)

    # ------------------------------------------------------------------
    # _run_query
    # ------------------------------------------------------------------
    def _run_query(self, query: str) -> Any:
        con = duckdb.connect(database=":memory:")
        con.execute(f"SET memory_limit='{self.mem_limit / (1024 ** 2)}MB'")
        con.execute("SET threads=2")
        try:
            for df, name in zip(self.dataframes, self.table_names):
                con.register(name, df)

            sanitized = re.sub(
                r'\bLIMIT\s+\d+\s*;?\s*$', '',
                query.rstrip(";").rstrip(),
                flags=re.IGNORECASE,
            ).rstrip()
            result = con.execute(sanitized + " LIMIT 100").df()
            return result

        except duckdb.ParserException as e:
            raise ValueError(
                f"Malformed SQL structure: {e}\n"
                "Hint: Do not mix UNION and JOIN syntax. "
                "UNION combines result sets of separate SELECT statements and does not use ON clauses. "
                "JOIN connects tables within a single SELECT statement using ON clauses. "
                "Rewrite the query using only JOINs to combine all required tables."
            ) from e

        except duckdb.BinderException as e:
            error_msg = str(e)
            if "not found in FROM clause" in error_msg or "Referenced column" in error_msg:
                hint = (
                    "Only use columns that exist in the schema. "
                    "Do not infer or guess column names by pattern."
                )
            elif "UNION" in error_msg and "different number" in error_msg:
                hint = (
                    "All SELECT statements in a UNION must have the same number of columns "
                    "and compatible types."
                )
            elif "Ambiguous reference" in error_msg:
                hint = (
                    "Ambiguous column reference. Qualify all column names with their table prefix, "
                    "e.g. Table_0.column_name instead of just column_name."
                )
            else:
                hint = "Check that all referenced columns, tables, and aliases are valid."
            raise ValueError(f"Binder error: {error_msg}\nHint: {hint}") from e

        finally:
            con.close()

    def _build_empty_result_feedback(self) -> str:
        return "\n".join([
            "The query returned 0 rows.",
            "This usually means a JOIN or WHERE condition matched nothing due to case or type mismatches.",
            "Suggestions:",
            "  1. String join keys: normalize casing on both sides using LOWER():",
            "       JOIN table2 ON LOWER(table1.key) = LOWER(table2.key)",
            "  2. String filters: wrap the column and the literal in LOWER():",
            "       WHERE LOWER(column_name) = 'expected value'",
            "  3. Numeric columns stored as text: cast before comparing or aggregating:",
            "       TRY_CAST(column_name AS DOUBLE)",
            "  4. Date/range filters: verify that the range or category value actually exists in the data.",
            "Rewrite the query applying LOWER() or TRY_CAST() where relevant.",
        ])

    #def _get_language_specific_rules(self) -> List[str]:
    #    return [
    #        "- Column names must exactly match the schema.",
    #        "- Amount may be stored as text; CAST to DOUBLE when using SUM or AVG.",
    #        f"- Available tables: {', '.join(self.table_names)}",
    #    ]

    def _get_language_name(self) -> str:
        return "SQL"