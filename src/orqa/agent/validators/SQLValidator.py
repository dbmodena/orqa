import duckdb
import pandas as pd
import polars as pl
from typing import List, Dict, Tuple, Any
from .QueryValidator import QueryValidator
import re
import json

# Matches any single-quoted SQL string literal, including '' escapes inside.
_LITERAL_RE = re.compile(r"'(?:''|[^'])*'")


class SQLValidator(QueryValidator):
    """Validator for SQL queries."""

    def _preprocess_query(self, query: str) -> str:
        query_code = self.remove_redundant_aliases(query)
        query_code = self.fix_aggregates(query_code)
        return query_code.replace("`", '"')

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

        # 2. Run the transformation on literal-free text.
        masked = self._fix_agg_core(masked)

        # 3. Restore the original literals.
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
    # _execute_query
    # ------------------------------------------------------------------
    def _execute_query(self, query: str) -> None:
        con = duckdb.connect(database=":memory:")
        try:
            for df, name in zip(self.dataframes, self.table_names):
                con.register(name, df)

            sanitized = re.sub(
                r'\bLIMIT\s+\d+\s*;?\s*$', '',
                query.rstrip(";").rstrip(),
                flags=re.IGNORECASE,
            ).rstrip()
            con.execute(sanitized + " LIMIT 1")

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

    def _get_language_specific_rules(self) -> List[str]:
        return [
            "- Column names must exactly match the schema.",
            "- Amount may be stored as text; CAST to DOUBLE when using SUM or AVG.",
            f"- Available tables: {', '.join(self.table_names)}",
        ]

    def _get_language_name(self) -> str:
        return "SQL"