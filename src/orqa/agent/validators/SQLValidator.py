import duckdb
import pandas as pd
import polars as pl
from typing import Any
from .QueryValidator import QueryValidator
import re

_LITERAL_RE = re.compile(r"'(?:''|[^'])*'")

UNAUTHORIZED_SQL_COMMANDS = [
    # Memory/config manipulation
    'set memory_limit', 'set threads', 'set max_memory',
    'set worker_threads', 'pragma', 'set enable',
    # File I/O
    'copy ', 'export database', 'import database',
    'read_csv', 'read_parquet', 'read_json', 'read_csv_auto',
    'attach ', 'detach ',
    # Extension loading
    'load ', 'install ',
    # DDL
    'create table', 'create view', 'create index',
    'drop table', 'drop view', 'drop index',
    'alter table', 'truncate',
    # DML
    'insert into', 'update ', 'delete from',
    # System access
    'information_schema', 'pg_', 'shell(', 'system(',
    'call dbms', 'exec ',
]

SQL_CARTESIAN_PATTERNS = [
    #(
    #    r'\bFROM\b[^;]+,[^;]+\bWHERE\b',
    #    "Implicit cross join via comma-separated tables in FROM clause"
    #),
    (
        r'\bCROSS\s+JOIN\b',
        "Explicit CROSS JOIN detected"
    ),
    (
        r'\bJOIN\s+\w+\s+(?!AS\s+\w+\s*)(?!ON\b)(?!USING\b)\b(?:WHERE|GROUP|ORDER|LIMIT|LEFT|RIGHT|INNER|OUTER|;|$)',
        "JOIN without ON or USING clause"
    ),
    (
        r'\bFROM\s+\w+\s*,\s*\w+(?:\s*,\s*\w+)*\s*(?:GROUP|ORDER|LIMIT|;|$)',
        "Multiple tables in FROM with no WHERE/JOIN condition"
    ),
    (
        r'\bON\s+1\s*=\s*1\b',
        "Always-true JOIN condition (ON 1=1)"
    ),
]


class SQLValidator(QueryValidator):
    """Validator for SQL queries."""

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------
    def _preprocess_query(self, query: str) -> str:
        self._check_banned_sql_commands(query)
        query = self._strip_set_statements(query)

        warnings = self._check_sql_cartesian(query)
        if warnings:
            raise ValueError(
                "Potentially dangerous query — possible Cartesian product detected:\n"
                + "\n".join(f"  - {w}" for w in warnings)
                + "\nRewrite using explicit JOIN ... ON conditions."
            )

        #self._check_query_connectivity(query)
        query = self._remove_redundant_aliases(query)
        query = self._fix_aggregates(query)
        return query.replace("`", '"')

    def _check_banned_sql_commands(self, query: str) -> None:
        query_lower = query.lower()
        found = [cmd for cmd in UNAUTHORIZED_SQL_COMMANDS if cmd in query_lower]
        if found:
            raise ValueError(
                f"Query contains unauthorized SQL commands: {', '.join(found)}\n"
                "These commands are not permitted for security and stability reasons."
            )

    def _strip_set_statements(self, query: str) -> str:
        return re.sub(r'\bSET\s+\w+\s*=\s*[^;]+;?\s*', '', query, flags=re.IGNORECASE).strip()

    def _check_sql_cartesian(self, query: str) -> list[str]:
        return [
            msg for pattern, msg in SQL_CARTESIAN_PATTERNS
            if re.search(pattern, query, re.IGNORECASE | re.DOTALL)
        ]

    def _remove_redundant_aliases(self, query: str) -> str:
        return re.compile(
            r'\b(FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s+\2\b',
            flags=re.IGNORECASE
        ).sub(r'\1 \2', query)

    def _fix_aggregates(self, query: str) -> str:
        """Mask string literals, run aggregate wrapping, restore literals."""
        literals: list[str] = []

        def _stash(m: re.Match) -> str:
            literals.append(m.group(0))
            return f"__LIT_{len(literals) - 1}__"

        masked = _LITERAL_RE.sub(_stash, query)
        # _fix_agg_core is available but currently disabled — uncomment to enable:
        # masked = self._fix_agg_core(masked)
        return re.sub(r"__LIT_(\d+)__", lambda m: literals[int(m.group(1))], masked)

    def _fix_agg_core(self, query: str) -> str:
        """Wraps bare SUM/AVG/MIN/MAX/COUNT arguments in TRY_CAST(... AS DOUBLE)."""
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

            depth, j = 0, match.end() - 1
            while j < len(query):
                if query[j] == '(':
                    depth += 1
                elif query[j] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1

            content = query[match.end():j].strip()

            if content == '*':
                result.append(f'{func}(*)')
            elif re.match(r'(TRY_CAST|CAST)\s*\(', content, re.IGNORECASE):
                result.append(f'{func}({content})')
            elif distinct_match := re.match(r'(DISTINCT\s+)(.*)', content, re.IGNORECASE | re.DOTALL):
                kw   = distinct_match.group(1)
                expr = distinct_match.group(2).strip()
                result.append(f'{func}({kw}*)' if expr == '*' else f'{func}({kw}TRY_CAST({expr} AS DOUBLE))')
            else:
                result.append(f'{func}(TRY_CAST({content} AS DOUBLE))')

            i = j + 1

        return ''.join(result)

    # ------------------------------------------------------------------
    # Connectivity check — detects disjoint sub-queries
    # ------------------------------------------------------------------
    def _check_query_connectivity(self, query: str) -> None:
        """
        Verify that all tables referenced in the query are connected into a
        single result through JOIN … ON conditions or UNION / UNION ALL.

        Strategy
        --------
        1. Find every table name (case-insensitive) actually referenced in the query.
        2. Build an alias → canonical-table map from every FROM / JOIN clause.
        3. Build an adjacency graph:
           - JOIN … ON left_alias.col = right_alias.col  →  edge between the two tables.
           - UNION / UNION ALL  →  fully connects all referenced tables (they share a
             result set by definition, even without an ON clause).
        4. Split on semicolons: each statement is treated as its own island unless the
           tables it touches are already linked via JOIN/UNION in that statement.
        5. BFS over the graph — if any referenced table is unreachable, raise.
        """
        from collections import defaultdict

        table_names_set = set(self.table_names)
        ql = query.lower()
        tl_names = [t.lower() for t in table_names_set]

        referenced = frozenset(
            t for t in table_names_set
            if re.search(rf'\b{re.escape(t.lower())}\b', ql)
        )
        if len(referenced) < 2:
            return  # single table or empty — nothing to check

        # ------------------------------------------------------------------
        # Build alias → canonical-table map from FROM / JOIN clauses
        # e.g.  "FROM orders o"  →  alias_map["o"] = "orders"
        # ------------------------------------------------------------------
        alias_map: dict[str, str] = {}
        from_join_pat = re.compile(
            r'(?:from|join)\s+(\w+)(?:\s+(?:as\s+)?(\w+))?', re.IGNORECASE
        )
        for m in from_join_pat.finditer(query):
            tbl_raw   = m.group(1).lower()
            alias_raw = m.group(2).lower() if m.group(2) else tbl_raw
            if tbl_raw in tl_names:
                canonical = next(t for t in table_names_set if t.lower() == tbl_raw)
                alias_map[alias_raw] = canonical
                alias_map[tbl_raw]   = canonical

        def resolve_alias(name: str) -> str | None:
            return alias_map.get(name.lower())

        # ------------------------------------------------------------------
        # Build adjacency
        # ------------------------------------------------------------------
        adjacency: dict[str, set] = defaultdict(set)

        # JOIN … ON left_alias.col = right_alias.col
        join_on_pat = re.compile(
            r'join\s+(\w+)(?:\s+(?:as\s+)?(\w+))?\s+on\s+(\w+)\.\w+\s*=\s*(\w+)\.\w+',
            re.IGNORECASE,
        )
        for m in join_on_pat.finditer(query):
            left_alias  = m.group(3)
            right_alias = m.group(4)
            left_tbl  = resolve_alias(left_alias)
            right_tbl = resolve_alias(right_alias)
            if left_tbl and right_tbl and left_tbl != right_tbl:
                adjacency[left_tbl].add(right_tbl)
                adjacency[right_tbl].add(left_tbl)

        # UNION / UNION ALL → fully connect all referenced tables
        if re.search(r'\bunion\b', ql):
            nodes = list(referenced)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    adjacency[nodes[i]].add(nodes[j])
                    adjacency[nodes[j]].add(nodes[i])

        # ------------------------------------------------------------------
        # BFS reachability
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Identify islands and raise
        # ------------------------------------------------------------------
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
            "JOIN … ON condition or UNION that links them into a single result.\n\n"
            "Common causes:\n"
            "  1. Multiple SELECT statements separated by semicolons with no JOIN or UNION\n"
            "  2. A JOIN written without an ON clause (already caught above)\n"
            "  3. A subquery that references a table but is never joined back\n\n"
            "Fix: ensure every table group is connected, e.g.:\n"
            "  ✓ SELECT … FROM orders o JOIN customers c ON o.customer_id = c.id\n"
            "  ✓ SELECT id FROM orders UNION SELECT id FROM customers\n"
            "  ✗ SELECT * FROM orders; SELECT * FROM customers"
        )

    # ------------------------------------------------------------------
    # Execution — runs inside sandbox process via base _execute_query
    # ------------------------------------------------------------------
    def _run_query(self, query: str, dataframes: list, table_names: list) -> Any:
        con = duckdb.connect(database=":memory:")
        con.execute(f"SET memory_limit='{self.mem_limit_mb}MB'")
        con.execute("SET threads=2")
        try:
            for df, name in zip(dataframes, table_names):
                con.register(name, df)

            sanitized = re.sub(
                r'\bLIMIT\s+\d+\s*;?\s*$', '',
                query.rstrip(";").rstrip(),
                flags=re.IGNORECASE,
            ).rstrip()
            return con.execute(sanitized + " LIMIT 100").df()

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
                hint = "Only use columns that exist in the schema. Do not infer or guess column names by pattern."
            elif "UNION" in error_msg and "different number" in error_msg:
                hint = "All SELECT statements in a UNION must have the same number of columns and compatible types."
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

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------
    def _build_empty_result_feedback(self) -> str:
        if len(self.table_names) == 1:
            return "\n".join([
                "Query returned 0 rows — filters may be too restrictive.",
                "Suggestions:",
                "  1. Relax filters; verify actual column values.",
                "  2. Normalize strings with LOWER(): WHERE LOWER(col) = 'value'",
                "  3. Cast text numbers: WHERE TRY_CAST(col AS DOUBLE) > x",
                "  4. Check date/range values exist.",
                "Rewrite using LOWER() or TRY_CAST() where needed.",
            ])
        return "\n".join([
            "Query returned 0 rows — JOIN/filters may not match.",
            "Suggestions:",
            "  1. Normalize join keys: LOWER(t1.key) = LOWER(t2.key)",
            "  2. Use LEFT JOIN to debug missing matches.",
            "  3. Normalize filters: WHERE LOWER(col) = 'value'",
            "  4. Cast text numbers: TRY_CAST(col AS DOUBLE)",
            "  5. Check date/range values exist.",
            "Rewrite using LOWER() or TRY_CAST() where needed.",
        ])

    def _get_language_name(self) -> str:
        return "SQL"