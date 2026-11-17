"""Static + sandboxed validation of generated DuckDB SQL, with feedback the
correction model can act on.

The query runs VERBATIM (no LIMIT appended, no rewriting) so that DuckDB's
own ``LINE n: ... ^`` excerpts point at the model's text; only the first
rows of the result are fetched. Every DuckDB error is translated by
``_explain_duckdb_error`` into the Column Statistics recipe the generation
prompt teaches for that situation (number text, date text, offsets, GROUP
BY, quoting), never a generic CAST.
"""

import re
from typing import Any

import duckdb
import pandas as pd

from .QueryValidator import QueryValidator
from orqa.utils.undecodable_names import undecodable_matches
from ...utils import summarize_large_value

_LITERAL_RE = re.compile(r"'(?:''|[^'])*'")
_QUOTED_IDENT_RE = re.compile(r'"(?:""|[^"])*"')
_BARE_IDENTIFIER_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)

# DuckDB's reserved keywords (`select keyword_name from duckdb_keywords()
# where keyword_category = 'reserved'`) — these are otherwise-legal bare
# identifiers (e.g. "from", "group", "limit") that still break unquoted when
# they collide with a real column name, so they're folded into the same
# "needs quoting" bucket as columns containing spaces/punctuation.
DUCKDB_RESERVED_KEYWORDS = frozenset({
    'all', 'analyse', 'analyze', 'and', 'any', 'array', 'as', 'asc', 'asymmetric',
    'both', 'case', 'cast', 'check', 'collate', 'column', 'constraint', 'create',
    'default', 'deferrable', 'desc', 'describe', 'distinct', 'do', 'else', 'end',
    'except', 'false', 'fetch', 'for', 'foreign', 'from', 'group', 'having', 'in',
    'initially', 'intersect', 'into', 'lambda', 'lateral', 'leading', 'limit',
    'not', 'null', 'offset', 'on', 'only', 'or', 'order', 'pivot', 'pivot_longer',
    'pivot_wider', 'placing', 'primary', 'qualify', 'references', 'returning',
    'select', 'show', 'some', 'summarize', 'symmetric', 'table', 'then', 'to',
    'trailing', 'true', 'union', 'unique', 'unpivot', 'using', 'variadic',
    'when', 'where', 'window', 'with',
})


def _needs_sql_quoting(col) -> bool:
    """True if ``col`` is not safe to reference as a bare SQL identifier —
    contains a space/punctuation/non-ASCII character, starts with a digit
    (or is purely numeric), or collides with a DuckDB reserved keyword."""
    s = str(col)
    return not _BARE_IDENTIFIER_RE.match(s) or s.lower() in DUCKDB_RESERVED_KEYWORDS


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

# A TIMESTAMPTZ -> TIMESTAMP cast follows the SESSION time zone, so the same
# query gives different values on different machines; the recipe is
# timezone('UTC', <timestamptz>).
_TZ_CAST_RE = re.compile(
    r"CAST\s*\((?:[^()]|\([^()]*\))*\bAS\s+TIMESTAMPTZ\s*\)\s*AS\s+TIMESTAMP\b"
    r"|::\s*TIMESTAMPTZ\s*::\s*TIMESTAMP\b",
    re.IGNORECASE | re.S,
)

_DATE_FUNCTIONS = {
    "year", "month", "day", "hour", "minute", "second", "quarter", "week", "weekday",
    "dayofweek", "dayofyear", "dayofmonth", "isodow", "isoyear", "yearweek", "epoch",
    "epoch_ms", "date_part", "datepart", "date_trunc", "datetrunc", "date_diff",
    "datediff", "date_sub", "datesub", "date_add", "dateadd", "age", "strftime",
    "last_day", "dayname", "monthname", "century", "decade", "era", "millennium",
    "timezone", "extract", "date_bin", "time_bucket", "to_days", "make_date",
}
_NUMERIC_FUNCTIONS = {
    "sum", "avg", "mean", "median", "quantile", "quantile_cont", "quantile_disc",
    "stddev", "stddev_samp", "stddev_pop", "var_samp", "var_pop", "variance", "corr",
    "covar_pop", "covar_samp", "round", "abs", "ceil", "ceiling", "floor", "sqrt",
    "power", "pow", "log", "ln", "log10", "log2", "exp", "sign", "+", "-", "*", "/",
    "//", "%", "^", "fdiv", "fmod", "regr_slope", "regr_intercept", "regr_r2", "sum_no_overflow",
}
_NUMERIC_TYPES = {
    "INTEGER", "INT", "INT32", "INT64", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT",
    "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "REAL", "INTEGER_LITERAL", "DECIMAL_LITERAL",
    "UBIGINT", "UINTEGER", "BIGNUM",
}
_DATE_TYPES = {"DATE", "TIMESTAMP", "TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE", "TIME", "INTERVAL",
               "TIMESTAMP_S", "TIMESTAMP_MS", "TIMESTAMP_NS"}

NUMBER_TEXT_SQL = (
    "Follow its `number text` fact in Column Statistics instead of a raw CAST or an implicit "
    "cast: `plain` -> `TRY_CAST(trim(col) AS DOUBLE)`; otherwise remove the shown separator/affix "
    "first — thousands comma: `TRY_CAST(replace(trim(col), ',', '') AS DOUBLE)`, decimal comma: "
    "`TRY_CAST(replace(trim(col), ',', '.') AS DOUBLE)`, a unit or symbol: `replace(...)` it away. "
    "Rows listed under `not numbers` / `missing tokens` become NULL, which is intended "
    "(aggregates skip NULL; filter them out explicitly where the question needs it)."
)
DATE_TEXT_SQL = (
    "Parse it with the format its `date text` fact in Column Statistics shows, before comparing "
    "it with a DATE/TIMESTAMP, extracting a part (YEAR/EXTRACT/date_trunc/strftime) or doing "
    "date arithmetic: `try_strptime(trim(col), '<format>')` (use the `(DuckDB: ...)` spelling when "
    "the fact gives one); ISO text with no `offsets` fact: `TRY_CAST(trim(col) AS TIMESTAMP)`; "
    "`offsets ... on every value`: `timezone('UTC', TRY_CAST(trim(col) AS TIMESTAMPTZ))`; "
    "`date as number yyyymmdd`: `try_strptime(CAST(TRY_CAST(col AS BIGINT) AS VARCHAR), '%Y%m%d')`. "
    "Rows under `not dates` become NULL. Never `CAST(... AS TIMESTAMPTZ) AS TIMESTAMP`."
)


def _duckdb_excerpt(message: str) -> str:
    """The first line of a DuckDB error plus its ``LINE n: ...`` / caret
    lines and any ``Candidate bindings`` line — without the candidate
    overload list, which says nothing the model can use."""
    lines = message.splitlines()
    keep = [lines[0]] if lines else []
    for i, line in enumerate(lines[1:], 1):
        if line.startswith("LINE ") or line.startswith("Candidate bindings") or set(line.strip()) == {"^"}:
            keep.append(line)
    return summarize_large_value("\n".join(keep))


class SQLValidator(QueryValidator):
    """Validator for SQL queries."""

    # Rows fetched for the result checks: the shape, emptiness and all-null
    # checks need the head, not the whole result.
    _CHUNKS_FETCHED = 50  # DuckDB chunks of 2048 rows

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------
    def _preprocess_query(self, query: str) -> str:
        self._check_banned_sql_commands(query)
        query = self._strip_set_statements(query)
        self._check_single_statement(query)
        self._check_session_zone_cast(query)

        warnings = self._check_sql_cartesian(query)
        if warnings:
            raise ValueError(
                "Potentially dangerous query — possible Cartesian product detected:\n"
                + "\n".join(f"  - {w}" for w in warnings)
                + "\nRewrite without any cross product (`FROM a, b`, `CROSS JOIN`, "
                "`JOIN ... ON 1=1` / `ON TRUE` are all rejected): independent "
                "per-table figures (one aggregate per table) go into scalar "
                "subqueries of ONE final SELECT — `SELECT (SELECT SUM(x) FROM a) + "
                "(SELECT SUM(x) FROM b) AS total` — or are stacked as labelled rows "
                "with UNION ALL; grouped results (one row per category/period) are "
                "combined only through a JOIN on the grouping key (FULL OUTER JOIN "
                "with COALESCE on the key when the categories differ between tables)."
            )

        query = self._remove_redundant_aliases(query)
        return query.replace("`", '"')

    def _check_banned_sql_commands(self, query: str) -> None:
        query_lower = query.lower()
        found = [cmd for cmd in UNAUTHORIZED_SQL_COMMANDS if cmd in query_lower]
        if found:
            raise ValueError(
                f"Query contains unauthorized SQL commands: {', '.join(found)}\n"
                "Only a read-only SELECT over the registered tables is allowed: no DDL/DML, "
                "no file access, no settings. The tables are already registered as Table_0, ..."
            )

    def _strip_set_statements(self, query: str) -> str:
        return re.sub(r'\bSET\s+\w+\s*=\s*[^;]+;?\s*', '', query, flags=re.IGNORECASE).strip()

    @staticmethod
    def _masked(query: str) -> str:
        """The query with string literals, quoted identifiers and comments
        blanked, for structural scans."""
        masked = _COMMENT_RE.sub(" ", query)
        masked = _LITERAL_RE.sub("''", masked)
        return _QUOTED_IDENT_RE.sub('""', masked)

    def _check_single_statement(self, query: str) -> None:
        """One SELECT: two statements separated by ';' run both, and only the
        last one's rows come back — the first is silently lost."""
        body = self._masked(query).strip().rstrip(";")
        if ";" in body:
            raise ValueError(
                "The code holds more than one SQL statement (a ';' between two SELECTs). "
                "Only ONE statement runs and only its result counts — the earlier one is "
                "discarded. Combine them into a single query: a CTE (`WITH t AS (...) SELECT "
                "...`), a JOIN on the shared key, scalar subqueries in one SELECT, or UNION ALL."
            )

    def _check_session_zone_cast(self, query: str) -> None:
        if _TZ_CAST_RE.search(self._masked(query)):
            raise ValueError(
                "`CAST(... AS TIMESTAMPTZ) AS TIMESTAMP` depends on the session time zone: the "
                "same query gives different clock values on different machines. For date text "
                "with an `offsets` fact use `timezone('UTC', TRY_CAST(trim(col) AS TIMESTAMPTZ))`, "
                "which yields the UTC wall clock everywhere (Column Statistics dates are UTC); "
                "ISO text without offsets is `TRY_CAST(trim(col) AS TIMESTAMP)`."
            )

    def _check_sql_cartesian(self, query: str) -> list[str]:
        return [
            msg for pattern, msg in SQL_CARTESIAN_PATTERNS
            if re.search(pattern, query, re.IGNORECASE | re.DOTALL)
        ]

    def _check_unquoted_special_columns(self, query: str, dataframes: list) -> None:
        """Catch a bare-numeric (or space/punctuation-bearing) column
        reference DuckDB would silently misinterpret, before execution.

        Columns are not dropped for having spaces, punctuation, or a purely
        numeric name (see ``utils.clean_columns``); they must be double-quoted.
        A name holding spaces/punctuation can appear unquoted only by mistake,
        so any bare occurrence is flagged. A purely NUMERIC name (year columns:
        "2018", "2019") is ambiguous: the bare number is also an ordinary
        literal (`WHERE year = 2018`, `IN (2019, 2020)`, `> 2018 - 2000`), and
        flagging every literal rejected correct queries. It is flagged only
        where a column is expected and a literal would be odd: in the SELECT
        list (`SELECT "Area", 2018 FROM`) and as an aggregate argument
        (`SUM(2018)`), where DuckDB would silently return the constant.
        Reserved-keyword collisions are left to DuckDB's own parser error
        (``FROM``, ``GROUP`` are also syntax everywhere else).
        """
        special_cols: set[str] = set()
        for df in dataframes:
            special_cols.update(
                str(c) for c in getattr(df, "columns", [])
                if not _BARE_IDENTIFIER_RE.match(str(c))
            )
        if not special_cols:
            return

        stripped = self._masked(query)
        offenders = []
        for col in sorted(special_cols):
            if re.fullmatch(r"\d+", col):
                # In-lists are literals; elsewhere a bare number in a column
                # position names the column.
                scan = re.sub(r"\bIN\s*\([^()]*\)", "IN ()", stripped, flags=re.IGNORECASE)
                position = (
                    rf"(?i)(?:\bSELECT\s+(?:DISTINCT\s+)?|,\s*|\b(?:SUM|AVG|MIN|MAX|COUNT|MEDIAN|"
                    rf"STDDEV\w*|VAR\w*|QUANTILE\w*|ROUND|ABS|CORR|TRY_CAST|CAST)\s*\(\s*)"
                    rf"{re.escape(col)}\b(?=\s*(?:AS\b|,|\)|FROM\b|$))"
                )
                if re.search(position, scan):
                    offenders.append(col)
            elif re.search(rf'(?<![\w"]){re.escape(col)}(?![\w"])', stripped, re.IGNORECASE):
                offenders.append(col)
        if not offenders:
            return

        examples = "\n".join(f'  - {col!r} -> "{col}"' for col in offenders[:8])
        raise ValueError(
            "Column name(s) used without double-quoting — these are not safe bare "
            "SQL identifiers (they contain spaces/punctuation/non-ASCII characters, "
            "or are purely numeric, which DuckDB reads as a number literal):\n"
            f"{examples}\n"
            'Wrap every reference to them in double quotes, e.g. "column name" — '
            'double any literal " inside the name itself (col"x" -> "col""x""). '
            "If a bare number above was meant as a constant, not the column, write it "
            "as an arithmetic expression or in a CTE so the intent is unambiguous."
        )

    def _remove_redundant_aliases(self, query: str) -> str:
        return re.compile(
            r'\b(FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s+\2\b',
            flags=re.IGNORECASE
        ).sub(r'\1 \2', query)

    # ------------------------------------------------------------------
    # DuckDB error explanation
    # ------------------------------------------------------------------
    def _column_suggestions(self, name: str) -> str:
        lines = []
        for alias, df in zip(self.table_names, self.dataframes):
            close = self._suggest_columns(name, [str(c) for c in getattr(df, "columns", [])], max_suggestions=4)
            if close:
                lines.append(f"  {alias}: " + ", ".join(f'"{c}"' for c in close))
        return ("\nClosest real columns, by table:\n" + "\n".join(lines)) if lines else ""

    def _explain_duckdb_error(self, exc: Exception, query: str) -> str:
        """The feedback for a DuckDB exception: its excerpt plus the recipe
        the correction model should apply."""
        full = str(exc)
        excerpt = _duckdb_excerpt(full)
        low = full.lower()

        # --- a function or operator received text where it needs a number/date
        m = re.search(
            r"(?:No function matches the given name and argument types|"
            r"Could not choose a best candidate function for the function call) ['\"]([^'\"(]+)\(([^)]*)\)",
            full,
        )
        if m:
            fn, args = m.group(1).strip(), m.group(2)
            types = [t.strip() for t in args.split(",")]
            fn_l = fn.lower()
            if fn_l == "~~":
                return (
                    f"{excerpt}\nLIKE on a non-text column: compare numerically "
                    "(`col = 1`, `col BETWEEN ...`) or cast to text first (`CAST(col AS VARCHAR) LIKE '1%'`)."
                )
            if "VARCHAR" in types and fn_l in _DATE_FUNCTIONS:
                return f"{excerpt}\n`{fn}` received date TEXT (VARCHAR). {DATE_TEXT_SQL}"
            if "VARCHAR" in types and fn_l in _NUMERIC_FUNCTIONS:
                return (
                    f"{excerpt}\n`{fn}` received TEXT (VARCHAR): the column is stored as number "
                    f"text. {NUMBER_TEXT_SQL} If the column is in fact an identifier or a code "
                    "(see its `identifier:` / `digit code` fact), it should not be summed or averaged at all."
                )
            if "VARCHAR" in types:
                return (
                    f"{excerpt}\n`{fn}` does not accept a VARCHAR argument: the column is stored as "
                    "text. Cast it to the type the function needs with the recipe of its Column "
                    "Statistics fact (`number text` -> TRY_CAST after removing separators; `date "
                    "text` -> try_strptime(format)), or use the text form of the function."
                )
            if any(t in _DATE_TYPES for t in types) and fn_l in _NUMERIC_FUNCTIONS:
                return (
                    f"{excerpt}\nArithmetic/aggregation on a DATE/TIMESTAMP: subtract two dates to "
                    "get an INTERVAL (`date_diff('day', a, b)` for a number of days), and aggregate "
                    "with min/max/count, not sum/avg."
                )
            return (
                f"{excerpt}\n`{fn}` does not exist for argument types ({', '.join(types)}): check "
                "the DuckDB function name and cast each argument to the type it expects "
                "(`CAST(x AS DOUBLE)` for a numeric column, try_strptime for date text)."
            )

        m = re.search(r"Cannot (?:compare values|mix values) of type (\w+(?: \w+)*) and (?:type )?(\w+(?: \w+)*)", full)
        if m:
            left, right = m.group(1).upper(), m.group(2).upper()
            if left == "VARCHAR" and right in _NUMERIC_TYPES or right == "VARCHAR" and left in _NUMERIC_TYPES:
                return (
                    f"{excerpt}\nA TEXT column is compared with a number: the column is stored as "
                    f"number text. {NUMBER_TEXT_SQL} If the column holds codes (zip, id), compare "
                    "with a quoted text literal instead."
                )
            if left == "VARCHAR" and right in _DATE_TYPES or right == "VARCHAR" and left in _DATE_TYPES:
                return f"{excerpt}\nA TEXT column is compared with a date: it is stored as date text. {DATE_TEXT_SQL}"
            return (
                f"{excerpt}\nThe two sides hold different kinds of value ({left} vs {right}). Cast the "
                "column to the literal's type with the recipe of its Column Statistics fact; never "
                "cast the numeric side to VARCHAR to compare as text."
            )

        # --- a cast failed on a stored value or on a literal from the query
        m = re.search(r"Could not convert string '(.*)' to (\w+)(?: when casting from source column (.+?))?$", full.splitlines()[0])
        if m:
            value, target, column = m.group(1), m.group(2).upper(), m.group(3)
            col = f'"{column}"' if column else "the column"
            in_query = f"'{value}'" in query
            if target in {"BOOL", "BOOLEAN"}:
                return (
                    f"{excerpt}\n{col} is not a boolean column: it stores the TEXT `{value}` (see its "
                    "values in Column Statistics). Test it as text (`col = '...'`, `col IS NOT NULL`), "
                    "not as TRUE/FALSE."
                )
            if in_query and target in _NUMERIC_TYPES:
                return (
                    f"{excerpt}\nThe LITERAL '{value}' in the query is not a number, but the column it is "
                    "compared with is numeric. Write a numeric literal (no quotes), or compare a text "
                    "column instead — check the column's type and values in Column Statistics."
                )
            if target in _NUMERIC_TYPES:
                return (
                    f"{excerpt}\n{col} is number TEXT: the stored value `{value}` is not a plain number "
                    f"({self._value_shape(value)}). {NUMBER_TEXT_SQL}"
                )
            if target in _DATE_TYPES or "DATE" in target or "TIME" in target:
                return f"{excerpt}\n{col} is date TEXT and `{value}` is not ISO. {DATE_TEXT_SQL}"
            return f"{excerpt}\nThe stored value `{value}` cannot become {target}: use TRY_CAST (NULL on failure) after cleaning the text per its Column Statistics fact."

        m = re.search(r"invalid (date|timestamp|time) field format: \"(.*)\", expected format is \(([^)]+)\)(?: when casting from source column (.+?))?$", full.splitlines()[0])
        if m or "field value out of range" in low:
            if m:
                value, column = m.group(2), m.group(4)
                col = f'"{column}"' if column else "the column"
                return f"{excerpt}\n{col} is date TEXT written as `{value}`, not ISO. {DATE_TEXT_SQL}"
            return f"{excerpt}\nThe date text does not parse with an implicit cast. {DATE_TEXT_SQL}"

        if "out of range for the destination type" in low or "overflow" in low:
            return (
                f"{excerpt}\nNumeric overflow: cast to a wider type (`CAST(col AS DOUBLE)` / `HUGEINT`) "
                "before summing, and check the column is a quantity, not an identifier or a "
                "yyyymmdd date number (Column Statistics)."
            )

        # --- binder: grouping, names, set operations
        m = re.search(r'column "(.+?)" must appear in the GROUP BY clause', full)
        if m:
            return (
                f"{excerpt}\n\"{m.group(1)}\" is selected without being aggregated: every non-aggregated "
                "SELECT column must be in GROUP BY (`GROUP BY \"{0}\"`), or be wrapped in an aggregate "
                "(MAX/MIN/ANY_VALUE) if one value per group is meant.".format(m.group(1))
            )
        m = re.search(r'Referenced column "(.+?)" not found in FROM clause', full) or re.search(
            r'(?:Table|Values list|Binding) "[^"]+" does not have a column named "(.+?)"', full
        )
        if m:
            name = m.group(1)
            held = [a for a, df in zip(self.table_names, self.dataframes) if name in [str(c) for c in getattr(df, "columns", [])]]
            if held:
                return (
                    f"{excerpt}\n\"{name}\" exists in {', '.join(held)} but not in the scope it is "
                    "referenced from: a CTE/subquery exposes only the columns it SELECTs (add it "
                    "there), a SELECT alias cannot be used in the same SELECT list, and a joined "
                    "table must be referenced through its alias."
                )
            every = [str(c) for df in self.dataframes for c in getattr(df, "columns", [])]
            damaged = undecodable_matches(name, every)
            if damaged:
                # "Total £" for the header "Total �": DuckDB has no escape for
                # U+FFFD in an identifier, so the character itself is the only
                # spelling — say that it is real, not a display glitch.
                real = " or ".join(f'"{r}"' for r in damaged)
                return (
                    f"{excerpt}\nNo column is named \"{name}\": the real header is {real}. Its "
                    "� is U+FFFD, a character the source file could not decode, and it is part "
                    "of the name: copy the name exactly, that character included, double-quoted — "
                    "never the symbol you think was meant."
                )
            return (
                f"{excerpt}\nNo column is named \"{name}\".{self._column_suggestions(name)}\n"
                "Use the real column names verbatim (double-quoted when they hold spaces, "
                "punctuation or digits); never infer a name."
            )
        m = re.search(r"Table with name (\S+) does not exist", full)
        if m:
            return (
                f"{excerpt}\nThe only tables are: {', '.join(self.table_names)} (use these aliases "
                "exactly; a CTE must be declared in WITH before it is used)."
            )
        if "set operations can only apply to expressions with the same number of result columns" in low:
            return (
                f"{excerpt}\nEvery SELECT in a UNION / UNION ALL must return the same number of "
                "columns, in the same order and of compatible types; pad a branch with a "
                "constant (`NULL AS x`) or a label where needed."
            )
        m = re.search(r'Ambiguous reference to column name "(.+?)"', full)
        if m:
            return (
                f"{excerpt}\n\"{m.group(1)}\" exists in more than one joined table: qualify every "
                "reference with the table alias (`t.\"{0}\"`), in SELECT, WHERE and GROUP BY.".format(m.group(1))
            )
        if "aggregate" in low and ("nested" in low or "cannot contain" in low or "window" in low):
            return (
                f"{excerpt}\nAggregates cannot be nested or used where DuckDB forbids them: compute the "
                "inner aggregate in a CTE/subquery (one row per group), then aggregate that."
            )
        if "having clause" in low or ("where clause" in low and "aggregate" in low):
            return (
                f"{excerpt}\nAn aggregate in WHERE is not allowed: filter aggregated groups in HAVING, "
                "or compute the aggregate in a CTE and filter it there."
            )

        # --- parser
        m = re.search(r'syntax error at or near "(.+?)"', full)
        if m:
            token = m.group(1)
            masked = self._masked(query)
            needs_quote = [
                str(c) for df in self.dataframes for c in getattr(df, "columns", [])
                if _needs_sql_quoting(c) and (str(c).lower() == token.lower() or str(c).lower().startswith(token.lower() + " "))
            ]
            # DuckDB names the token AFTER a bare reserved-word column
            # (`SELECT order FROM t` -> near "FROM"): any keyword-named column
            # used bare in the query is the likelier culprit.
            needs_quote += [
                str(c) for df in self.dataframes for c in getattr(df, "columns", [])
                if str(c).lower() in DUCKDB_RESERVED_KEYWORDS and str(c) not in needs_quote
                and re.search(rf"\b{re.escape(str(c))}\b(?!\s+BY\b)", masked, re.IGNORECASE)
            ]
            hint = (
                f"The column(s) {', '.join(repr(c) for c in needs_quote[:3])} must be double-quoted wherever "
                f"they are meant as columns (\"{token}\" is where the parser gave up): a bare reserved word "
                "or a name holding spaces/punctuation reads as syntax."
                if needs_quote else
                "Check the keyword and punctuation at the caret (a misspelled keyword such as FORM for FROM, "
                "a missing comma or parenthesis, a reserved word used as a bare column name — quote it)."
            )
            if re.search(r"\bUNION\b", query, re.I) and re.search(r"\bON\b", query, re.I):
                hint += (" UNION combines whole SELECTs and takes no ON clause; JOIN ... ON belongs inside "
                         "one SELECT.")
            return f"{excerpt}\n{hint}"

        if "out of memory" in low:
            return (
                f"{excerpt}\nThe query ran out of memory. Aggregate earlier, filter before joining, "
                "and avoid DISTINCT or ORDER BY over the whole raw table when a grouped result is asked."
            )
        return f"{excerpt}\nCheck that all referenced columns, tables and aliases exist and that each argument has the type its function expects."

    @staticmethod
    def _value_shape(value: str) -> str:
        v = value.strip()
        if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", v):
            return "it carries a thousands comma"
        if re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+(,\d+)?", v) or re.fullmatch(r"[-+]?\d+,\d+", v):
            return "it uses a decimal comma and/or a thousands dot"
        if re.fullmatch(r"[-+]?\d{1,3}([  ']\d{3})+([.,]\d+)?", v):
            return "it carries a thousands space or apostrophe"
        if re.search(r"\d", v):
            return "digits with an affix, unit or other characters around them"
        return "no digits at all — a `not numbers` / `missing tokens` value that must become NULL"

    # ------------------------------------------------------------------
    # Execution — runs inside sandbox process via base _execute_query
    # ------------------------------------------------------------------
    def _run_query(self, query: str, dataframes: list, table_names: list) -> Any:
        self._check_unquoted_special_columns(query, dataframes)

        con = duckdb.connect(database=":memory:")
        # UTC session, as in QueryExecutor._execute_sql: a TIMESTAMPTZ ->
        # TIMESTAMP cast must give the same value here and at execution.
        con.execute("SET TimeZone='UTC'")
        con.execute(f"SET memory_limit='{self.mem_limit_mb}MB'")
        con.execute("SET threads=4")
        try:
            for df, name in zip(dataframes, table_names):
                con.register(name, df)
            cursor = con.execute(query.rstrip().rstrip(";"))
            chunks = []
            for _ in range(self._CHUNKS_FETCHED):
                chunk = cursor.fetch_df_chunk(1)
                if chunk is None or len(chunk) == 0:
                    if not chunks:
                        chunks.append(chunk if chunk is not None else pd.DataFrame())
                    break
                chunks.append(chunk)
            result = pd.concat(chunks, ignore_index=True) if len(chunks) > 1 else chunks[0]
            result.attrs["orqa_null_columns"] = self._all_null_columns(result)
            return result
        except duckdb.Error as e:
            raise ValueError(self._explain_duckdb_error(e, query)) from e
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------
    _EMPTY_RESULT_TAIL = (
        "Only widen a predicate the question does not actually require. If every "
        "filter is faithful to the question and the data genuinely has no matching "
        "rows, keep the query as it is and say so in `motivation` — do not broaden "
        "the scope to force rows."
    )

    def _build_empty_result_feedback(self) -> str:
        if len(self.table_names) == 1:
            head = (
                "Empty result — the WHERE clause matched no rows.\n"
                "Fix, in order: (1) drop the predicates one at a time to find which one "
                "empties the result; (2) compare values exactly as Column Statistics shows "
                "them stored (case, accents, whitespace — `lower(trim(col)) = lower('value')` "
                "when the question allows); (3) a text column compared with a number or date: "
                "cast it first with its number text / date text recipe (TRY_CAST after removing "
                "separators, try_strptime with the shown format); (4) check the date range "
                "against the column's parsed min..max in Column Statistics."
            )
        else:
            head = (
                "Empty result — the JOIN matched no rows, or a WHERE after it removed them all.\n"
                "Fix, in order: (1) switch to LEFT JOIN to see which side fails to match; "
                "(2) normalise the join keys on BOTH sides — `ON lower(trim(t1.key)) = "
                "lower(trim(t2.key))` — or cast both to the same type when one side is number "
                "text (compare the two columns' values and shapes in Column Statistics); "
                "(3) then drop the WHERE predicates one at a time."
            )
        return f"{head}\n{self._EMPTY_RESULT_TAIL}"

    def _get_language_name(self) -> str:
        return "SQL"
