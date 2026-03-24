from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import json
import re
import multiprocessing as mp
import pandas as pd

TECHNICAL_TERMS = [
    # Data structures
    'dataframe', 'dataframes', 'dataset', 'datasets',
    'schema', 'dtype', 'index', 'indices',
    'null', 'nan', 'none',
    # Table/data terminology
    'record', 'records', 'row', 'rows',
    'column', 'columns', 'field', 'fields',
    'table', 'tables', 'entry', 'entries',
    # Database/query operations
    'query', 'select', 'distinct',
    'join', 'merge', 'union', 'concat',
    'groupby', 'group by', 'order by',
    'pivot', 'unpivot', 'melt',
    'primary key', 'foreign key',
    # Code/library specific
    'pd.', 'df.', 'sql', 'duckdb',
    'str.lower', 'astype', 'fillna', 'dropna',
    # Statistical/math jargon
    'correlate', 'correlation', 'coefficient',
    'pearson', 'spearman', 'kendall',
    'covariance', 'r-squared',
    'percentile', 'quantile', 'variance',
    'p-value', 'hypothesis', 'significance',
    # Vague-but-technical
    'aggregate', 'aggregation',
    'reshape', 'subset', 'slice',
]

TERM_SUGGESTIONS = {
    'join':        'Instead of "join", say "combine", "match", or "link" — e.g. "link customer info with their orders".',
    'merge':       'Instead of "merge", say "combine" or "bring together" — e.g. "combine sales data with location info".',
    'correlate':   'Instead of "correlate", say "tend to increase/decrease together" or "relationship between".',
    'correlation': 'Instead of "correlation", say "relationship" or "pattern" — e.g. "Is there a pattern between X and Y?".',
    'groupby':     'Instead of "groupby", say "for each" or "broken down by" — e.g. "What is the average revenue for each region?".',
    'group by':    'Instead of "group by", say "for each" or "per" — e.g. "total sales per category".',
    'aggregate':   'Instead of "aggregate", say "total", "overall", or "combined" — e.g. "What is the total revenue per store?".',
    'aggregation': 'Instead of "aggregation", say "summary" or "total" — e.g. "Give me a summary of sales by region".',
    'filter':      'Instead of "filter", say "where", "only", or "that have" — e.g. "restaurants that have more than 10 inspections".',
    'query':       'Instead of "query", describe the business question directly — e.g. "Which customers spent the most last month?".',
    'select':      'Instead of "select", say "find", "show", or "list" — e.g. "Show the top 10 restaurants by revenue".',
    'pivot':       'Instead of "pivot", say "broken down by" or "compared across" — e.g. "Revenue compared across regions and categories".',
    'null':        'Instead of "null", say "missing" or "without a value" — e.g. "restaurants without a listed address".',
    'nan':         'Instead of "NaN", say "missing" or "not available" — e.g. "entries where the phone number is not available".',
    'schema':      'Instead of "schema", describe the data directly — e.g. "restaurant name, address, and inspection date".',
    'dataframe':   'Instead of "dataframe", say "data" or describe the subject — e.g. "the restaurant data".',
    'dataset':     'Instead of "dataset", say "data" or name the subject — e.g. "the inspection records".',
    'row':         'Instead of "row", say "entry", "restaurant", or whatever the subject is — e.g. "each restaurant".',
    'column':      'Instead of "column", name the actual piece of information — e.g. "the restaurant name" instead of "the name column".',
    'union':       'Instead of "union", say "combined" or "across both" — e.g. "restaurants across both lists".',
}


def _sandbox_worker(queue: mp.Queue, fn, args: tuple) -> None:
    """Top-level worker function executed inside the sandbox process.
    Must be defined at module level so it is picklable on Windows."""
    try:
        queue.put(("ok", fn(*args)))
    except Exception as e:
        queue.put(("error", type(e).__name__, str(e)))


class QueryValidator(ABC):
    """Base class for query validation."""

    DEFAULT_TIMEOUT   = 30   # seconds
    DEFAULT_MEM_LIMIT = 512  # MB

    def __init__(
        self,
        dataframes:   List,
        table_names:  List[str],
        lookup_dict:  dict,
        mem_limit_mb: int = DEFAULT_MEM_LIMIT,
        timeout:      int = DEFAULT_TIMEOUT,
    ):
        self.dataframes   = dataframes
        self.table_names  = table_names
        self.lookup_dict  = lookup_dict
        self.mem_limit_mb = mem_limit_mb
        self.mem_limit    = mem_limit_mb * 1024 * 1024  # bytes — kept for DuckDB SET
        self.timeout      = timeout

        self.validation_errors: list = []
        self.good_queries:      dict = {}
        self.errors:            list = []

    # ------------------------------------------------------------------
    # Sandbox
    # ------------------------------------------------------------------
    def _run_in_sandbox(self, fn, args: tuple = ()) -> Any:
        """
        Run fn(*args) in a child process.
        On timeout the process is killed and the OS reclaims all its memory.
        """
        queue = mp.Queue()
        p = mp.Process(target=_sandbox_worker, args=(queue, fn, args), daemon=True)
        p.start()
        p.join(timeout=self.timeout)

        if p.is_alive():
            p.kill()
            p.join()
            raise TimeoutError(
                f"Query exceeded the {self.timeout}s time limit and was aborted.\n"
                "This usually means a Cartesian product or a missing join condition.\n"
                "Simplify the query or add more specific join/filter conditions."
            )

        if queue.empty():
            raise RuntimeError("Sandbox process exited without returning a result.")

        status, *rest = queue.get()
        if status == "error":
            exc_type_name, exc_msg = rest[0], rest[1]
            # Reconstruct the original exception type so that callers such as
            # _normalize_pandas_error can match on isinstance(e, KeyError) etc.
            exc_type = {
                "KeyError":          KeyError,
                "ValueError":        ValueError,
                "TypeError":         TypeError,
                "MemoryError":       MemoryError,
                "SyntaxError":       SyntaxError,
                "TimeoutError":      TimeoutError,
                "NameError":         NameError,
                "AttributeError":    AttributeError,
                "IndexError":        IndexError,
                "ZeroDivisionError": ZeroDivisionError,
                "OverflowError":     OverflowError,
                # pandas-specific — arrive as their class name from the worker
                "MergeError":           ValueError,
                "ParserError":          ValueError,
                "OutOfBoundsDatetime":  ValueError,
                "InvalidIndexError":    KeyError,
            }.get(exc_type_name, RuntimeError)
            raise exc_type(exc_msg)
        return rest[0]

    # ------------------------------------------------------------------
    # Memory pre-check
    # ------------------------------------------------------------------
    def _check_input_memory(self) -> None:
        """Raise MemoryError if the total size of input DataFrames exceeds mem_limit_mb."""
        total_mb = sum(
            df.memory_usage(deep=True).sum()
            for df in self.dataframes
            if isinstance(df, pd.DataFrame)
        ) / (1024 ** 2)

        if total_mb > self.mem_limit_mb:
            raise MemoryError(
                f"Input data is {total_mb:.1f}MB, which exceeds the "
                f"{self.mem_limit_mb}MB limit.\n"
                "Consider pre-filtering your data before running this query."
            )

    # ------------------------------------------------------------------
    # Main validation loop
    # ------------------------------------------------------------------
    def validate_queries(self, result: Dict) -> Tuple[bool, Dict, Dict]:
        all_valid = True

        try:
            self._check_input_memory()
        except MemoryError as e:
            return False, [{"role": "user", "content": f"MemoryError: {e}"}], {}, [str(e)]

        for idx, q in enumerate(result["queries"]):
            actual_query = q
            dataframes, ordered_names = self.prefilter_dataframes(actual_query['tables'])
            try:
                raw_code = q.get("code") or ""
                if not raw_code.strip():
                    raise ValueError("query 'code' field is missing or empty")

                query_code = self.replace_aliases(raw_code, self.lookup_dict)
                query_code = self._preprocess_query(query_code.strip())
                actual_query["code"] = query_code
                result_data = self._execute_query(query_code, dataframes, ordered_names)

                if self._is_empty_result(result_data):
                    raise ValueError(self._build_empty_result_feedback())
                if not self._check_table_usage(query_code):
                    raise ValueError(self._build_unused_tables_feedback())
                if not self._check_tables_field_coverage(actual_query):
                    raise ValueError(self._build_tables_field_coverage_feedback())
                if self._check_table_names_in_question(actual_query["question"]):
                    raise ValueError(self._build_question_tables_feedback())

                technical_terms = self._check_technical_terms_in_question(actual_query["question"])
                if technical_terms:
                    raise ValueError(self._build_technical_terms_feedback(technical_terms))
                self.good_queries[idx] = actual_query

            except Exception as e:
                all_valid = False
                error_msg = e.args[0] if e.args else str(e)
                self.validation_errors.append({"query": actual_query, "error": f"{type(e).__name__}: {error_msg}"})
                self.errors.append(f"Error {type(e).__name__}: {error_msg}")

        if all_valid:
            return True, {}, self.good_queries, self.errors
        return False, self._build_feedback(), self.good_queries, self.errors

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def _preprocess_query(self, query: str) -> str:
        pass

    @abstractmethod
    def _run_query(self, query: str, dataframes: list, table_names: list) -> Any:
        """Execute the query and return a DataFrame result."""
        pass

    def _execute_query(self, query: str, dataframes: list, table_names: list) -> Any:
        """Wrap _run_query in a sandbox. Override in subclass if needed."""
        return self._run_in_sandbox(self._run_query, args=(query, dataframes, table_names))

    @abstractmethod
    def _build_empty_result_feedback(self) -> str:
        pass

    @abstractmethod
    def _get_language_name(self) -> str:
        pass

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def prefilter_dataframes(self, tables):
        # Build a name→DataFrame lookup from the constructor-ordered lists.
        # This is the only safe way to match — positional zip is wrong because
        # the query's 'tables' field can list tables in any order.
        df_by_name = dict(zip(self.table_names, self.dataframes))

        dataframes = []
        ordered_names = []
        for table in tables:
            name = table["name"]
            dataframe = df_by_name.get(name)
            if dataframe is None:
                raise KeyError(
                    f"Table '{name}' referenced in the query was not found in the "
                    f"available tables: {self.table_names}.\n"
                    "Check for a typo or a stale table name."
                )
            cols = table.get("columns_involved") or []
            if cols:
                existing = [c for c in cols if c in dataframe.columns]
                missing  = [c for c in cols if c not in dataframe.columns]
                if missing:
                    raise KeyError(
                        f"columns_involved references columns that do not exist in table "
                        f"'{name}': {missing}.\n"
                        "Check column names for typos or stale references after a rename."
                    )
                dataframes.append(dataframe[existing])
            else:
                dataframes.append(dataframe)
            ordered_names.append(name)

        return dataframes, ordered_names


    def replace_aliases(self, code: str, aliases: dict) -> str:
        if code is None:
            return ""
        for table_name, alias in aliases.items():
            # Quoted version: "alias" → table_name (exact string match, safe)
            code = code.replace(f'"{alias}"', table_name)
            # Unquoted: word-boundary match only, so substrings inside column
            # names like 'category' are never accidentally mangled.
            code = re.sub(rf'\b{re.escape(alias)}\b', table_name, code)
        return code

    def _is_empty_result(self, result: Any) -> bool:
        if result is None:
            return True
        if hasattr(result, 'empty'):
            return result.empty
        if hasattr(result, 'is_empty'):
            return result.is_empty()
        if hasattr(result, '__len__'):
            return len(result) == 0
        return False

    def _check_table_usage(self, query_text: str) -> bool:
        self.unused_tables = {t for t in self.table_names if t not in query_text}
        return len(self.unused_tables) == 0

    def _check_table_names_in_question(self, question: str) -> bool:
        """Returns True if any table name leaks into the question."""
        question_lower = question.lower()
        found = {t for t in self.table_names if t.lower() in question_lower}
        self.unused_tables = set(self.table_names) - found
        return len(found) > 0

    def _check_technical_terms_in_question(self, question: str) -> List[str]:
        question_lower = question.lower()
        return [t for t in TECHNICAL_TERMS if re.search(rf'\b{re.escape(t)}\b', question_lower)]

    def _check_tables_field_coverage(self, query: dict) -> bool:
        alias_by_name = {v: k for k, v in self.lookup_dict.items()}
        raw_tables = query.get("tables") or query.get("Tables") or []

        normalised = []
        for t in raw_tables:
            entry = dict(t)
            raw_name = entry.get("name", "").strip()
            entry["name"] = alias_by_name.get(raw_name, raw_name)
            normalised.append(entry)

        query.pop("Tables", None)
        query["tables"] = normalised

        declared = {t["name"] for t in normalised}
        self.tables_field_missing = set(self.table_names) - declared
        self.tables_field_extra   = declared - set(self.table_names)
        return len(self.tables_field_missing) == 0

    # ------------------------------------------------------------------
    # Feedback builders
    # ------------------------------------------------------------------
    def _build_tables_field_coverage_feedback(self) -> str:
        lines = ["The 'tables' field must contain one entry for every table used in the query."]
        if self.tables_field_missing:
            lines.append("Missing entries:\n" + "\n".join(f"  - {t}" for t in sorted(self.tables_field_missing)))
        if self.tables_field_extra:
            lines.append("Unknown entries to remove:\n" + "\n".join(f"  - {t}" for t in sorted(self.tables_field_extra)))
        lines.append(
            "Each entry must set 'name' to the exact table alias, "
            "'reason' to why the table is needed, and "
            "'join_justification' to why it is combined with the other tables in this way."
        )
        return "\n".join(lines)

    def _build_unused_tables_feedback(self) -> str:
        missing_details = "\n".join(
            f"  {t}: {self.lookup_dict.get(t, 'unknown')}" for t in sorted(self.unused_tables)
        )
        return "\n".join([
            "The query must reference ALL provided tables.",
            f"Missing tables:\n{missing_details}",
            f"All required tables: {', '.join(self.table_names)}.",
            "Rewrite the query ensuring each table is joined and contributes to the result.",
        ])

    def _build_question_tables_feedback(self) -> str:
        return "\n".join([
            "Do not use the table names in the natural language question.",
            "Make use of the metadata of the tables if needed.",
            "The question should look like it was written by a user with no knowledge of the table names.",
        ])

    def _build_technical_terms_feedback(self, technical_terms_found: List[str]) -> str:
        lines = [
            "The natural language question must not contain technical data terms.",
            f"Found: {', '.join(technical_terms_found)}.",
            "Rephrase the question as a business user with no knowledge of the underlying data structure would ask it.",
            "",
        ]
        specific = [TERM_SUGGESTIONS[t] for t in technical_terms_found if t in TERM_SUGGESTIONS]
        if specific:
            lines.append("Suggestions for the flagged terms:")
            lines.extend(f"  - {s}" for s in specific)
        else:
            lines.append("Example: instead of 'join the records from both tables', say 'combine sales with customer info'.")
        return "\n".join(lines)

    def _find_repeated_errors(self, error_texts: list) -> list:
        """Return consolidated rules for error types that appear more than once."""
        rules = []
        checked = set()

        patterns = [
            (
                "str.lower() on string literal",
                lambda e: "was called on a string literal" in e,
                "NEVER call .str.lower() on a quoted string like 'col'.str.lower(). "
                "Always use DataFrame['col'].str.lower(). "
                "To normalise join keys use .assign(_key=Table_X['col'].str.lower()) "
                "BEFORE the merge, then join on='_key'."
            ),
            (
                "stale Series in chained merge",
                lambda e: "stale series" in e.lower() or "stale" in e.lower() and "left_on" in e.lower(),
                "NEVER use Table_X['col'] as left_on/right_on after a prior .merge(). "
                "Normalise ALL keys with .assign() before the chain starts."
            ),
            (
                "merge() with no keys",
                lambda e: "no join keys" in e.lower(),
                "Always specify on=, left_on=, or right_on= in every .merge() call."
            ),
            (
                "NameError: merge not defined",
                lambda e: "name 'merge' is not defined" in e,
                "Call merge as a method: Table_A.merge(Table_B, ...) — not as a bare function."
            ),
            (
                "MergeError: duplicate suffixes",
                lambda e: "duplicate columns" in e.lower() and "suffixes" in e.lower(),
                "Suffixes must produce unique column names across ALL merge steps. "
                "Use progressive suffixes: ('_T1','_T2'), ('_T12','_T3'), ('_T123','_T4')."
            ),
            (
                "missing tables",
                lambda e: "must reference all provided tables" in e.lower(),
                "Every query MUST join ALL provided tables. "
                "No table can be omitted — each one contributes required columns."
            ),
            (
                "disjoint sub-queries",
                lambda e: "disjoint query detected" in e.lower(),
                "Every query must form a SINGLE connected result. "
                "Do not produce two separate merge chains — all table groups must be "
                "linked together via merge(), join(), or concat() into one final expression."
            ),
        ]

        for label, matcher, rule in patterns:
            count = sum(1 for e in error_texts if matcher(e))
            if count >= 2 and label not in checked:
                rules.append(f"[repeated {count}x] {rule}")
                checked.add(label)

        return rules

    def _build_feedback(self) -> list:
        feedback_lines = [
            f"The following of the generated {self._get_language_name()} queries are invalid.",
            "Fix the following queries listed below:\n",
        ]
        queries = {"queries": []}

        # Prepend a consolidated rule block for any mistake that recurred 2+ times
        repeated = self._find_repeated_errors([err["error"] for err in self.validation_errors])
        if repeated:
            feedback_lines.insert(0,
                "RECURRING MISTAKES — apply these rules to ALL queries before attempting a fix:\n"
                + "\n".join(f"  * {r}" for r in repeated)
                + "\n"
            )

        for idx, err in enumerate(self.validation_errors, start=1):
            queries["queries"].append(err["query"])
            feedback_lines.append(f"Query number: {idx}:\nError:\n{err['error']}\n")

        return [
            {"role": "system", "content": json.dumps(queries, indent=2)},
            {"role": "user",   "content": "\n".join(feedback_lines)},
        ]