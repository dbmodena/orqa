from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import json
import re
import multiprocessing as mp
import pandas as pd

TECHNICAL_TERMS = [
    'dataframe', 'dataframes', 'dataset', 'datasets',
    'schema', 'dtype', 'index', 'indices',
    'null', 'nan', 'none',
    'record', 'records', 'row', 'rows',
    'column', 'columns', 'field', 'fields',
    'table', 'tables', 'entry', 'entries',
    'query', 'select', 'distinct',
    'join', 'merge', 'union', 'concat',
    'groupby', 'group by', 'order by',
    'pivot', 'unpivot', 'melt',
    'primary key', 'foreign key',
    'pd.', 'df.', 'sql', 'duckdb',
    'str.lower', 'astype', 'fillna', 'dropna',
    'correlate', 'correlation', 'coefficient',
    'pearson', 'spearman', 'kendall',
    'covariance', 'r-squared',
    'percentile', 'quantile', 'variance',
    'p-value', 'hypothesis', 'significance',
    'aggregate', 'aggregation',
    'reshape', 'subset', 'slice',
]

TERM_SUGGESTIONS = {
    'join':        '"combine", "match", or "link" — e.g. "link customers with their orders".',
    'merge':       '"combine" or "bring together" — e.g. "combine sales data with location info".',
    'correlate':   '"tend to increase/decrease together" or "relationship between".',
    'correlation': '"relationship" or "pattern" — e.g. "Is there a pattern between X and Y?".',
    'groupby':     '"for each" or "broken down by" — e.g. "average revenue for each region".',
    'group by':    '"for each" or "per" — e.g. "total sales per category".',
    'aggregate':   '"total" or "combined" — e.g. "total revenue per store".',
    'aggregation': '"summary" or "total" — e.g. "summary of sales by region".',
    'filter':      '"where", "only", or "that have" — e.g. "restaurants with more than 10 inspections".',
    'query':       'Describe the business question directly — e.g. "Which customers spent the most last month?".',
    'select':      '"find", "show", or "list" — e.g. "Show the top 10 restaurants by revenue".',
    'pivot':       '"broken down by" or "compared across" — e.g. "Revenue across regions and categories".',
    'null':        '"missing" or "without a value" — e.g. "restaurants without a listed address".',
    'nan':         '"missing" or "not available" — e.g. "entries where the phone number is not available".',
    'schema':      'Describe the data directly — e.g. "restaurant name, address, and inspection date".',
    'dataframe':   '"data" or describe the subject — e.g. "the restaurant data".',
    'dataset':     '"data" or name the subject — e.g. "the inspection records".',
    'row':         'Name the subject — e.g. "each restaurant" instead of "each row".',
    'column':      'Name the information — e.g. "the restaurant name" instead of "the name column".',
    'union':       '"combined" or "across both" — e.g. "restaurants across both lists".',
}


def _sandbox_worker(queue: mp.Queue, fn, args: tuple) -> None:
    try:
        queue.put(("ok", fn(*args)))
    except Exception as e:
        queue.put(("error", type(e).__name__, str(e)))


class QueryValidator(ABC):

    DEFAULT_TIMEOUT   = 30
    DEFAULT_MEM_LIMIT = 512

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
        self.mem_limit    = mem_limit_mb * 1024 * 1024
        self.timeout      = timeout

        self.validation_errors: list = []
        self.good_queries:      dict = {}
        self.errors:            list = []

    # ------------------------------------------------------------------
    # Sandbox
    # ------------------------------------------------------------------
    def _run_in_sandbox(self, fn, args: tuple = ()) -> Any:
        queue = mp.Queue()
        p = mp.Process(target=_sandbox_worker, args=(queue, fn, args), daemon=True)
        p.start()
        p.join(timeout=self.timeout)

        if p.is_alive():
            p.kill()
            p.join()
            raise TimeoutError(
                f"Query exceeded {self.timeout}s limit — likely a Cartesian product or missing join condition.\n"
                "Add more specific join/filter conditions."
            )

        if queue.empty():
            raise RuntimeError("Sandbox process exited without returning a result.")

        status, *rest = queue.get()
        if status == "error":
            exc_type_name, exc_msg = rest[0], rest[1]
            exc_type = {
                "KeyError": KeyError, "ValueError": ValueError, "TypeError": TypeError,
                "MemoryError": MemoryError, "SyntaxError": SyntaxError,
                "TimeoutError": TimeoutError, "NameError": NameError,
                "AttributeError": AttributeError, "IndexError": IndexError,
                "ZeroDivisionError": ZeroDivisionError, "OverflowError": OverflowError,
                "MergeError": ValueError, "ParserError": ValueError,
                "OutOfBoundsDatetime": ValueError, "InvalidIndexError": KeyError,
            }.get(exc_type_name, RuntimeError)
            raise exc_type(exc_msg)
        return rest[0]

    # ------------------------------------------------------------------
    # Memory pre-check
    # ------------------------------------------------------------------
    def _check_input_memory(self) -> None:
        total_mb = sum(
            df.memory_usage(deep=True).sum()
            for df in self.dataframes
            if isinstance(df, pd.DataFrame)
        ) / (1024 ** 2)
        if total_mb > self.mem_limit_mb:
            raise MemoryError(
                f"Input data is {total_mb:.1f}MB, exceeds {self.mem_limit_mb}MB limit.\n"
                "Pre-filter your data before running this query."
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
            try:
                dataframes, ordered_names = self.prefilter_dataframes(actual_query['tables'])
                raw_code = q.get("code") or ""
                if not raw_code.strip():
                    raise ValueError("query 'code' field is missing or empty")

                query_code = self.replace_aliases(raw_code, self.lookup_dict)
                query_code = self._preprocess_query(query_code.strip())
                actual_query["code"] = query_code
                result_data = self._execute_query(query_code, dataframes, ordered_names)

                # FIX: empty-result rejection is now gated through _empty_result_is_error(),
                # which subclasses can override. For multi-table queries, sample data may
                # have no overlapping keys, so a logically correct merge will return 0 rows —
                # rejecting it here generates spurious correction cycles and token waste.
                if self._is_empty_result(result_data) and self._empty_result_is_error(ordered_names):
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
        pass

    def _execute_query(self, query: str, dataframes: list, table_names: list) -> Any:
        return self._run_in_sandbox(self._run_query, args=(query, dataframes, table_names))

    @abstractmethod
    def _build_empty_result_feedback(self) -> str:
        pass

    @abstractmethod
    def _get_language_name(self) -> str:
        pass

    def _empty_result_is_error(self, ordered_names: list) -> bool:
        """
        Controls whether an empty execution result is treated as a hard validation error.
        Default: always reject empty results.
        Subclasses can override to be more lenient — e.g. skip rejection for multi-table
        queries where sample data may have no overlapping keys.
        """
        return True

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def prefilter_dataframes(self, tables):
        df_by_name = dict(zip(self.table_names, self.dataframes))
        dataframes, ordered_names = [], []
        for table in tables:
            name = table["name"]
            dataframe = df_by_name.get(name)
            if dataframe is None:
                raise KeyError(
                    f"Table '{name}' not found. Available: {self.table_names}.\n"
                    "Check for a typo or stale table name."
                )
            cols = table.get("columns_involved") or []
            if cols:
                missing = [c for c in cols if c not in dataframe.columns]
                if missing:
                    raise KeyError(
                        f"columns_involved has unknown columns in '{name}': {missing}.\n"
                        "Check for typos or stale references."
                    )
                dataframes.append(dataframe[[c for c in cols if c in dataframe.columns]])
            else:
                dataframes.append(dataframe)
            ordered_names.append(name)
        return dataframes, ordered_names

    def replace_aliases(self, code: str, aliases: dict) -> str:
        if code is None:
            return ""
        for table_name, alias in aliases.items():
            code = code.replace(f'"{alias}"', table_name)
            code = code.replace(f"'{alias}'", table_name)
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
        if len(self.table_names) == 1:
            if self.table_names[0] in query_text:
                return True
            self.unused_tables = {self.table_names[0]}
            return False
        self.unused_tables = {t for t in self.table_names if t not in query_text}
        return len(self.unused_tables) == 0

    def _check_table_names_in_question(self, question: str) -> bool:
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
        lines = ["'tables' field must have one entry per table used."]
        if self.tables_field_missing:
            lines.append("Missing: " + ", ".join(sorted(self.tables_field_missing)))
        if self.tables_field_extra:
            lines.append("Unknown (remove): " + ", ".join(sorted(self.tables_field_extra)))
        lines.append("Each entry needs: 'name' (exact alias), 'reason', 'join_justification'.")
        return "\n".join(lines)

    def _build_unused_tables_feedback(self) -> str:
        missing = ", ".join(
            f"{t} ({self.lookup_dict.get(t, '?')})" for t in sorted(self.unused_tables)
        )
        return (
            f"Query must reference ALL tables. Missing: {missing}.\n"
            f"Required: {', '.join(self.table_names)}. Join all tables into a single result."
        )

    def _build_question_tables_feedback(self) -> str:
        return (
            "Do not use table names in the question. "
            "Write it as a business user with no knowledge of table names would."
        )

    def _build_technical_terms_feedback(self, technical_terms_found: List[str]) -> str:
        found_str = ", ".join(technical_terms_found)
        lines = [
            f"Question contains technical terms: {found_str}.",
            "Rephrase as a non-technical business question.",
        ]
        specific = [f"{t} → {TERM_SUGGESTIONS[t]}" for t in technical_terms_found if t in TERM_SUGGESTIONS]
        if specific:
            lines.append("Suggestions: " + " | ".join(specific))
        return "\n".join(lines)

    def _find_repeated_errors(self, error_texts: list) -> list:
        rules = []
        checked = set()
        patterns = [
            (
                "str.lower() on literal",
                lambda e: "was called on a string literal" in e,
                "NEVER call .str.lower() on a quoted string. Use DataFrame['col'].str.lower(). "
                "Normalise via .assign(_key=Table_X['col'].str.lower()) before merge, then on='_key'."
            ),
            (
                "stale Series in chained merge",
                lambda e: "stale series" in e.lower() or ("stale" in e.lower() and "left_on" in e.lower()),
                "NEVER use Table_X['col'] as left_on/right_on after a prior .merge(). "
                "Normalise ALL keys with .assign() before the chain."
            ),
            (
                "merge() with no keys",
                lambda e: "no join keys" in e.lower(),
                "Always specify on=, left_on=, or right_on= in every .merge() call."
            ),
            (
                "NameError: merge not defined",
                lambda e: "name 'merge' is not defined" in e,
                "Use Table_A.merge(Table_B, ...) — not bare merge(...)."
            ),
            (
                "duplicate suffixes",
                lambda e: "duplicate columns" in e.lower() and "suffixes" in e.lower(),
                "Suffixes must be unique across ALL merge steps: ('_T1','_T2'), ('_T12','_T3'), etc."
            ),
            (
                "missing tables",
                lambda e: "must reference all provided tables" in e.lower(),
                "Every query MUST join ALL provided tables — none can be omitted."
            ),
            (
                "disjoint sub-queries",
                lambda e: "disjoint query" in e.lower(),
                "All table groups must form ONE connected result via merge/join/concat — no parallel chains."
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
            f"Invalid {self._get_language_name()} queries — fix the ones listed below:\n",
        ]
        queries = {"queries": []}

        repeated = self._find_repeated_errors([err["error"] for err in self.validation_errors])
        if repeated:
            feedback_lines.insert(0,
                "RECURRING MISTAKES — apply to ALL queries before fixing:\n"
                + "\n".join(f"  * {r}" for r in repeated) + "\n"
            )

        for idx, err in enumerate(self.validation_errors, start=1):
            queries["queries"].append(err["query"])
            feedback_lines.append(f"Query {idx}:\n{err['error']}\n")

        return [
            {"role": "system", "content": json.dumps(queries, indent=2)},
            {"role": "user",   "content": "\n".join(feedback_lines)},
        ]