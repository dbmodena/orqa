from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import json 
import re
import pandas as pd
import threading

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

# Term-specific rephrasing suggestions shown in feedback when a banned term is found.
TERM_SUGGESTIONS = {
    'join':        'Instead of "join", say "combine", "match", or "link" — e.g. "link customer info with their orders".',
    'merge':       'Instead of "merge", say "combine" or "bring together" — e.g. "combine sales data with location info".',
    'correlate':   'Instead of "correlate", say "tend to increase/decrease together" or "relationship between" — e.g. "Do restaurants with higher X tend to have higher Y?".',
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

class QueryValidator(ABC):
    """Base class for query validation."""
    
    def __init__(self, dataframes: List, table_names: List[str], lookup_dict: dict,mem_limit=512):
        self.dataframes = dataframes
        self.table_names = table_names
        self.validation_errors = []
        self.good_queries = {}
        self.lookup_dict=lookup_dict
        self.errors = []
        self.mem_limit = mem_limit *1024 *1024


    def _check_input_memory(self, limit_mb: int = 500):
        """Check total memory usage of input DataFrames before execution."""
        total_bytes = 0
        for df in self.dataframes:
            if isinstance(df, pd.DataFrame):
                total_bytes += df.memory_usage(deep=True).sum()
            #elif hasattr(df, 'estimated_size'):  # polars
            #    total_bytes += df.estimated_size()

        total_mb = total_bytes / (1024 ** 2)
        if total_mb > self.mem_limit:
            raise MemoryError(
                f"Input data is {total_mb:.1f}MB, which exceeds the {limit_mb}MB limit.\n"
                "Consider pre-filtering your data before running this query."
            )

    def _check_technical_terms_in_question(self, question: str) -> List[str]:
        """Returns list of found technical terms, empty if none found."""
        question_lower = question.lower()
        return [term for term in TECHNICAL_TERMS if re.search(rf'\b{term}\b', question_lower)]
    
    def validate_queries(self, result: Dict) -> Tuple[bool, Dict, Dict]:
        """
        Validate all queries and return status, feedback, and valid queries.
        
        Returns:
            Tuple of (all_valid, feedback_message, good_queries)
        """
        all_valid = True
        
        for idx, q in enumerate(result["queries"]):
            try:
                actual_query = q
                # FIX: q["code"] may be None if the LLM omits the field or returns null.
                # Treat it as an empty string so the rest of the pipeline gets a str.
                raw_code = q.get("code") or ""
                if not raw_code.strip():
                    all_valid = False
                    self.validation_errors.append({
                        "query": actual_query,
                        "error": "ValueError: query 'code' field is missing or empty"
                    })
                    self.errors.append("Error ValueError: query 'code' field is missing or empty")
                    continue
                query_code = self.replace_aliases(raw_code, self.lookup_dict)
                query_code = self._preprocess_query(query_code.strip())
                actual_query["code"]=query_code
                result_data = self._execute_query(query_code)
                if self._is_empty_result(result_data):
                    raise ValueError(self._build_empty_result_feedback())
                tables_used = self._check_table_usage(query_code)
                if not tables_used:
                    raise ValueError(self._build_unused_tables_feedback())

                if not self._check_tables_field_coverage(actual_query):
                    raise ValueError(self._build_tables_field_coverage_feedback())
                
                tables_used = self._check_table_names_in_question(actual_query["question"])
                if tables_used:
                     raise ValueError(self._build_question_tables_feedback())
                technical_terms_found = self._check_technical_terms_in_question(actual_query["question"])
                if technical_terms_found:
                    raise ValueError(self._build_technical_terms_feedback(technical_terms_found))
                self.good_queries[idx] = actual_query
                

            except Exception as e:
                all_valid = False
                #print(f"Query code: {actual_query["code"]}")
                #print(f"Error {type(e).__name__}: {str(e)}")
                self.validation_errors.append({
                    "query": actual_query,
                    "error": f"{type(e).__name__}: {str(e)}"
                })
                self.errors.append(f"Error {type(e).__name__}: {str(e)}")
        
        if all_valid:
            return True, {}, self.good_queries,self.errors
        
        return False, self._build_feedback(), self.good_queries,self.errors
    

    def replace_aliases(self, code: str, aliases: dict) -> str:
        # FIX: guard against None input (LLM may emit null for the code field)
        if code is None:
            return ""
        for table_name, alias in aliases.items():
            # Replace both quoted and unquoted versions of the alias
            code = code.replace(f'"{alias}"', table_name)  # handles "4dx7-axux"
            code = code.replace(alias, table_name)          # handles 4dx7-axux (fallback)
        return code

    def _check_table_names_in_question(self,question):
        tables_used = set()
        question = question.lower()
        for table_name in self.table_names:
                if table_name.lower() in question:
                    tables_used.add(table_name)
        self.unused_tables = set(self.table_names) - tables_used
        return len(self.unused_tables) != len(self.table_names)


    def _check_table_usage(self, query_text) -> bool:
        tables_used = set()
        for table_name in self.table_names:
            if table_name in query_text:
                tables_used.add(table_name)
        self.unused_tables = set(self.table_names) - tables_used
        return len(self.unused_tables) == 0

    def _check_tables_field_coverage(self, query: dict) -> bool:
        """Returns True if the 'tables' field contains one entry per expected table.
        If an entry uses the real dataset name instead of the alias, it is normalised
        to the alias before the comparison so the check is not tripped by naming style."""
        # Build reverse map: dataset_name -> alias
        alias_by_name = {v: k for k, v in self.lookup_dict.items()}

        # Accept both 'tables' and 'Tables' from the LLM response.
        raw_tables = query.get("tables") or query.get("Tables") or []

        normalised_tables = []
        for t in raw_tables:
            entry = dict(t)
            raw_name = entry.get("name", "").strip()
            # If the LLM used the real dataset name, swap it for the alias.
            entry["name"] = alias_by_name.get(raw_name, raw_name)
            normalised_tables.append(entry)

        # Write normalised entries back under the canonical lowercase key.
        query.pop("Tables", None)
        query["tables"] = normalised_tables

        declared = {t["name"] for t in normalised_tables}
        self.tables_field_missing = set(self.table_names) - declared
        self.tables_field_extra = declared - set(self.table_names)
        return len(self.tables_field_missing) == 0

    def _build_tables_field_coverage_feedback(self) -> str:
        lines = [
            "The 'tables' field must contain one entry for every table used in the query.",
        ]
        if self.tables_field_missing:
            missing_list = "\n".join(f"  - {t}" for t in sorted(self.tables_field_missing))
            lines.append(f"Missing entries for the following tables:\n{missing_list}")
        if self.tables_field_extra:
            extra_list = "\n".join(f"  - {t}" for t in sorted(self.tables_field_extra))
            lines.append(
                f"The following entries do not match any known table alias and must be removed:\n{extra_list}"
            )
        lines.append(
            "Each entry must set 'name' to the exact table alias, "
            "'reason' to why the table is needed, and "
            "'join_justification' to why it is combined with the other tables in this way."
        )
        return "\n".join(lines)

    @abstractmethod
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query before execution."""
        pass
    
    def _execute_query(self, query: str) -> Any:
        return self.run_with_timeout(self._run_query, args=(query,), timeout=30)
    
    @abstractmethod
    def _run_query(self, query: str) -> Any:
        """Execute query in appropriate environment. Raises exception on error.
        Must return the query result (DataFrame or similar) for empty-result detection."""
        pass

    def _is_empty_result(self, result: Any) -> bool:
        """Return True if the query produced no rows. Works for pandas/polars DataFrames."""
        if result is None:
            return True
        if hasattr(result, 'empty'):       # pandas DataFrame / Series
            return result.empty
        if hasattr(result, 'is_empty'):    # polars DataFrame
            return result.is_empty()
        if hasattr(result, '__len__'):
            return len(result) == 0
        return False

    @abstractmethod
    def _build_empty_result_feedback(self) -> str:
        """Return a language-specific hint when the query returns no rows."""
        pass

    

    def _build_technical_terms_feedback(self, technical_terms_found: List[str]) -> str:
        lines = [
            "The natural language question must not contain technical data terms.",
            f"Found: {', '.join(technical_terms_found)}.",
            "Rephrase the question as a business user with no knowledge of the underlying data structure would ask it.",
            "",
        ]
        specific = [
            TERM_SUGGESTIONS[term]
            for term in technical_terms_found
            if term in TERM_SUGGESTIONS
        ]
        if specific:
            lines.append("Suggestions for the flagged terms:")
            lines.extend(f"  - {s}" for s in specific)
        else:
            lines.append("Example: instead of 'join the records from both tables', say 'combine sales with customer info'.")
        return "\n".join(lines)
    
    def _build_unused_tables_feedback(self) -> Dict:
        """Build feedback message for unused tables."""
        missing_details = "\n".join(
            f"  {t}: {self.lookup_dict.get(t, 'unknown')}"
            for t in sorted(self.unused_tables)
        )
        feedback_lines = [
            "The query must reference ALL provided tables. Every table exists for a reason and contains information required to answer the question.",
            f"The following tables are missing from the query:\n{missing_details}",
            f"All tables that must appear in the query: {', '.join(self.table_names)}.",
            "Rewrite the query ensuring each table is joined and contributes to the result."
        ]

        
        return"\n".join(feedback_lines)

    def _build_question_tables_feedback(self):
        """Build feedback message for used tables in the natural language question."""
        feedback_lines = [
            "Do not use the table names in the natual language question.",
            "Make use of the metadata of the tables if needed."
            "Question should look like it's made by a user that has no knowledge of the tables."
        ]
        
        return "\n".join(feedback_lines)
        
    def _build_feedback(self):
        """Build feedback message for validation errors."""
        feedback_lines = [
            f"The following of the generated {self._get_language_name()} queries are invalid.",
            "Fix the following queries listed below:\n"
        ]
        queries = {"queries":[]}
        for idx, err in enumerate(self.validation_errors, start=1):
            queries["queries"].append(err['query'])
            feedback_lines.append(
                f"Query number: {idx}:\n"
                f"Error:\n{err['error']}\n"
            )
        
        message_llm = {
            "role": "system",
            "content": json.dumps(queries, indent=2)
        }
        message_feedback = {"role": "user", "content": "\n".join(feedback_lines)}


        return [message_llm, message_feedback]
    
    @abstractmethod
    def _get_language_name(self) -> str:
        """Return the language name for error messages."""
        pass


    def run_with_timeout(self,func, args=(), timeout=30):
        """
        Runs func(*args) in a thread. Raises TimeoutError if it exceeds timeout seconds.
        Windows-compatible (no resource/subprocess needed).
        """
        result = [None]
        exception = [None]

        def target():
            try:
                result[0] = func(*args)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            # Thread is still running — query took too long
            raise TimeoutError(
                f"Query exceeded the {timeout}s time limit and was aborted.\n"
                "This usually means a Cartesian product or missing join condition.\n"
                "Simplify the query or add more specific join/filter conditions."
            )

        if exception[0] is not None:
            raise exception[0]

        return result[0]