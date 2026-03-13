from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import json 
import re

TECHNICAL_TERMS = [
    'record', 'records', 'dataset', 'datasets', 'dataframe', 'dataframes',
    'table', 'tables', 'row', 'rows', 'column', 'columns', 'schema',
    'index', 'indices', 'null', 'nan', 'dtype', 'merge', 'join'
]

class QueryValidator(ABC):
    """Base class for query validation."""
    
    def __init__(self, dataframes: List, table_names: List[str], lookup_dict: dict):
        self.dataframes = dataframes
        self.table_names = table_names
        self.validation_errors = []
        self.good_queries = {}
        self.lookup_dict=lookup_dict
        self.errors = []

    def _check_technical_terms_in_question(self, question: str) -> bool:
        """Returns True if the question contains technical/data terms a user shouldn't know."""
        question_lower = question.lower()
        found = [term for term in TECHNICAL_TERMS if re.search(rf'\b{term}\b', question_lower)]
        self.technical_terms_found = found
        return len(found) > 0
    
    def validate_queries(self, result: Dict) -> Tuple[bool, Dict, Dict]:
        """
        Validate all queries and return status, feedback, and valid queries.
        
        Returns:
            Tuple of (all_valid, feedback_message, good_queries)
        """
        all_valid = True
        
        for idx, q in enumerate(result["queries"]):
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
            try:
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
                if self._check_technical_terms_in_question(actual_query["question"]):
                    raise ValueError(self._build_technical_terms_feedback())
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

        normalised_tables = []
        for t in query.get("tables", []):
            entry = dict(t)
            raw_name = entry.get("name", "").strip()
            # If the LLM used the real dataset name, swap it for the alias.
            entry["name"] = alias_by_name.get(raw_name, raw_name)
            normalised_tables.append(entry)

        # Write normalised entries back so downstream code sees the corrected names.
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
    
    @abstractmethod
    def _execute_query(self, query: str) -> Any:
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

    def _build_technical_terms_feedback(self) -> str:
        feedback_lines = [
            "The natural language question must not contain technical data terms.",
            f"Found: {', '.join(self.technical_terms_found)}.",
            "Rephrase the question as a business user with no knowledge of the underlying data structure would ask it.",
            "Example: instead of 'join the records from both tables', say 'combine sales with customer info'."
        ]
        return "\n".join(feedback_lines)
    
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