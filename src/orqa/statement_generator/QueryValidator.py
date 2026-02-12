from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any


class QueryValidator(ABC):
    """Base class for query validation."""
    
    def __init__(self, dataframes: List, table_names: List[str]):
        self.dataframes = dataframes
        self.table_names = table_names
        self.validation_errors = []
        self.good_queries = {}
    
    def validate_queries(self, result: Dict) -> Tuple[bool, Dict, Dict]:
        """
        Validate all queries and return status, feedback, and valid queries.
        
        Returns:
            Tuple of (all_valid, feedback_message, good_queries)
        """
        all_valid = True
        
        for idx, q in enumerate(result["queries"]):
            query_code = self._preprocess_query(q["code"].strip())
            
            try:
                self._execute_query(query_code)
                self.good_queries[idx] = q
                
            except Exception as e:
                all_valid = False
                print(e)
                self.validation_errors.append({
                    "id": idx,
                    "query": q,
                    "error": f"{type(e).__name__}: {str(e)}"
                })
        
        if all_valid:
            # Check if all tables are used
            tables_used = self._check_table_usage(result["queries"])
            if not tables_used:
                return False, self._build_unused_tables_feedback(), self.good_queries
            tables_used = self._check_table_names_in_question(result["queries"])
            if tables_used:
                return False, self._build_question_tables_feedback(), self.good_queries
            return True, {}, self.good_queries
        
        return False, self._build_feedback(), self.good_queries
    
    def _check_table_names_in_question(self,queries):
        tables_used = set()
        
        for q in queries:
            question = q["question"].lower()
            for table_name in self.table_names:
                if table_name.lower() in question:
                    tables_used.add(table_name)
        self.unused_tables = set(self.table_names) - tables_used
        return len(self.unused_tables) != len(self.table_names)


    @abstractmethod
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query before execution."""
        pass
    
    @abstractmethod
    def _execute_query(self, query: str) -> None:
        """Execute query in appropriate environment. Raises exception on error."""
        pass
    
    @abstractmethod
    def _get_language_specific_rules(self) -> List[str]:
        """Return language-specific validation rules."""
        pass
    
    def _check_table_usage(self, queries: List[Dict]) -> bool:
        """
        Check if all tables are referenced in at least one query.
        
        Returns:
            True if all tables are used, False otherwise
        """
        tables_used = set()
        
        for q in queries:
            query_text = q["code"].lower()
            for table_name in self.table_names:
                if table_name.lower() in query_text:
                    tables_used.add(table_name)
        
        self.unused_tables = set(self.table_names) - tables_used
        return len(self.unused_tables) == 0
    
    def _build_unused_tables_feedback(self) -> Dict:
        """Build feedback message for unused tables."""
        feedback_lines = [
            "Not all tables are being used in the queries.",
            f"Missing tables: {', '.join(sorted(self.unused_tables))}\n",
            f"Available tables: {', '.join(self.table_names)}\n",
            "Please ensure all tables are referenced in at least one query."
        ]
        
        return {
            "role": "user",
            "content": "\n".join(feedback_lines)
        }
    def _build_question_tables_feedback(self) -> Dict:
        """Build feedback message for used tables in the natural language question."""
        feedback_lines = [
            "Do not use the table names in the natual language question.",
            "Please ensure that the associated question does not the refer to specific tables."
            "Make use of the metadata of the tables if needed."
            "Question should look like it's made by a user that has no knowledge of the tables."
        ]
        
        return {
            "role": "user",
            "content": "\n".join(feedback_lines)
        }
    def _build_feedback(self) -> Dict:
        """Build feedback message for validation errors."""
        feedback_lines = [
            f"Some of the generated {self._get_language_name()} queries are invalid.",
            "Fix ONLY the queries listed below. Do not modify valid queries.\n"
        ]
        
        for err in self.validation_errors:
            feedback_lines.append(
                f"Query {err['id'] + 1}:\n"
                f"{err['query']}\n\n"
                f"Error:\n{err['error']}\n"
            )
        
        feedback_lines.append("General rules:")
        feedback_lines.extend(self._get_language_specific_rules())
        feedback_lines.append("- Do not change queries that are not listed above.")
        
        return {
            "role": "user",
            "content": "\n".join(feedback_lines)
        }
    
    @abstractmethod
    def _get_language_name(self) -> str:
        """Return the language name for error messages."""
        pass