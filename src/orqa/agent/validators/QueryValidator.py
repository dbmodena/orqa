from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import re

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
                tables_used = self._check_table_usage(q)
                if not tables_used:
                    raise ValueError(self._build_unused_tables_feedback())
                tables_used = self._check_table_names_in_question(q)
                if tables_used:
                     raise ValueError(self._build_question_tables_feedback())
                self.good_queries[idx] = q
                
            except Exception as e:
                all_valid = False
                self.validation_errors.append({
                    "query": q,
                    "error": f"{type(e).__name__}: {str(e)}"
                })
        
        if all_valid:
            return True, {}, self.good_queries
        
        return False, self._build_feedback(), self.good_queries
    
    def _check_table_names_in_question(self,query):
        tables_used = set()
        question = query["question"].lower()
        for table_name in self.table_names:
                if table_name.lower() in question:
                    tables_used.add(table_name)
        self.unused_tables = set(self.table_names) - tables_used
        return len(self.unused_tables) != len(self.table_names)


    def _check_table_usage(self, query) -> bool:
        tables_used = set()
        query_text = query["code"]
        for table_name in self.table_names:
            if table_name in query_text:
                tables_used.add(table_name)
        self.unused_tables = set(self.table_names) - tables_used
        return len(self.unused_tables) == 0

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
    

    
    
    def _build_unused_tables_feedback(self) -> Dict:
        """Build feedback message for unused tables."""
        feedback_lines = [
            "Not all tables are being used in the query.",
            f"Missing tables: {', '.join(sorted(self.unused_tables))}\n",
            f"Available tables: {', '.join(self.table_names)}\n"
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
        
        message_llm =  {"role": "user", "content": "\n".join(feedback_lines)}
        message_feedback = {"role": "user", "content": "\n".join(feedback_lines)}


        return [message_llm, message_feedback]
    
    @abstractmethod
    def _get_language_name(self) -> str:
        """Return the language name for error messages."""
        pass