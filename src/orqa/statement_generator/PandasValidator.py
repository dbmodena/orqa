import pandas as pd
import polars as pl
from io import StringIO
import sys
from typing import List, Dict, Tuple, Any
from statement_generator.QueryValidator import QueryValidator


class PandasValidator(QueryValidator):
    """Validator for pandas/polars queries."""
    
    def _preprocess_query(self, query: str) -> str:
        return self.clean_pandas(query)
    
    def _execute_query(self, query: str) -> None:
        # Safe builtins
        safe_builtins = {
            'abs': abs, 'min': min, 'max': max, 'sum': sum,
            'len': len, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter,
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
            'True': True, 'False': False, 'None': None,
        }
        
        # Create namespace
        namespace = {
            'pd': pd,
            'pl': pl,
            '__builtins__': safe_builtins
        }
        
        # Map table names to dataframes
        for df, name in zip(self.dataframes, self.table_names):
            namespace[name] = df
        
        local_namespace = namespace.copy()
        
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        try:
            compiled_code = compile(query, '<string>', 'exec')
            exec(compiled_code, local_namespace)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def _get_language_specific_rules(self) -> List[str]:
        return [
            "- Column names must exactly match the DataFrame schema.",
            "- Use correct pandas/polars syntax (e.g., df['column'] for pandas, df['column'] for polars).",
            "- For numeric operations on string columns, convert with .astype(float) (pandas) or .cast(pl.Float64) (polars).",
            f"- IMPORTANT: Available DataFrames are named EXACTLY: {', '.join(self.table_names)}",
            "- DO NOT add 'df_' prefix or any other modification to these names.",
        ]
    
    def _get_language_name(self) -> str:
        return "Python"
    
    def clean_pandas(self,query: str) -> str:
        """Clean pandas query code."""
        lines = query.split(';')
        cleaned = [line.strip() for line in lines if 'import' not in line.lower()]
        return '; '.join(cleaned).strip()
         