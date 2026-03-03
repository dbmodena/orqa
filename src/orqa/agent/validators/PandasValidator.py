import pandas as pd
import polars as pl
from io import StringIO
import sys
from typing import List, Dict, Tuple, Any
from .QueryValidator import QueryValidator
import re

UNAUTHORIZED_COMMANDS = [
    'read_csv', 'read_excel', 'read_json', 'read_parquet',
    'read_sql', 'read_table', 'read_html', 'read_pickle',
    'to_csv', 'to_excel', 'to_json', 'to_parquet', 'to_pickle',
    'open(', 'os.', 'sys.', 'exec(', 'eval('
]

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
        except KeyError as e:
            raise KeyError(
                f"Column {str(e)} not found at runtime.\n\n"
                f"Possible causes:\n"
                f"  1. Column does not exist in any of the original tables — check spelling and case\n"
                f"  2. Column was incorrectly suffixed — a suffix is ONLY valid if the column name exists in BOTH tables\n"
                f"  3. Column was incorrectly left unsuffixed — if a column name exists in BOTH tables it MUST be suffixed\n\n"
                f"Strictly follow the suffix constraints provided — do not infer, assume, or apply suffixes beyond what is defined there.\n"
            )
        except SyntaxError as e:
            raise SyntaxError(
                f"Invalid Python syntax in query.\n"
                f"Error: {str(e)}\n"
                f"Tip: Ensure multiple statements are separated by newlines (\\n)"
            )
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
    
    def clean_pandas(self, query_code: str) -> str:
        """Clean pandas query code."""
        query = query_code.replace("`", "'").replace('"', "'")
        lines = query.split(';')
        cleaned = [line.strip() for line in lines if 'import' not in line.lower()]
        # Remove comments from each line
        cleaned = [re.sub(r'#.*$', '', line).strip() for line in cleaned]
        # Remove unauthorized commands
        cleaned = [line for line in cleaned if not any(cmd in line for cmd in UNAUTHORIZED_COMMANDS)]
        # Filter out empty lines
        cleaned = [line for line in cleaned if line]
        return '; '.join(cleaned).strip()
         