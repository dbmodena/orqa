import duckdb
import pandas as pd
import polars as pl
from typing import List, Dict, Tuple, Any
from .QueryValidator import QueryValidator


class SQLValidator(QueryValidator):
    """Validator for SQL queries."""
    
    def _preprocess_query(self, query: str) -> str:
        return query.replace("`", '"')
    
    def _execute_query(self, query: str) -> None:
        con = duckdb.connect(database=":memory:")
        try:
            for df, name in zip(self.dataframes, self.table_names):
                con.register(name, df)
            
            # Validate syntax + binding only
            #print(query.rstrip(";") + " LIMIT 1")
            con.execute(query.rstrip(";") + " LIMIT 1")
        finally:
            con.close()
    
    def _get_language_specific_rules(self) -> List[str]:
        return [
            "- Column names must exactly match the schema.",
            "- Amount may be stored as text; CAST to DOUBLE when using SUM or AVG.",
            f"- Available tables: {', '.join(self.table_names)}",
        ]
    
    def _get_language_name(self) -> str:
        return "SQL"