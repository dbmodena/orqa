import duckdb
import pandas as pd
import polars as pl
from typing import List, Dict, Tuple, Any
from .QueryValidator import QueryValidator
import re
import json

class SQLValidator(QueryValidator):
    """Validator for SQL queries."""
    
    def _preprocess_query(self, query: str) -> str:
        query_code = self.remove_redundant_aliases(query)
        query_code = self.fix_aggregates(query_code)
        return query_code.replace("`", '"')

    def fix_aggregates(self, query: str) -> str:
        pattern = r'\b(SUM|AVG|MIN|MAX|COUNT)\s*\(\s*(?!TRY_CAST|CAST|REPLACE)([^)]+?)\s*\)'
        replacement = r'\1(TRY_CAST(\2 AS DOUBLE))'
        return re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    

    def remove_redundant_aliases(self, query: str) -> str:
        """
        Removes redundant aliases like:
        FROM table AS table
        JOIN table AS table
        """
        pattern = re.compile(
            r'\b(FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s+\2\b',
            flags=re.IGNORECASE
        )

        return pattern.sub(r'\1 \2', query)


    def _execute_query(self, query: str) -> None:
        con = duckdb.connect(database=":memory:")
        #print(f"testing over {self.table_names}")
        try:
            for df, name in zip(self.dataframes, self.table_names):
                con.register(name, df)
            
            # Validate syntax + binding only
            #print(query.rstrip(";") + " LIMIT 1")
            sanitized = re.sub(r'\bLIMIT\s+\d+\s*;?\s*$', '', query.rstrip(";").rstrip(), flags=re.IGNORECASE).rstrip()
            con.execute(sanitized + " LIMIT 1")
        except duckdb.ParserException as e:
            raise ValueError(
                f"Malformed SQL structure: {e}\n"
                "Hint: Do not mix UNION and JOIN syntax. "
                "UNION combines result sets of separate SELECT statements and does not use ON clauses. "
                "JOIN connects tables within a single SELECT statement using ON clauses. "
                "Rewrite the query using only JOINs to combine all required tables."
            ) from e
        
        except duckdb.BinderException as e:
            error_msg = str(e)
            
            if "not found in FROM clause" in error_msg or "Referenced column" in error_msg:
                hint = (
                    "Only use columns that exist in the schema. "
                    "Do not infer or guess column names by pattern. "
                )
            elif "UNION" in error_msg and "different number" in error_msg:
                hint = (
                    "All SELECT statements in a UNION must have the same number of columns "
                    "and compatible types."
                )
            elif "Ambiguous reference" in error_msg:
                hint = (
                    "Ambiguous column reference. Qualify all column names with their table prefix, "
                    "e.g. Table_0.column_name instead of just column_name."
                )
            else:
                hint = "Check that all referenced columns, tables, and aliases are valid."

            raise ValueError(f"Binder error: {error_msg}\nHint: {hint}") from e
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