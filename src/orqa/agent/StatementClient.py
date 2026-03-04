import importlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Type

import yaml
import litellm
from litellm import completion, Router
from pydantic import BaseModel, ValidationError


import pandas as pd
from .prompting import DatasetDescription, _load_prompt
from pathlib import Path
from .structured_outputs import QuerySet, Query
import duckdb


import pandas as pd
import polars as pl
from typing import List, Dict, Tuple, Union, Any

import sys
from io import StringIO

import pandas as pd
from pathlib import Path
import duckdb
from .LLMClientStructured import LLMClientStructured
from .validators.SQLValidator import SQLValidator
from .validators.PandasValidator import PandasValidator
import re

SQL_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING",
    "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN", "OUTER JOIN", "FULL JOIN", "CROSS JOIN",
    "UNION", "UNION ALL", "INTERSECT", "EXCEPT",
    "SUM", "AVG", "MIN", "MAX", "COUNT", "RANK", "ROW_NUMBER", "DENSE_RANK",
    "CAST", "COALESCE", "NULLIF", "CASE", "WHEN", "THEN", "ELSE",
    "DISTINCT", "LIMIT", "OFFSET", "WITH", "AS", "ON", "AND", "OR", "NOT", "IN", "EXISTS",
}

PANDAS_KEYWORDS = {
    "merge", "join", "concat", "groupby", "agg", "aggregate",
    "sum", "mean", "avg", "min", "max", "count", "nunique", "rank",
    "sort_values", "sort_index", "drop_duplicates", "fillna", "dropna",
    "apply", "map", "filter", "query", "where", "assign", "pivot", "melt",
    "stack", "unstack", "explode", "resample", "rolling", "expanding",
}



class LLMClientStatementGenerator(LLMClientStructured):
    """
    LiteLLM client with YAML configuration and structured output support.
    """

    def __init__(self, config_path: Path):
        """
        Initialize LLM client with configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
        """
        # 1. Load configuration
        super().__init__(config_path,"querying")
        self.fallback_response_model = self._load_pydantic_response_model("fallback_querying")


    def complete(
        self,
        prompt: str,
        dataframes,table_names, typology="SQL",involved_cols=None
    ) -> Any:
        """
        Make a completion request with optional structured output.

        :param prompt: The prompt to send to the model
        :param **kwargs: Additional arguments to pass to litellm.completion
        """
        usage_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        initial_message = prompt
        #if typology == "PANDAS":
        #    initial_message = self.add_suffix_constraint(initial_message, dataframes,table_names,involved_cols)
        initial_message = self.reform_prompt_constraint(initial_message)
        #print(involved_cols)
        #print(initial_message)
        messages = [{"role": "system", "content": "You are an expert Data Engineer"},{"role": "user", "content": initial_message}]
        completion_args = {
            "model": self.config["model"],
            "messages": messages,
            "temperature": self.temperature,
        }
        
        last_content = ""
        last_error = None

        good_queries: dict[str, dict] = {}
        all_errors = []
        for attempt in range(self.max_retries):
            try:
                #print(f"Attempt {attempt + 1}/{self.max_retries}...")

                response = self.router.completion(**completion_args)
                usage = response["usage"]
                usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
                usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
                usage_total["total_tokens"] += usage.get("total_tokens", 0)
                content = response["choices"][0]["message"]["content"]
                # Parse structured output
                last_content = content
                cleaned_content = self._clean_json_response(content)
                try:
                    json_data = self.fix_json_with_triple_quotes(cleaned_content)
                    json_data = self._normalize_response(json_data)
                    # Then validate with Pydantic
                    #print(json_data)
                    result = self.response_model.model_validate(json_data)
                    result = result.model_dump()
                    outcome, errors, accepted_queries,error_messages = self.validate_queries(dataframes, result,table_names,typology)
                    for key, accepted_query in accepted_queries.items():
                            accepted_query["keywords"] = self.count_keywords(accepted_query["code"], typology)
                            good_queries[accepted_query["id"]] = accepted_query
                    if not outcome:
                       messages = [{"role": "system", "content": "You are an expert Data Engineer."},{"role": "user", "content": prompt}]
                       #print(errors)
                       messages.extend(errors)
                       all_errors.extend(error_messages)
                       pydantic = self.reform_prompt_constraint("")
                       messages.append({"role": "user", "content":pydantic})
                       continue

                    #print(f"✓ Success on attempt {attempt + 1}\n")
                    return {"queries": list(good_queries.values())}, usage_total,all_errors
                except json.JSONDecodeError as e:
                    # JSON parsing failed
                    last_error = e
                    all_errors.append(e)
                    error_msg = self._format_json_error(cleaned_content, e)
                    if attempt < self.max_retries - 1:
                        messages = [{"role": "system", "content": "You are an expert Data Engineer, below are listed the instructions for generating and correcting queries."},{"role": "user", "content": initial_message}]
                        messages.append({"role": "system", "content": last_content})
                        pydantic = self.reform_prompt_constraint("")
                        messages.append({"role": "user", "content": f"You generated a bad formatted output, that gave the following error message:\n{error_msg}\n{pydantic}"})
                        time.sleep(self.retry_delay)
                        print(last_error)
                        continue
                except ValidationError as e:
                    last_error = e
                    all_errors.append(e)
                    error_msg = self._format_validation_error(e)
                    if attempt < self.max_retries - 1:
                        messages = [{"role": "system", "content": "You are an expert Data Engineer."},{"role": "user", "content": prompt}]
                        pydantic = self.reform_prompt_constraint("")
                        messages.append({"role": "system", "content":last_content})
                        messages.append({"role": "user","content": f"Genereated queries are not valid, the error message geneated:\n{error_msg}\n{pydantic}"})
                        #print("💬 Sending validation errors to LLM...\n")
                        print(last_error)
                        time.sleep(self.retry_delay)
                        continue

            except Exception as e:
                last_error = e
                #print(f"✗ Error on attempt {attempt + 1}: {e}")

                # Wait before retry
                if attempt < self.max_retries - 1:
                        messages = [{"role": "system", "content": "You are an expert Data Engineer"},{"role": "user", "content": prompt}]
                        # Add assistant's failed response
                        messages.append({"role": "system", "content": last_content})
                        # Add error feedback as user message
                        messages.append({"role": "user", "content": f"Genereated queries that are not valid, error message geneated\n{e}\nMake a unique JSON compliant to the Pydantic format:\n{json.dumps(self.response_model.model_json_schema(), indent=2)}"})
                        #print("💬 Sending validation errors to LLM...\n")
                        print(last_error)
                        time.sleep(self.retry_delay)
        return {"queries": list(good_queries.values())}, usage_total,all_errors


    def count_keywords(self,query: str, kind: str = "SQL") -> dict[str, int]:
        """
        Count keyword occurrences in a SQL or Pandas query string.
        kind: "sql" or "pandas"
        """
        keywords = SQL_KEYWORDS if kind == "SQL" else PANDAS_KEYWORDS
        query_upper = query.upper() if kind == "SQL" else query
        counts = {}

        for kw in sorted(keywords):
            kw_pattern = kw.upper() if kind == "SQL" else kw
            # Use word boundaries to avoid partial matches (e.g. IN inside INNER)
            pattern = rf'\b{re.escape(kw_pattern)}\b'
            matches = re.findall(pattern, query_upper, flags=re.IGNORECASE)
            if matches:
                counts[kw] = len(matches)

        return counts

    def validate_queries(self, dataframes, result, table_names,type):
        if type=="SQL":
            return self.validate_sql_queries(dataframes, result, table_names)
        else:
            return self.validate_dataframe_queries(dataframes, result, table_names)


    def validate_dataframe_queries(self, dataframes, result, table_names):
        """Validate pandas/polars queries."""
        validator = PandasValidator(dataframes, list(table_names.keys()),table_names)
        return validator.validate_queries(result)


    def validate_sql_queries(self, dataframes, result, table_names):
        """Validate SQL queries."""
        validator = SQLValidator(dataframes, list(table_names.keys()),table_names)
        return validator.validate_queries(result)

    def fix_json_with_triple_quotes(self,content: str) -> dict:
        """
        Fix JSON that contains Python triple-quoted strings.
        """
        # Step 1: Trova tutte le occorrenze di triple quotes
        pattern = r'"""[\s\S]*?"""'
        
        matches = list(re.finditer(pattern, content))
        
        # Step 2: Sostituisci ogni match con una stringa JSON valida
        offset = 0
        result = content
        
        for match in matches:
            original = match.group(0)
            # Rimuovi le triple virgolette
            inner_content = original[3:-3].strip()
            
            # Escape dei caratteri speciali
            escaped = (inner_content
                    .replace('\\', '\\\\')
                    .replace('"', '\\"')
                    .replace('\n', '\\n')  # Rimuovi newline o usa \\n se vuoi preservarli
                    .replace('\r', '')
                    .replace('\t', ' '))
            
            # Sostituisci nel risultato
            start = match.start() + offset
            end = match.end() + offset
            replacement = f'"{escaped}"'
            result = result[:start] + replacement + result[end:]
            
            # Aggiorna l'offset
            offset += len(replacement) - len(original)
        
        # Step 3: Parse del JSON
        return json.loads(result)


    def _normalize_response(self, json_data):
        """Normalize response to match QuerySet schema."""
        # Se è già una lista, wrappala in {"queries": [...]}
        if isinstance(json_data, list):
            return {"queries": json_data}
        
        # Se è un dict, verifica che abbia la chiave queries
        if isinstance(json_data, dict):
            if "queries" not in json_data:
                # Se ha altre chiavi che sembrano query, prova a recuperare
                if len(json_data) > 0:
                    # Potrebbe essere un singolo query object
                    return {"queries": [json_data]}
            return json_data
        
        # Fallback: wrappa in struttura corretta
        return {"queries": [json_data]}

    def add_suffix_constraint(self, prompt, dataframes, table_names, involved_cols):
        from itertools import combinations

        # Find common columns between each pair of tables
        common_cols_info = []
        for (name_a, df_a), (name_b, df_b) in combinations(zip(table_names, dataframes), 2):
            common = set(df_a.columns) & set(df_b.columns)
            common_cols_info.append(
                f"  {name_a} ∩ {name_b}: {sorted(common) if common else '(no common columns)'}"
            )

        constraint = (
            "### Suffix Constraint Information\n"
            "Columns suffixed after merge (ONLY these get a suffix):\n"
            + "\n".join(common_cols_info) + "\n\n"
            + (
                f"Join key columns (NEVER suffixed — kept as-is after merge):\n"
                f"  {involved_cols}\n\n"
                if involved_cols else ""
            )
            + "All other columns keep their original name — do not add any suffix to them.\n"
        )

        return f"{prompt}\n{constraint}"






