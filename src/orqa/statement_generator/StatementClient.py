import importlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Type

import yaml
from litellm import completion, Router
from pydantic import BaseModel, ValidationError


import pandas as pd
from statement_generator.prompting import DatasetDescription, _load_prompt
from pathlib import Path
from statement_generator.structured_outputs import QuerySet, Query
import duckdb


import pandas as pd
import polars as pl
from typing import List, Dict, Tuple, Union, Any

import sys
from io import StringIO

import pandas as pd
from pathlib import Path
import duckdb
from statement_generator.LLMClientStructured import LLMClientStructured


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

    def complete(
        self,
        prompt: str,
        dataframes,table_names, typology="SQL"
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
        initial_message = self.reform_prompt_constraint(prompt)
        messages = [{"role": "system", "content": initial_message}]
        completion_args = {
            "model": self.config["model"],
            "messages": messages,
            "temperature": self.temperature,
            "num_retries":3,
        }
        
        last_content = None
        last_error = None

        good_queries: dict[str, self.response_model] = {}
        for attempt in range(self.max_retries):
            try:
                print(f"Attempt {attempt + 1}/{self.max_retries}...")

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
                    # First try to parse as JSON
                    json_data = json.loads(cleaned_content)
                    # Then validate with Pydantic
                    result = self.response_model.model_validate(json_data)
                    result = result.model_dump()
                    outcome, errors, accepted_queries = self.validate_queries(dataframes, result,table_names,type)
                    good_queries.update(accepted_queries)
                    if not outcome:
                       messages.append(errors)
                       print(errors)
                       continue

                    print(f"✓ Success on attempt {attempt + 1}\n")
                    return {"queries": list(good_queries.values())}, usage_total
                except json.JSONDecodeError as e:
                    # JSON parsing failed
                    last_error = e
                    error_msg = self._format_json_error(cleaned_content, e)
                    print(f"⚠️ JSON parsing error on attempt {attempt + 1}")

                    if attempt < self.max_retries - 1:
                        # Add assistant's failed response
                        messages = [{"role": "system", "content": initial_message}]
                        messages.append({"role": "assistant", "content": content})
                        # Add error feedback as user message
                        messages.append({"role": "user", "content": error_msg})
                        time.sleep(self.retry_delay)
                        continue
                except ValidationError as e:
                    # Pydantic validation failed
                    last_error = e
                    error_msg = self._format_validation_error(e)
                    print(f"⚠️ Validation error on attempt {attempt + 1}")

                    if attempt < self.max_retries - 1:
                        # Add assistant's failed response
                        messages = [{"role": "system", "content": initial_message}]
                        messages.append({"role": "system", "content": content})
                        # Add error feedback as user message
                        messages.append({"role": "user", "content": error_msg})
                        print("💬 Sending validation errors to LLM...\n")
                        time.sleep(self.retry_delay)
                        continue

            except Exception as e:
                last_error = e
                print(f"✗ Error on attempt {attempt + 1}: {e}")

                # Wait before retry
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...\n")
                    time.sleep(self.retry_delay)

        # All retries exhausted
        print(f"\n❌ Failed after {self.max_retries} attempts")
        print(f"Last error: {last_error}")
        if last_content:
            print(f"\nLast response preview:\n{last_content[:300]}...\n")

        return {"queries": list(good_queries.values())}, usage_total

    def validate_queries(self, dataframes, result, table_names,type):
        if type=="SQL":
            return self.validate_sql_queries(dataframes, result, table_names)
        else:
            return self.validate_dataframe_queries(dataframes, result, table_names)

    def clean_pandas(self,query: str) -> str:
        """Rimuove import statements dalle query"""
        lines = query.split(';')
        # Filtra righe che contengono import
        cleaned = [line.strip() for line in lines if 'import' not in line.lower()]
        return '; '.join(cleaned).strip() 

    def validate_dataframe_queries(self, dataframes, result, table_names):
        """
        Validate pandas/polars queries by executing them in a sandboxed environment.
        """
        import sys
        from io import StringIO
        
        validation_errors = []
        all_valid = True
        good_queries = {}
        
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
        for df, name in zip(dataframes, table_names):
            namespace[name] = df
        
        for idx, q in enumerate(result["queries"]):
            query_code = self.clean_pandas(q["code"].strip()) 
            
            try:
                local_namespace = namespace.copy()
                
                # Capture stdout/stderr
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = StringIO()
                sys.stderr = StringIO()
                
                try:
                    compiled_code = compile(query_code, '<string>', 'exec')
                    exec(compiled_code, local_namespace)
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                
                good_queries[idx] = q
                
            except Exception as e:
                all_valid = False
                print(e)
                error_type = type(e).__name__
                validation_errors.append({
                    "id": idx,
                    "query": query_code,
                    "error": f"{error_type}: {str(e)}"
                })
        
        if all_valid:
            return True, {}, good_queries
        
        # Build feedback - USA I NOMI ESATTI
        feedback_lines = [
            "Some of the generated Python queries are invalid.",
            "Fix ONLY the queries listed below. Do not modify valid queries.\n"
        ]
        
        for err in validation_errors:
            feedback_lines.append(
                f"Query {err['id'] + 1}:\n"
                f"{err['query']}\n\n"
                f"Error:\n{err['error']}\n"
            )
        
        # IMPORTANTE: Mostra i nomi ESATTI delle variabili disponibili
        feedback_lines.append(
            "General rules:\n"
            "- Column names must exactly match the DataFrame schema.\n"
            "- Use correct pandas/polars syntax (e.g., df['column'] for pandas, df['column'] for polars).\n"
            "- For numeric operations on string columns, convert with .astype(float) (pandas) or .cast(pl.Float64) (polars).\n"
            f"- IMPORTANT: Available DataFrames are named EXACTLY: {', '.join(table_names)}\n"  # <-- Enfatizza questo
            f"- DO NOT add 'df_' prefix or any other modification to these names.\n"  # <-- Aggiungi questa riga
            "- Do not change queries that are not listed above."
        )
        
        return False, {
            "role": "user",
            "content": "\n".join(feedback_lines)
        }, good_queries

    def validate_sql_queries(self,dataframes, result, table_names):
        validation_errors = []
        all_valid = True
        good_queries = {}
        for idx, q in enumerate(result["queries"]):
            sql = q["code"].replace("`", '"')
            #id = q["id"]
            con = duckdb.connect(database=":memory:")
            try:
                for df, name in zip(dataframes, table_names):
                    con.register(name, df)

                # Validate syntax + binding only
                con.execute(sql.rstrip(";") + " LIMIT 1")
                good_queries[idx] = q

            except Exception as e:
                all_valid = False
                validation_errors.append({
                    "id": idx,#str(id),
                    "sql": sql,
                    "error": str(e)
                })

            finally:
                con.close()

        if all_valid:
            return True, {},good_queries

        # Build ONE aggregated feedback message
        feedback_lines = [
            "Some of the generated SQL queries are invalid.",
            "Fix ONLY the queries listed below. Do not modify valid queries.\n"
        ]

        for err in validation_errors:
            feedback_lines.append(
                f"Query {err['id'] + 1}:\n"
                f"{err['sql']}\n\n"
                f"Error:\n{err['error']}\n"
            )

        feedback_lines.append(
            "General rules:\n"
            "- Column names must exactly match the schema.\n"
            "- Amount may be stored as text; CAST to DOUBLE when using SUM or AVG.\n"
            "- Do not change queries that are not listed above."
        )

        return False, {
            "role": "user",
            "content": "\n".join(feedback_lines)
        },good_queries











