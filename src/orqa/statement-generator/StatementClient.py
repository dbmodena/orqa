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
import prompting
from prompting import DatasetDescription
from pathlib import Path
from structured_outputs import QuerySet, Query
import duckdb


import pandas as pd
import polars as pl
from typing import List, Dict, Tuple, Union, Any

import sys
from io import StringIO

import pandas as pd
import prompting
from prompting import DatasetDescription
from pathlib import Path
import duckdb
from LLMClientStructured import LLMClientStructured


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
        dataframes,table_names, typology="SQL",
        **kwargs,
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
        messages = [{"role": "system", "content": self.reform_prompt_constraint(prompt)}]
        completion_args = {
            "model": "primary",  # Router handles the actual model selection
            "messages": messages,
            "temperature": self.temperature,
             "response_format" : {"type": "json_object"},
            **kwargs,
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
                        messages.append({"role": "assistant", "content": content})
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
            query_code = q["code"].strip()  # <-- Cambiato da "query" a "code"
            
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






def parse_matches(matches,TABLE1,TABLE2):
    """Generate a textual description of database operations from match data between two tables."""
    operations = []
    all_join_conditions = []
    
    for match in matches:
        table1 = match["table1"]
        table2 = match["table2"]
        operation = match["operation"]
        
        # Build join conditions for this match
        join_conditions = []
        for couple in match["couples"]:
            condition = f"{TABLE1}.{couple[0]} {couple[1]} {TABLE2}.{couple[2]}"
            join_conditions.append(condition)
            all_join_conditions.append(condition)
        
        # Build operation description
        if operation == "U":
            operations.append(f"{TABLE1} UNIONS {TABLE2}")
        elif operation == "JC":
            operations.append(f"{TABLE1} JOIN-CORRELATION {TABLE2}")
        elif operation == "MJ":
            # Format: "table1 joins table2 on columns (condition1, condition2, ...)"
            conditions_str = ", ".join(f"({cond})" for cond in join_conditions)
            operations.append(f"{TABLE1} joins {TABLE2} on columns {conditions_str}")
    
    # Determine the primary operation type
    operation_types = [m["operation"] for m in matches]
    
    if "MJ" in operation_types:
        operation_name = "MULTI-JOIN"
    elif "JC" in operation_types:
        operation_name = "JOIN-CORRELATION"
    elif "U" in operation_types:
        operation_name = "UNION"
    else:
        operation_name = "operation"
    
    # Build final description
    description = f"The query must be done on the {operation_name} operation where:\n"
    description += "\n".join(f" {op}" for op in operations)
    
    return description









if __name__=="__main__":
    #### we fetch the dataframes
    folder = r"D:\uk_small\uk_small_copy\datasets\csv"
    path = Path("litellm.yaml")
    D1 = "2019-March-return__3f436d14-4e17-476c-a3e4-66d18e7f6c90"
    D2 = "2019-January-return__e07d11d5-afcb-4b8d-909f-d4952c4f183a"
    ### env 
    
    # Load CSVs
    df1 = pd.read_csv(Path(folder) / f"{D1}.csv")
    df1["Amount"] = df1["Amount"].str.replace(",", "").astype(float)
    df2 = pd.read_csv(Path(folder) / f"{D2}.csv")
    df2["Amount"] = df2["Amount"].str.replace(",", "").astype(float)

    ### define the tables aliases
    TABLE1=f"df_{D1.split("__")[0].replace("-","_")}"
    TABLE2=f"df_{D2.split("__")[0].replace("-","_")}"
    # Define matches with real columns
    matches = [
        {
            "table1": TABLE1,
            "table2": TABLE2,
            "operation": "MJ",
            "couples": [("Transaction number", "=", df2.columns[6]),("Supplier", "=",df2.columns[5])]
        },
    ]
    # Create table descriptions
    descriptor = DatasetDescription()
    table = descriptor.update(TABLE1, df1.shape[0], df1.shape[1], "", "", df1.head(3))
    table += f"\n{descriptor.update(TABLE2, df2.shape[0], df2.shape[1], '', '', df2.head(3))}"

    # Load prompt
    prompt = prompting._load_prompt("prompt.md", "PandasCodeGeneration", matches=parse_matches(matches,TABLE1,TABLE2), table=table)
    client = LLMClientStatementGenerator(Path("litellm.yaml"))
    ### testing queries
    dataframes = [df1,df2]
    table_names = [TABLE1,TABLE2]
    print(prompt)
    queries=client.complete(prompt,dataframes,table_names,typology="PANDAS")
    with open("validated_queries.json", "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2)   
    print(queries)

