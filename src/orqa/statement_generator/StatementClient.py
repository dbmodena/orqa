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
from statement_generator.SQLValidator import SQLValidator
from statement_generator.PandasValidator import PandasValidator


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


    def add_difficulty(self,prompt, difficulty):
                actual_dif = difficulty if difficulty in [1, 3, 5] else (3 if difficulty == 2 else 5)
                return f"{prompt}\n{_load_prompt('statement_generator/prompt.md', f'Difficulty Level:{actual_dif}')}"



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
        messages = [{"role": "system", "content": "You are an expert Data Engineer"},{"role": "user", "content": initial_message}]
        completion_args = {
            "model": self.config["model"],
            "messages": messages,
            "temperature": self.temperature,
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
                print(content)
                cleaned_content = self._clean_json_response(content)
                try:
                    # First try to parse as JSON
                    json_data = json.loads(cleaned_content)
                    # Then validate with Pydantic
                    result = self.response_model.model_validate(json_data)
                    result = result.model_dump()
                    outcome, errors, accepted_queries = self.validate_queries(dataframes, result,table_names,typology)
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


    def validate_dataframe_queries(self, dataframes, result, table_names):
        """Validate pandas/polars queries."""
        validator = PandasValidator(dataframes, table_names)
        return validator.validate_queries(result)


    def validate_sql_queries(self, dataframes, result, table_names):
        """Validate SQL queries."""
        validator = SQLValidator(dataframes, table_names)
        return validator.validate_queries(result)


   











