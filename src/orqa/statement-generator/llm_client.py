import importlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Type

import yaml
from litellm import completion, Router
from pydantic import BaseModel, ValidationError


class LLMClient:
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
        self.config = self._load_config(config_path)
        
        # 2. Set basic attributes
        self.temperature = self.config.get("temperature", 0.2)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        self.enable_json_mode = self.config.get("enable_json_mode", True)
        self.provider_params = self.config.get("provider_params", {}) or {}
        
        # 3. Load response model
        self.response_model = self._load_pydantic_response_model()
        self.raw = self.response_model is None
        
        # 4. Setup router LAST (depends on provider_params being set)
        self.router = self._setup_router()


    def _get_provider_from_model(self, model: str) -> str:
        """Extract provider name from model string."""
        if "/" in model:
            return model.split("/")[0].replace("_chat", "")
        return "unknown"

    def _get_provider_specific_params(self, model: str) -> dict[str, Any]:
        """
        Get provider-specific parameters for a given model.
        
        Args:
            model: Model string (e.g., "ollama_chat/llama3.3:latest")
            
        Returns:
            Dictionary of provider-specific parameters
        """
        provider = self._get_provider_from_model(model)
        
        # Defensive programming: handle None and missing keys
        if not self.provider_params:
            return {}
        
        provider_config = self.provider_params.get(provider, {})
        
        # Handle case where provider_config might be None
        if provider_config is None:
            return {}
        return provider_config.copy()


    def _setup_router(self) -> Router:
        """Setup LiteLLM Router with primary and fallback models."""
        model_list = []
        
        # Primary model
        primary_model = self.config["model"]
        primary_params = self._get_provider_specific_params(primary_model)
        model_list.append({
            "model_name": "primary",
            "litellm_params": {
                "model": primary_model,
                **primary_params
            }
        })
        
        # Fallback models
        fallback_names = []
        for idx, fallback_model in enumerate(self.config.get("fallback_models", [])):
            fallback_name = f"fallback_{idx}"
            fallback_names.append(fallback_name)
            fallback_params = self._get_provider_specific_params(fallback_model)
            model_list.append({
                "model_name": fallback_name,
                "litellm_params": {
                    "model": fallback_model,
                    **fallback_params
                }
            })
        
        # Setup fallback chain: primary -> fallback_0 -> fallback_1 -> ...
        fallbacks = [{"primary": fallback_names}] if fallback_names else []
        
        return Router(
            model_list=model_list,
            fallbacks=fallbacks,
            num_retries=1,  # Router handles retries per model
            timeout=60,
            set_verbose=True  # Shows which model is being used
        )


    def _load_config(self, config_path) -> dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _clean_json_response(self, content: str) -> str:
        """Clean up JSON response by extracting valid JSON"""
        content = content.strip()

        # Remove markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        # Try to find JSON object or array
        # Look for content between outermost { } or [ ]
        brace_start = content.find("{")
        bracket_start = content.find("[")

        # Determine which comes first
        if brace_start == -1:
            start = bracket_start
        elif bracket_start == -1:
            start = brace_start
        else:
            start = min(brace_start, bracket_start)

        if start == -1:
            return content

        # Find matching closing character
        if content[start] == "{":
            # Find the last closing brace
            depth = 0
            for i in range(start, len(content)):
                if content[i] == "{":
                    depth += 1
                elif content[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return content[start : i + 1].strip()
        else:
            # Find the last closing bracket
            depth = 0
            for i in range(start, len(content)):
                if content[i] == "[":
                    depth += 1
                elif content[i] == "]":
                    depth -= 1
                    if depth == 0:
                        return content[start : i + 1].strip()

        return content.strip()

    def _load_pydantic_response_model(self) -> Optional[Type[BaseModel]]:
        """
        Dynamically load Pydantic model from config.

        Returns:
            Pydantic model class or None if not specified
        """
        response_model_config = self.config.get("response_model")

        if not response_model_config:
            return None

        module_name = response_model_config.get("module")
        class_name = response_model_config.get("class")

        if not module_name or not class_name:
            raise ValueError("response_model must specify both 'module' and 'class'")

        try:
            # Import the module
            module = importlib.import_module(module_name)

            # Get the class from the module
            model_class = getattr(module, class_name)

            # Verify it's a Pydantic model
            if not issubclass(model_class, BaseModel):
                raise TypeError(f"{class_name} is not a Pydantic BaseModel")

            return model_class

        except ImportError as e:
            raise ImportError(f"Could not import module '{module_name}': {e}")
        except AttributeError as e:
            raise AttributeError(
                f"Could not find class '{class_name}' in module '{module_name}': {e}"
            )

    def _format_json_error(self, content: str, error: Exception) -> str:
        """
        Format JSON parsing error with context.

        :param content: The content that failed to parse
        :param error: The JSON parsing exception
        :return: Human-readable error message for the LLM
        """
        formatted_error = (
            "❌ JSON PARSING ERROR - Your response is not valid JSON.\n\n"
            f"Error: {str(error)}\n\n"
            # f"Your response (first 200 chars):\n{snippet}\n\n"
            "Required schema:\n"
            f"{json.dumps(self.response_model.model_json_schema(), indent=2)}\n\n"
            "⚠️ Common issues:\n"
            "  - Missing quotes around strings\n"
            "  - Trailing commas\n"
            "  - Unescaped special characters\n"
            "  - Text before or after the JSON object\n\n"
            "Please generate ONLY a valid JSON object matching the schema above."
        )

        return formatted_error

    def _format_validation_error(self, error: ValidationError) -> str:
        """
        Format Pydantic validation error in a clear, actionable way.

        :param error: Pydantic ValidationError
        :return: Human-readable error message for the LLM
        """
        error_messages = []

        for err in error.errors():
            field_path = " -> ".join(str(x) for x in err["loc"])
            error_type = err["type"]
            message = err["msg"]

            error_messages.append(
                f"  • Field '{field_path}': {message} (error type: {error_type})"
            )

        formatted_error = (
            "❌ VALIDATION ERROR - Your JSON response does not match the required schema.\n\n"
            "Required schema:\n"
            f"{json.dumps(self.response_model.model_json_schema(), indent=2)}\n\n"
            "Validation errors found:\n"
            + "\n".join(error_messages)
            + "\n\n⚠️ Please fix these issues and generate a valid JSON response."
        )

        return formatted_error

    def _value_validation_error(
        self, content: dict, schema: list
    ) -> tuple[bool, str | None]:
        """
        Validate that all columns in the content exist in the schema.
        Works generically with any task structure.
        Returns (has_error, error_message)
        """
        hallucinated_cols = set()

        def extract_columns(obj):
            """Recursively extract all column values from nested structures"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Check if this is a column-related field
                    if key in ["columns", "join_column", "correlation_column"]:
                        if isinstance(value, list):
                            for col in value:
                                if col not in schema:
                                    hallucinated_cols.add(col)
                        elif isinstance(value, str):
                            if value not in schema:
                                hallucinated_cols.add(value)
                    else:
                        # Recurse into nested structures
                        extract_columns(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_columns(item)

        # Extract all columns from the content
        extract_columns(content)

        # If hallucinations found, return error
        if hallucinated_cols:
            formatted_error = (
                "❌ Hallucination ERROR - Your response contains columns that do not exist.\n\n"
                f"The following columns do not exist: {sorted(hallucinated_cols)}\n\n"
                "Please generate ONLY a valid JSON object that contains only the following real columns:\n"
                f"{schema}\n"
            )
            return True, formatted_error

        return False, None





    def complete_sql(
        self,
        prompt: str,
        dataframes,table_names,
        reply_model: Optional[Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """
        Make a completion request with optional structured output.

        :param prompt: The prompt to send to the model
        :param response_model: Optional Pydantic model for structured output
        :param temperature: Override default temperature
        :param max_retries: Override default max_retries
        :param **kwargs: Additional arguments to pass to litellm.completion

        :return: If response_model is provided, instance of the Pydantic model,
                otherwise a raw string response
        """
        temp = temperature if temperature is not None else self.temperature
        retries = max_retries if max_retries is not None else self.max_retries
        usage_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        messages = [{"role": "system", "content": prompt}]
        completion_args = {
            "model": "primary",  # Router handles the actual model selection
            "messages": messages,
            "temperature": temp,
            **kwargs,
        }

        if reply_model:
            self.response_model = reply_model
            self.raw = False
            
        if self.response_model and self.enable_json_mode:
            completion_args["response_format"] = {"type": "json_object"}
        
        assert self.response_model is not None or self.raw
        last_content = None
        last_error = None

        good_queries: dict[str, SQLQuery] = {}
        for attempt in range(retries):
            try:
                print(f"Attempt {attempt + 1}/{retries}...")

                response = self.router.completion(**completion_args)
                usage = response["usage"]
                usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
                usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
                usage_total["total_tokens"] += usage.get("total_tokens", 0)
                content = response["choices"][0]["message"]["content"]

                # If no response model, return raw content
                if self.raw:
                    print(f"✓ Success on attempt {attempt + 1}\n")
                    return content, usage_total

                # Parse structured output
                last_content = content
                cleaned_content = self._clean_json_response(content)

                try:
                    # First try to parse as JSON
                    json_data = json.loads(cleaned_content)

                    # Then validate with Pydantic
                    result = self.response_model.model_validate(json_data)
                    result = result.model_dump()
                    ##################### TODO
                    outcome, errors, accepted_queries = validate_sql_queries(dataframes, result,table_names)
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

                    if attempt < retries - 1:
                        # Add assistant's failed response
                        messages.append({"role": "assistant", "content": content})
                        # Add error feedback as user message
                        messages.append({"role": "user", "content": error_msg})
                        print("💬 Sending error feedback to LLM...\n")
                        time.sleep(self.retry_delay)
                        continue
                except ValidationError as e:
                    # Pydantic validation failed
                    last_error = e
                    error_msg = self._format_validation_error(e)
                    print(f"⚠️ Validation error on attempt {attempt + 1}")

                    if attempt < retries - 1:
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
                if attempt < retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...\n")
                    time.sleep(self.retry_delay)

        # All retries failed
        # All retries exhausted
        print(f"\n❌ Failed after {retries} attempts")
        print(f"Last error: {last_error}")
        if last_content and not self.raw:
            print(f"\nLast response preview:\n{last_content[:300]}...\n")

        return self.response_model.model_dump_json(indent=2), usage_total


import pandas as pd
import prompting
from prompting import DatasetDescription
from pathlib import Path
from structured_outputs import SQLQuerySet
import duckdb

def parse_matches(matches):
    """Generate a textual description of database operations from match data."""
    
    operations = []
    all_join_conditions = []
    
    for match in matches:
        table1 = match["table1"]
        table2 = match["table2"]
        operation = match["operation"]
        
        # Build join conditions for this match
        join_conditions = []
        for couple in match["couples"]:
            condition = f"{table1}.{couple[0]} {couple[1]} {table2}.{couple[2]}"
            join_conditions.append(condition)
            all_join_conditions.append(condition)
        
        # Build operation description
        if operation == "U":
            operations.append(f"{table1} UNIONS {table2}")
        elif operation == "JC":
            operations.append(f"{table1} JOIN-CORRELATION {table2}")
        elif operation == "MJ":
            # Format: "table1 joins table2 on columns (condition1, condition2, ...)"
            conditions_str = ", ".join(f"({cond})" for cond in join_conditions)
            operations.append(f"{table1} joins {table2} on columns {conditions_str}")
    
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


def validate_sql_queries(dataframes, result, table_names):
    validation_errors = []
    all_valid = True
    good_queries = {}
    for idx, q in enumerate(result["queries"]):
        sql = q["query"].replace("`", '"')
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



if __name__=="__main__":
    #### we fetch the dataframes
    folder = r"D:\uk_small\uk_small_copy\datasets\csv"
    path = Path("litellm.yaml")
    D1 = "2019-March-return__3f436d14-4e17-476c-a3e4-66d18e7f6c90"
    D2 = "2019-January-return__e07d11d5-afcb-4b8d-909f-d4952c4f183a"

    # Load CSVs
    df1 = pd.read_csv(Path(folder) / f"{D1}.csv")
    df1["Amount"] = df1["Amount"].str.replace(",", "").astype(float)

    df2 = pd.read_csv(Path(folder) / f"{D2}.csv")
    df2["Amount"] = df2["Amount"].str.replace(",", "").astype(float)
    # Define matches with real columns
    matches = [
        {
            "table1": D1.split("__")[0],
            "table2": D2.split("__")[0],
            "operation": "MJ",
            "couples": [("Transaction number", "=", df2.columns[6]),("Supplier", "=",df2.columns[5])]
        },
    ]

    # Create table descriptions
    descriptor = DatasetDescription()
    table = descriptor.update(D1, df1.shape[0], df1.shape[1], "", "", df1.head(3))
    table += f"\n{descriptor.update(D2, df2.shape[0], df2.shape[1], '', '', df2.head(3))}"

    # Load prompt
    prompt = prompting._load_prompt("prompt.md", "SQLGeneration", matches=parse_matches(matches), table=table)
    # fetch the queries
    prompt =prompt + "\n" +  prompting._load_prompt("prompt.md", "Pydantic", format= SQLQuerySet.schema_json(indent=2))
    client = LLMClient(Path("litellm.yaml"))
    ### testing queries
    dataframes = [df1,df2]
    table_names = [D1.split("__")[0],D2.split("__")[0]]
    queries=client.complete_sql(prompt,dataframes,table_names, reply_model=SQLQuerySet)
    with open("validated_queries.json", "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2)   
    print(queries)

