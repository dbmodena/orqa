import json
import time
from pathlib import Path
from typing import Any

from .LLMClientStructured import LLMClientStructured
from pydantic import ValidationError


class TaskProposerLLMClient(LLMClientStructured):
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
        super().__init__(config_path, "table_analyzer")

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

    def complete(
        self,
        prompt: str,
        schema=None,
        column_typings=None,
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
        messages = [
            {"role": "system", "content": self.reform_prompt_constraint(prompt)}
        ]
        completion_args = {
            "model": "primary",  # Router handles the actual model selection
            "messages": messages,
            "temperature": self.temperature,
            # "response_format": {"type": "json_object"},
            **kwargs,
        }
        last_content = None
        last_error = None

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

                    if schema is not None:
                        invalid, error_msg = self._value_validation_error(
                            result, schema
                        )
                        if invalid:
                            messages.append({"role": "user", "content": error_msg})
                            continue
                        elif column_typings is not None:
                            for tasks in result["join_correlation_tasks"]:
                                if tasks["correlation_column"]:
                                    if not column_typings[tasks["correlation_column"]]:
                                        error_msg = (
                                            "❌ Correlation ERROR - Non-numerical column used.\n\n"
                                            f"The result: {result}\n\n"
                                            f"The column '{tasks['correlation_column']}' has dtype '{column_typings[tasks['correlation_column']]}', "
                                            "which is not numerical.\n\n"
                                            "Correlation can only be computed on numerical columns "
                                            "(Int*, UInt*, Float*).\n"
                                        )
                                        messages.append(
                                            {"role": "user", "content": error_msg}
                                        )
                                        continue

                    print(f"✓ Success on attempt {attempt + 1}\n")
                    return result, usage_total
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
                        print("💬 Sending error feedback to LLM...\n")
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

        # All retries failed
        # All retries exhausted
        print(f"\n❌ Failed after {self.max_retries} attempts")
        print(f"Last error: {last_error}")
        if last_content:
            print(f"\nLast response preview:\n{last_content[:300]}...\n")

        return {}, usage_total


if __name__ == "__main__":
    ### testing main
    import pandas as pd
    import prompting
    from prompting import DatasetDescription

    #### we fetch the dataframes
    folder = r"D:\uk_small\uk_small_copy\datasets\csv"
    path = Path("litellm.yaml")
    D1 = "2019-March-return__3f436d14-4e17-476c-a3e4-66d18e7f6c90"
    # Load CSVs
    df1 = pd.read_csv(Path(folder) / f"{D1}.csv")
    df1["Amount"] = df1["Amount"].str.replace(",", "").astype(float)

    ### define the tables aliases
    TABLE1 = f"df_{D1.split('__')[0].replace('-', '_')}"
    # Create table descriptions
    descriptor = DatasetDescription()
    table = descriptor.update(TABLE1, df1.shape[0], df1.shape[1], "", "", df1.head(3))

    # Load prompt
    prompt = prompting._load_prompt("prompt.md", "Analyze", table=table)
    client = LLMClientTableAnalyzer(Path("litellm.yaml"))
    ### testing queries
    print(prompt)
    result = client.complete(prompt)
    print(result)

