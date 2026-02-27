import importlib
import json
import time
from pathlib import Path
from typing import Any, Optional, Type

from .prompting import DatasetDescription, _load_prompt
from pydantic import BaseModel, ValidationError
from .LLMClient import LLMClient

class LLMClientStructured(LLMClient):
    """
    LiteLLM client with YAML configuration and structured output support.
    """

    def __init__(self, config_path: Path, response_model="response_model"):
        """
        Initialize LLM client with configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
        """
        # 1. Inherits the methods and proprierties
        super().__init__(config_path)

        # 2. Load response model
        self.response_model = self._load_pydantic_response_model(response_model)

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

    def _load_pydantic_response_model(self, model) -> Optional[Type[BaseModel]]:
        """
        Dynamically load Pydantic model from config.

        Returns:
            Pydantic model class or None if not specified
        """
        response_model_config = self.config.get(model)

        if not response_model_config:
            return None

        module_name = response_model_config.get("module")
        class_name = response_model_config.get("class")

        if not module_name or not class_name:
            raise ValueError("response_model must specify both 'module' and 'class'")

        try:
            # Import the module
            module = importlib.import_module(f".{module_name}", package=__package__)

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
            "Required Pydantic schema:\n"
            f"{json.dumps(self.response_model.model_json_schema(), indent=2)}\n\n"
            "⚠️ Common issues:\n"
            "  - Missing quotes around strings\n"
            "  - Trailing commas\n"
            "  - Unescaped special characters\n"
            "  - Text before or after the JSON object\n\n"
            "Please generate ONLY a valid JSON object matching the schema above."
        )

        return formatted_error

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
            "Required Pydantic schema:\n"
            f"{json.dumps(self.response_model.model_json_schema(), indent=2)}\n\n"
            "Validation errors found:\n"
            + "\n".join(error_messages)
            + "\n\n⚠️ Please fix these issues and generate a valid JSON response."
        )

        return formatted_error

    def reform_prompt_constraint(self, prompt: str):
        format_dict = self.response_model.model_json_schema()
        format_str = json.dumps(format_dict, indent=2)

        constraint = (
            "### Output format instructions\n"
            "Return the final answer **strictly in JSON** and **exactly matching** "
            "the Pydantic following schema provided:\n\n"
            f"{format_str}\n\n"
            "All required fields must be present, field names and data types must match exactly, "
            "and all validation constraints (including value ranges, uniqueness, and completeness rules) "
            "must be satisfied. Do not add extra fields, omit required fields, "
            "or include explanatory text outside the JSON."
        )

        return f"{prompt}\n{constraint}"

    def complete(
        self,
        prompt: str,
        **kwargs,
    ) -> Any:
        """
        Make a completion request with optional structured output.
        :param prompt: The prompt to send to the model
        :param **kwargs: Additional arguments to pass to litellm.completion
        """
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        messages = [
            {"role": "system", "content": self.reform_prompt_constraint(prompt)}
        ]
        completion_args = {
            "model": "primary",  # Router handles the actual model selection
            "messages": messages,
            "temperature": self.temperature,
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

        # All retries exhausted
        print(f"\n❌ Failed after {self.max_retries} attempts")
        print(f"Last error: {last_error}")
        if last_content:
            print(f"\nLast response preview:\n{last_content[:300]}...\n")

        return {}, usage_total
