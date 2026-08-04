import json
import time
from pathlib import Path
from typing import Any

from ..llm_client.LLMClientStructured import LLMClientStructured
from ..utility.message_builder import sanitize_messages
from pydantic import ValidationError

import logging

logger = logging.getLogger(__name__)


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
        # The prompt goes in a USER message (with a minimal system preamble),
        # not a lone system message — some providers (e.g. Cohere via OCI)
        # reject requests with no user message at all.
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": self.reform_prompt_constraint(prompt)},
        ]
        completion_args = {
            "model": "primary",  # Router handles the actual model selection
            "messages": messages,
            **self._default_temperature_kwargs(self.config["model"]),
            # "response_format": {"type": "json_object"},
            **kwargs,
        }
        last_content = None
        last_error = None

        for attempt in range(self.max_retries):
            try:
                logger.debug("Attempt %d/%d...", attempt + 1, self.max_retries)

                completion_args["messages"] = sanitize_messages(messages)
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

                    logger.debug("Success on attempt %d", attempt + 1)
                    return result, usage_total
                except json.JSONDecodeError as e:
                    # JSON parsing failed
                    last_error = e
                    error_msg = self._format_json_error(cleaned_content, e)
                    logger.warning("JSON parsing error on attempt %d", attempt + 1)

                    if attempt < self.max_retries - 1:
                        # Add assistant's failed response
                        messages.append({"role": "assistant", "content": content})
                        # Add error feedback as user message
                        messages.append({"role": "user", "content": error_msg})
                        logger.debug("Sending error feedback to LLM...")
                        time.sleep(self.retry_delay)
                        continue
                except ValidationError as e:
                    # Pydantic validation failed
                    last_error = e
                    error_msg = self._format_validation_error(e)
                    logger.warning("Validation error on attempt %d", attempt + 1)

                    if attempt < self.max_retries - 1:
                        # Add assistant's failed response
                        messages.append({"role": "assistant", "content": content})
                        # Add error feedback as user message
                        messages.append({"role": "user", "content": error_msg})
                        logger.debug("Sending validation errors to LLM...")
                        time.sleep(self.retry_delay)
                        continue

            except Exception as e:
                last_error = e
                logger.error("Error on attempt %d: %s", attempt + 1, e)

                # Wait before retry
                if attempt < self.max_retries - 1:
                    logger.debug("Retrying in %s seconds...", self.retry_delay)
                    time.sleep(self.retry_delay)

        # All retries failed
        # All retries exhausted
        logger.error("Failed after %d attempts. Last error: %s", self.max_retries, last_error)
        if last_content:
            logger.debug("Last response preview:\n%s...", last_content[:300])

        return {}, usage_total


# Fields of PairTaskSelection whose values must exist in the Q-side schema
# vs the R-side schema.
_Q_SIDE_FIELDS = ("q_columns", "q_key", "q_target")
_R_SIDE_FIELDS = ("r_columns", "r_key", "r_target")


class PairTaskSelectorLLMClient(LLMClientStructured):
    """
    Structured client selecting discovery operations for one concrete
    (Q, R) dataset pair: validates each side's columns against its own
    schema and requires numeric correlation targets on both sides.
    """

    def __init__(self, config_path: Path):
        super().__init__(config_path, "pair_task_selector")

    def _value_validation_error(
        self, content: dict, q_schema: list, r_schema: list
    ) -> tuple[bool, str | None]:
        """Route each field to the schema of its side (q_* → Q, r_* → R)."""
        hallucinated: dict[str, set] = {"Q": set(), "R": set()}

        def check(value, schema, side):
            if isinstance(value, list):
                hallucinated[side].update(c for c in value if c not in schema)
            elif isinstance(value, str) and value not in schema:
                hallucinated[side].add(value)

        def walk(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in _Q_SIDE_FIELDS:
                        check(value, q_schema, "Q")
                    elif key in _R_SIDE_FIELDS:
                        check(value, r_schema, "R")
                    else:
                        walk(value)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(content)

        if hallucinated["Q"] or hallucinated["R"]:
            parts = []
            if hallucinated["Q"]:
                parts.append(
                    f"Columns not in dataset Q: {sorted(hallucinated['Q'])}\n"
                    f"Dataset Q columns: {q_schema}"
                )
            if hallucinated["R"]:
                parts.append(
                    f"Columns not in dataset R: {sorted(hallucinated['R'])}\n"
                    f"Dataset R columns: {r_schema}"
                )
            formatted_error = (
                "❌ Hallucination ERROR - Your response contains columns that "
                "do not exist.\n\n" + "\n\n".join(parts) + "\n\n"
                "Please generate ONLY a valid JSON object using q_-prefixed "
                "fields with Q's real columns and r_-prefixed fields with R's "
                "real columns.\n"
            )
            return True, formatted_error

        return False, None

    def _correlation_validation_error(
        self, result: dict, q_column_typings: dict, r_column_typings: dict
    ) -> str | None:
        """Correlation targets must be numeric on their respective sides."""
        for task in result.get("join_correlation_tasks", []):
            for target, typings, side in (
                (task.get("q_target"), q_column_typings, "Q"),
                (task.get("r_target"), r_column_typings, "R"),
            ):
                if target and not typings.get(target, False):
                    return (
                        "❌ Correlation ERROR - Non-numerical column used.\n\n"
                        f"The result: {result}\n\n"
                        f"The column '{target}' of dataset {side} is not "
                        "numerical. Correlation can only be computed on "
                        "numerical columns (Int*, UInt*, Float*).\n"
                    )
        return None

    def complete(
        self,
        prompt: str,
        q_schema: list,
        r_schema: list,
        q_column_typings: dict,
        r_column_typings: dict,
        **kwargs,
    ) -> Any:
        usage_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": self.reform_prompt_constraint(prompt)},
        ]
        completion_args = {
            "model": "primary",
            "messages": messages,
            **self._default_temperature_kwargs(self.config["model"]),
            **kwargs,
        }
        last_content = None
        last_error = None

        for attempt in range(self.max_retries):
            try:
                logger.debug("Attempt %d/%d...", attempt + 1, self.max_retries)

                completion_args["messages"] = sanitize_messages(messages)
                response = self.router.completion(**completion_args)
                usage = response["usage"]
                usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
                usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
                usage_total["total_tokens"] += usage.get("total_tokens", 0)
                content = response["choices"][0]["message"]["content"]
                last_content = content
                cleaned_content = self._clean_json_response(content)

                try:
                    json_data = json.loads(cleaned_content)
                    result = self.response_model.model_validate(json_data)
                    result = result.model_dump()

                    invalid, error_msg = self._value_validation_error(
                        result, q_schema, r_schema
                    )
                    if not invalid:
                        error_msg = self._correlation_validation_error(
                            result, q_column_typings, r_column_typings
                        )
                        invalid = error_msg is not None
                    if invalid:
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": error_msg})
                        continue

                    logger.debug("Success on attempt %d", attempt + 1)
                    return result, usage_total
                except json.JSONDecodeError as e:
                    last_error = e
                    error_msg = self._format_json_error(cleaned_content, e)
                    logger.warning("JSON parsing error on attempt %d", attempt + 1)
                    if attempt < self.max_retries - 1:
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": error_msg})
                        time.sleep(self.retry_delay)
                        continue
                except ValidationError as e:
                    last_error = e
                    error_msg = self._format_validation_error(e)
                    logger.warning("Validation error on attempt %d", attempt + 1)
                    if attempt < self.max_retries - 1:
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": error_msg})
                        time.sleep(self.retry_delay)
                        continue

            except Exception as e:
                last_error = e
                logger.error("Error on attempt %d: %s", attempt + 1, e)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        logger.error(
            "Failed after %d attempts. Last error: %s", self.max_retries, last_error
        )
        if last_content:
            logger.debug("Last response preview:\n%s...", last_content[:300])

        return {}, usage_total

