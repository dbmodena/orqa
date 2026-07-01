import importlib
import json
import re
import time
from pathlib import Path
from typing import Any, Optional, Type

from pydantic import BaseModel, ValidationError

from .prompting import DatasetDescription, _load_prompt
from .LLMClient import LLMClient


class LLMClientStructured(LLMClient):
    """
    LiteLLM client with YAML configuration and structured output support.

    Adds on top of LLMClient:
    - Pydantic response-model loading and validation
    - A robust JSON repair pipeline shared by all subclasses
    - Retry loops that feed structured error messages back to the LLM
    """

    def __init__(self, config_path: Path, response_model: str = "response_model"):
        super().__init__(config_path)
        self.response_model = self._load_pydantic_response_model(response_model)

    # ------------------------------------------------------------------
    # Pydantic helpers
    # ------------------------------------------------------------------

    def _load_pydantic_response_model(self, model: str) -> Optional[Type[BaseModel]]:
        """Dynamically load a Pydantic model class from the YAML config."""
        response_model_config = self.config.get(model)
        if not response_model_config:
            return None

        module_name = response_model_config.get("module")
        class_name = response_model_config.get("class")
        if not module_name or not class_name:
            raise ValueError("response_model must specify both 'module' and 'class'")

        try:
            module = importlib.import_module(f".{module_name}", package=__package__)
            model_class = getattr(module, class_name)
            if not issubclass(model_class, BaseModel):
                raise TypeError(f"{class_name} is not a Pydantic BaseModel")
            return model_class
        except ImportError as e:
            raise ImportError(f"Could not import module '{module_name}': {e}") from e
        except AttributeError as e:
            raise AttributeError(
                f"Could not find class '{class_name}' in module '{module_name}': {e}"
            ) from e

    def reform_prompt_constraint(self, prompt: str) -> str:
        """Append the Pydantic JSON-schema constraint to a prompt."""
        format_str = json.dumps(self.response_model.model_json_schema(), indent=2)
        constraint = (
            "### Output format instructions\n"
            "Return the final answer **strictly in JSON** and **exactly matching** "
            "the Pydantic following schema provided:\n\n"
            f"{format_str}\n\n"
            "All required fields must be present, field names and data types must match exactly, "
            "and all validation constraints (including value ranges, uniqueness, and completeness "
            "rules) must be satisfied. Do not add extra fields, omit required fields, "
            "or include explanatory text outside the JSON."
        )
        return f"{prompt}\n{constraint}"

    # ------------------------------------------------------------------
    # JSON cleaning & repair pipeline
    # ------------------------------------------------------------------

    def _clean_json_response(self, content: str) -> str:
        """
        Strip markdown fences and extract the outermost JSON object or array.
        Returns the trimmed content even if no valid JSON delimiters are found,
        so the repair pipeline can attempt further fixes.
        """
        content = content.strip()

        # Remove markdown code fences
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        # Locate the first opening delimiter
        brace_start = content.find("{")
        bracket_start = content.find("[")

        if brace_start == -1 and bracket_start == -1:
            return content

        if brace_start == -1:
            start = bracket_start
        elif bracket_start == -1:
            start = brace_start
        else:
            start = min(brace_start, bracket_start)

        opener = content[start]
        closer = "}" if opener == "{" else "]"
        depth = 0
        for i in range(start, len(content)):
            if content[i] == opener:
                depth += 1
            elif content[i] == closer:
                depth -= 1
                if depth == 0:
                    return content[start : i + 1].strip()

        return content.strip()

    def _fix_triple_quotes(self, content: str) -> str:
        """Replace triple-quoted strings with a valid single-quoted JSON string."""
        pattern = r'"""[\s\S]*?"""'
        offset = 0
        result = content
        for match in re.finditer(pattern, content):
            original = match.group(0)
            inner = original[3:-3].strip()
            escaped = (
                inner
                .replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "")
                .replace("\t", " ")
            )
            replacement = f'"{escaped}"'
            start = match.start() + offset
            end = match.end() + offset
            result = result[:start] + replacement + result[end:]
            offset += len(replacement) - len(original)
        return result

    def _escape_literal_controls(self, content: str) -> str:
        """
        Escape literal control characters (newlines, tabs, carriage returns) that
        appear inside JSON string values but were not escaped by the LLM.
        """
        result: list[str] = []
        in_string = False
        i = 0
        while i < len(content):
            ch = content[i]
            if in_string:
                if ch == "\\":
                    result.append(ch)
                    i += 1
                    if i < len(content):
                        result.append(content[i])
                        i += 1
                    continue
                elif ch == '"':
                    in_string = False
                    result.append(ch)
                elif ch == "\n":
                    result.append("\\n")
                elif ch == "\r":
                    result.append("\\r")
                elif ch == "\t":
                    result.append("\\t")
                else:
                    result.append(ch)
            else:
                if ch == '"':
                    in_string = True
                result.append(ch)
            i += 1
        return "".join(result)

    def _fix_structural_issues(self, content: str) -> str:
        """
        Fix common structural JSON problems produced by LLMs:
        - Trailing commas before } or ]
        - Unterminated string at end of document
        - Unclosed braces/brackets (truncated output)
        """
        # 1. Remove trailing commas before closing delimiters
        fixed = re.sub(r",\s*(?=[}\]])", "", content)

        # 2. Close any unterminated string (odd number of unescaped quotes)
        unescaped_quotes = re.findall(r'(?<!\\)"', fixed)
        if len(unescaped_quotes) % 2 != 0:
            fixed = fixed.rstrip() + '"'

        # 3. Close unclosed braces/brackets (handles truncated LLM output)
        closer_map = {"{": "}", "[": "]"}
        stack: list[str] = []
        in_str = False
        i = 0
        while i < len(fixed):
            ch = fixed[i]
            if in_str:
                if ch == "\\":
                    i += 2
                    continue
                if ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch in ("{", "["):
                    stack.append(closer_map[ch])
                elif ch in ("}", "]"):
                    if stack and stack[-1] == ch:
                        stack.pop()
            i += 1

        fixed = fixed.rstrip()
        for closer in reversed(stack):
            fixed += closer

        return fixed

    def _repair_json(self, content: str) -> dict:
        """
        Multi-stage JSON repair pipeline.

        Stage 1 – fast path (valid JSON)
        Stage 2 – fix triple-quoted strings
        Stage 3 – escape literal control characters inside strings
        Stage 4 – structural fixes (trailing commas, unclosed delimiters)

        Raises ``json.JSONDecodeError`` if all stages fail.
        """
        if not isinstance(content, str):
            raise json.JSONDecodeError("content is not a string", str(content), 0)

        # Stage 1
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Stage 2
        patched = self._fix_triple_quotes(content)
        try:
            return json.loads(patched)
        except json.JSONDecodeError:
            pass

        # Stage 3
        escaped = self._escape_literal_controls(patched)
        try:
            return json.loads(escaped)
        except json.JSONDecodeError:
            pass

        # Stage 4 – will raise if still broken, giving the caller a meaningful error
        structural = self._fix_structural_issues(escaped)
        return json.loads(structural)

    def _normalize_response(self, json_data: Any, root_key: str | None = "queries") -> Any:
        """
        Normalize the LLM response for Pydantic validation.

        If ``root_key`` is None, return the parsed JSON unchanged.
        Otherwise, coerce the top-level JSON into ``{root_key: [...]}`` form.
        """
        if root_key is None:
            return json_data
        if isinstance(json_data, list):
            return {root_key: json_data}
        if isinstance(json_data, dict):
            if root_key in json_data:
                return json_data
            if json_data:
                return {root_key: [json_data]}
        return {root_key: []}

    # ------------------------------------------------------------------
    # Error formatting helpers
    # ------------------------------------------------------------------

    def _format_json_error(self, content: str, error: Exception) -> str:
        return (
            "❌ JSON PARSING ERROR – Your response is not valid JSON.\n\n"
            f"Error: {error}\n\n"
            "Required Pydantic schema:\n"
            f"{json.dumps(self.response_model.model_json_schema(), indent=2)}\n\n"
            "⚠️ Common issues:\n"
            "  - Missing quotes around strings\n"
            "  - Trailing commas\n"
            "  - Unescaped special characters\n"
            "  - Text before or after the JSON object\n\n"
            "Please generate ONLY a valid JSON object matching the schema above."
        )

    def _format_validation_error(self, error: ValidationError) -> str:
        lines = []
        for err in error.errors():
            field_path = " -> ".join(str(x) for x in err["loc"])
            lines.append(f"  • Field '{field_path}': {err['msg']} (type: {err['type']})")
        return (
            "❌ VALIDATION ERROR – Your JSON response does not match the required schema.\n\n"
            "Required Pydantic schema:\n"
            f"{json.dumps(self.response_model.model_json_schema(), indent=2)}\n\n"
            "Validation errors found:\n"
            + "\n".join(lines)
            + "\n\n⚠️ Please fix these issues and generate a valid JSON response."
        )

    # ------------------------------------------------------------------
    # Generic completion loop
    # ------------------------------------------------------------------

    def complete(self, prompt: str, root_key: str | None = "queries", **kwargs) -> Any:
        """
        Make a structured completion request with automatic retry and error feedback.

        Returns ``(result_dict, usage_total)`` on success; ``({}, usage_total)`` on
        exhausted retries.
        """
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        messages = [{"role": "system", "content": self.reform_prompt_constraint(prompt)}]
        completion_args = {
            "model": "primary",
            "messages": messages,
            "temperature": self.temperature,
            **kwargs,
        }
        last_content: Optional[str] = None
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                print(f"Attempt {attempt + 1}/{self.max_retries}...")
                response = self.router.completion(**completion_args)

                usage = response["usage"]
                usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
                usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
                usage_total["total_tokens"] += usage.get("total_tokens", 0)

                content = response["choices"][0]["message"]["content"]
                if content is None:
                    raise ValueError("LLM returned None content.")
                last_content = content

                cleaned = self._clean_json_response(content)

                try:
                    json_data = self._repair_json(cleaned)
                    json_data = self._normalize_response(json_data, root_key=root_key)
                    result = self.response_model.model_validate(json_data)
                    print(f"✓ Success on attempt {attempt + 1}")
                    return result.model_dump(), usage_total

                except json.JSONDecodeError as e:
                    last_error = e
                    print(f"⚠️ JSON parsing error on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": self._format_json_error(cleaned, e)})
                        time.sleep(self.retry_delay)

                except ValidationError as e:
                    last_error = e
                    print(f"⚠️ Validation error on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": self._format_validation_error(e)})
                        time.sleep(self.retry_delay)

            except Exception as e:
                last_error = e
                print(f"✗ Error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.retry_delay}s…")
                    time.sleep(self.retry_delay)

        print(f"\n❌ Failed after {self.max_retries} attempts. Last error: {last_error}")
        if last_content:
            print(f"Last response preview:\n{last_content[:300]}…")
        return {}, usage_total
