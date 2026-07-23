import json
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from ..llm_client.LLMClientStructured import LLMClientStructured
from ..utility.message_builder import JudgeMessageBuilder, sanitize_messages




class LLMStatementJudge(LLMClientStructured):
    """
    LiteLLM client with YAML configuration and structured output support.

    Uses a fixed 2-block message structure (system + query evaluation) that is
    rebuilt from scratch on each call — no retained history between calls.
    """

    def __init__(self, config_path: Path):
        super().__init__(config_path, "statement_judge")
        self._message_builder = JudgeMessageBuilder()

    def complete(self, prompt: str, data: Any = None, **kwargs) -> Any:
        """
        Make a structured completion request with fixed 2-block messages.

        ``prompt`` carries ONLY the static judge instructions; ``data`` carries
        the per-query evaluation payload. The instructions plus the (equally
        static) output-schema constraint form the system message, so the system
        message is byte-identical across every judge call in a run — the
        provider's prompt cache serves that whole prefix, and only the short
        user message (the payload) is uncached.

        Each call rebuilds the message array from scratch using JudgeMessageBuilder,
        ensuring no history is retained between calls. On parse errors, the user
        payload is rebuilt with error feedback (the system message never changes).

        Returns ``(result_dict, usage_total)`` on success; ``({}, usage_total)`` on
        exhausted retries.
        """
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        system_prompt = self.reform_prompt_constraint(prompt)
        last_content: Optional[str] = None
        last_error: Optional[Exception] = None

        # First-attempt user message: the queries to evaluate. On retries it is
        # replaced with error feedback. Never empty — providers reject
        # zero-token messages (OCI 400: "message must be at least 1 token long").
        query_payload = (
            f"Queries:\n{data}\n\n"
            "Evaluate the queries above following the instructions and "
            "return only the JSON verdict."
        )

        for attempt in range(self.max_retries):
            try:
                # Fixed 2-block rebuild: system + query evaluation
                messages = sanitize_messages(
                    self._message_builder.build(system_prompt, query_payload)
                )

                completion_args = {
                    "model": "primary",
                    "messages": messages,
                    "temperature": self.temperature,
                    **kwargs,
                }

                print(f"Attempt {attempt + 1}/{self.max_retries}...")
                response = self.router.completion(**completion_args)

                usage = response["usage"]
                usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
                usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
                usage_total["total_tokens"] += usage.get("total_tokens", 0)

                content = response["choices"][0]["message"]["content"]
                if content is None or not str(content).strip():
                    raise ValueError("LLM returned empty content.")
                last_content = content

                cleaned = self._clean_json_response(content)

                try:
                    json_data = self._repair_json(cleaned)
                    json_data = self._normalize_response(json_data)
                    result = self.response_model.model_validate(json_data)
                    print(f"✓ Success on attempt {attempt + 1}")
                    return result.model_dump(), usage_total

                except json.JSONDecodeError as e:
                    last_error = e
                    print(f"⚠️ JSON parsing error on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        # Rebuild with error feedback — no accumulation. The
                        # queries stay in the payload (the system message no
                        # longer carries them) so the retry can still see them.
                        query_payload = (
                            f"Queries:\n{data}\n\n{self._format_json_error(cleaned, e)}"
                        )
                        time.sleep(self.retry_delay)

                except ValidationError as e:
                    last_error = e
                    print(f"⚠️ Validation error on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        # Rebuild with error feedback — no accumulation. Keep
                        # the queries in the payload (see above).
                        query_payload = (
                            f"Queries:\n{data}\n\n{self._format_validation_error(e)}"
                        )
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