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
from .utility.message_builder import JudgeMessageBuilder
from .validators.SQLValidator import SQLValidator
from .validators.PandasValidator import PandasValidator
import re




class LLMStatementJudge(LLMClientStructured):
    """
    LiteLLM client with YAML configuration and structured output support.

    Uses a fixed 2-block message structure (system + query evaluation) that is
    rebuilt from scratch on each call — no retained history between calls.
    """

    def __init__(self, config_path: Path):
        super().__init__(config_path, "statement_judge")
        self._message_builder = JudgeMessageBuilder()

    def complete(self, prompt: str, **kwargs) -> Any:
        """
        Make a structured completion request with fixed 2-block messages.

        Each call rebuilds the message array from scratch using JudgeMessageBuilder,
        ensuring no history is retained between calls. On parse errors, the prompt
        is rebuilt with error feedback rather than appending to history.

        Returns ``(result_dict, usage_total)`` on success; ``({}, usage_total)`` on
        exhausted retries.
        """
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        system_prompt = self.reform_prompt_constraint(prompt)
        last_content: Optional[str] = None
        last_error: Optional[Exception] = None

        # The query_payload starts as empty; on retries it includes error feedback
        query_payload = ""

        for attempt in range(self.max_retries):
            try:
                # Fixed 2-block rebuild: system + query evaluation
                messages = self._message_builder.build(system_prompt, query_payload)

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
                if content is None:
                    raise ValueError("LLM returned None content.")
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
                        # Rebuild with error feedback — no accumulation
                        query_payload = self._format_json_error(cleaned, e)
                        time.sleep(self.retry_delay)

                except ValidationError as e:
                    last_error = e
                    print(f"⚠️ Validation error on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        # Rebuild with error feedback — no accumulation
                        query_payload = self._format_validation_error(e)
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