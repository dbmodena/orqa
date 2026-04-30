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
import re


class LLMClientStatementGenerator(LLMClientStructured):
    """
    LiteLLM client with YAML configuration and structured output support.

    Responsibility: generate the *initial* set of queries from a prompt.
    Static validation and correction are now handled by LLMStatementValidator
    (see StatementValidator.py).  This class only retries on JSON / Pydantic
    parse errors — it does NOT run PandasValidator / SQLValidator internally.
    """

    def __init__(self, config_path: Path):
        super().__init__(config_path, "querying")
        self.fallback_response_model = self._load_pydantic_response_model("fallback_querying")

    # ------------------------------------------------------------------
    # JSON repair utilities
    # ------------------------------------------------------------------

    def _repair_json(self, content: str) -> dict:

        if not isinstance(content, str):
            raise json.JSONDecodeError("content is not a string", str(content), 0)

        # Stage 1 – fast path
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Stage 2 – fix triple-quoted strings
        patched = self._fix_triple_quotes(content)
        try:
            return json.loads(patched)
        except json.JSONDecodeError:
            pass

        # Stage 3 – escape literal control characters inside JSON strings
        escaped = self._escape_literal_controls(patched)
        try:
            return json.loads(escaped)
        except json.JSONDecodeError:
            pass

        # Stage 4 – structural fixes (trailing commas, unclosed delimiters)
        structural = self._fix_structural_issues(escaped)
        try:
            return json.loads(structural)
        except json.JSONDecodeError:
            pass

        # Final raise so the caller receives a meaningful JSONDecodeError
        return json.loads(structural)

    def _fix_structural_issues(self, content: str) -> str:
        """
        Fix common structural JSON problems produced by LLMs:
        - Trailing commas before } or ]
        - Unterminated string at end of document
        - Unclosed braces / brackets (truncated LLM output)
        """
        # 1. Remove trailing commas before a closing delimiter
        fixed = re.sub(r',\s*(?=[}\]])', '', content)

        # 2. Close any unterminated string at the very end of the document.
        unescaped_quotes = re.findall(r'(?<!\\)"', fixed)
        if len(unescaped_quotes) % 2 != 0:
            fixed = fixed.rstrip() + '"'

        # 3. Count open braces/brackets and close any that were never closed
        closer_map = {'{': '}', '[': ']'}
        stack: list[str] = []
        in_str = False
        i = 0
        while i < len(fixed):
            ch = fixed[i]
            if in_str:
                if ch == '\\':
                    i += 2
                    continue
                if ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch in ('{', '['):
                    stack.append(closer_map[ch])
                elif ch in ('}', ']'):
                    if stack and stack[-1] == ch:
                        stack.pop()
            i += 1

        fixed = fixed.rstrip()
        for closer in reversed(stack):
            fixed += closer

        return fixed

    def _fix_triple_quotes(self, content: str) -> str:
        """Replace \"\"\"...\"\"\" with a properly escaped single-quoted JSON string."""
        pattern = r'"""[\s\S]*?"""'
        matches = list(re.finditer(pattern, content))
        offset = 0
        result = content
        for match in matches:
            original = match.group(0)
            inner = original[3:-3].strip()
            escaped = (
                inner
                .replace('\\', '\\\\')   # must be first
                .replace('"', '\\"')
                .replace('\n', '\\n')
                .replace('\r', '')
                .replace('\t', ' ')
            )
            replacement = f'"{escaped}"'
            start = match.start() + offset
            end = match.end() + offset
            result = result[:start] + replacement + result[end:]
            offset += len(replacement) - len(original)
        return result

    def _escape_literal_controls(self, content: str) -> str:
        """
        Scan JSON text and escape any literal control characters that appear
        inside string values (i.e. between unescaped double-quotes).
        """
        result = []
        in_string = False
        i = 0
        while i < len(content):
            ch = content[i]
            if in_string:
                if ch == '\\':
                    result.append(ch)
                    i += 1
                    if i < len(content):
                        result.append(content[i])
                        i += 1
                    continue
                elif ch == '"':
                    in_string = False
                    result.append(ch)
                elif ch == '\n':
                    result.append('\\n')
                elif ch == '\r':
                    result.append('\\r')
                elif ch == '\t':
                    result.append('\\t')
                else:
                    result.append(ch)
            else:
                if ch == '"':
                    in_string = True
                    result.append(ch)
                else:
                    result.append(ch)
            i += 1
        return ''.join(result)

    # Keep the old name as a thin wrapper so nothing external breaks.
    def fix_json_with_triple_quotes(self, content: str) -> dict:
        return self._repair_json(content)

    # ------------------------------------------------------------------
    # Table-alias validation
    # ------------------------------------------------------------------

    def _find_missing_tables(self, query_code: str, all_aliases: list[str]) -> list[str]:
        """
        Return the aliases from *all_aliases* that are NOT referenced in
        *query_code*.  Matching is word-boundary aware so e.g. "Table_0"
        inside "Table_00" is not counted as a hit.
        """
        return [
            alias for alias in all_aliases
            if not re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', query_code)
        ]

    def _partition_queries(
        self, queries: list[dict], all_aliases: list[str]
    ) -> tuple[list[dict], list[dict]]:
        """
        Split *queries* into (good, bad) based on whether every alias in
        *all_aliases* appears in each query's ``code`` field.

        Returns:
            good: queries that reference all aliases.
            bad:  queries that are missing one or more aliases.
        """
        good, bad = [], []
        for q in queries:
            missing = self._find_missing_tables(q.get("code", ""), all_aliases)
            if missing:
                bad.append({**q, "_missing_tables": missing})
            else:
                good.append(q)
        return good, bad

    def _build_unused_tables_feedback(
        self, bad_queries: list[dict], all_aliases: list[str]
    ) -> str:
        """Build a feedback message listing each offending query and its missing tables."""
        lines = [
            "Some generated queries do not reference all required tables.",
            "Rewrite ONLY the queries listed below so that every table alias is used.",
            f"Required aliases: {', '.join(all_aliases)}\n",
        ]
        for i, q in enumerate(bad_queries, 1):
            missing = q.get("_missing_tables", [])
            # Show a short excerpt of the code so the LLM can identify the query
            code_preview = q.get("code", "")[:120].replace("\n", " ")
            lines.append(
                f"  Query {i} — missing {', '.join(sorted(missing))}:\n"
                f"    code preview: {code_preview!r}"
            )
        lines.append(
            "\nReturn ONLY the corrected queries (same JSON schema as before)."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        dataframes,
        table_names,
        typology: str = "SQL",
        involved_cols=None,
    ) -> Any:
        """
        Generate the initial set of queries from *prompt*, then validate that
        every query references all table aliases.  Queries that pass are kept;
        only the failing ones are sent back for correction (up to
        ``MAX_TABLE_ALIAS_RETRIES`` extra rounds).  After all retries, any
        still-failing queries are silently dropped and only the good ones are
        returned.
        """
        MAX_TABLE_ALIAS_RETRIES = 3

        usage_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        all_errors: list[str] = []
        all_aliases: list[str] = list(table_names.keys())  # e.g. ["Table_0", "Table_1"]

        # Optionally enrich the prompt with suffix-constraint context
        enriched_prompt = prompt
        if involved_cols is not None and typology == "PANDAS":
            enriched_prompt = self.add_suffix_constraint(
                prompt, dataframes, list(table_names.keys()), involved_cols
            )

        initial_message = self.reform_prompt_constraint(enriched_prompt)

        messages = [
            {"role": "system", "content": "You are an expert Data Engineer"},
            {"role": "user", "content": initial_message},
        ]
        last_content = ""

        # ── Phase 1: initial generation with JSON/Pydantic retry loop ──────────
        for attempt in range(self.max_retries):
            try:
                completion_args = {
                    "model": self.config["model"],
                    "messages": messages,
                    "temperature": self.temperature,
                }

                response = self.router.completion(**completion_args)
                usage = response["usage"]
                usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
                usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
                usage_total["total_tokens"] += usage.get("total_tokens", 0)

                content = response["choices"][0]["message"]["content"]
                if content is None:
                    raise ValueError(
                        "LLM returned None content — the model may have emitted a "
                        "tool-call response instead of a text completion."
                    )
                last_content = content

                cleaned_content = self._clean_json_response(content)
                json_data = self._repair_json(cleaned_content)
                json_data = self._normalize_response(json_data)
                result = self.response_model.model_validate(json_data)
                result_dict = result.model_dump()
                result_dict = self.clean_query_set(result_dict, table_names)

                # Successfully parsed — break out and move to alias validation
                break

            except json.JSONDecodeError as e:
                all_errors.append(f"JSONDecodeError on attempt {attempt + 1}: {e}")
                error_msg = self._format_json_error(last_content, e)
                if attempt < self.max_retries - 1:
                    pydantic = self.reform_prompt_constraint("")
                    messages = [
                        {"role": "system", "content": "You are an expert Data Engineer"},
                        {"role": "user", "content": initial_message},
                        {"role": "assistant", "content": last_content},
                        {
                            "role": "user",
                            "content": (
                                f"Your output could not be parsed as JSON:\n{error_msg}\n{pydantic}"
                            ),
                        },
                    ]
                    time.sleep(self.retry_delay)

            except ValidationError as e:
                all_errors.append(f"ValidationError on attempt {attempt + 1}: {e}")
                error_msg = self._format_validation_error(e)
                if attempt < self.max_retries - 1:
                    pydantic = self.reform_prompt_constraint("")
                    messages = [
                        {"role": "system", "content": "You are an expert Data Engineer"},
                        {"role": "user", "content": enriched_prompt},
                        {"role": "assistant", "content": last_content},
                        {
                            "role": "user",
                            "content": (
                                f"Your output failed schema validation:\n{error_msg}\n{pydantic}"
                            ),
                        },
                    ]
                    time.sleep(self.retry_delay)

            except Exception as e:
                all_errors.append(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    messages = [
                        {"role": "system", "content": "You are an expert Data Engineer"},
                        {"role": "user", "content": enriched_prompt},
                        {"role": "assistant", "content": last_content},
                        {
                            "role": "user",
                            "content": (
                                f"An error occurred: {e}\n"
                                f"Return a JSON object matching:\n"
                                f"{json.dumps(self.response_model.model_json_schema(), indent=2)}"
                            ),
                        },
                    ]
                    time.sleep(self.retry_delay)
        else:
            # All JSON/Pydantic retries exhausted
            return (
                {"queries": []},
                usage_total,
                all_errors,
                self.config["model"],
            )

        # ── Phase 2: table-alias validation with targeted retry loop ───────────
        good_queries, bad_queries = self._partition_queries(
            result_dict.get("queries", []), all_aliases
        )

        for alias_attempt in range(MAX_TABLE_ALIAS_RETRIES):
            if not bad_queries:
                break  # Nothing left to fix

            feedback = self._build_unused_tables_feedback(bad_queries, all_aliases)
            # Strip the internal _missing_tables key before sending to LLM
            bad_queries_clean = [
                {k: v for k, v in q.items() if k != "_missing_tables"}
                for q in bad_queries
            ]
            all_errors.append(
                f"Table-alias check (attempt {alias_attempt + 1}): "
                f"{len(bad_queries)} query/queries missing aliases."
            )

            retry_messages = [
                {"role": "system", "content": "You are an expert Data Engineer"},
                {"role": "user", "content": initial_message},
                {"role": "assistant", "content": json.dumps({"queries": bad_queries_clean})},
                {"role": "user", "content": feedback},
            ]

            try:
                response = self.router.completion(
                    model=self.config["model"],
                    messages=retry_messages,
                    temperature=self.temperature,
                )
                usage = response["usage"]
                usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
                usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
                usage_total["total_tokens"] += usage.get("total_tokens", 0)

                content = response["choices"][0]["message"]["content"]
                if content is None:
                    raise ValueError("LLM returned None content during alias-retry.")

                cleaned = self._clean_json_response(content)
                json_data = self._repair_json(cleaned)
                json_data = self._normalize_response(json_data)
                retried = self.response_model.model_validate(json_data)
                retried_dict = self.clean_query_set(retried.model_dump(), table_names)

                # Re-partition: some may now be fixed, others still broken
                newly_good, bad_queries = self._partition_queries(
                    retried_dict.get("queries", []), all_aliases
                )
                good_queries.extend(newly_good)

            except (json.JSONDecodeError, ValidationError, Exception) as e:
                all_errors.append(
                    f"Table-alias retry {alias_attempt + 1} failed with {type(e).__name__}: {e}"
                )
                time.sleep(self.retry_delay)

        # After all alias retries, silently drop any still-failing queries
        if bad_queries:
            all_errors.append(
                f"Dropping {len(bad_queries)} query/queries that still fail alias check "
                f"after {MAX_TABLE_ALIAS_RETRIES} retries."
            )

        return (
            {"queries": good_queries},
            usage_total,
            all_errors,
            self.config["model"],
        )

    def _normalize_response(self, json_data):
        if isinstance(json_data, list):
            return {"queries": json_data}
        if isinstance(json_data, dict):
            if "queries" not in json_data:
                if len(json_data) > 0:
                    return {"queries": [json_data]}
            return json_data
        return {"queries": [json_data]}

    def add_suffix_constraint(self, prompt, dataframes, table_names, involved_cols):
        from itertools import combinations
        common_cols_info = []
        for (name_a, df_a), (name_b, df_b) in combinations(zip(table_names, dataframes), 2):
            common = set(df_a.columns) & set(df_b.columns)
            common_cols_info.append(
                f"  {name_a} ∩ {name_b}: {sorted(common) if common else '(no common columns)'}"
            )
        constraint = (
            "### Suffix Constraint Information\n"
            "Columns suffixed after merge (ONLY these get a suffix):\n"
            + "\n".join(common_cols_info) + "\n\n"
            + (
                f"Join key columns (NEVER suffixed — kept as-is after merge):\n"
                f"  {involved_cols}\n\n"
                if involved_cols else ""
            )
            + "All other columns keep their original name — do not add any suffix to them.\n"
        )
        return f"{prompt}\n{constraint}"

    @staticmethod
    def clean_table_names(query_code: str, aliases: dict) -> str:
        inverted = {v: k for k, v in aliases.items()}

        result = query_code
        for table_name, alias in inverted.items():
            underscore_variant = table_name.replace("-", "_")

            for name_to_replace in {table_name, underscore_variant}:
                result = re.sub(
                    r'(?<![.\w])' + re.escape(name_to_replace) + r'(?!\w)',
                    alias,
                    result,
                )
        return result

    @staticmethod
    def clean_query_set(query_set: dict, aliases: dict) -> dict:
        cleaned = {**query_set, "queries": []}
        for query in query_set.get("queries", []):
            q = {**query, "code": LLMClientStatementGenerator.clean_table_names(query["code"], aliases)}
            cleaned["queries"].append(q)
        return cleaned