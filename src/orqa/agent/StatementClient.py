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
from .validators.SQLValidator import SQLValidator
from .validators.PandasValidator import PandasValidator
import re

SQL_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING",
    "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN", "OUTER JOIN", "FULL JOIN", "CROSS JOIN",
    "UNION", "UNION ALL", "INTERSECT", "EXCEPT",
    "SUM", "AVG", "MIN", "MAX", "COUNT", "RANK", "ROW_NUMBER", "DENSE_RANK", "CORR",
    "CAST", "COALESCE", "NULLIF", "CASE", "WHEN", "THEN", "ELSE",
    "DISTINCT", "LIMIT", "OFFSET", "WITH", "AS", "ON", "AND", "OR", "NOT", "IN", "EXISTS",
}

PANDAS_KEYWORDS = {
    "merge", "join", "concat", "groupby", "agg", "aggregate","corr",
    "sum", "mean", "avg", "min", "max", "count", "nunique", "rank",
    "sort_values", "sort_index", "drop_duplicates", "fillna", "dropna",
    "apply", "map", "filter", "query", "where", "assign", "pivot", "melt",
    "stack", "unstack", "explode", "resample", "rolling", "expanding",
}



class LLMClientStatementGenerator(LLMClientStructured):
    """
    LiteLLM client with YAML configuration and structured output support.
    """

    def __init__(self, config_path: Path):
        super().__init__(config_path, "querying")
        self.fallback_response_model = self._load_pydantic_response_model("fallback_querying")

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
        #    Heuristic: odd number of unescaped double-quotes → append one.
        unescaped_quotes = re.findall(r'(?<!\\)"', fixed)
        if len(unescaped_quotes) % 2 != 0:
            fixed = fixed.rstrip() + '"'

        # 3. Count open braces/brackets and close any that were never closed
        #    (handles truncated LLM output).
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
        This fixes the common LLM output pattern of:
            {"code": "df.merge(...)\n.head(5)"}
        where \n is a real newline, not the two-character escape sequence.
        """
        result = []
        in_string = False
        i = 0
        while i < len(content):
            ch = content[i]
            if in_string:
                if ch == '\\':
                    # Consume the escape sequence as-is (already valid JSON)
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

    def complete(
        self,
        prompt: str,
        dataframes, table_names, typology="SQL", involved_cols=None, feedback:list = None
    ) -> Any:
        usage_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        initial_message = self.reform_prompt_constraint(prompt)
        count = 0
        # FIX 2: do NOT bake `messages` into completion_args up-front.
        # Instead, update completion_args["messages"] right before every
        # router.completion call so that retry feedback is actually sent.
        messages = [
            {"role": "system", "content": "You are an expert Data Engineer"},
            {"role": "user", "content": initial_message},
        ]
        if feedback is not None:
            messages.extend(feedback)
        last_content = ""
        last_error = None
        good_queries: dict[str, dict] = {}
        all_errors = []

        for attempt in range(self.max_retries):
            try:
                # FIX 2: always build completion_args from the current messages list.
                completion_args = {
                    "model": self.config["model"],
                    "messages": messages,
                    "temperature": self.temperature,
                    #"max_tokens": 16000,
                }

                response = self.router.completion(**completion_args)
                usage = response["usage"]
                usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
                usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
                usage_total["total_tokens"] += usage.get("total_tokens", 0)
                content = response["choices"][0]["message"]["content"]
                # Guard: some models return None for content (e.g. tool-call-only responses)
                if content is None:
                    raise ValueError(
                        "LLM returned None content – the model may have emitted a "
                        "tool-call response instead of a text completion."
                    )
                last_content = content
                ##print(last_content)
                cleaned_content = self._clean_json_response(content)
                try:
                    # FIX 1 & 3: use the robust repair pipeline
                    json_data = self._repair_json(cleaned_content)
                    json_data = self._normalize_response(json_data)
                    result = self.response_model.model_validate(json_data)
                    result = result.model_dump()
                    outcome, errors, accepted_queries, error_messages = self.validate_queries(
                        dataframes, result, table_names, typology
                    )
                    for key, accepted_query in accepted_queries.items():
                        #accepted_query["keywords"] = self.count_keywords(accepted_query["code"], typology)
                        accepted_query["id"] = f"{count}"
                        good_queries[f"{count}"] = accepted_query
                        count = count + 1

                    if not outcome:
                        # FIX 2: reassign messages and it will be picked up next iteration
                        messages = [
                            {"role": "system", "content": "You are an expert Data Engineer."},
                            {"role": "user", "content": prompt},
                        ]
                        messages.extend(errors)
                        all_errors.extend(error_messages)
                        pydantic = self.reform_prompt_constraint("")
                        messages.append({"role": "user", "content": pydantic})
                        continue

                    return (
                        {"queries": list(good_queries.values())},
                        usage_total,
                        [e if isinstance(e, str) else str(e) for e in all_errors],
                        self.config["model"],
                    )

                except json.JSONDecodeError as e:
                    last_error = e
                    all_errors.append(f"Error JSONDecodeError: {e}")
                    error_msg = self._format_json_error(cleaned_content, e)
                    if attempt < self.max_retries - 1:
                        # FIX 2: rebuild messages so the next iteration sends them
                        messages = [
                            {"role": "system", "content": "You are an expert Data Engineer, below are listed the instructions for generating and correcting queries."},
                            {"role": "user", "content": initial_message},
                            {"role": "system", "content": last_content},
                        ]
                        pydantic = self.reform_prompt_constraint("")
                        messages.append({
                            "role": "user",
                            "content": f"You generated a bad formatted output, that gave the following error message:\n{error_msg}\n{pydantic}",
                        })
                        time.sleep(self.retry_delay)
                        #print(last_error)
                        continue

                except ValidationError as e:
                    last_error = e
                    all_errors.append(f"Error ValidationError: {e}")
                    error_msg = self._format_validation_error(e)
                    if attempt < self.max_retries - 1:
                        # FIX 2: rebuild messages
                        messages = [
                            {"role": "system", "content": "You are an expert Data Engineer."},
                            {"role": "user", "content": prompt},
                            {"role": "system", "content": last_content},
                        ]
                        pydantic = self.reform_prompt_constraint("")
                        messages.append({
                            "role": "user",
                            "content": f"Generated queries are not valid, the error message generated:\n{error_msg}\n{pydantic}",
                        })
                        #print(last_error)
                        time.sleep(self.retry_delay)
                        continue

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # FIX 2: rebuild messages
                    messages = [
                        {"role": "system", "content": "You are an expert Data Engineer"},
                        {"role": "user", "content": prompt},
                        {"role": "system", "content": last_content},
                    ]
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Generated queries that are not valid, error message generated\n{e}\n"
                            f"Make a unique JSON compliant to the Pydantic format:\n"
                            f"{json.dumps(self.response_model.model_json_schema(), indent=2)}"
                        ),
                    })
                    #print(last_error)
                    time.sleep(self.retry_delay)

        return (
            {"queries": list(good_queries.values())},
            usage_total,
            [e if isinstance(e, str) else str(e) for e in all_errors],
            self.config["model"],
        )

    def count_keywords(self, query: str, kind: str = "SQL") -> dict[str, int]:
        keywords = SQL_KEYWORDS if kind == "SQL" else PANDAS_KEYWORDS
        query_upper = query.upper() if kind == "SQL" else query
        counts = {}
        for kw in sorted(keywords):
            kw_pattern = kw.upper() if kind == "SQL" else kw
            pattern = rf'\b{re.escape(kw_pattern)}\b'
            matches = re.findall(pattern, query_upper, flags=re.IGNORECASE)
            if matches:
                counts[kw] = len(matches)
        return counts

    def validate_queries(self, dataframes, result, table_names, type):
        if type == "SQL":
            return self.validate_sql_queries(dataframes, result, table_names)
        else:
            return self.validate_dataframe_queries(dataframes, result, table_names)

    def validate_dataframe_queries(self, dataframes, result, table_names):
        validator = PandasValidator(dataframes, list(table_names.keys()), table_names)
        return validator.validate_queries(result)

    def validate_sql_queries(self, dataframes, result, table_names):
        validator = SQLValidator(dataframes, list(table_names.keys()), table_names)
        return validator.validate_queries(result)

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