import importlib
import json
import os
import re
import time
from io import StringIO
from pathlib import Path
from typing import Any, Optional, Type

import litellm
import pandas as pd
import polars as pl
import yaml
from litellm import completion, Router
from pydantic import BaseModel, ValidationError

from .LLMClientStructured import LLMClientStructured
from .utility.alias_substitution import AliasSubstitution
from .utility.message_builder import ClientMessageBuilder
from .prompting import DatasetDescription, _load_prompt
from .structured_outputs import QuerySet, Query, TableAnalyses, QueryPlan
import duckdb
import sys


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
        self.table_analyzer_model = self._load_pydantic_response_model("table_analyzer")
        self.query_planner_model = self._load_pydantic_response_model("query_planner")
        self.fallback_response_model = self._load_pydantic_response_model("fallback_querying")

    def _response_root_key(self, model_name: str) -> str | None:
        if model_name == "table_analyzer":
            return "tables"
        if model_name == "query_planner":
            return None
        return "queries"

    def _complete_with_model(self, prompt: str, response_model_name: str, **kwargs) -> Any:
        original_model = self.response_model
        self.response_model = self._load_pydantic_response_model(response_model_name)
        try:
            return super().complete(
                prompt,
                root_key=self._response_root_key(response_model_name),
                **kwargs,
            )
        finally:
            self.response_model = original_model

    def _build_table_analysis_prompt(
        self,
        alias: str,
        metadata: dict,
        columns: list[str],
        sample: list[dict],
        languages: list[str],
    ) -> str:
        return (
            "Analyze a single table and return a JSON object matching the provided schema. "
            "Do not include any explanatory text outside the JSON.\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "- Extract up to 10 keywords (max) that best capture the essential concepts and domain concepts in this table.\n"
            "- Keywords should be meaningful terms that a domain expert would use to describe what this table contains.\n"
            "- Keywords are crucial for helping non-expert users understand what data this table contains.\n"
            f"\n\nAlias: {alias}"
            f"\nDetected languages: {json.dumps(languages, ensure_ascii=False)}"
            f"\nColumns:\n{json.dumps(columns, indent=2, ensure_ascii=False)}"
            f"\n\nMetadata:\n{json.dumps(metadata, indent=2, ensure_ascii=False, default=str)}"
            f"\n\nSample rows:\n{json.dumps(sample, indent=2, ensure_ascii=False, default=str)}"
        )

    def _run_table_analysis(
        self,
        alias: str,
        metadata: dict,
        columns: list[str],
        sample: list[dict],
        languages: list[str],
    ) -> dict:
        prompt = self._build_table_analysis_prompt(alias, metadata, columns, sample, languages)
        result, _ = self._complete_with_model(prompt, "table_analyzer")
        tables = result.get("tables", [])
        if tables:
            return tables[0]
        return {"alias": alias, "table_description": "", "table_keywords": []}

    def _build_query_planner_prompt(
        self,
        analyses: list[dict],
        aliases: dict,
        matches: Any,
        languages: list[str],
    ) -> str:
        return (
            "Use the following per-table analysis results to produce a query plan, "
            "a business question, and keyword-level guidance. Return only valid JSON.\n\n"
            "IMPORTANT INSTRUCTIONS FOR QUESTION GENERATION:\n"
            "- Generate a question as if asked by an AVERAGE, NON-EXPERT USER who:\n"
            "  * Does NOT know table or column names\n"
            "  * Has general business knowledge\n"
            "  * Phrases questions naturally and conversationally\n"
            "- USE TABLE KEYWORDS STRATEGICALLY: incorporate the keywords from each table's analysis "
            "to help pinpoint to the correct tables without naming them directly.\n"
            "  Example: if a table has keywords 'sales', 'revenue', 'orders', use them naturally in the question.\n"
            "- Generate up to 10 keywords (max) for the question that capture its intent and incorporate table keywords.\n"
            "- The question should read naturally and allow someone to infer which tables are being queried.\n"
            f"\n\nAliases: {json.dumps(list(aliases.keys()), indent=2, ensure_ascii=False)}"
            f"\n\nTable analysis:\n{json.dumps({'tables': analyses}, indent=2, ensure_ascii=False, default=str)}"
            f"\n\nMatch requirements:\n{json.dumps(matches, indent=2, ensure_ascii=False, default=str)}"
            f"\n\nDetected languages: {json.dumps(languages, ensure_ascii=False)}"
        )

    def _run_query_planner(
        self,
        analyses: list[dict],
        aliases: dict,
        matches: Any,
        languages: list[str],
    ) -> dict:
        prompt = self._build_query_planner_prompt(analyses, aliases, matches, languages)
        result, _ = self._complete_with_model(prompt, "query_planner")
        return result

    def _enrich_prompt_for_final_generation(
        self,
        prompt: str,
        table_analyses: list[dict],
        planner: dict,
    ) -> str:
        return (
            f"{prompt}\n\n"
            "### TABLE-LEVEL ANALYSIS\n"
            f"{json.dumps({'tables': table_analyses}, indent=2, ensure_ascii=False, default=str)}\n\n"
            "### QUERY PLANNING GUIDANCE\n"
            f"{json.dumps(planner, indent=2, ensure_ascii=False, default=str)}\n\n"
            "### KEYWORD CONSTRAINTS\n"
            "- Each table must provide up to 10 keywords (max) that capture its domain concepts.\n"
            "- Each question must include up to 10 keywords (max) that incorporate table keywords when relevant.\n"
            "- Keywords help non-expert users understand which tables the query touches.\n\n"
            "When generating the final output, include the following fields in each query:\n"
            "- query_plan\n"
            "- question_keywords (max 10, should include relevant table keywords)\n"
            "- translated_question_keywords (max 10)\n"
            "- For each table, description, keywords (max 10), and translated_keywords (max 10) copied from the table analysis\n"
            "- Ensure the question is phrased as a non-expert user would ask it, using table keywords as hints.\n"
            "Return only valid JSON matching the expected schema."
        )

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
        matches=None,
        metadata=None,
        languages=None,
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

        # Build analysis context for each table before generating the final queries.
        metadata_list = metadata or []
        if isinstance(metadata_list, dict):
            metadata_list = [metadata_list.get(alias, {}) for alias in table_names]

        detected_languages = languages or []
        if isinstance(detected_languages, str):
            detected_languages = [detected_languages]

        table_analyses = []
        for idx, (alias, _) in enumerate(table_names.items()):
            df = dataframes[idx]
            table_metadata = metadata_list[idx] if idx < len(metadata_list) else {}
            table_analyses.append(
                self._run_table_analysis(
                    alias=alias,
                    metadata=table_metadata,
                    columns=[f"{col} ({df[col].dtype})" for col in df.columns],
                    sample=df.head(3).to_dict(orient="records"),
                    languages=detected_languages,
                )
            )

        plan = self._run_query_planner(
            table_analyses,
            table_names,
            matches=matches,
            languages=detected_languages,
        )

        enriched_prompt = self._enrich_prompt_for_final_generation(
            prompt,
            table_analyses,
            plan,
        )

        initial_message = self.reform_prompt_constraint(enriched_prompt)

        message_builder = ClientMessageBuilder()
        system_prompt = "You are an expert Data Engineer"
        messages = message_builder.build(system_prompt, initial_message)
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

                # Merge table-level analysis into each generated query's tables[] so
                # keywords/descriptions are preserved centrally (not duplicated per-query).
                analyses_map = {t.get('alias') or t.get('name'): t for t in table_analyses}
                for q in result_dict.get('queries', []):
                    if not isinstance(q.get('tables'), list):
                        continue
                    for tbl in q['tables']:
                        tbl_name = tbl.get('name') or tbl.get('alias')
                        analysis = analyses_map.get(tbl_name, {})
                        if 'description' not in tbl or not tbl.get('description'):
                            tbl['description'] = analysis.get('table_description') or analysis.get('description', tbl.get('description', ''))
                        if 'keywords' not in tbl or not tbl.get('keywords'):
                            tbl['keywords'] = analysis.get('table_keywords') or analysis.get('keywords', tbl.get('keywords', []))
                        if 'translated_keywords' not in tbl:
                            tbl['translated_keywords'] = analysis.get('translated_keywords', [])

                # Successfully parsed — break out and move to alias validation
                break

            except json.JSONDecodeError as e:
                all_errors.append(f"JSONDecodeError on attempt {attempt + 1}: {e}")
                error_msg = self._format_json_error(last_content, e)
                if attempt < self.max_retries - 1:
                    pydantic = self.reform_prompt_constraint("")
                    # Rebuild from scratch with error feedback
                    error_prompt = (
                        f"{initial_message}\n\n"
                        f"Your previous output could not be parsed as JSON:\n{error_msg}\n{pydantic}"
                    )
                    messages = message_builder.build(system_prompt, error_prompt)
                    time.sleep(self.retry_delay)

            except ValidationError as e:
                all_errors.append(f"ValidationError on attempt {attempt + 1}: {e}")
                error_msg = self._format_validation_error(e)
                if attempt < self.max_retries - 1:
                    pydantic = self.reform_prompt_constraint("")
                    # Rebuild from scratch with error feedback
                    error_prompt = (
                        f"{enriched_prompt}\n\n"
                        f"Your previous output failed schema validation:\n{error_msg}\n{pydantic}"
                    )
                    messages = message_builder.build(system_prompt, error_prompt)
                    time.sleep(self.retry_delay)

            except Exception as e:
                all_errors.append(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    # Rebuild from scratch with error feedback
                    error_prompt = (
                        f"{enriched_prompt}\n\n"
                        f"An error occurred: {e}\n"
                        f"Return a JSON object matching:\n"
                        f"{json.dumps(self.response_model.model_json_schema(), indent=2)}"
                    )
                    messages = message_builder.build(system_prompt, error_prompt)
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

            retry_messages = message_builder.build(
                system_prompt,
                f"{initial_message}\n\n"
                f"Previous output (with issues):\n{json.dumps({'queries': bad_queries_clean})}\n\n"
                f"{feedback}",
            )

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

                # Merge table-level analysis into retried results as well
                analyses_map = {t.get('alias') or t.get('name'): t for t in table_analyses}
                for q in retried_dict.get('queries', []):
                    if not isinstance(q.get('tables'), list):
                        continue
                    for tbl in q['tables']:
                        tbl_name = tbl.get('name') or tbl.get('alias')
                        analysis = analyses_map.get(tbl_name, {})
                        if 'description' not in tbl or not tbl.get('description'):
                            tbl['description'] = analysis.get('table_description') or analysis.get('description', tbl.get('description', ''))
                        if 'keywords' not in tbl or not tbl.get('keywords'):
                            tbl['keywords'] = analysis.get('table_keywords') or analysis.get('keywords', tbl.get('keywords', []))
                        if 'translated_keywords' not in tbl:
                            tbl['translated_keywords'] = analysis.get('translated_keywords', [])

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

    def _normalize_response(self, json_data: Any, root_key: str | None = "queries") -> Any:
        if root_key is None:
            return json_data
        if isinstance(json_data, list):
            return {root_key: json_data}
        if isinstance(json_data, dict):
            if root_key not in json_data:
                if len(json_data) > 0:
                    return {root_key: [json_data]}
            return json_data
        return {root_key: [json_data]}

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
        """Deprecated: use AliasSubstitution.substitute() instead."""
        sub = AliasSubstitution(aliases)
        return sub.substitute(query_code)

    @staticmethod
    def clean_query_set(query_set: dict, aliases: dict) -> dict:
        """Apply alias substitution to all queries in a query set.

        Uses AliasSubstitution to replace original table names with canonical
        aliases in the code field of each query, as well as question,
        motivation, and tables[].name fields.
        """
        sub = AliasSubstitution(aliases)
        cleaned = {**query_set, "queries": []}
        for query in query_set.get("queries", []):
            cleaned["queries"].append(sub.substitute_query(query))
        return cleaned