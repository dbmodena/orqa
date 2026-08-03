import importlib
import json
import logging
import os
import re
import time
from io import StringIO
from pathlib import Path
from typing import Any, Type

import litellm
import pandas as pd
import polars as pl
import yaml
from litellm import completion, Router
from pydantic import BaseModel, ValidationError

from ..llm_client.LLMClientStructured import LLMClientStructured
from ..utility.alias_substitution import AliasSubstitution
from ..utility.message_builder import ClientMessageBuilder, sanitize_messages
from ..prompting import DatasetDescription, GenerationEnrichmentPrompt
from ..utility.structured_outputs import Query, TableAnalyses, QueryPlan
from ...utils.pipeline_logger import PipelineLogger
import duckdb
import sys

logger = logging.getLogger(__name__)


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
        self._log = PipelineLogger()

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

    def enrich_prompt_with_table_analysis(
        self,
        prompt: str,
        table_analyses: list[dict],
    ) -> str:
        """Append the table-analysis enrichment block to ``prompt``.

        Called ONCE per run on the run-stable base prompt (see
        ``StatementOrchestrator._run`` in ``agent.py``) — before the per-plan generation loop — so the
        enrichment sits in the shared, cacheable prefix of every generation
        call rather than trailing each call's unique plan section.
        """
        enrichment = GenerationEnrichmentPrompt().update(
            table_analysis=json.dumps(
                {"tables": table_analyses}, indent=2, ensure_ascii=False, default=str
            ),
            planning_section="",
        )
        return f"{prompt}\n\n{enrichment}"

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

    def _finalize_query_set(self, queries: list[dict]) -> dict:
        """Validate every generated query as a :class:`Query`, return a query-set dict.

        This is the final gate before ``complete()`` hands its result back to the
        caller: each dict in ``queries`` (already schema-checked once at parse
        time via ``self.response_model.model_validate``, but possibly mutated
        since — by ``clean_query_set``, the table-analysis merge, or the
        table-alias retry loop) is re-validated as a :class:`Query`. A query
        that fails re-validation is dropped (logged, not raised) rather than
        failing the whole batch — one malformed query shouldn't sink every
        other query in the same run.

        ``Query`` itself no longer carries the planner-owned question-level
        metadata (``query_plan``, ``question_keywords``,
        ``translated_question_keywords``, ``translated_question``,
        ``detected_language``, ``topic``, ``story`` — see
        ``prompting.models.SQLQueryPlan``/``PandasQueryPlan``), which
        ``complete()`` merges onto each query dict from the plan before this
        method runs. Re-validating strictly through ``Query`` would silently
        drop those fields, so each validated query's canonical fields are
        merged back onto the ORIGINAL dict (which still carries them) rather
        than the other way around.
        """
        validated: list[dict] = []
        for q in queries:
            try:
                query_obj = Query.model_validate(q)
            except ValidationError as exc:
                logger.warning(
                    "Dropping a generated query that failed Query validation "
                    "in the final gate (client_id=%r): %s",
                    q.get("client_id") if isinstance(q, dict) else None,
                    exc,
                )
                continue
            merged = dict(q) if isinstance(q, dict) else {}
            merged.update(query_obj.model_dump())
            validated.append(merged)
        return {"queries": validated}

    def _enforce_single_query(self, queries: list[dict]) -> list[dict]:
        """Keep at most one query — every generation call is bound to a single plan.

        Every call to :meth:`complete` is scoped to exactly one query plan
        (``precomputed_plan`` — see ``QueryPlanner.plan_batch``) and the prompt
        asks for exactly one query in return. If the model nonetheless returns
        more than one, keep only the first — the plan this call was given
        only applies to one query, so any extras aren't traceable to a
        plan of their own and would silently break the one-plan-to-one-query
        contract downstream.
        """
        if len(queries) <= 1:
            return queries
        logger.warning(
            "Generation call bound to a single plan returned %d queries; "
            "keeping only the first and dropping the rest.",
            len(queries),
        )
        return queries[:1]

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
        *,
        typology: str = "SQL",
        involved_cols=None,
        matches=None,
        metadata=None,
        languages=None,
        precomputed_plan: dict,
    ) -> Any:
        """
        Generate the initial set of queries from *prompt*, then validate that
        every query references all table aliases.  Queries that pass are kept;
        only the failing ones are sent back for correction (up to
        ``MAX_TABLE_ALIAS_RETRIES`` extra rounds).  After all retries, any
        still-failing queries are silently dropped and only the good ones are
        returned.

        Args:
            precomputed_plan: The specific plan (dict form) this generation
                call is bound to — one plan out of ``QueryPlanner.plan_batch``'s
                several independent plans. It is already injected into
                ``prompt`` upstream (by ``GenerationCoordinator``), so no plan
                section is added here and the model sees exactly one plan.
                Its own ``tables`` (name/reason/columns_involved/description/
                keywords, already judged by the plan panel) is what gets
                stamped onto every generated query below — table-level
                analysis is no longer merged in here separately, since the
                generation LLM no longer produces a ``tables`` field at all.
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

        # Reuse the caller's already-computed plan verbatim — this method never
        # runs its own query-planner LLM call. Both the plan AND the
        # table-analysis enrichment are already injected into `prompt`
        # upstream (GenerationCoordinator adds the plan; the orchestrator
        # enriches the run-stable base prompt once via
        # ``enrich_prompt_with_table_analysis``), so nothing is appended here
        # and the run's generation calls keep their shared prompt prefix.
        plan = precomputed_plan
        self._log.query_plan(plan)

        initial_message = prompt

        message_builder = ClientMessageBuilder()
        # The output-schema constraint is static per response model, so it
        # rides in the system message — ahead of all dynamic content — where
        # it extends the cacheable prefix shared by every generation call.
        system_prompt = f"You are an expert Data Engineer.\n\n{self.schema_constraint()}"
        messages = message_builder.build(system_prompt, initial_message)
        last_content = ""

        # ── Phase 1: initial generation with JSON/Pydantic retry loop ──────────
        for attempt in range(self.max_retries):
            try:
                completion_args = {
                    "model": self.config["model"],
                    "messages": sanitize_messages(messages),
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

                # Copy the planner-owned question-level metadata onto every
                # generated query. These fields are no longer part of the
                # Query schema (the generation LLM never produces them) —
                # they were already decided once, during planning, and every
                # query bound to this plan/call shares that same metadata.
                # `tables` (name/reason/columns_involved/description/keywords)
                # is one of them: the plan's own `tables[]` (List[Table]) was
                # already judged by the plan panel (PlanJudgment.table_check)
                # and checked for full alias coverage by
                # QueryPlanner.validate_plan, so it is simply copied here
                # rather than asked of the generation LLM a second time.
                plan_fields = {
                    "query_plan": plan.get("query_plan", ""),
                    "question_keywords": plan.get("question_keywords", []),
                    "translated_question_keywords": plan.get("translated_question_keywords", []),
                    "translated_question": plan.get("translated_question", ""),
                    "detected_language": plan.get("detected_language", ""),
                    "topic": plan.get("topic", ""),
                    "story": plan.get("story", ""),
                    "tables": plan.get("tables", []),
                    # Structural-complexity tier, decided at planning time and
                    # judged by the plan panel (PlanJudgment.difficulty_check)
                    # — no longer part of the generation LLM's own output.
                    "difficulty": plan.get("difficulty", ""),
                    # Plan-declared result contract, judged by the plan panel
                    # (PlanJudgment.expected_result_check) and mechanically
                    # enforced against the executed result by the validators
                    # (QueryValidator._check_expected_result_type).
                    "expected_result_type": plan.get("expected_result_type", ""),
                    "expected_result_description": plan.get("expected_result_description", ""),
                }
                for q in result_dict.get('queries', []):
                    q.update(plan_fields)

                result_dict['queries'] = self._enforce_single_query(
                    result_dict.get('queries', [])
                )

                # Successfully parsed — break out and move to alias validation
                break

            except json.JSONDecodeError as e:
                all_errors.append(f"JSONDecodeError on attempt {attempt + 1}: {e}")
                error_msg = self._format_json_error(last_content, e)
                if attempt < self.max_retries - 1:
                    # Rebuild from scratch with error feedback appended AFTER
                    # the unchanged initial message, so the retry still shares
                    # the same cached prompt prefix. The schema itself already
                    # rides in the (static) system message.
                    error_prompt = (
                        f"{initial_message}\n\n"
                        f"Your previous output could not be parsed as JSON:\n{error_msg}"
                    )
                    messages = message_builder.build(system_prompt, error_prompt)
                    time.sleep(self.retry_delay)

            except ValidationError as e:
                all_errors.append(f"ValidationError on attempt {attempt + 1}: {e}")
                error_msg = self._format_validation_error(e)
                if attempt < self.max_retries - 1:
                    # Rebuild from scratch with error feedback
                    error_prompt = (
                        f"{initial_message}\n\n"
                        f"Your previous output failed schema validation:\n{error_msg}"
                    )
                    messages = message_builder.build(system_prompt, error_prompt)
                    time.sleep(self.retry_delay)

            except Exception as e:
                all_errors.append(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    # Rebuild from scratch with error feedback
                    error_prompt = (
                        f"{initial_message}\n\n"
                        f"An error occurred: {e}\n"
                        f"Return a JSON object matching the schema in the instructions."
                    )
                    messages = message_builder.build(system_prompt, error_prompt)
                    time.sleep(self.retry_delay)
        else:
            # All JSON/Pydantic retries exhausted
            return (
                self._finalize_query_set([]),
                usage_total,
                all_errors,
                self.config["model"],
                plan,
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
                    messages=sanitize_messages(retry_messages),
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

                # Re-apply the same planner-owned fields (including `tables`)
                # the phase-1 queries got above — this is a FRESH LLM response,
                # re-validated through the (tables-less) Query schema, so it
                # never carries them on its own.
                for q in retried_dict.get('queries', []):
                    q.update(plan_fields)

                # Re-partition: some may now be fixed, others still broken
                newly_good, bad_queries = self._partition_queries(
                    retried_dict.get("queries", []), all_aliases
                )
                good_queries.extend(self._enforce_single_query(newly_good))

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
            self._finalize_query_set(good_queries),
            usage_total,
            all_errors,
            self.config["model"],
            plan,
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