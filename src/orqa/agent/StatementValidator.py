"""
StatementValidator
==================
Decoupled validation + correction layer that sits between StatementClient
(initial generation) and StatementJudge (quality judgment).

Responsibilities
----------------
1. Assign every incoming query a deterministic positional integer id (1, 2, 3…)
   so the correction LLM always works with simple, stable identifiers.
2. Run the static Python validator (PandasValidator / SQLValidator).
3. Collect judge feedback for queries rejected in a previous judge iteration.
4. Maintain two lists throughout the retry loop:
     - approved  : queries that passed static validation (grown each attempt)
     - pending   : queries still failing (shrunk each attempt, or discarded on
                   the final attempt)
5. When pending is non-empty and attempts remain, call the correction LLM once
   with a fully self-contained prompt: schema context + failing queries + errors.
6. Return (approved, usage, errors) — queries still failing after the final
   cycle are dropped from the result (not returned as best-effort).

Design notes
------------
* IDs are reassigned to sequential integers at the top of validate_and_correct
  and a mapping back to the caller's original ids is kept so the output list
  can be returned with the original ids restored.
* The correction LLM never sees UUIDs or auto_* strings — only small integers —
  which makes id round-tripping reliable.
* If the LLM omits or mangles a query id, the original query is silently kept
  for that slot; no warning is raised since the next validation attempt will
  catch any remaining errors.
"""
import logging
import re
import json
import time
from pathlib import Path

from pydantic import ValidationError

from .alias_substitution import AliasSubstitution
from .error_formatter import ErrorFormatter
from .message_builder import ValidatorMessageBuilder
from .StatementClient import LLMClientStructured
from .prompting import PandasValidatorCorrectionPrompt, SQLValidatorCorrectionPrompt
from .validators.SQLValidator import SQLValidator
from .validators.PandasValidator import PandasValidator

logger = logging.getLogger(__name__)

# Width of the separator lines used in console output
_SEP_WIDTH = 68


class LLMStatementValidator(LLMClientStructured):
    """
    Validates generated queries with static validators (Pandas / SQL) and, when
    errors are found, corrects them via a dedicated LLM correction call.

    Also incorporates judge feedback so that queries rejected by the judge are
    corrected in the same LLM call as any static validation failures.

    The validate-and-correct cycle is retried up to MAX_VALIDATION_RETRIES times.
    Each attempt moves passing queries into the approved list; the pending list
    is replaced by the corrected-but-still-failing remainder.  On the final
    cycle any still-failing queries are dropped from the result so the caller
    only receives validated output.
    """

    MAX_VALIDATION_RETRIES: int = 3

    def __init__(self, config_path: Path, kind: str):
        super().__init__(config_path, "querying")
        self.fallback_response_model = self._load_pydantic_response_model("fallback_querying")
        self.kind = kind
        if kind == "PANDAS":
            self._correction_prompt = PandasValidatorCorrectionPrompt()
        else:
            self._correction_prompt = SQLValidatorCorrectionPrompt()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_and_correct(
        self,
        queries: list,
        dataframes: list,
        aliases: dict,
        table_schemas: str,
        judge_feedback: list | None = None,
    ) -> tuple[list, dict, list]:
        """
        Validate *queries* and correct any failures, retrying up to
        MAX_VALIDATION_RETRIES times.

        Args:
            queries:        Query dicts to validate.
            dataframes:     Pre-loaded DataFrames for the static validator.
            aliases:        Alias → file-path mapping.
            table_schemas:  Pre-formatted schema string for the correction prompt.
            judge_feedback: Structured feedback from JudgementResponseAgent,
                            each element: {"id", "query", "error"}.
                            The ids here are the caller's original ids (may be
                            auto_* UUIDs); they are remapped internally.

        Returns:
            (final_queries, token_usage_dict, error_strings)
            final_queries preserves the caller's original ids.
        """
        if not queries:
            return queries, _empty_usage(), []

        # ------------------------------------------------------------------
        # Alias substitution: centralized, applied immediately on entry.
        # ------------------------------------------------------------------
        alias_sub = AliasSubstitution(aliases)

        # ------------------------------------------------------------------
        # Assign deterministic sequential integer ids for the duration of this
        # call.  Keep a mapping so we can restore original ids before returning.
        # ------------------------------------------------------------------
        original_id_by_pos: dict[int, object] = {}   # pos_id → caller's original id
        print(f"[Validator] actual aliases {aliases}")
        pending: list[dict] = []
        print("==================Pending queries ====================================")
        for pos, q in enumerate(queries, start=1):
            q_copy = dict(q)
            original_id_by_pos[pos] = q_copy.get("id")
            q_copy["id"] = pos
            q_copy["code"] = alias_sub.substitute(q_copy["code"])
            pending.append(q_copy)
            print(f"#{q_copy["id"]} {q_copy["question"]}")
            print(f"## code: {q_copy["code"]}")

        # Verification: log warning and re-apply if original names still detected
        for q in pending:
            remaining = alias_sub.verify(q.get("code", ""))
            if remaining:
                logger.warning(
                    "[Validator] Original names still present after substitution "
                    "in query #%s: %s — re-applying.",
                    q.get("id"), remaining,
                )
                q["code"] = alias_sub.substitute(q["code"])

        # Build reverse map: caller's original id (as str) → positional int
        orig_to_pos: dict[str, int] = {
            str(orig_id): pos for pos, orig_id in original_id_by_pos.items()
        }

        # Remap judge_feedback ids to positional integers
        remapped_judge_feedback: list[dict] | None = None
        if judge_feedback:
            remapped_judge_feedback = []
            for fb in judge_feedback:
                fb_copy = dict(fb)
                pos = orig_to_pos.get(str(fb.get("id")))
                if pos is not None:
                    fb_copy["id"] = pos
                remapped_judge_feedback.append(fb_copy)

        self._print_judge_feedback(remapped_judge_feedback)

        # ------------------------------------------------------------------
        # Main retry loop
        # ------------------------------------------------------------------
        usage_total = _empty_usage()
        all_errors: list[str] = []
        approved: list[dict] = []   # queries that have cleared static validation

        # Track errors per query across attempts for escalation
        # Maps query_id (str) → list of error texts from previous attempt
        previous_errors_by_id: dict[str, list[str]] = {}

        has_judge_feedback = bool(remapped_judge_feedback)

        for cycle in range(1, self.MAX_VALIDATION_RETRIES + 1):

            is_last_cycle = (cycle == self.MAX_VALIDATION_RETRIES)

            # ==============================================================
            # JUDGE FEEDBACK PATH: cycle 1 with judge feedback present
            # Skip static validation, send queries + judge feedback to LLM
            # ==============================================================
            if has_judge_feedback and cycle == 1:
                print(
                    f"[Validator] Cycle {cycle}/{self.MAX_VALIDATION_RETRIES}: "
                    f"Judge feedback path — skipping static validation, "
                    f"sending {len(pending)} query/queries with judge feedback to LLM."
                )

                corrected_queries, tokens, llm_errors, assistant_content = (
                    self._run_judge_feedback_cycle(
                        pending, remapped_judge_feedback, table_schemas, aliases,
                    )
                )
                _accumulate_usage(usage_total, tokens)
                existing = set(all_errors)
                all_errors.extend(e for e in llm_errors if e not in existing)

                # After judge-feedback cycle, pending becomes the corrected output.
                # Next cycle will run static validation on these.
                pending = corrected_queries
                if not pending:
                    break
                continue  # proceed to next cycle (normal static validation)

            # ==============================================================
            # NORMAL VALIDATION PATH: run QueryValidator
            # ==============================================================
            passing, failing, static_errors = self._split_valid_invalid(
                pending, dataframes, aliases
            )
            approved.extend(passing)
            pending = failing

            if not pending:
                print(f"[Validator] All queries passed on cycle {cycle}.")
                break  # All passed

            if is_last_cycle:
                # No more retries — drop the still-failing queries so the caller
                # never receives invalid output.
                all_errors.append(
                    f"[Validator] {len(pending)} query/queries still failing "
                    f"after {self.MAX_VALIDATION_RETRIES} correction cycle(s); "
                    f"dropping from result."
                )
                break

            print(
                f"[Validator] Cycle {cycle}/{self.MAX_VALIDATION_RETRIES}: "
                f"{len(pending)} query/queries need correction (static failures)."
            )

            print(f"[Validator] Error↔Query pairing for cycle {cycle}:")

            corrected_queries, tokens, llm_errors, assistant_content = (
                self._run_static_correction_cycle(
                    pending, static_errors, table_schemas, dataframes, aliases,
                    previous_errors_by_id=previous_errors_by_id,
                )
            )
            _accumulate_usage(usage_total, tokens)
            existing = set(all_errors)
            all_errors.extend(e for e in llm_errors if e not in existing)

            # Store current errors for next-attempt escalation comparison
            current_errors_by_id = _build_errors_by_id(
                pending, static_errors, None
            )
            previous_errors_by_id = {
                qid: list(errs) for qid, errs in current_errors_by_id.items()
            }

            # pending becomes the corrected queries; next cycle will re-validate
            pending = corrected_queries
            if not pending:
                break

        # ------------------------------------------------------------------
        # Restore original ids before returning
        # ------------------------------------------------------------------
        pos_to_orig: dict[int, object] = original_id_by_pos  # pos → original id
        final_queries: list[dict] = []
        for q in approved:
            q_out = dict(q)
            orig = pos_to_orig.get(q_out.get("id"))
            if orig is not None:
                q_out["id"] = orig
            final_queries.append(q_out)

        # Re-sort to match the original input order (by positional id)
        orig_to_pos_int: dict[object, int] = {
            orig: pos for pos, orig in original_id_by_pos.items()
        }
        final_queries.sort(key=lambda q: orig_to_pos_int.get(q.get("id"), 0))

        return final_queries, usage_total, all_errors

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_static_correction_cycle(
        self,
        failing: list,
        static_errors: list,
        table_schemas: str,
        dataframes: list,
        aliases: dict,
        previous_errors_by_id: dict[str, list[str]] | None = None,
    ) -> tuple[list, dict, list, str | None]:
        """Normal correction cycle: format static errors and call LLM.

        Builds per-query error map, handles error escalation for recurring
        failures, formats errors with ``[STATIC VALIDATION ERROR]`` prefix via
        ``ErrorFormatter.format_static_error()``, detects recurring patterns,
        builds the correction prompt, assembles the stateless 2-message array,
        and calls the correction LLM.

        Args:
            failing: Non-empty list of query dicts that failed static validation.
            static_errors: Error messages from the static validator, positionally
                corresponding to *failing* queries.
            table_schemas: Pre-formatted schema string for the correction prompt.
            dataframes: Pre-loaded DataFrames for diagnostic context.
            aliases: Alias → file-path mapping.
            previous_errors_by_id: Error map from the previous cycle for
                escalation comparison. ``None`` on the first correction attempt.

        Returns:
            (corrected_queries, usage_dict, error_strings, assistant_content)
            assistant_content is the raw string from the LLM response (or None).
        """
        error_formatter = ErrorFormatter()
        message_builder = ValidatorMessageBuilder()

        system_prompt = "You are a helpful assistant."

        if previous_errors_by_id is None:
            previous_errors_by_id = {}

        # ---- Build per-query error map with source labels -----------------
        errors_by_id = _build_errors_by_id(
            failing, static_errors, None  # No judge feedback in normal path
        )

        # ---- Escalate errors that repeat across attempts ------------------
        for qid, error_list in errors_by_id.items():
            prev_errors = previous_errors_by_id.get(qid, [])
            if prev_errors:
                escalated = []
                for err in error_list:
                    err_core = _strip_source_prefix(err)
                    prev_cores = [_strip_source_prefix(e) for e in prev_errors]
                    if err_core in prev_cores:
                        context = _build_diagnostic_context(
                            qid, dataframes, aliases
                        )
                        escalated.append(
                            error_formatter.escalate_error(err, context)
                        )
                    else:
                        escalated.append(err)
                errors_by_id[qid] = escalated

        # ---- Log error↔query pairing -------------------------------------
        for q in failing:
            qid = str(q.get("id", "?"))
            errs = errors_by_id.get(qid, ["(none)"])
            print(f"  #{qid} → {len(errs)} error(s): {errs[0][:120]!r}{'…' if len(errs[0]) > 120 else ''}")

        # ---- Build structured entries for ErrorFormatter ------------------
        queries_with_errors_entries = _build_formatter_entries(
            failing, errors_by_id
        )

        # ---- Detect recurring patterns across all errors ------------------
        all_current_errors = []
        for errs in errors_by_id.values():
            all_current_errors.extend(errs)
        recurring = error_formatter.detect_recurring(all_current_errors)

        # ---- Use ErrorFormatter to build the correction prompt text --------
        queries_with_errors_text = error_formatter.build_correction_prompt(
            queries_with_errors_entries, recurring
        )

        # ---- Generate pydantic constraint text ----------------------------
        pydantic_constraint = self.reform_prompt_constraint("").strip()

        # ---- Render the full user message from the correction template ----
        user_content = self._correction_prompt.update(
            table_schemas=table_schemas,
            queries_with_errors=queries_with_errors_text,
            pydantic_constraint=pydantic_constraint,
        )

        # ---- Fixed-structure rebuild: stateless 2 blocks -------------------
        messages_to_send = message_builder.build(
            system_content=system_prompt,
            user_content=user_content,
        )

        corrected_queries, tokens, llm_errors, assistant_message = (
            self._correct_with_llm(messages_to_send, failing)
        )
        assistant_content = assistant_message.get("content") if assistant_message else None
        return corrected_queries, tokens, llm_errors, assistant_content

    def _run_judge_feedback_cycle(
        self,
        pending: list,
        judge_feedback: list,
        table_schemas: str,
        aliases: dict | None = None,
    ) -> tuple[list, dict, list, str | None]:
        """First cycle when judge feedback is present — skips static validation.

        Sends ALL pending queries together with their judge feedback (including
        suggestions) to the Correction LLM.  No static validation is performed.

        Args:
            pending: Non-empty list of query dicts to correct.
            judge_feedback: Non-empty list of feedback entries, each with
                ``"id"``, ``"error"``, and optionally ``"suggestion"``.
            table_schemas: Pre-formatted schema string for the correction prompt.
            aliases: Optional alias mapping (unused, kept for interface compat).

        Returns:
            (corrected_queries, usage_dict, error_strings, assistant_content)
            assistant_content is the raw string from the LLM response (or None).
        """
        error_formatter = ErrorFormatter()
        message_builder = ValidatorMessageBuilder()

        system_prompt = "You are a helpful assistant."

        # Build feedback block using ErrorFormatter
        feedback_text = error_formatter.build_judge_feedback_block(
            pending, judge_feedback
        )

        # Build queries text block for the LLM
        queries_text_parts = []
        for q in pending:
            qid = q.get("id", "?")
            question = q.get("question", "(no question)")
            code = q.get("code", q.get("query", "(no code)"))
            difficulty = q.get("difficulty", "unknown")
            queries_text_parts.append(
                f"--- Query ID: {qid}  (difficulty: {difficulty}) ---\n"
                f"Question: {question}\n"
                f"Code:\n{code}"
            )
        queries_text = "\n\n".join(queries_text_parts)

        # Combine queries text and judge feedback into queries_with_errors
        queries_with_errors_text = f"{queries_text}\n\n{feedback_text}"

        # ---- Generate pydantic constraint text ----------------------------
        pydantic_constraint = self.reform_prompt_constraint("").strip()

        # ---- Render the full user message from the correction template ----
        user_content = self._correction_prompt.update(
            table_schemas=table_schemas,
            queries_with_errors=queries_with_errors_text,
            pydantic_constraint=pydantic_constraint,
        )

        # ---- Fixed-structure rebuild: stateless 2 blocks -------------------
        messages = message_builder.build(
            system_content=system_prompt,
            user_content=user_content,
        )

        corrected_queries, usage, errors, assistant_message = (
            self._correct_with_llm(messages, pending)
        )
        assistant_content = assistant_message.get("content") if assistant_message else None
        return corrected_queries, usage, errors, assistant_content

    def _split_valid_invalid(
        self, queries: list, dataframes: list, aliases: dict
    ) -> tuple[list, list, list]:
        """
        Run the static validator and partition *queries* into passing / failing.

        ``passing`` is taken directly from the validator's accepted output so that
        alias substitutions applied inside ``validate_queries`` (raw dataset IDs →
        canonical Table_N names) are preserved in the returned query dicts.
        Matching against the original list by *id* is safe because positional
        integer ids are assigned at the top of ``validate_and_correct``.

        Returns:
            (passing_queries, failing_queries, error_message_strings)
        """
        
        accepted, static_errors = self._run_static_validator(queries, dataframes, aliases)
        # Use the validator-transformed dicts directly so alias-substituted code
        # is not silently discarded by re-filtering from the raw originals.
        accepted_ids = {q.get("id") for q in accepted}
        passing = accepted                                          # canonical names intact
        failing = [q for q in queries if q.get("id") not in accepted_ids]
        return passing, failing, static_errors
    
    @staticmethod
    def clean_table_names(query_code: str, aliases: dict) -> str:
        """Deprecated: use AliasSubstitution.substitute() instead."""
        sub = AliasSubstitution(aliases)
        return sub.substitute(query_code)

    @staticmethod
    def _print_judge_feedback(judge_feedback: list | None) -> None:
        if not judge_feedback:
            return

        print(f"\n{'─' * _SEP_WIDTH}")
        print(
            f"[Validator] Received judge feedback for "
            f"{len(judge_feedback)} query/queries:"
        )
        for i, fb in enumerate(judge_feedback, start=1):
            qid      = fb.get("id", "?")
            error    = fb.get("error", "(no error text)")
            q        = fb.get("query", {})
            question = q.get("question", "(no question)")
            code     = q.get("code", "(no code)")

            max_code_len = 1000
            code_display = (
                code[:max_code_len] + "…" if len(code) > max_code_len else code
            )
            error_lines   = error.splitlines()
            error_display = error_lines[0]
            if len(error_lines) > 1:
                continuation = "\n".join(
                    f"               {line}" for line in error_lines[1:]
                )
                error_display = f"{error_display}\n{continuation}"

            print(
                f"\n  [{i}] id        = {qid}\n"
                f"      question  = {question}\n"
                f"      code      = {code_display}\n"
                f"      error     = {error_display}"
            )
        print(f"{'─' * _SEP_WIDTH}\n")

    def _run_static_validator(
        self, queries: list, dataframes: list, aliases: dict
    ) -> tuple[list, list]:
        result = {"queries": queries}
        try:
            if self.kind == "PANDAS":
                validator = PandasValidator(dataframes, list(aliases.keys()), aliases)
            else:
                validator = SQLValidator(dataframes, list(aliases.keys()), aliases)
            _outcome, _conv_errors, accepted_dict, error_messages = (
                validator.validate_queries(result)
            )
            return list(accepted_dict.values()), list(error_messages)
        except Exception as exc:
            return [], [str(exc)]

    def _format_queries_with_errors(
        self, queries: list, errors_by_id: dict[str, list[str]]
    ) -> str:
        """Deprecated: use ErrorFormatter.build_correction_prompt() instead.
        
        Kept for backward compatibility but no longer called in the main loop.
        """
        parts: list[str] = []
        for q in queries:
            qid        = str(q.get("id", "?"))
            difficulty = q.get("difficulty", "unknown")
            question   = q.get("question", "(no question)")
            code       = q.get("code", q.get("query", "(no code)"))
            errors     = errors_by_id.get(qid, [])
            errors_text = (
                "\n".join(f"  - {e}" for e in errors)
                if errors
                else "  - (no specific error — please review for correctness)"
            )
            parts.append(
                f"--- Query ID: {qid}  (difficulty: {difficulty}) ---\n"
                f"Question: {question}\n"
                f"Code:\n{code}\n"
                f"Errors / Feedback:\n{errors_text}"
            )
        return "\n\n".join(parts)

    def _correct_with_llm(
        self,
        messages: list[dict],
        queries_to_fix: list,
    ) -> tuple[list, dict, list, dict]:
        """
        Call the correction LLM and return corrected queries.

        If the LLM omits or mangles a query id, the original query is silently
        kept for that slot — no warning is raised since the caller will
        re-validate and handle any remaining failures in the next attempt.

        Returns:
            (corrected_queries, usage_dict, error_strings, assistant_message_dict)
        """
        usage_total = _empty_usage()
        all_errors: list[str] = []
        local_messages = list(messages)
        last_content = ""

        for attempt in range(self.max_retries):
            try:
                response = self.router.completion(
                    model=self.config["model"],
                    messages=local_messages,
                    temperature=self.temperature,
                )
                usage = response["usage"]
                usage_total["prompt_tokens"]     += usage.get("prompt_tokens", 0)
                usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
                usage_total["total_tokens"]      += usage.get("total_tokens", 0)

                content = response["choices"][0]["message"]["content"]
                if content is None:
                    raise ValueError("LLM returned None content during correction.")
                last_content = content

                cleaned   = self._clean_json_response(content)
                json_data = self._repair_json(cleaned)
                json_data = self._normalize_response(json_data)
                validated = self.response_model.model_validate(json_data)
                corrected = validated.model_dump().get("queries", [])

                # For each query we sent, prefer the LLM-corrected version;
                # fall back silently to the original if the id was dropped/mangled.
                original_by_id  = {str(q.get("id")): q for q in queries_to_fix}
                corrected_by_id = {str(q.get("id")): q for q in corrected}
                merged = [
                    corrected_by_id.get(qid, original_by_id[qid])
                    for qid in original_by_id
                ]

                assistant_message = {"role": "assistant", "content": last_content}
                return merged, usage_total, all_errors, assistant_message

            except (json.JSONDecodeError, ValidationError) as e:
                all_errors.append(f"Correction attempt {attempt + 1} parse error: {e}")
                if attempt < self.max_retries - 1:
                    local_messages = [
                        *local_messages,
                        {"role": "assistant", "content": last_content},
                        {
                            "role": "user",
                            "content": self.reform_prompt_constraint(
                                f"Your response could not be parsed. Error: {e}\n"
                                "Return valid JSON only."
                            ),
                        },
                    ]
                    time.sleep(self.retry_delay)

            except Exception as e:
                all_errors.append(f"Correction attempt {attempt + 1} error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        all_errors.append(
            "All correction retries exhausted — returning original queries unchanged."
        )
        fallback_assistant = {"role": "assistant", "content": last_content or ""}
        return queries_to_fix, usage_total, all_errors, fallback_assistant

    def _normalize_response(self, json_data):
        if isinstance(json_data, list):
            return {"queries": json_data}
        if isinstance(json_data, dict):
            if "queries" not in json_data:
                if json_data:
                    return {"queries": [json_data]}
            return json_data
        return {"queries": [json_data]}


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _query_code(q: dict) -> str:
    """Return the executable code string regardless of key name (code vs query)."""
    return q.get("code", q.get("query", ""))


def _build_errors_by_id(
    failing: list,
    static_errors: list,
    judge_feedback: list | None,
) -> dict[str, list[str]]:
    """Build a per-query-id error list combining static and judge errors.

    Static errors are matched to failing queries positionally (index 0 → first
    failing query, index 1 → second, …).  If the validator returns fewer error
    strings than failing queries the extras get a generic fallback message; if it
    returns more they are silently ignored (shouldn't happen in practice).

    Each error is explicitly labeled with its source:
    - "[STATIC VALIDATION ERROR] ..." for static validation failures
    - "[JUDGE FEEDBACK] ..." for judge-rejected queries
    """
    errors_by_id: dict[str, list[str]] = {}

    if failing:
        for i, q in enumerate(failing):
            qid = str(q.get("id", "?"))
            if i < len(static_errors):
                error_text = str(static_errors[i])
            elif static_errors:
                # Fewer errors than failing queries — attach the last one as a
                # best-effort fallback rather than leaving the slot empty.
                error_text = str(static_errors[-1])
            else:
                error_text = "(no specific error — please review for correctness)"
            errors_by_id.setdefault(qid, []).append(
                f"[STATIC VALIDATION ERROR] {error_text}"
            )
            print(f"[Validator] #{qid} [STATIC VALIDATION ERROR] {error_text}")

    for fb in (judge_feedback or []):
        qid = str(fb["id"])
        errors_by_id.setdefault(qid, []).append(
            f"[JUDGE FEEDBACK] {fb['error']}"
        )
    return errors_by_id


def _strip_source_prefix(error: str) -> str:
    """Strip the source label prefix from an error message for comparison.

    Removes leading '[STATIC VALIDATION ERROR] ' or '[JUDGE FEEDBACK] ' prefixes
    so that the core error text can be compared across attempts.
    """
    prefixes = ["[STATIC VALIDATION ERROR] ", "[JUDGE FEEDBACK] "]
    for prefix in prefixes:
        if error.startswith(prefix):
            return error[len(prefix):]
    return error


def _build_diagnostic_context(
    query_id: str, dataframes: list, aliases: dict
) -> dict:
    """Build diagnostic context dict for error escalation.

    Extracts column names and DataFrame shape from available dataframes
    to provide the LLM with additional context for fixing persistent errors.
    """
    context: dict = {}

    if dataframes:
        # Collect column info from all available DataFrames
        all_columns: list[str] = []
        shapes: list[tuple] = []
        dtypes: dict[str, str] = {}

        for df in dataframes:
            if hasattr(df, "columns"):
                all_columns.extend([str(c) for c in df.columns])
                shapes.append(df.shape)
                for col in df.columns:
                    dtypes[str(col)] = str(df[col].dtype)

        if all_columns:
            context["columns"] = all_columns[:50]  # Limit to avoid huge prompts
        if shapes:
            context["shape"] = shapes[0]  # Primary DataFrame shape
        if dtypes:
            # Limit dtypes to first 20 columns
            limited_dtypes = dict(list(dtypes.items())[:20])
            context["dtypes"] = limited_dtypes

    return context


def _build_formatter_entries(
    queries_to_fix: list, errors_by_id: dict[str, list[str]]
) -> list[dict]:
    """Build the queries_with_errors entries for ErrorFormatter.build_correction_prompt().

    Converts the internal errors_by_id map into the structured format expected
    by ErrorFormatter, with explicit source labels extracted from error prefixes.
    """
    entries = []
    for q in queries_to_fix:
        qid = str(q.get("id", "?"))
        raw_errors = errors_by_id.get(qid, [])

        errors: list[str] = []
        source_labels: list[str] = []

        for err in raw_errors:
            if err.startswith("[STATIC VALIDATION ERROR] "):
                source_labels.append("Static validation error")
                errors.append(err[len("[STATIC VALIDATION ERROR] "):])
            elif err.startswith("[JUDGE FEEDBACK] "):
                source_labels.append("Judge feedback")
                errors.append(err[len("[JUDGE FEEDBACK] "):])
            else:
                source_labels.append("Unknown")
                errors.append(err)

        entries.append({
            "query_id": int(qid) if qid.isdigit() else 0,
            "errors": errors,
            "source_labels": source_labels,
        })

    return entries


def _empty_usage() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _accumulate_usage(total: dict, partial: dict) -> None:
    total["prompt_tokens"]     += partial.get("prompt_tokens", 0)
    total["completion_tokens"] += partial.get("completion_tokens", 0)
    total["total_tokens"]      += partial.get("total_tokens", 0)