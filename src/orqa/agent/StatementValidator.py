"""
StatementValidator
==================
Validates and corrects generated queries through up to MAX_VALIDATION_RETRIES
cycles of static validation (Pandas/SQL) + LLM correction.

Flow
----
Each cycle:
  1. If judge feedback is present on cycle 1: skip static validation,
     send all pending queries + feedback directly to the LLM.
  2. Otherwise: run static validator, collect passing into `approved`,
     send failing + their errors to the LLM.
  3. The LLM returns corrected queries.  These completely replace `pending`.
     Matching is positional — LLM-returned IDs are never trusted.
  4. On the final cycle, any still-failing queries are dropped.

LLM call contract
-----------------
* Messages are stateless per call: [system, user] only.
* On a Pydantic/JSON parse failure the parent retry pattern is used:
  append [assistant, user(error)] and retry — but only for format errors.
* If all Pydantic retries are exhausted, return [] so the caller keeps
  whatever `approved` queries it already has.

Error lifetime — two tiers
--------------------------
Tier 1 — audit trail  (`all_errors` in validate_and_correct):
  Append-only list that accumulates every error that occurs across all cycles:
  LLM parse failures, dropped-query warnings, exhaustion messages.
  Never read by the loop, never influences control flow.
  Returned to the caller at the end purely for logging/inspection.

Tier 2 — prompt errors  (`errors_by_id` local to each cycle):
  Built fresh every cycle from the *current* failing queries via
  _positional_errors(failing, static_errors).  Only contains errors for
  queries that are alive in the current `pending` list.  Consumed by the
  prompt builder and discarded at the end of the cycle — never carried
  forward.  When `pending` is overwritten with corrected queries, all
  association with the old evicted queries and their errors is gone.
"""
import json
import logging
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


class LLMStatementValidator(LLMClientStructured):

    MAX_VALIDATION_RETRIES: int = 3

    def __init__(self, config_path: Path, kind: str):
        super().__init__(config_path, "querying")
        self.fallback_response_model = self._load_pydantic_response_model("fallback_querying")
        self.kind = kind
        self._correction_prompt = (
            PandasValidatorCorrectionPrompt()
            if kind == "PANDAS"
            else SQLValidatorCorrectionPrompt()
        )

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
        Validate and correct *queries*, retrying up to MAX_VALIDATION_RETRIES times.

        Returns:
            (approved_queries, token_usage, error_strings)
            approved_queries carry the caller's original IDs.
        """
        if not queries:
            return [], _empty_usage(), []

        # --- Alias substitution & sequential ID assignment ----------------
        alias_sub = AliasSubstitution(aliases)
        original_ids: list = []
        pending: list[dict] = []

        for pos, q in enumerate(queries, start=1):
            q_copy = dict(q)
            original_ids.append(q_copy.get("id"))
            q_copy["id"] = pos
            q_copy["code"] = alias_sub.substitute(q_copy.get("code", ""))
            pending.append(q_copy)

        # --- Remap judge feedback to positional IDs -----------------------
        remapped_feedback: list[dict] | None = None
        if judge_feedback:
            orig_to_pos = {
                str(orig): pos
                for pos, orig in enumerate(original_ids, start=1)
            }
            remapped_feedback = []
            for fb in judge_feedback:
                fb_copy = dict(fb)
                pos = orig_to_pos.get(str(fb.get("id")))
                if pos is not None:
                    fb_copy["id"] = pos
                remapped_feedback.append(fb_copy)

        usage_total = _empty_usage()
        # Audit trail — append-only, never read by the loop, returned to caller.
        # Accumulates LLM parse failures, dropped-query notices, exhaustion
        # messages across all cycles.  Does not influence control flow.
        all_errors: list[str] = []
        approved: list[dict] = []
        has_judge_feedback = bool(remapped_feedback)

        # ------------------------------------------------------------------
        # Main correction loop
        # ------------------------------------------------------------------
        for cycle in range(1, self.MAX_VALIDATION_RETRIES + 1):
            is_last_cycle = cycle == self.MAX_VALIDATION_RETRIES

            # ----------------------------------------------------------
            # JUDGE FEEDBACK PATH — cycle 1 only, skips static validation
            # ----------------------------------------------------------
            if has_judge_feedback and cycle == 1:
                logger.info(
                    "[Validator] Cycle %d/%d: judge-feedback path — "
                    "%d queries → LLM (no static validation)",
                    cycle, self.MAX_VALIDATION_RETRIES, len(pending),
                )
                user_content = self._build_judge_feedback_prompt(
                    pending, remapped_feedback, table_schemas
                )
                corrected, tokens, errors = self._call_correction_llm(
                    pending, user_content
                )
                _accumulate_usage(usage_total, tokens)
                all_errors.extend(errors)

                if not corrected:
                    logger.warning(
                        "[Validator] Judge-feedback LLM call returned nothing; "
                        "skipping remaining cycles."
                    )
                    break

                _print_query_diff(pending, corrected, cycle, "judge-feedback")
                pending = corrected
                has_judge_feedback = False   # only runs once
                continue                     # next cycle does static validation

            # ----------------------------------------------------------
            # STATIC VALIDATION PATH
            # ----------------------------------------------------------
            passing, failing, static_errors = self._split_valid_invalid(
                pending, dataframes, aliases
            )
            approved.extend(passing)
            pending = failing

            logger.info(
                "[Validator] Cycle %d/%d: %d passed, %d failing",
                cycle, self.MAX_VALIDATION_RETRIES, len(passing), len(failing),
            )

            if not pending:
                logger.info("[Validator] All queries passed on cycle %d.", cycle)
                break

            if is_last_cycle:
                logger.warning(
                    "[Validator] Cycle %d (final): dropping %d still-failing queries.",
                    cycle, len(pending),
                )
                all_errors.append(
                    f"[Validator] {len(pending)} query/queries dropped after "
                    f"{self.MAX_VALIDATION_RETRIES} correction cycle(s)."
                )
                break

            # Build per-query error map for this cycle's LLM prompt.
            # Tier-2 errors: ephemeral, scoped to this cycle only.
            # Built from the *current* failing queries — no association with
            # queries evicted in previous cycles.  Discarded after prompt build.
            errors_by_id = _positional_errors(failing, static_errors)

            user_content = self._build_static_correction_prompt(
                failing, errors_by_id, table_schemas
            )
            corrected, tokens, errors = self._call_correction_llm(
                failing, user_content
            )
            _accumulate_usage(usage_total, tokens)
            all_errors.extend(errors)

            if not corrected:
                logger.warning(
                    "[Validator] Static-correction LLM call returned nothing on "
                    "cycle %d; dropping %d queries.",
                    cycle, len(failing),
                )
                all_errors.append(
                    f"[Validator] LLM correction failed on cycle {cycle}; "
                    f"dropping {len(failing)} queries."
                )
                break

            # Completely replace pending with corrected output.
            # Old queries and their errors_by_id are now evicted — the next
            # cycle's _positional_errors call will build a fresh error map
            # bound only to these new queries.
            _print_query_diff(failing, corrected, cycle, "static")
            pending = corrected   # re-validated next cycle

        # --- Restore original IDs -----------------------------------------
        pos_to_orig = {pos: orig for pos, orig in enumerate(original_ids, start=1)}
        final: list[dict] = []
        for q in approved:
            q_out = dict(q)
            q_out["id"] = pos_to_orig.get(q_out.get("id"), q_out.get("id"))
            final.append(q_out)

        orig_order = {orig: i for i, orig in enumerate(original_ids)}
        final.sort(key=lambda q: orig_order.get(q.get("id"), 0))

        return final, usage_total, all_errors

    # ------------------------------------------------------------------
    # LLM call — single entry point for all correction calls
    # ------------------------------------------------------------------

    def _call_correction_llm(
        self,
        source_queries: list[dict],
        user_content: str,
    ) -> tuple[list[dict], dict, list[str]]:
        """
        Send [system, user] to the correction LLM and return corrected queries
        matched to *source_queries* by position (not by ID).

        On Pydantic/JSON parse failure: retry following the parent pattern
        (append [assistant, user(error_message)] to the message list).
        If all retries fail: return ([], usage, errors) so the caller can
        keep whatever approved queries it already holds.

        Args:
            source_queries: The original query dicts being corrected.
                            Used as positional fallbacks and ID donors.
            user_content:   The fully rendered user message for this cycle.

        Returns:
            (corrected_queries, usage_dict, error_strings)
        """
        system_content = "You are a helpful assistant."

        # Stateless base — rebuilt fresh for every correction cycle
        base_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        messages = list(base_messages)

        usage = _empty_usage()
        # Local to this call — only holds Pydantic/parse retry errors.
        # Appended to the caller's audit trail (all_errors) on return.
        # Never carried between correction cycles.
        errors: list[str] = []
        last_content = ""

        for attempt in range(self.max_retries):
            try:
                response = self.router.completion(
                    model=self.config["model"],
                    messages=messages,
                    temperature=self.temperature,
                )
                _accumulate_usage(usage, response["usage"])

                content = response["choices"][0]["message"]["content"]
                if content is None:
                    raise ValueError("LLM returned None content during correction.")
                last_content = content

                cleaned   = self._clean_json_response(content)
                json_data = self._repair_json(cleaned)
                json_data = self._normalize_response(json_data)
                validated = self.response_model.model_validate(json_data)
                corrected = validated.model_dump().get("queries", [])

                # ----------------------------------------------------------
                # Position-based merge: take LLM code by index, preserve all
                # other fields (question, difficulty, id) from the source.
                # If the LLM returns fewer items than sent, keep the source
                # query unchanged for the missing slots.
                # ----------------------------------------------------------
                result: list[dict] = []
                for i, original in enumerate(source_queries):
                    if i < len(corrected):
                        merged = dict(original)
                        merged["code"] = corrected[i].get(
                            "code", original.get("code", "")
                        )
                        result.append(merged)
                    else:
                        logger.warning(
                            "[Validator] LLM returned only %d/%d queries; "
                            "keeping original at position %d.",
                            len(corrected), len(source_queries), i,
                        )
                        result.append(dict(original))

                return result, usage, errors

            except (json.JSONDecodeError, ValidationError) as e:
                msg = f"Correction attempt {attempt + 1} parse/validation error: {e}"
                errors.append(msg)
                logger.warning("[Validator] %s\nRaw content: %.500s", msg, last_content)

                if attempt < self.max_retries - 1:
                    # Follow parent pattern: keep context, append error feedback
                    messages.append({"role": "assistant", "content": last_content})
                    messages.append({
                        "role": "user",
                        "content": self.reform_prompt_constraint(
                            f"Your response could not be parsed. Error: {e}\n"
                            "Return valid JSON only."
                        ),
                    })
                    time.sleep(self.retry_delay)

            except Exception as e:
                msg = f"Correction attempt {attempt + 1} error: {e}"
                errors.append(msg)
                logger.error("[Validator] %s", msg)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        errors.append(
            f"[Validator] All {self.max_retries} correction retries exhausted — "
            "returning empty result; approved queries preserved."
        )
        logger.error(
            "[Validator] Correction retries exhausted. Last raw content:\n%.1000s",
            last_content,
        )
        return [], usage, errors

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_static_correction_prompt(
        self,
        failing: list[dict],
        errors_by_id: dict[str, str],
        table_schemas: str,
    ) -> str:
        """Render the user message for a static-validation correction cycle."""
        error_formatter = ErrorFormatter()

        entries = [
            {
                "query_id": q.get("id"),
                "errors": [errors_by_id.get(str(q.get("id")), "(no error)")],
                "source_labels": ["Static validation error"],
            }
            for q in failing
        ]
        queries_with_errors_text = error_formatter.build_correction_prompt(entries, [])

        return self._correction_prompt.update(
            table_schemas=table_schemas,
            queries_with_errors=queries_with_errors_text,
            pydantic_constraint=self.reform_prompt_constraint("").strip(),
        )

    def _build_judge_feedback_prompt(
        self,
        pending: list[dict],
        judge_feedback: list[dict],
        table_schemas: str,
    ) -> str:
        """Render the user message for a judge-feedback correction cycle."""
        error_formatter = ErrorFormatter()

        queries_parts = []
        for q in pending:
            queries_parts.append(
                f"--- Query ID: {q.get('id')}  "
                f"(difficulty: {q.get('difficulty', 'unknown')}) ---\n"
                f"Question: {q.get('question', '(no question)')}\n"
                f"Code:\n{q.get('code', '(no code)')}"
            )
        queries_text = "\n\n".join(queries_parts)

        feedback_text = error_formatter.build_judge_feedback_block(
            pending, judge_feedback
        )
        queries_with_errors_text = f"{queries_text}\n\n{feedback_text}"

        return self._correction_prompt.update(
            table_schemas=table_schemas,
            queries_with_errors=queries_with_errors_text,
            pydantic_constraint=self.reform_prompt_constraint("").strip(),
        )

    # ------------------------------------------------------------------
    # Static validation
    # ------------------------------------------------------------------

    def _split_valid_invalid(
        self, queries: list, dataframes: list, aliases: dict
    ) -> tuple[list, list, list]:
        """Run the static validator and split queries into passing / failing."""
        accepted, static_errors = self._run_static_validator(
            queries, dataframes, aliases
        )
        accepted_ids = {q.get("id") for q in accepted}
        failing = [q for q in queries if q.get("id") not in accepted_ids]
        return accepted, failing, static_errors

    def _run_static_validator(
        self, queries: list, dataframes: list, aliases: dict
    ) -> tuple[list, list]:
        result = {"queries": queries}
        try:
            if self.kind == "PANDAS":
                validator = PandasValidator(
                    dataframes, list(aliases.keys()), aliases
                )
            else:
                validator = SQLValidator(
                    dataframes, list(aliases.keys()), aliases
                )
            _outcome, _conv_errors, accepted_dict, error_messages = (
                validator.validate_queries(result)
            )
            accepted     = list(accepted_dict.values())
            accepted_ids = {q.get("id") for q in accepted}
            failing      = [q for q in queries if q.get("id") not in accepted_ids]

            _check_error_alignment(failing, validator.validation_errors, error_messages)

            return accepted, list(error_messages)
        except Exception as exc:
            logger.exception("[Validator] Static validator raised an exception.")
            return [], [str(exc)]

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

def _check_error_alignment(
    failing: list[dict],
    validation_errors: list[dict],
    error_messages: list[str],
) -> None:
    """Verify that static_errors[i] is paired with failing[i].

    Three things must hold for positional pairing to be safe:
      1. len(validation_errors) == len(error_messages)  — validator internals consistent
      2. len(validation_errors) == len(failing)          — one error per failing query
      3. validation_errors[i]["query"]["id"] == failing[i]["id"]  — same order

    Mismatches are logged as errors but never raise — the caller proceeds with
    whatever alignment exists and the worst outcome is a slightly mismatched
    error message in the LLM prompt, caught on the next validation cycle.
    """
    sep = "─" * 68

    # 1. Internal consistency: validation_errors vs error_messages counts
    if len(validation_errors) != len(error_messages):
        logger.error(
            "[Validator] Alignment check FAILED — "
            "validation_errors count (%d) != error_messages count (%d). "
            "Validator internal state is inconsistent.",
            len(validation_errors), len(error_messages),
        )
        print(f"\n{sep}")
        print(
            f"[Validator] ⚠ ALIGNMENT MISMATCH: "
            f"validation_errors={len(validation_errors)}, "
            f"error_messages={len(error_messages)}"
        )
        print(sep)

    # 2. Count: one error entry per failing query
    if len(validation_errors) != len(failing):
        logger.error(
            "[Validator] Alignment check FAILED — "
            "failing queries count (%d) != validation_errors count (%d).",
            len(failing), len(validation_errors),
        )
        print(f"\n{sep}")
        print(
            f"[Validator] ⚠ ALIGNMENT MISMATCH: "
            f"failing={len(failing)}, validation_errors={len(validation_errors)}"
        )
        print(sep)
        return   # positional check below is meaningless if counts differ

    # 3. Order: validation_errors[i] must belong to the same query as failing[i]
    mismatches: list[str] = []
    for i, (ve, fail_q) in enumerate(zip(validation_errors, failing)):
        ve_id   = ve.get("query", {}).get("id")
        fail_id = fail_q.get("id")
        if ve_id != fail_id:
            mismatches.append(
                f"  position {i}: validation_errors has id={ve_id!r}, "
                f"failing has id={fail_id!r}"
            )

    if mismatches:
        logger.error(
            "[Validator] Alignment check FAILED — error order does not match "
            "failing query order at %d position(s):\n%s",
            len(mismatches), "\n".join(mismatches),
        )
        print(f"\n{sep}")
        print(f"[Validator] ⚠ ERROR ORDER MISMATCH ({len(mismatches)} position(s)):")
        for m in mismatches:
            print(m)
        print(sep)
    else:
        print(
            f"[Validator] ✓ Error alignment OK — "
            f"{len(failing)} failing queries paired correctly."
        )


def _print_query_diff(
    old_queries: list[dict],
    new_queries: list[dict],
    cycle: int,
    path: str,
) -> None:
    """Print old and new query lists independently — no positional pairing assumed."""
    sep = "─" * 68
    print(f"\n{sep}")
    print(f"[Validator] Cycle {cycle} ({path}) — before correction "
          f"({len(old_queries)} queries)")
    print(sep)
    for q in old_queries:
        print(f"  #{q.get('id', '?')}  {q.get('question', '(no question)')}")
        print(f"  code: {q.get('code', '(no code)')}\n")

    print(sep)
    print(f"[Validator] Cycle {cycle} ({path}) — after correction "
          f"({len(new_queries)} queries)")
    print(sep)
    for q in new_queries:
        print(f"  #{q.get('id', '?')}  {q.get('question', '(no question)')}")
        print(f"  code: {q.get('code', '(no code)')}\n")
    print(f"{sep}\n")


def _positional_errors(
    failing: list[dict], static_errors: list[str]
) -> dict[str, str]:
    """
    Build a query-id → error-string map by matching errors positionally to
    failing queries (index 0 → first failing query, etc.).

    If the validator returns fewer errors than failing queries the last error
    is reused as a best-effort fallback.  If no errors at all, a generic
    message is used.
    """
    result: dict[str, str] = {}
    for i, q in enumerate(failing):
        qid = str(q.get("id", i))
        if static_errors:
            result[qid] = static_errors[i] if i < len(static_errors) else static_errors[-1]
        else:
            result[qid] = "(no specific error — please review for correctness)"
    return result


def _empty_usage() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _accumulate_usage(total: dict, partial: dict) -> None:
    total["prompt_tokens"]     += partial.get("prompt_tokens", 0)
    total["completion_tokens"] += partial.get("completion_tokens", 0)
    total["total_tokens"]      += partial.get("total_tokens", 0)