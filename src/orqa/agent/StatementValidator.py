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
6. Return (approved + any still-pending-after-last-attempt, usage, errors).

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
import re
import json
import time
from pathlib import Path

from pydantic import ValidationError

from .StatementClient import LLMClientStructured
from .prompting import PandasValidatorCorrectionPrompt, SQLValidatorCorrectionPrompt
from .validators.SQLValidator import SQLValidator
from .validators.PandasValidator import PandasValidator

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
    attempt any still-failing queries are appended to approved as-is (best
    effort) so the caller always gets back the same number of queries it sent.
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
            q_copy["code"]=self.clean_table_names(q_copy["code"], aliases)
            pending.append(q_copy)
            print(f"#{q_copy["id"]} {q_copy["question"]}")
            print(f"## code: {q_copy["code"]}")
            

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

        system_message = {
            "role": "system",
            "content": (
                "You are an expert Data Engineer specialising in query correction. "
                "Fix the listed queries exactly as instructed and return valid JSON."
            ),
        }
        conversation_history: list[dict] = []

        for attempt in range(1, self.MAX_VALIDATION_RETRIES + 1):

            is_last_attempt = attempt == self.MAX_VALIDATION_RETRIES

            # ---- Step 1: static validation --------------------------------
            passing, failing, static_errors = self._split_valid_invalid(
                pending, dataframes, aliases
            )

            # ---- Step 2: determine which queries need LLM correction ------
            # Judge feedback is only applied on the first attempt; subsequent
            # passes rely solely on fresh static validation of corrected output.
            active_judge_feedback = remapped_judge_feedback if attempt == 1 else None
            judge_ids = {str(fb["id"]) for fb in (active_judge_feedback or [])}

            failing_ids = {str(q.get("id")) for q in failing}
            ids_to_fix  = failing_ids | judge_ids

            # Queries that passed static validation AND are not judge-rejected
            # move into the approved list immediately.
            newly_approved = [q for q in passing if str(q.get("id")) not in judge_ids]
            approved.extend(newly_approved)

            # Queries to send to the correction LLM
            queries_to_fix = [q for q in pending if str(q.get("id")) in ids_to_fix]

            if not queries_to_fix:
                print(f"[Validator] All queries passed on attempt {attempt}.")
                break

            print(
                f"[Validator] Attempt {attempt}/{self.MAX_VALIDATION_RETRIES}: "
                f"{len(queries_to_fix)} query/queries need correction "
                f"({len(failing)} static, {len(judge_ids)} judge)."
            )

            if is_last_attempt:
                # No more retries — drop the still-failing queries so the caller
                # never receives invalid output.  If approved is also empty the
                # caller gets an empty list (documented contract).
                all_errors.append(
                    f"[Validator] {len(queries_to_fix)} query/queries still failing "
                    f"after {self.MAX_VALIDATION_RETRIES} correction attempt(s); "
                    f"dropping from result."
                )
                break

            # ---- Step 3: build per-query error map ------------------------
            errors_by_id = _build_errors_by_id(
                failing, static_errors, active_judge_feedback
            )

            # ---- Step 4: correction LLM call ------------------------------
            # Diagnostic: confirm each query is paired with its own error only
            print(f"[Validator] Error↔Query pairing for attempt {attempt}:")
            for q in queries_to_fix:
                qid = str(q.get("id", "?"))
                errs = errors_by_id.get(qid, ["(none)"])
                print(f"  #{qid} → {len(errs)} error(s): {errs[0][:120]!r}{'…' if len(errs[0]) > 120 else ''}")
            queries_with_errors_text = self._format_queries_with_errors(
                queries_to_fix, errors_by_id
            )
            correction_prompt = self._correction_prompt.update(
                table_schemas=table_schemas,
                aliases=json.dumps(aliases, indent=2),
                queries_with_errors=queries_with_errors_text,
            )
            new_user_message = {
                "role": "user",
                "content": self.reform_prompt_constraint(correction_prompt),
            }

            # Sliding window: keep at most 1 prior exchange before the new message
            prior_exchange = conversation_history[-2:] if len(conversation_history) >= 2 else []
            messages_to_send = [system_message, *prior_exchange, new_user_message]

            corrected_queries, tokens, llm_errors, assistant_message = (
                self._correct_with_llm(messages_to_send, queries_to_fix)
            )
            _accumulate_usage(usage_total, tokens)
            # Deduplicate: only append errors not already recorded
            existing = set(all_errors)
            all_errors.extend(e for e in llm_errors if e not in existing)
            conversation_history.extend([new_user_message, assistant_message])

            # ---- Step 5: pending becomes the corrected queries_to_fix -----
            # The next iteration will re-validate these; any that now pass will
            # be moved to approved, any still failing will be corrected again.
            pending = corrected_queries

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
        # Invert: table_name -> alias  (e.g. "qvir-knu3" -> "Table_0")
        inverted = {v: k for k, v in aliases.items()}

        result = query_code
        for table_name, alias in inverted.items():
            # Also match the underscore variant (e.g. "qvir_knu3") since LLMs
            # convert dashes to underscores to produce valid SQL identifiers.
            underscore_variant = table_name.replace("-", "_")

            for name_to_replace in {table_name, underscore_variant}:
                result = re.sub(
                    r'(?<![.\w])' + re.escape(name_to_replace) + r'(?!\w)',
                    alias,
                    result,
                )
        return result

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
                f"Static validation error:\n{error_text}"
            )
            print(f"[Validator] #{qid} Static validation error:\n{error_text}")

    for fb in (judge_feedback or []):
        qid = str(fb["id"])
        errors_by_id.setdefault(qid, []).append(
            f"Judge feedback: {fb['error']}"
        )
    return errors_by_id


def _empty_usage() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _accumulate_usage(total: dict, partial: dict) -> None:
    total["prompt_tokens"]     += partial.get("prompt_tokens", 0)
    total["completion_tokens"] += partial.get("completion_tokens", 0)
    total["total_tokens"]      += partial.get("total_tokens", 0)