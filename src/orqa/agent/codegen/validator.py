"""
StatementValidator
==================
Validates and corrects generated queries through up to MAX_VALIDATION_RETRIES
cycles of static validation (Pandas/SQL) + LLM correction.

Flow
----
Each cycle:
  1. If judge feedback is present on cycle 1: skip static validation,
     send each pending query + its feedback to the LLM.
  2. Otherwise: run static validator, collect passing into `approved`,
     send each failing query + its error to the LLM.
  3. Every correction call is for exactly ONE query — never batched — run
     concurrently on a bounded thread pool (mirrors
     ``JudgementResponseAgent._judge_all_concurrent``). Each query is matched
     back to its result by a pre-recorded index, not by anything in the
     response, so a query can never be misattributed regardless of which
     call completes first or how the model behaves. This also means one
     query's correction failing (retries exhausted) only affects that query
     — it falls back to its own unchanged original — instead of aborting the
     whole cycle for every other query too.
  4. On the final cycle, any still-failing queries are dropped.

LLM call contract
-----------------
* Messages are stateless per call: [system, user] only.
* On a Pydantic/JSON parse failure the parent retry pattern is used:
  append [assistant, user(error)] and retry — but only for format errors.
* If all Pydantic retries are exhausted for a query, that query's original
  (unchanged) dict is kept — never the whole cycle's queries.

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
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


from orqa.agent.codegen.generator import GENERATION_MAX_TOKENS, AliasSubstitution
from orqa.agent.llm_client.LLMClientStructured import LLMClientStructured
from orqa.agent.prompting import PandasValidatorCorrectionPrompt, SQLValidatorCorrectionPrompt
from orqa.agent.validators.SQLValidator import SQLValidator
from orqa.agent.validators.PandasValidator import PandasValidator


# ── Error formatter ───────────────────────────────────────────────────────────
# ErrorFormatter
# ==============
# Formats validation errors for the correction LLM prompt.
#
# Each query is rendered as a JSON object matching the Query Pydantic schema,
# followed by its errors/feedback — giving the LLM the exact structure it must
# return alongside the context it needs to fix it. The JSON block includes the
# question-level metadata bundle (question/question_keywords/translated_question/
# translated_question_keywords/topic/story) that planning produces as one unit,
# so a correction that rewrites the question can regenerate all of it
# consistently rather than leaving the rest stale.
#
# Every correction call is for exactly ONE query (see
# ``StatementValidator._correct_queries_concurrently``) — there is no
# cross-query batching, so there's nothing to compare across queries within a
# single call.

class ErrorFormatter:

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format_per_query(
        self,
        query_id: int,
        errors: list[str],
        source_labels: list[str],
        question: str = "",
        translated_question: str = "",
        detected_language: str = "",
        question_keywords: list[str] | None = None,
        translated_question_keywords: list[str] | None = None,
        topic: str = "",
        story: str = "",
        code: str = "",
        tables: list = None,
    ) -> str:
        """Format a single query as a JSON block (matching the Query Pydantic schema)
        followed by its errors/feedback.

        The JSON block gives the LLM the exact structure it must reproduce in its
        response, with every field populated so it has full context to correct only
        what is broken.

        ``question``, ``question_keywords``, ``translated_question``,
        ``translated_question_keywords``, ``topic``, and ``story`` are shown as
        ONE linked bundle (they were all produced together during planning —
        see ``prompting.models.SQLQueryPlan``/``PandasQueryPlan``): the
        correction prompt instructs the model to regenerate all of them
        together if it rewrites ``question``, and to return them unchanged
        otherwise.
        """
        # --- Build the query JSON object (mirrors Query Pydantic schema) ---
        # Normalise tables into the Table schema shape regardless of how they
        # arrived (dict with name/reason/columns_involved, or plain string).
        normalised_tables: list[dict] = []
        for t in (tables or []):
            if isinstance(t, dict):
                normalised_tables.append({
                    "name": t.get("name", ""),
                    "reason": t.get("reason", ""),
                    "columns_involved": t.get("columns_involved", []),
                })
            else:
                normalised_tables.append({
                    "name": str(t),
                    "reason": "",
                    "columns_involved": [],
                })

        query_obj = {
            "question": question,
            "question_keywords": question_keywords or [],
            "translated_question": translated_question,
            "translated_question_keywords": translated_question_keywords or [],
            "detected_language": detected_language,
            "topic": topic,
            "story": story,
            "code": code,
            "tables": normalised_tables,
        }
        query_json = json.dumps(query_obj, ensure_ascii=False, indent=2)

        # --- Build the error/feedback block --------------------------------
        error_lines: list[str] = []
        for error, label in zip(errors, source_labels):
            error_lines.append(f"[{label}] {error}")
        if not error_lines:
            error_lines.append("[No errors]")

        return (
            f"--- Query #{query_id} ---\n"
            f"{query_json}\n\n"
            + "\n".join(error_lines)
        )

    def build_correction_prompt(self, queries_with_errors: list) -> str:
        """Build the correction prompt for a (single-entry) list of queries-with-errors."""
        sections: list[str] = ["=== PER-QUERY ERRORS ==="]
        for entry in queries_with_errors:
            formatted = self.format_per_query(
                query_id=entry["query_id"],
                errors=entry["errors"],
                source_labels=entry["source_labels"],
                question=entry.get("question", ""),
                translated_question=entry.get("translated_question", ""),
                detected_language=entry.get("detected_language", ""),
                question_keywords=entry.get("question_keywords"),
                translated_question_keywords=entry.get("translated_question_keywords"),
                topic=entry.get("topic", ""),
                story=entry.get("story", ""),
                code=entry.get("code", ""),
                tables=entry.get("tables"),
            )
            sections.append(formatted)

        return "\n\n".join(sections)


logger = logging.getLogger(__name__)

# Bounded worker pool for concurrent one-query-at-a-time correction calls.
# Mirrors JUDGE_CONCURRENCY in agent.py — same rationale, same order of magnitude.
CORRECTION_CONCURRENCY = 6


class LLMStatementValidator(LLMClientStructured):

    # Static-validation cycles (validate -> LLM correction -> validate) before
    # a still-failing query is dropped. Measured on the code stage
    # (experiments/structured_stability, 2026-11-03): on a 22-step plan the
    # query passed static validation on the 4th and the 5th cycle in two
    # runs — beyond the former 3, which dropped both queries of that group
    # in one run.
    MAX_VALIDATION_RETRIES: int = 5

    # The correction prompt already carries the schema (``pydantic_constraint``
    # in the user message), so the system message does not repeat it; code is
    # a long output, so native JSON-schema decoding stays off (see
    # StatementClient).
    schema_in_prompt = False
    native_schema = False

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
        *,
        column_statistics: str,
    ) -> tuple[list, dict, list]:
        """
        Validate and correct *queries*, retrying up to MAX_VALIDATION_RETRIES times.

        ``column_statistics`` is the run's column-statistics block, rendered
        once by the caller: every correction prompt carries it byte-identical
        (a shared cacheable prefix), and it is what tells the corrector how a
        value that broke a cast or a filter is actually stored. Required, so
        the empty-block wording lives only in render_column_statistics.

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
                    "%d queries → LLM, one call per query (no static validation)",
                    cycle, self.MAX_VALIDATION_RETRIES, len(pending),
                )
                feedback_by_id = {
                    str(fb.get("id")): fb for fb in remapped_feedback
                }
                prompts = []
                for q in pending:
                    qid = str(q.get("id"))
                    fb = feedback_by_id.get(qid, {})
                    error = fb.get("error", "")
                    suggestion = fb.get("suggestion", "")
                    error_text = f"{error}\n  Suggestion: {suggestion}" if suggestion else (
                        error or "(no judge feedback available)"
                    )
                    prompts.append(self._build_one_correction_prompt(
                        q, error_text, "Judge Feedback", table_schemas, column_statistics,
                    ))

                corrected, tokens, errors = self._correct_queries_concurrently(
                    pending, prompts
                )
                _accumulate_usage(usage_total, tokens)
                all_errors.extend(errors)

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
                dropped_errors_by_id = _positional_errors(failing, static_errors)
                for q in pending:
                    qid = q.get("id")
                    err = dropped_errors_by_id.get(str(qid), "(no specific error)")
                    code = q.get("code", "(no code)")
                    logger.warning(
                        "[Validator]   dropped #%s — %s\n    code: %s",
                        qid, err, code,
                    )
                    all_errors.append(
                        f"[Validator] Dropped query #{qid}: {err}\n    code: {code}"
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

            prompts = [
                self._build_one_correction_prompt(
                    q,
                    errors_by_id.get(str(q.get("id")), "(no error)"),
                    "Static validation error",
                    table_schemas,
                    column_statistics,
                )
                for q in failing
            ]
            corrected, tokens, errors = self._correct_queries_concurrently(
                failing, prompts
            )
            _accumulate_usage(usage_total, tokens)
            all_errors.extend(errors)

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
    # LLM calls — one query per call, run concurrently
    # ------------------------------------------------------------------

    def _correct_queries_concurrently(
        self,
        source_queries: list[dict],
        prompts: list[str],
    ) -> tuple[list[dict], dict, list[str]]:
        """Correct each query with its own isolated LLM call, run concurrently.

        Mirrors ``JudgementResponseAgent._judge_all_concurrent``: every query
        gets exactly one call (never batched), submitted on a bounded thread
        pool, and matched back to its result by a pre-recorded index — never
        by anything in the response. A query can therefore never be
        misattributed regardless of completion order or model behaviour, and
        one query's correction failing only affects that query (it falls
        back to its own unchanged original) rather than the whole cycle.

        Args:
            source_queries: The queries being corrected, in order.
            prompts: The fully-rendered user message for each query, same
                order/length as ``source_queries`` (``prompts[i]`` corrects
                ``source_queries[i]``).

        Returns:
            (corrected_queries, usage_dict, error_strings) — corrected_queries
            is always the same length as source_queries.
        """
        usage_total = _empty_usage()
        all_errors: list[str] = []
        if not source_queries:
            return [], usage_total, all_errors

        results: list[dict] = list(source_queries)
        max_workers = max(1, min(CORRECTION_CONCURRENCY, len(source_queries)))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_index = {
                pool.submit(self._call_correction_llm_one, query, prompt): i
                for i, (query, prompt) in enumerate(zip(source_queries, prompts))
            }

            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    corrected_query, tokens, errors = future.result()
                except Exception as exc:  # noqa: BLE001 — isolate per-query failures
                    logger.error(
                        "[Validator] Correction call for query #%s raised: %s",
                        source_queries[i].get("id"), exc,
                    )
                    corrected_query = dict(source_queries[i])
                    tokens = _empty_usage()
                    errors = [
                        f"[Validator] Correction call for query "
                        f"#{source_queries[i].get('id')} raised "
                        f"{type(exc).__name__}: {exc}"
                    ]
                results[i] = corrected_query
                _accumulate_usage(usage_total, tokens)
                all_errors.extend(errors)

        return results, usage_total, all_errors

    def _call_correction_llm_one(
        self,
        source_query: dict,
        user_content: str,
    ) -> tuple[dict, dict, list[str]]:
        """
        Send [system, user] to the correction LLM for exactly ONE query and
        return the corrected query — or the original, unchanged, if every
        retry fails.

        The call is the shared budget-aware structured loop
        (``LLMClientStructured.complete``: explicit budget with escalation on
        a cut answer, shape coercion, one repair call, the repetition guard).
        ``user_content`` already carries the schema via ``pydantic_constraint``,
        so the system message is the bare preamble (``schema_in_prompt``).

        Returns:
            (corrected_query, usage_dict, error_strings)
        """

        def backfill(json_data: Any) -> Any:
            # The merge below only consumes `code` and the question bundle;
            # every other schema-required field (difficulty, tables, ...) is
            # discarded in favour of the source query anyway. Backfill
            # missing fields from the source BEFORE validation so a lean
            # correction (just the fixed code) never fails the full QuerySet
            # schema over a field nobody reads — demanding the model
            # regenerate unchanged fields is pure retry overhead.
            items = json_data.get("queries") if isinstance(json_data, dict) else None
            if items and isinstance(items[0], dict):
                for field, value in source_query.items():
                    if field not in ("id", "client_id"):
                        items[0].setdefault(field, value)
            return json_data

        result, usage = LLMClientStructured.complete(
            self, user_content, root_key="queries", pre_validate=backfill,
            max_tokens=GENERATION_MAX_TOKENS,
        )
        if not result:
            message = (
                f"[Validator] All {self.max_retries} correction retries exhausted for "
                f"query #{source_query.get('id')}; keeping original code."
            )
            logger.error(message)
            return dict(source_query), usage, [message]

        corrected_list = result.get("queries", []) or []
        corrected_item = corrected_list[0] if corrected_list else {}

        merged = dict(source_query)
        merged["code"] = corrected_item.get("code", source_query.get("code", ""))
        # `question`, `question_keywords`, `translated_question`,
        # `translated_question_keywords`, `topic`, and `story` are one
        # linked bundle produced together during planning (see
        # prompting.models.SQLQueryPlan/PandasQueryPlan). The correction
        # prompt instructs the LLM to regenerate all of them consistently
        # whenever it rewrites `question`, or echo them unchanged otherwise
        # — so trust whatever comes back, falling back to the original only
        # when a field is omitted/empty (the LLM not bothering to echo it)
        # rather than treating that as a deliberate clear.
        for field, default in (
            ("question", ""),
            ("question_keywords", []),
            ("translated_question", ""),
            ("translated_question_keywords", []),
            ("topic", ""),
            ("story", ""),
        ):
            merged[field] = corrected_item.get(field) or source_query.get(field, default)

        return merged, usage, []

    # ------------------------------------------------------------------
    # Prompt builder — one query per prompt, shared by both correction paths
    # ------------------------------------------------------------------

    def _build_one_correction_prompt(
        self,
        query: dict,
        error_text: str,
        source_label: str,
        table_schemas: str,
        column_statistics: str,
    ) -> str:
        """Render the user message correcting exactly ONE query.

        Shared by the static-validation path (``source_label="Static
        validation error"``) and the judge-feedback path
        (``source_label="Judge Feedback"``) — the only difference between the
        two is the error text and its label; the rendered structure (full
        question/keyword/translation bundle + code + tables) is identical
        either way, matching what ``format_per_query`` always produced for a
        single entry.
        """
        error_formatter = ErrorFormatter()
        entry = {
            "query_id": query.get("id"),
            "question": query.get("question", ""),
            "translated_question": query.get("translated_question", ""),
            "detected_language": query.get("detected_language", ""),
            "question_keywords": query.get("question_keywords", []),
            "translated_question_keywords": query.get("translated_question_keywords", []),
            "topic": query.get("topic", ""),
            "story": query.get("story", ""),
            "code": query.get("code", ""),
            "tables": query.get("tables"),
            "errors": [error_text],
            "source_labels": [source_label],
        }
        queries_with_errors_text = error_formatter.build_correction_prompt([entry])

        return self._correction_prompt.update(
            table_schemas=table_schemas,
            column_statistics=column_statistics,
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

    def _normalize_response(self, json_data, root_key: str | None = "queries"):
        key = root_key or "queries"
        if isinstance(json_data, list):
            return {key: json_data}
        if isinstance(json_data, dict):
            if key not in json_data:
                if json_data:
                    return {key: [json_data]}
            return json_data
        return {key: [json_data]}


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
    # 1. Internal consistency: validation_errors vs error_messages counts
    if len(validation_errors) != len(error_messages):
        logger.error(
            "[Validator] Alignment check FAILED — "
            "validation_errors count (%d) != error_messages count (%d). "
            "Validator internal state is inconsistent.",
            len(validation_errors), len(error_messages),
        )

    # 2. Count: one error entry per failing query
    if len(validation_errors) != len(failing):
        logger.error(
            "[Validator] Alignment check FAILED — "
            "failing queries count (%d) != validation_errors count (%d).",
            len(failing), len(validation_errors),
        )
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
    else:
        logger.debug(
            "[Validator] Error alignment OK — %d failing queries paired correctly.",
            len(failing),
        )


def _print_query_diff(
    old_queries: list[dict],
    new_queries: list[dict],
    cycle: int,
    path: str,
) -> None:
    """Log old and new query lists independently — no positional pairing assumed."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    sep = "─" * 68
    lines = [
        "",
        sep,
        f"[Validator] Cycle {cycle} ({path}) — before correction "
        f"({len(old_queries)} queries)",
        sep,
    ]
    for q in old_queries:
        lines.append(f"  #{q.get('id', '?')}  {q.get('question', '(no question)')}")
        lines.append(f"  code: {q.get('code', '(no code)')}\n")

    lines.append(sep)
    lines.append(
        f"[Validator] Cycle {cycle} ({path}) — after correction "
        f"({len(new_queries)} queries)"
    )
    lines.append(sep)
    for q in new_queries:
        lines.append(f"  #{q.get('id', '?')}  {q.get('question', '(no question)')}")
        lines.append(f"  code: {q.get('code', '(no code)')}\n")
    lines.append(f"{sep}\n")

    logger.debug("%s", "\n".join(lines))


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
