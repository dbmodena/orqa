"""Majority-vote judge panels.

A :class:`JudgePanel` replaces a single primary-model judge call with N
independent small-model judges that each receive the identical (static
instructions + schema) system message and per-item payload, vote in parallel,
and whose verdicts are aggregated by majority. Two panels exist in the
pipeline, configured under the ``judge_profiles`` key of the LLM YAML:

* ``plan``  — judges structured query plans before code generation
  (results saved under the run's ``plan_feedback``),
* ``code``  — judges the generated (question, code, result) triples
  (results saved per query under ``code_feedback``).

Rationale: the panel's judges are deliberately DIFFERENT models from the
generation model (a judge from the generator's own family tends to approve its
own reasoning style), and majority voting only helps when the judges' errors
are decorrelated — so profiles should mix model families, not sizes of one.

Each ``judge_profiles.<panel>`` entry is a model reference exactly like the
primary model (``LLMClient._resolve_model_spec``): a LiteLLM model
string, the name of a ``deployments`` entry (e.g. a dedicated OCI endpoint),
or a mapping with a ``model`` key plus per-judge litellm overrides; provider
params are fetched from ``provider_params`` by the resolved model's provider
prefix. A panel with no configured profiles reports
``is_configured == False`` and the caller falls back to the legacy
single-judge path, so older YAMLs keep working unchanged.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Optional

from pydantic import ValidationError

from orqa.agent.llm_client.LLMClientStructured import (
    LLMClientStructured,
    _looks_like_format_rejection,
    _needed_structural_repair,
    _usage_of,
    budget_exhausted,
    degenerate_repetition,
)
from orqa.agent.llm_client.messages import JudgeMessageBuilder, sanitize_messages
from orqa.agent.llm_client.coercion import errors_brief, validate_leniently


logger = logging.getLogger(__name__)

# Seconds a vote may take when the judge's own params set no ``timeout``.
JUDGE_TIMEOUT = 120


class JudgePanel(LLMClientStructured):
    """N-model majority-vote judge panel over one response schema.

    Each vote runs the same budget-aware loop as every structured call
    (see ``LLMClientStructured``): an explicit output budget (the caller's
    ``max_tokens``, ``JUDGE_MAX_TOKENS`` in agent.py) raised when a verdict is
    cut by the judge's own reasoning, native JSON-schema decoding (verdicts
    are short, where it is safe), shape coercion of ``"yes"`` votes and the
    like, one short repair call, and the repetition guard.

    The panel reuses the shared JSON-repair / Pydantic validation pipeline of
    :class:`LLMClientStructured`, but routes each vote to its own judge model
    as its own call target — a judge that fails after
    the configured retries is a failed vote by design, never silently served
    by a different model (that would collapse the panel's independence).

    Args:
        config_path: Path to the LLM YAML configuration.
        panel: Which ``judge_profiles`` list to load (``"plan"`` / ``"code"``).
        response_model: YAML config key of the per-judge response schema
            (e.g. ``"statement_judge"`` or ``"plan_judge"``).
        root_key: When the response schema wraps its items in a list (the
            ``Judgments.queries`` shape), the wrapper key to unwrap — the
            FIRST item is the judge's verdict. ``None`` for flat schemas.
        judge_count: Cap the panel to the first N entries of
            ``judge_profiles.<panel>`` (see the workflow yaml's
            ``tasks.judges.plan_mode``/``code_mode`` — "mono" resolves to
            1, "trio" to 3, translated by the caller before construction;
            this class only knows the resulting count, not the mono/trio
            vocabulary). ``None`` (the default) uses every configured
            profile unchanged — the pre-existing behavior. Aggregation
            itself (``_aggregate``) is already generic over N, so capping
            here needs no changes there: majority voting over 1 judge
            degenerates to trusting that judge's own verdict outright.
    """

    def __init__(
        self,
        config_path: Path,
        panel: str,
        response_model: str,
        root_key: Optional[str] = None,
        vote_fields: Optional[List[str]] = None,
        judge_count: Optional[int] = None,
    ):
        super().__init__(config_path, response_model)
        self.panel = panel
        self._root_key = root_key
        # Native JSON-schema decoding is per judge ENTRY (target): a
        # provider that rejects the parameter turns it off for that judge
        # only — two judges may share a model name on different endpoints.
        self._native_ok_by_judge: dict[str, bool] = {}
        # Layered voting (see _aggregate): when set, each named boolean field
        # is majority-voted independently across the judges, and the panel
        # approves only if EVERY layer's majority approves. When None (code
        # panel), the single per-judge `approved` verdict is majority-voted
        # as before.
        self.vote_fields = list(vote_fields) if vote_fields else None
        self._message_builder = JudgeMessageBuilder()
        raw_profiles = (self.config.get("judge_profiles") or {}).get(panel) or []
        if judge_count is not None:
            raw_profiles = raw_profiles[:judge_count]
        # A panel needs both its judges AND its response schema; missing
        # either (older YAMLs) simply leaves the panel unconfigured.
        self.judges = (
            self._normalize_profiles(raw_profiles)
            if self.response_model is not None
            else []
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def is_configured(self) -> bool:
        """True when at least one judge profile (and the schema) is configured."""
        return bool(self.judges)

    def _normalize_profiles(self, raw_profiles: Any) -> List[dict]:
        """Coerce ``judge_profiles.<panel>`` entries into judge descriptors.

        Accepts a plain model string, a deployment name, or a mapping
        carrying ``model`` plus per-judge litellm overrides. Each descriptor
        carries ``model`` (the label the votes and logs name the judge by —
        the deployment name when one was used), ``litellm_model`` (the model
        string that is called), ``target`` (the name its calls pass as
        ``model``), ``params`` (provider params + deployment params +
        overrides) and ``max_output_tokens`` (the deployment's cap, if any).
        Invalid entries are skipped with a log.
        """
        judges: List[dict] = []
        if not isinstance(raw_profiles, (list, tuple)):
            logger.warning(
                "judge_profiles.%s must be a list; got %r — panel disabled.",
                self.panel, type(raw_profiles).__name__,
            )
            return judges
        for idx, entry in enumerate(raw_profiles):
            # Outside the try: a judge other than $ORQA_ONLY_LLM stops the
            # run instead of being skipped.
            self._require_allowed(entry)
            try:
                spec = self._resolve_model_spec(entry)
            except ValueError as exc:
                logger.warning(
                    "Skipping invalid judge_profiles.%s entry #%d: %r (%s)",
                    self.panel, idx, entry, exc,
                )
                continue
            self._register_target(f"{self.panel}_judge_{idx}", spec, default_timeout=JUDGE_TIMEOUT)
            judges.append(
                {
                    "index": idx,
                    "model": spec["label"],
                    "litellm_model": spec["model"],
                    "target": f"{self.panel}_judge_{idx}",
                    "params": spec["params"],
                    # Alias kept for subclasses written against the former
                    # descriptor (experiments/structured_stability).
                    "overrides": spec["params"],
                    "max_output_tokens": spec["max_output_tokens"],
                }
            )
        return judges

    def _judge_response_format(self) -> dict:
        return self.response_format()

    def _judge_call(self, judge: dict, messages: list[dict], **kwargs) -> tuple[Any, bool]:
        """One vote's call to its judge's own target (throttle backoff
        shared with every other client). Returns
        ``(response, native_used)``."""
        native = self.native_schema and self._native_ok_by_judge.get(judge["target"], True)
        args = {
            "model": judge["target"],
            "messages": sanitize_messages(messages),
            **self._default_temperature_kwargs(judge["litellm_model"], judge["params"]),
            **kwargs,
        }
        if native:
            try:
                return self._completion_with_backoff(
                    {**args, "response_format": self._judge_response_format()}
                ), True
            except Exception as exc:  # noqa: BLE001
                if not _looks_like_format_rejection(exc):
                    raise
                logger.warning("Panel %s judge %s rejected response_format (%s); native schema off for it.",
                               self.panel, judge["model"], str(exc)[:160])
                self._native_ok_by_judge[judge["target"]] = False
        return self._completion_with_backoff(args), False

    def _validate_vote(self, json_data: Any, record: dict) -> Optional[dict]:
        """Strict, then coerced; unwraps ``root_key`` to the judge's verdict."""
        model = self.response_model
        try:
            result = model.model_validate(json_data).model_dump()
            record["outcome"] = "strict"
        except ValidationError as exc:
            record["strict_errors"] = errors_brief(exc.errors(include_url=False))
            inst, _, fixes, remaining = validate_leniently(json_data, model)
            record["fixes"] = fixes
            if inst is None:
                record["coerce_errors"] = errors_brief(remaining)
                return None
            record["outcome"] = "coerced"
            logger.info("Panel %s: verdict validated after shape coercion (%s)", self.panel, "; ".join(fixes[:6]))
            result = inst.model_dump()
        if self._root_key:
            items = result.get(self._root_key) or []
            if not items:
                record["coerce_errors"] = [f"$.{self._root_key}: no verdict in the list"]
                return None
            return items[0]
        return result

    def _judge_once(
        self, judge: dict, system_prompt: str, payload: str, **kwargs
    ) -> tuple[dict, dict]:
        """One judge's vote: fixed 2-block messages, budget-aware retry.

        The system message never changes across retries; parse errors only
        rebuild the user payload. A verdict cut by the output budget (a
        reasoning judge thinking through a 6-table payload) is retried with
        a bigger budget, not the same call again. Returns ``({}, usage)``
        when this judge exhausts its retries.
        """
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        user_payload = payload
        last_error: Optional[Exception] = None
        call_kwargs = dict(kwargs)
        cap = judge.get("max_output_tokens")
        self._initial_budget(call_kwargs, cap)

        for attempt in range(self.max_retries):
            record: dict = {"client": self._label, "kind": "call", "judge": judge["model"],
                            "attempt": attempt + 1, "max_tokens": call_kwargs.get("max_tokens")}
            t0 = time.perf_counter()
            try:
                messages = self._message_builder.build(system_prompt, user_payload)
                record["prompt"] = user_payload
                response, native_used = self._judge_call(judge, messages, **call_kwargs)
                record["native_schema"] = native_used
                record["model"] = response.get("model")
                usage = _usage_of(response)
                for key in usage_total:
                    usage_total[key] += usage.get(key, 0)
                choice = response["choices"][0]
                content = choice["message"]["content"]
                content = "" if content is None else str(content)
                record.update(finish_reason=choice.get("finish_reason"), usage=usage, raw=content,
                              elapsed_s=round(time.perf_counter() - t0, 2), raw_chars=len(content))
                truncated = budget_exhausted(choice.get("finish_reason"), usage, call_kwargs.get("max_tokens"), content)
                record["truncated"] = truncated
                if truncated and (not content.strip()
                                  or _needed_structural_repair(self._clean_json_response(content))):
                    record["outcome"] = "truncated_empty" if not content.strip() else "truncated_repaired"
                    last_error = ValueError("verdict cut at the output budget")
                    logger.warning("Panel %s judge %s: verdict cut at %s tokens; raising the budget",
                                   self.panel, judge["model"], call_kwargs.get("max_tokens"))
                    self._truncation_feedback(call_kwargs, cap)  # raises the budget in place
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    continue
                if not content.strip():
                    raise ValueError("Judge returned empty content.")
                loop = degenerate_repetition(content)
                if loop:
                    record["outcome"] = "degenerate_loop"
                    record["loop_unit"] = loop
                    last_error = ValueError(f"degenerate repetition: {loop!r}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    continue

                cleaned = self._clean_json_response(content)
                try:
                    json_data = self._repair_json(cleaned)
                except json.JSONDecodeError as e:
                    record["outcome"] = "json_error"
                    record["json_error"] = str(e)[:200]
                    last_error = e
                    if attempt < self.max_retries - 1:
                        user_payload = f"{payload}\n\n{self._format_json_error(cleaned, e)}"
                        time.sleep(self.retry_delay)
                    continue
                json_data = self._normalize_response(json_data, root_key=self._root_key)
                result = self._validate_vote(json_data, record)
                if result is not None:
                    return result, usage_total
                record["outcome"] = "invalid"
                errors = record.get("coerce_errors") or record.get("strict_errors") or []
                last_error = ValueError("; ".join(errors[:5]))
                if self.repair_call:
                    repaired, r_usage, r_record = self._repair_vote(
                        judge, system_prompt, json_data, errors, **call_kwargs
                    )
                    for key in usage_total:
                        usage_total[key] += r_usage.get(key, 0)
                    self._observe({**r_record, "attempt": attempt + 1})
                    if repaired is not None:
                        record["outcome"] = "invalid->repaired"
                        return repaired, usage_total
                if attempt < self.max_retries - 1:
                    user_payload = (
                        f"{payload}\n\nSCHEMA VALIDATION ERROR — your JSON did not match the required "
                        "schema on these fields:\n" + "\n".join(f"  - {e}" for e in errors[:40])
                        + "\nFix exactly these fields and return the full JSON object again."
                    )
                    time.sleep(self.retry_delay)
            except Exception as e:  # noqa: BLE001
                record["outcome"] = "error"
                record["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = (
                        max(self.retry_delay, self.throttle_max_delay)
                        if self.is_throttled(e) else self.retry_delay
                    )
                    time.sleep(delay)
            finally:
                self._observe(record)

        logger.warning(
            "Panel %s judge %s produced no valid verdict after %d attempts: %s",
            self.panel, judge["model"], self.max_retries, last_error,
        )
        return {}, usage_total

    def _repair_vote(self, judge: dict, system_prompt: str, json_data: Any, errors: List[str],
                     **kwargs) -> tuple[Optional[dict], dict, dict]:
        """One short correction call over the judge's own JSON verdict."""
        record: dict = {"client": self._label, "kind": "repair", "judge": judge["model"]}
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        user = (
            "The JSON verdict below does not satisfy the required schema. Validation errors:\n"
            + "\n".join(f"- {e}" for e in errors[:40])
            + "\n\nReturn the complete corrected JSON object only, keeping every value that is already "
            "valid and changing only what the errors require.\n\n```json\n"
            + json.dumps(json_data, ensure_ascii=False, default=str) + "\n```"
        )
        record["prompt"] = user
        t0 = time.perf_counter()
        try:
            response, native_used = self._judge_call(judge, self._message_builder.build(system_prompt, user), **kwargs)
            record["model"] = response.get("model")
            usage = _usage_of(response)
            choice = response["choices"][0]
            content = choice["message"]["content"] or ""
            record.update(native_schema=native_used, finish_reason=choice.get("finish_reason"), usage=usage,
                          raw=content, elapsed_s=round(time.perf_counter() - t0, 2))
            data = self._normalize_response(self._repair_json(self._clean_json_response(content)),
                                            root_key=self._root_key)
            result = self._validate_vote(data, record)
            if result is not None:
                record["outcome"] = "repaired:" + record["outcome"]
                return result, usage, record
            record["outcome"] = "repair_failed"
        except Exception as exc:  # noqa: BLE001
            record["outcome"] = "repair_error"
            record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return None, usage, record

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, votes: List[dict]) -> dict:
        """Merge per-judge verdicts into one majority judgment.

        Default (code panel): approval requires a strict majority of the
        VALID votes on the per-judge ``approved`` verdict.

        Layered (``vote_fields`` set, plan panel): each named boolean field
        is majority-voted independently — e.g. ``plan_approval`` and
        ``table_usage_approval`` — and the panel approves only when EVERY
        layer's majority approves. This is deliberately NOT a majority over
        per-judge overall verdicts: two judges each failing a *different*
        layer must not sink a plan both layers' majorities accept, and a
        layer a majority rejects must sink it even if every judge's other
        layer vote passes.

        In both modes a failed judge abstains rather than rejects, and a tie
        or an all-failed panel rejects, so infrastructure failures degrade
        toward the correction loop, never toward silently approving.

        The merged dict keeps the schema fields downstream consumers read
        (``feedback``/``suggestions``/``response``/... taken from the leading
        approving judgment when approved, or joined across the blaming
        judgments when rejected — in layered mode the judges that voted a
        FAILING layer down, not every overall rejector) and always carries
        the per-layer majority results plus the full per-judge record under
        ``panel`` so the votes can be saved as plan/code feedback.
        """
        valid = [v for v in votes if v["judgment"]]
        approving = [v for v in valid if v["judgment"].get("approved")]
        rejecting = [v for v in valid if not v["judgment"].get("approved")]

        layer_results: dict = {}
        if self.vote_fields:
            for field in self.vote_fields:
                yes = sum(1 for v in valid if v["judgment"].get(field))
                layer_results[field] = yes > len(valid) - yes
            approved = bool(valid) and all(layer_results.values())
        else:
            approved = len(approving) > len(rejecting)

        if approved:
            # In layered mode every-judge-rejects can still aggregate to
            # approved (each failing a different layer) — fall back to any
            # valid judgment for the non-vote fields then.
            leading = approving[0] if approving else valid[0]
            merged = dict(leading["judgment"])
        elif valid:
            # Whose feedback explains the rejection: in layered mode, the
            # judges that voted a failing layer down; otherwise the overall
            # rejectors.
            if layer_results:
                failing = [f for f, ok in layer_results.items() if not ok]
                blaming = [
                    v for v in valid
                    if any(not v["judgment"].get(f) for f in failing)
                ] or rejecting or valid
            else:
                blaming = rejecting
            merged = dict(blaming[0]["judgment"])
            merged["feedback"] = "\n".join(
                f"[{v['judge']}] {v['judgment'].get('feedback', '')}".strip()
                for v in blaming
            )
            merged["suggestions"] = "\n".join(
                f"[{v['judge']}] {v['judgment'].get('suggestions', '')}".strip()
                for v in blaming
                if str(v["judgment"].get("suggestions", "")).strip()
            )
            # Union of the blaming judges' list verdicts, first-appearance
            # order: violated_criteria (code panel) and unjustified_tables
            # (plan panel — read by the unjustifiable-group early abort).
            for list_field in ("violated_criteria", "unjustified_tables"):
                union: List[Any] = []
                for v in blaming:
                    for item in v["judgment"].get(list_field, []) or []:
                        if item not in union:
                            union.append(item)
                if union or list_field in merged:
                    merged[list_field] = union
        else:
            merged = {
                "feedback": "No judge in the panel returned a valid verdict.",
                "suggestions": "",
            }
        # Per-layer majority results (layered mode only): downstream reads
        # e.g. merged["table_usage_approval"] to tell a table-driven
        # rejection from a plan-quality one.
        merged.update(layer_results)

        merged["approved"] = approved
        if not approved:
            # The business-insight fields are only meaningful on approval
            # (judge.md contract: empty string if not approved).
            for field in ("response", "translated_response"):
                if field in merged:
                    merged[field] = ""

        merged["panel"] = {
            "panel": self.panel,
            "approve_votes": len(approving),
            "reject_votes": len(rejecting),
            "failed_votes": len(votes) - len(valid),
            "votes": [
                {
                    "judge": v["judge"],
                    **(
                        v["judgment"]
                        if v["judgment"]
                        else {"error": "no valid verdict"}
                    ),
                }
                for v in votes
            ],
        }
        return merged


# ── Single primary-model code judge ───────────────────────────────────────────

class LLMStatementJudge(LLMClientStructured):
    """
    Single primary-model code judge (the fallback when the LLM yaml configures
    no ``judge_profiles.code`` panel — see ``JudgePanel``).

    Uses a fixed 2-block message structure (system + query evaluation) that is
    rebuilt from scratch on each call — no retained history between calls.
    """

    # A verdict is a short output: native JSON-schema decoding is safe there
    # (the judge panels run it too).
    native_schema = True

    def __init__(self, config_path: Path):
        super().__init__(config_path, "statement_judge")

    def complete(self, prompt: str, data: Any = None, **kwargs) -> Any:
        """
        Make a structured completion request with fixed 2-block messages.

        ``prompt`` carries ONLY the static judge instructions; ``data`` carries
        the per-query evaluation payload. The instructions plus the (equally
        static) output-schema constraint form the system message, so the system
        message is byte-identical across every judge call in a run — the
        provider's prompt cache serves that whole prefix, and only the short
        user message (the payload) is uncached.

        The call is the shared budget-aware structured loop
        (``LLMClientStructured.complete``): on a parse error the user payload
        is rebuilt with the error feedback (the system message never changes),
        a verdict cut by the output budget is retried with a bigger one.

        Returns ``(result_dict, usage_total)`` on success; ``({}, usage_total)`` on
        exhausted retries.
        """
        # Never empty — providers reject zero-token messages (OCI 400:
        # "message must be at least 1 token long").
        query_payload = (
            f"Queries:\n{data}\n\n"
            "Evaluate the queries above following the instructions and "
            "return only the JSON verdict."
        )
        return LLMClientStructured.complete(
            self, query_payload, root_key="queries", system_prompt=prompt, **kwargs
        )
