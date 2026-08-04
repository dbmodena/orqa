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

Each ``judge_profiles.<panel>`` entry is a LiteLLM model string (or a mapping
with a ``model`` key plus per-judge litellm overrides); provider params are
fetched from ``provider_params`` by the model's provider prefix, exactly like
the primary/fallback models. A panel with no configured profiles reports
``is_configured == False`` and the caller falls back to the legacy
single-judge path, so older YAMLs keep working unchanged.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Optional

from litellm import Router
from pydantic import ValidationError

from ..llm_client.LLMClientStructured import LLMClientStructured
from ..utility.message_builder import JudgeMessageBuilder, sanitize_messages

logger = logging.getLogger(__name__)


class JudgePanel(LLMClientStructured):
    """N-model majority-vote judge panel over one response schema.

    The panel reuses the shared JSON-repair / Pydantic validation pipeline of
    :class:`LLMClientStructured`, but routes each vote to its own judge model
    via a dedicated router with NO fallback chain — a judge that fails after
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
        self.panel_router = self._build_panel_router() if self.judges else None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def is_configured(self) -> bool:
        """True when at least one judge profile (and the schema) is configured."""
        return bool(self.judges)

    def _normalize_profiles(self, raw_profiles: Any) -> List[dict]:
        """Coerce ``judge_profiles.<panel>`` entries into judge descriptors.

        Accepts a plain model string or a mapping carrying ``model`` plus
        per-judge litellm overrides. Invalid entries are skipped with a log.
        """
        judges: List[dict] = []
        if not isinstance(raw_profiles, (list, tuple)):
            logger.warning(
                "judge_profiles.%s must be a list; got %r — panel disabled.",
                self.panel, type(raw_profiles).__name__,
            )
            return judges
        for idx, entry in enumerate(raw_profiles):
            if isinstance(entry, str) and entry.strip():
                model, overrides = entry.strip(), {}
            elif isinstance(entry, dict) and str(entry.get("model", "")).strip():
                entry = dict(entry)
                model = str(entry.pop("model")).strip()
                overrides = entry
            else:
                logger.warning(
                    "Skipping invalid judge_profiles.%s entry #%d: %r",
                    self.panel, idx, entry,
                )
                continue
            judges.append(
                {
                    "index": idx,
                    "model": model,
                    "router_name": f"{self.panel}_judge_{idx}",
                    "overrides": overrides,
                }
            )
        return judges

    def _build_panel_router(self) -> Router:
        """One router entry per judge; provider params resolved by prefix.

        No fallback chain: each vote must come from its own judge model or
        count as failed.
        """
        model_list = []
        for judge in self.judges:
            provider_params = self._get_provider_specific_params(judge["model"])
            model_list.append(
                {
                    "model_name": judge["router_name"],
                    "litellm_params": {
                        "model": judge["model"],
                        **provider_params,
                        **judge["overrides"],
                    },
                }
            )
        return Router(
            model_list=model_list,
            fallbacks=[],
            num_retries=0,
            timeout=120,
            set_verbose=False,
        )

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    def evaluate(
        self, instructions: str, payload: str, **kwargs
    ) -> tuple[dict, dict]:
        """Run every judge on the same payload and majority-aggregate.

        ``instructions`` carries ONLY the static judge instructions; they are
        combined with the (equally static) schema constraint into the system
        message, so each judge model sees a byte-identical prefix on every
        call and its provider prompt cache is hit. ``payload`` is the per-item
        user message.

        Returns ``(judgment, usage_total)`` where ``judgment`` is the
        aggregated verdict dict — schema fields merged from the majority side,
        plus a ``panel`` sub-dict holding every per-judge vote — and is never
        empty (an all-judges-failed panel yields a rejection with the reason).
        """
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not self.is_configured:
            return (
                {"approved": False, "feedback": f"Judge panel {self.panel!r} is not configured.", "suggestions": ""},
                usage_total,
            )

        system_prompt = self.reform_prompt_constraint(instructions)
        votes: List[dict] = []
        with ThreadPoolExecutor(max_workers=len(self.judges)) as pool:
            future_to_judge = {
                pool.submit(self._judge_once, judge, system_prompt, payload, **kwargs): judge
                for judge in self.judges
            }
            for future in as_completed(future_to_judge):
                judge = future_to_judge[future]
                try:
                    judgment, usage = future.result()
                except Exception as exc:  # noqa: BLE001 — one judge must not sink the panel
                    logger.warning(
                        "Panel %s judge %s failed: %s", self.panel, judge["model"], exc
                    )
                    judgment, usage = {}, {}
                for key in usage_total:
                    usage_total[key] += (usage or {}).get(key, 0)
                votes.append({"index": judge["index"], "judge": judge["model"], "judgment": judgment})

        votes.sort(key=lambda v: v["index"])
        return self._aggregate(votes), usage_total

    def _judge_once(
        self, judge: dict, system_prompt: str, payload: str, **kwargs
    ) -> tuple[dict, dict]:
        """One judge's vote: fixed 2-block messages, retry-with-error-feedback.

        Mirrors ``LLMStatementJudge.complete`` — the system message never
        changes across retries; parse errors only rebuild the user payload.
        Returns ``({}, usage)`` when this judge exhausts its retries.
        """
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        user_payload = payload
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                messages = sanitize_messages(
                    self._message_builder.build(system_prompt, user_payload)
                )
                response = self.panel_router.completion(
                    model=judge["router_name"],
                    messages=messages,
                    **self._default_temperature_kwargs(judge["model"], judge["overrides"]),
                    **kwargs,
                )
                usage = response["usage"]
                usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
                usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
                usage_total["total_tokens"] += usage.get("total_tokens", 0)

                content = response["choices"][0]["message"]["content"]
                if content is None or not str(content).strip():
                    raise ValueError("Judge returned empty content.")

                cleaned = self._clean_json_response(content)
                try:
                    json_data = self._repair_json(cleaned)
                    json_data = self._normalize_response(json_data, root_key=self._root_key)
                    result = self.response_model.model_validate(json_data).model_dump()
                    if self._root_key:
                        items = result.get(self._root_key) or []
                        return (items[0] if items else {}), usage_total
                    return result, usage_total
                except json.JSONDecodeError as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        user_payload = f"{payload}\n\n{self._format_json_error(cleaned, e)}"
                        time.sleep(self.retry_delay)
                except ValidationError as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        user_payload = f"{payload}\n\n{self._format_validation_error(e)}"
                        time.sleep(self.retry_delay)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        logger.warning(
            "Panel %s judge %s produced no valid verdict after %d attempts: %s",
            self.panel, judge["model"], self.max_retries, last_error,
        )
        return {}, usage_total

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
