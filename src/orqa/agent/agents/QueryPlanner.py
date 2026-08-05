"""Structured query planner (task 5.3).

:class:`QueryPlanner` turns per-table analyses plus the upstream ``match`` /
``involved_cols`` relationship constraints into a structured query plan (see
:mod:`orqa.agent.prompting.models`): an ordered list of plan-step decomposition
steps and the ``table_links`` carried over from the mandatory upstream
constraints. The planner is **kind-aware**: a ``"SQL"`` planner produces
:class:`~orqa.agent.prompting.models.SQLQueryPlan`, while a ``"PANDAS"``
planner produces :class:`~orqa.agent.prompting.models.PandasQueryPlan`.

Design constraints implemented here (Requirements 5.1, 5.2, 6.1, 6.2):

* The provided ``match`` / ``involved_cols`` links are the only **verified**
  relationships. The planner prompt states tables may only be combined through
  them (in any composition shape — chained or independent branches), and the
  produced plan preserves them **unchanged** in ``table_links`` regardless of
  what the language model returns for that field.
* Column statistics (``TableStats``) are injected into the planner prompt.

Plan *validation* and the re-request / free-text fallback (task 5.4) are
implemented here too: :meth:`QueryPlanner.validate_plan` assigns contiguous
``1..N`` step orders and checks table/column references, and :meth:`plan`
re-requests the plan once on failure before falling back to a schema-valid
free-text plan (Requirements 5.3, 5.4, 5.5, 5.6).
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union, get_args

from pydantic import BaseModel, ValidationError

from ..llm_client.LLMClientStructured import LLMClientStructured
from ..prompting.models import (
    _DIFFICULTY_LEVELS,
    _RESULT_TYPES,
    PandasPlanStep,
    PandasQueryPlan,
    PandasQueryPlanSet,
    SQLPlanStep,
    SQLQueryPlan,
    SQLQueryPlanSet,
    TableStats,
)
from ..prompting.prompts import QueryPlannerPrompt
from ..utility.difficulty_estimator import build_reconciliation_feedback, estimate_plan_tier
from ..utility.structured_outputs import QueryLink, Table
from ...utils import shield_dataframe_for_prompt

logger = logging.getLogger(__name__)

# Ops whose entire purpose is to produce a NEW column rather than read an
# existing one (a groupby's count/sum output, a derive's computed column).
# validate_plan's column-existence check (Requirement 5.5) exempts these ops:
# an unrecognized column name here is the step's own output, not a bad
# reference.
_COLUMN_PRODUCING_OPS = frozenset({"aggregate", "derive"})

# An underscore between two letters in a QUESTION is a near-certain sign a
# raw column identifier got pasted in verbatim (natural language never uses
# underscores) — e.g. "the increase in Pre-K seats (grade_pk_half_day_full_day)"
# or "the store_and_fwd_flag is 'N'". Zero known false positives in practice,
# so this alone is grounds for rejection — see validate_plan.
_UNDERSCORE_IN_QUESTION_RE = re.compile(r"[a-zA-Z]_[a-zA-Z]")


def _question_leaks_implementation(question: str) -> Optional[str]:
    """Human-readable reason ``question`` leaks a raw column identifier, or
    ``None`` when it's clean. See ``validate_plan``."""
    text = question or ""
    if _UNDERSCORE_IN_QUESTION_RE.search(text):
        return (
            "contains an underscore, which almost always means a raw column "
            "identifier was pasted into the question text verbatim"
        )
    return None


QueryPlan = Union[SQLQueryPlan, PandasQueryPlan]
PlanStep = Union[SQLPlanStep, PandasPlanStep]

# Neither request path below set max_tokens at all before this — relying on
# whatever OCI's own server-side default happens to be for reasoning models
# (openai.gpt-oss-*, google.gemini-2.5-flash) that burn hidden thinking
# tokens out of the same budget as the visible response (see the judges'
# own JUDGE_MAX_TOKENS in agent.py, raised after a real starvation incident
# there). A live instrumented run never observed truncation with no cap set
# (usage up to ~4900 tokens, always finish_reason="stop") — so this isn't a
# fix for an observed failure, just closing the same unmanaged-parameter gap
# the judges already had closed: an explicit, generous, inspectable ceiling
# instead of an opaque provider default with no lever to raise it.
GENERATION_MAX_TOKENS = 8000


class PlanValidationError(ValueError):
    """Raised when a structured query plan fails structural validation.

    Signals that a plan references an unknown table alias, references a column
    that does not exist in any referenced table, or has no steps
    (Requirements 5.3-5.5).
    """


class QueryPlannerClient(LLMClientStructured):
    """Structured LLM client that returns a single kind-appropriate query plan.

    Reuses the shared JSON-repair / retry pipeline from
    :class:`LLMClientStructured` but pins the response model to
    :class:`SQLQueryPlan` or :class:`PandasQueryPlan` depending on ``kind``.
    """

    def __init__(self, config_path: Path, kind: str):
        # ``query_planner`` is a valid config key (legacy ``QueryPlan``); load it
        # to satisfy the base constructor, then override with the kind-specific
        # structured model.
        super().__init__(config_path, response_model="query_planner")
        self.kind = kind
        self.response_model = SQLQueryPlan if kind == "SQL" else PandasQueryPlan

    def request_plan(self, prompt: str, **kwargs) -> tuple[dict, dict]:
        """Return ``(plan_dict, usage)``; ``plan_dict`` is ``{}`` on failure."""
        return self.complete(prompt, root_key=None, **kwargs)


class QueryPlanBatchClient(LLMClientStructured):
    """Structured LLM client that returns a kind-appropriate batch of query plans.

    Sibling of :class:`QueryPlannerClient` for the multi-plan request (several
    query plans in one call). Kept as a separate client (rather than toggling
    ``response_model`` on the same instance) so a single
    :class:`QueryPlanner` can freely mix single-plan and batch-plan requests
    without cross-contaminating response-model state.
    """

    def __init__(self, config_path: Path, kind: str):
        super().__init__(config_path, response_model="query_planner")
        self.kind = kind
        self.response_model = SQLQueryPlanSet if kind == "SQL" else PandasQueryPlanSet

    def request_plan_batch(self, prompt: str, **kwargs) -> tuple[dict, dict]:
        """Return ``(plan_set_dict, usage)``; ``plan_set_dict`` is ``{}`` on failure."""
        return self.complete(prompt, root_key=None, **kwargs)


class QueryPlanner:
    """Produces a kind-appropriate structured query plan from analyses and constraints."""

    def __init__(self, config_path: Path, kind: str, client: Optional[Any] = None):
        """Create a planner.

        Args:
            config_path: Path to the LLM YAML configuration.
            kind: Generation kind (``"PANDAS"`` or ``"SQL"``). Determines which
                plan model (and prompt wording) this planner produces.
            client: An object exposing ``request_plan(prompt) -> (dict, usage)``.
                Injected for testing; when omitted a :class:`QueryPlannerClient`
                is constructed lazily on first use.
        """
        self.config_path = config_path
        self.kind = kind
        self._client = client
        # Separate lazily-constructed client for the multi-plan batch request
        # (``plan_batch``); kept distinct from ``self._client`` so the two
        # request shapes (single plan vs. a ``plans: [...]`` list) never share
        # response-model state. May also be injected for testing.
        self._batch_client: Optional[Any] = None

    @property
    def _is_pandas(self) -> bool:
        return self.kind == "PANDAS"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        analyses: Sequence[dict],
        aliases: dict,
        match: Any,
        involved_cols: Optional[dict],
        stats: Sequence[TableStats],
        languages: Optional[Sequence[str]] = None,
        dfs: Optional[Sequence[Any]] = None,
    ) -> QueryPlan:
        """Produce a structured query plan.

        Args:
            analyses: Per-table analysis dicts (one per alias).
            aliases: Mapping of alias -> dataset name.
            match: The upstream match constraint. Either a preformatted string,
                ``None``, or an already-structured list of link dicts / QueryLinks.
            involved_cols: Mapping of alias -> the columns that participate in the
                mandatory relationship for that table.
            stats: Per-table column statistics injected into the prompt.
            languages: Detected languages for question phrasing.
            dfs: The tables in scope (one per alias, same order), used to inject
                a real up-to-10-row sample per table so questions ground their
                concrete values in actually observed data rather than invented
                ones.

        Returns:
            A :class:`SQLQueryPlan` or :class:`PandasQueryPlan` (per ``self.kind``)
            whose ``table_links`` preserve the provided constraints unchanged.
        """
        languages = list(languages or [])

        # Build the mandatory relationship constraints ONCE. These are preserved
        # verbatim in the produced plan (Requirements 6.1, 6.2).
        constraint_links = self._build_constraint_links(match, involved_cols, aliases)

        prompt = self._build_prompt(
            analyses, aliases, constraint_links, stats, languages, dfs=dfs,
        )

        # Known columns per alias are derived from the column statistics so plan
        # validation (Requirement 5.5) can check every referenced column exists.
        known_columns = self._known_columns(stats)

        raw_plan, _usage = self._request_plan(prompt)
        candidate = self._assemble_plan(raw_plan, constraint_links)

        # First validation attempt. On success the normalised plan (contiguous
        # 1..N orders) is returned directly (Requirement 5.3).
        try:
            return self.validate_plan(candidate, aliases, known_columns)
        except PlanValidationError as exc:
            first_error = str(exc)
            logger.warning(
                "Structured plan validation failed (%s); re-requesting once.\n"
                "  raw plan: %s",
                first_error,
                json.dumps(raw_plan, ensure_ascii=False, default=str),
            )

        # Re-request the plan exactly once, feeding the validation error back to
        # the model so it can correct the offending references (Requirement 5.6).
        retry_prompt = self._retry_prompt(prompt, first_error)
        raw_plan_retry, _usage_retry = self._request_plan(retry_prompt)
        retry_candidate = self._assemble_plan(raw_plan_retry, constraint_links)

        try:
            return self.validate_plan(retry_candidate, aliases, known_columns)
        except PlanValidationError as exc_retry:
            logger.warning(
                "Structured plan re-request also failed (%s); "
                "falling back to a free-text plan.\n  raw plan: %s",
                exc_retry,
                json.dumps(raw_plan_retry, ensure_ascii=False, default=str),
            )
            return self._free_text_fallback(retry_candidate, aliases, constraint_links)

    def plan_batch(
        self,
        analyses: Sequence[dict],
        aliases: dict,
        match: Any,
        involved_cols: Optional[dict],
        stats: Sequence[TableStats],
        languages: Optional[Sequence[str]] = None,
        num_plans: int = 3,
        dfs: Optional[Sequence[Any]] = None,
        retrievable_keywords: Optional[list[str]] = None,
    ) -> List[QueryPlan]:
        """Produce several independent query plans in a single LLM call.

        Unlike :meth:`plan` (one plan, one question), this asks the model for
        ``num_plans`` *distinct* business questions over the same tables, each
        with its own ``question``/``question_keywords`` and its own ordered
        ``steps``.

        Each returned plan preserves the same mandatory ``table_links`` as
        :meth:`plan` (the upstream ``match``/``involved_cols`` constraints are
        never altered), and each plan is independently validated: a plan that
        fails validation is re-requested once on its own (not the whole batch),
        then falls back to a schema-valid free-text plan if the retry also
        fails (mirrors the per-plan behaviour of :meth:`plan`/:meth:`validate_plan`).

        Args:
            analyses: Per-table analysis dicts (one per alias).
            aliases: Mapping of alias -> dataset name.
            match: The upstream match constraint (see :meth:`plan`).
            involved_cols: Mapping of alias -> mandatory relationship columns.
            stats: Per-table column statistics injected into the prompt.
            languages: Detected languages for question phrasing.
            num_plans: How many distinct plans to request. The model may return
                fewer (e.g. a single-table run with little to decompose); it is
                never padded back up to ``num_plans``.
            dfs: The tables in scope (one per alias, same order), used to inject
                a real up-to-10-row sample per table so questions ground their
                concrete values in actually observed data rather than invented
                ones.
            retrievable_keywords: A keyword set EMPIRICALLY VERIFIED (see
                ``orqa.agent.utility.keyword_suggestion.
                suggest_retrievable_keywords``) to surface every table in
                ``aliases`` within the plan judge's own keyword-searchability
                top-K, computed BEFORE this call against the real reverse
                index — not a guess. When given, every plan's `question`
                must weave these terms in naturally (not just list them in
                `question_keywords`), so the run starts from a proven-
                retrievable footing instead of discovering retrievability by
                trial and error across correction rounds. ``None``/empty
                when unavailable (no index configured, or the group could
                not be resolved — see the pre-planning check in
                ``StatementOrchestrator._run``) — the prompt then falls back
                to its ordinary retrievability guidance alone.

        Returns:
            A non-empty list of kind-appropriate query plans, each independently
            schema-valid, each preserving the mandatory ``table_links``.
        """
        languages = list(languages or [])
        num_plans = max(1, int(num_plans))

        constraint_links = self._build_constraint_links(match, involved_cols, aliases)
        known_columns = self._known_columns(stats)

        prompt = self._build_prompt(
            analyses, aliases, constraint_links, stats, languages,
            num_plans=num_plans, dfs=dfs,
            retrievable_keywords=retrievable_keywords,
        )
        raw_set, _usage = self._request_plan_batch(prompt)
        raw_plans = self._extract_raw_plans(raw_set)

        plans: List[QueryPlan] = [
            self._validate_or_retry_one(
                raw_plan, prompt, constraint_links, aliases, known_columns,
            )
            for raw_plan in raw_plans
        ]

        if not plans:
            # The model returned nothing usable at all — fall back to a single
            # free-text plan so the caller always gets at least one plan back.
            empty = self._empty_plan(constraint_links)
            plans = [self._free_text_fallback(empty, aliases, constraint_links)]

        plans = self._dedupe_difficulties(
            plans, analyses, aliases, match, involved_cols, stats, languages,
            dfs, retrievable_keywords,
        )
        plans = self._reconcile_difficulty(
            plans, analyses, aliases, match, involved_cols, stats, languages,
            dfs, retrievable_keywords,
        )
        plans = [self._pin_table_descriptions(p, analyses) for p in plans]
        plans = [self._pin_retrievable_keywords(p, retrievable_keywords) for p in plans]

        return plans

    def _dedupe_difficulties(
        self,
        plans: List[QueryPlan],
        analyses: Sequence[dict],
        aliases: dict,
        match: Any,
        involved_cols: Optional[dict],
        stats: Sequence[TableStats],
        languages: Sequence[str],
        dfs: Optional[Sequence[Any]],
        retrievable_keywords: Optional[list[str]],
    ) -> List[QueryPlan]:
        """Ensure a batch has no duplicate difficulty tiers — deterministic,
        code-level, no judge involved.

        Keeps each tier's FIRST occurrence unchanged. Every later occurrence
        of an already-used tier is reassigned to one of the tiers the batch
        is still missing (in easy -> medium -> hard order) and sent through
        a forced revision so its STEPS are redesigned to genuinely earn that
        new tier — never just relabeled. Complements (does not overlap with)
        the plan judge's per-plan Check 7: this pass guarantees tier
        UNIQUENESS across the batch; Check 7 guarantees each individual
        plan's tier is honest.
        """
        if len(plans) < 2:
            return plans

        all_tiers = ["easy", "medium", "hard"]
        used = {p.difficulty for p in plans}
        missing = iter(t for t in all_tiers if t not in used)
        seen: set = set()
        result: List[QueryPlan] = []
        for plan in plans:
            if plan.difficulty not in seen:
                seen.add(plan.difficulty)
                result.append(plan)
                continue
            target = next(missing, None)
            if target is None:
                # No missing tier left to reassign to (only possible with a
                # batch of more than 3 plans, which the live pipeline never
                # requests) — leave the duplicate as-is rather than guess.
                result.append(plan)
                continue
            logger.info(
                "Plan difficulty dedup: reassigning duplicate '%s' -> '%s'",
                plan.difficulty, target,
            )
            retarget = plan.model_copy(update={"difficulty": target})
            feedback = (
                f"This plan's difficulty (`{plan.difficulty}`) duplicates "
                "another plan already in this batch. It is REASSIGNED to "
                f"`{target}` — redesign the STEPS (add or remove operations "
                f"or tables) so the plan's actual complexity genuinely earns "
                f"the `{target}` tier per the DIFFICULTY rubric, not just the "
                "label."
            )
            revised, _usage = self.revise_plan(
                retarget, feedback, analyses, aliases, match, involved_cols,
                stats, languages, dfs=dfs,
                retrievable_keywords=retrievable_keywords,
            )
            seen.add(target)
            result.append(revised if revised is not None else retarget)
        return result

    def _reconcile_difficulty(
        self,
        plans: List[QueryPlan],
        analyses: Sequence[dict],
        aliases: dict,
        match: Any,
        involved_cols: Optional[dict],
        stats: Sequence[TableStats],
        languages: Sequence[str],
        dfs: Optional[Sequence[Any]],
        retrievable_keywords: Optional[list[str]],
    ) -> List[QueryPlan]:
        """Verify each plan's declared `difficulty` against the deterministic
        structural/data-engineering estimate (see
        ``orqa.agent.utility.difficulty_estimator``) — code-level, no judge
        involved, same "deterministic, no judge involved" spirit as
        :meth:`_dedupe_difficulties` — and force one corrective revision on
        a mismatch, feeding the estimate's own named gaps back as feedback
        rather than a vague "make it harder/easier" instruction.

        Runs AFTER :meth:`_dedupe_difficulties` (tier uniqueness first, so a
        reassigned duplicate's revision is what gets checked here) and
        BEFORE the plan ever reaches the judge panel — the same
        "catch what's mechanically checkable before spending a judge call"
        principle already used for keyword-searchability (see
        ``agent._judge_plans``). The plan judge's own ``difficulty_approval``
        layer (``plan_judge.md`` Check 5) stays the final backstop for the
        two things this estimator can't compute — whether flag/bucket steps
        preserve genuinely DIFFERENT messy-data patterns, and whether a
        bucket step's branching reflects real domain judgment rather than
        an arbitrary split — so a mismatch this pass misses is still caught
        downstream; this pass just spares most plans that round-trip.

        One revision attempt per plan, mirroring the rest of this file's
        single-retry convention — a plan still mismatched after its
        revision is passed through unchanged and left to the judge panel.
        """
        result: List[QueryPlan] = []
        for plan in plans:
            estimate = estimate_plan_tier(plan)
            if estimate.tier == plan.difficulty:
                result.append(plan)
                continue
            feedback = build_reconciliation_feedback(estimate, plan.difficulty)
            logger.info(
                "Plan difficulty reconciliation: declared '%s', computed "
                "'%s' — revising once. %s",
                plan.difficulty, estimate.tier, estimate.explanation,
            )
            revised, _usage = self.revise_plan(
                plan, feedback, analyses, aliases, match, involved_cols,
                stats, languages, dfs=dfs,
                retrievable_keywords=retrievable_keywords,
            )
            candidate = revised if revised is not None else plan
            re_estimate = estimate_plan_tier(candidate)
            if re_estimate.tier != candidate.difficulty:
                logger.warning(
                    "Plan difficulty reconciliation: still mismatched after "
                    "revision (declared '%s', computed '%s') — leaving as-is "
                    "for the judge panel to catch.",
                    candidate.difficulty, re_estimate.tier,
                )
            result.append(candidate)
        return result

    def _pin_table_descriptions(
        self, plan: QueryPlan, analyses: Sequence[dict]
    ) -> QueryPlan:
        """Overwrite each `tables[].description`/`.keywords` with the cached
        table-analysis value for that alias — deterministic, no judge
        involved, same spirit as :meth:`_dedupe_difficulties` /
        :meth:`_reconcile_difficulty` above.

        The prompt (query_planner.md) already asks the model to copy these
        verbatim from TABLE-LEVEL ANALYSIS, but that is a request, not a
        guarantee: the model transcribes free text into its own structured
        output, and was observed to reproduce a shorter/differently-worded
        description for the SAME table across different plans in the same
        batch (or across a correction round's fresh call). Since every
        table's canonical description/keywords already live in `analyses`
        (cached once per table by `TableAnalysisAgent`), pinning them here
        removes the drift for free — no extra LLM call — and keeps every
        plan's copy byte-identical, which is what downstream topic/temporal
        grounding checks (and the judge panels reading `tables[].reason`
        alongside it) assume it already was.
        """
        by_alias = {
            a.get("alias"): a
            for a in analyses
            if isinstance(a, dict) and a.get("alias")
        }
        new_tables = []
        changed = False
        for table in plan.tables:
            source = by_alias.get(table.name)
            if source is None:
                new_tables.append(table)
                continue
            canonical_description = source.get("table_description") or table.description
            canonical_keywords = source.get("table_keywords") or table.keywords
            if (
                canonical_description != table.description
                or canonical_keywords != table.keywords
            ):
                changed = True
                table = table.model_copy(
                    update={
                        "description": canonical_description,
                        "keywords": canonical_keywords,
                    }
                )
            new_tables.append(table)
        return plan.model_copy(update={"tables": new_tables}) if changed else plan

    def _pin_retrievable_keywords(
        self, plan: QueryPlan, retrievable_keywords: Optional[list[str]]
    ) -> QueryPlan:
        """Guarantee `question_keywords` contains every keyword the
        pre-planning search (see `keyword_suggestion.suggest_retrievable_keywords`,
        called before this planner ever runs — `retrievable_keywords` is
        only ever passed in already EMPIRICALLY VERIFIED against the real
        reverse index) proved necessary and sufficient to surface this
        plan's tables.

        Same category of gap as `_pin_table_descriptions`: the prompt asks
        the model to "weave these terms in naturally" into the question,
        but that's advisory — the model still freely writes its own
        `question_keywords`, and nothing forces the verified set to survive
        into it. A plan can read as perfectly retrievable and still lose
        the one term that actually made it so, if the model paraphrases or
        drops it. This makes retrievability a guarantee instead of a hope,
        with no extra LLM call: union the verified terms into whatever the
        model already wrote, verified terms first — no count cap on
        `question_keywords` (see `QueryPlan.limit_question_keywords`), so
        nothing here can ever trim one away either.
        """
        if not retrievable_keywords:
            return plan
        existing = list(plan.question_keywords or [])
        missing = [kw for kw in retrievable_keywords if kw not in existing]
        if not missing:
            return plan
        combined = list(dict.fromkeys(missing + existing))
        return plan.model_copy(update={"question_keywords": combined})

    def revise_plan(
        self,
        plan: QueryPlan,
        feedback: str,
        analyses: Sequence[dict],
        aliases: dict,
        match: Any,
        involved_cols: Optional[dict],
        stats: Sequence[TableStats],
        languages: Optional[Sequence[str]] = None,
        dfs: Optional[Sequence[Any]] = None,
        retrievable_keywords: Optional[list[str]] = None,
    ) -> tuple[Optional[QueryPlan], dict]:
        """Re-request ONE plan corrected against reviewer feedback.

        Used by the plan judge panel's correction loop: when the panel rejects
        a plan, its aggregated feedback/suggestions are handed back here so
        the planner can rewrite the question and/or steps. The revised plan
        goes through the same structural validation as any other plan, with
        one validation-error retry; the mandatory ``table_links`` constraints
        are preserved unchanged, exactly as in :meth:`plan`.

        Args:
            retrievable_keywords: Same pre-verified keyword set as
                :meth:`plan_batch` (must be the SAME set passed to the
                original ``plan_batch`` call this plan came from, so a
                revision never drifts away from the anchor the run started
                with) — re-included here because a correction round rebuilds
                the base prompt from scratch rather than reusing the
                original call's.

        Returns:
            ``(revised_plan, usage_total)`` — ``revised_plan`` is ``None``
            when the model could not produce a structurally valid revision,
            so the caller keeps the previous version of the plan.
        """
        languages = list(languages or [])
        constraint_links = self._build_constraint_links(match, involved_cols, aliases)
        known_columns = self._known_columns(stats)
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        base_prompt = self._build_prompt(
            analyses, aliases, constraint_links, stats, languages, dfs=dfs,
            retrievable_keywords=retrievable_keywords,
        )
        correction_prompt = (
            f"{base_prompt}\n\n"
            "### PLAN CORRECTION REQUEST\n"
            "A review panel rejected the plan below. Revise it so every point "
            "of the review feedback is addressed — rewrite the question and/or "
            "the steps as needed, keeping the question concise, anchored to a "
            "specific topic, and phrased like an average non-technical user "
            "seeking an insight. Return a SINGLE flat JSON object matching the "
            "plan schema directly (`question`, `steps`, `table_links`, ...) at "
            "the top level. Do NOT wrap it in a `plans` list or any other key. "
            "Do not try to fix a difficulty-correspondence rejection by "
            "changing the `difficulty` value itself — it is fixed for this "
            "plan; only add or remove STEPS so the plan's actual complexity "
            "matches it (any `difficulty` you return here is ignored).\n\n"
            "### PLAN TO CORRECT\n"
            f"{json.dumps(plan.model_dump(), indent=2, ensure_ascii=False, default=str)}\n\n"
            "### REVIEW FEEDBACK\n"
            f"{feedback}"
        )

        def _accumulate(usage: dict) -> None:
            for key in usage_total:
                usage_total[key] += (usage or {}).get(key, 0)

        raw_plan, usage = self._request_plan(correction_prompt)
        _accumulate(usage)
        candidate = self._assemble_plan(raw_plan, constraint_links)
        # The difficulty tier is fixed for a correction round — the model may
        # only change the STEPS to match it, never relabel. Force-override
        # whatever the model returned, mirroring how table_links is already
        # force-preserved above (see _assemble_plan's docstring).
        candidate = candidate.model_copy(update={"difficulty": plan.difficulty})
        try:
            validated = self.validate_plan(candidate, aliases, known_columns)
            validated = self._pin_table_descriptions(validated, analyses)
            validated = self._pin_retrievable_keywords(validated, retrievable_keywords)
            return validated, usage_total
        except PlanValidationError as exc:
            first_error = str(exc)
            logger.warning(
                "Revised plan failed structural validation (%s); "
                "re-requesting once.", first_error,
            )

        retry_prompt = self._retry_prompt(correction_prompt, first_error)
        raw_retry, usage_retry = self._request_plan(retry_prompt)
        _accumulate(usage_retry)
        retry_candidate = self._assemble_plan(raw_retry, constraint_links)
        retry_candidate = retry_candidate.model_copy(update={"difficulty": plan.difficulty})
        try:
            validated_retry = self.validate_plan(retry_candidate, aliases, known_columns)
            validated_retry = self._pin_table_descriptions(validated_retry, analyses)
            validated_retry = self._pin_retrievable_keywords(validated_retry, retrievable_keywords)
            return validated_retry, usage_total
        except PlanValidationError as exc_retry:
            logger.warning(
                "Revised plan re-request also failed structural validation "
                "(%s); keeping the previous plan version.", exc_retry,
            )
            return None, usage_total

    def _validate_or_retry_one(
        self,
        raw_plan: dict,
        base_prompt: str,
        constraint_links: List[QueryLink],
        aliases: dict,
        known_columns: dict,
    ) -> QueryPlan:
        """Validate a single raw plan from a batch, re-requesting just that one.

        Mirrors the retry-once-then-fallback shape of :meth:`plan`, but scoped
        to a single plan within the batch so one bad plan never forces a
        re-request (and re-validation) of the whole set.
        """
        candidate = self._assemble_plan(raw_plan, constraint_links)
        try:
            return self.validate_plan(candidate, aliases, known_columns)
        except PlanValidationError as exc:
            first_error = str(exc)
            logger.warning(
                "Structured plan validation failed in batch (%s); "
                "re-requesting this plan once.\n  raw plan: %s",
                first_error,
                json.dumps(raw_plan, ensure_ascii=False, default=str),
            )

        # NOTE: base_prompt is the BATCH prompt (built with num_plans>1) and
        # still contains its "produce N independent plans, return a `plans`
        # list" instructions. But this retry is single-plan: _request_plan
        # validates the response against the singular PandasQueryPlan/
        # SQLQueryPlan schema (via QueryPlannerClient), not the batch *Set*
        # schema. Without an explicit override, the model follows the
        # leftover batch instructions and wraps its fix in {"plans": [...]},
        # which fails schema validation with "field required" for every
        # top-level field (question, steps, ...) since none of them are at
        # the top level of that wrapped response.
        retry_prompt = self._retry_prompt(
            f"{base_prompt}\n\n"
            "### SINGLE-PLAN CORRECTION OVERRIDE\n"
            "Disregard the multi-plan instructions above for this request — "
            "you are correcting exactly ONE plan, shown below. Return a "
            "SINGLE flat JSON object matching the plan schema directly "
            "(`question`, `steps`, `table_links`, ...) at the top level. Do "
            "NOT wrap it in a `plans` list or any other key, and do not "
            "return a bare list.\n\n"
            "### PLAN TO CORRECT\n"
            f"{json.dumps(raw_plan, indent=2, ensure_ascii=False, default=str)}",
            first_error,
        )
        raw_retry, _usage = self._request_plan(retry_prompt)
        retry_candidate = self._assemble_plan(raw_retry, constraint_links)
        try:
            return self.validate_plan(retry_candidate, aliases, known_columns)
        except PlanValidationError as exc_retry:
            logger.warning(
                "Structured plan re-request also failed in batch (%s); "
                "falling back to a free-text plan for this entry.\n  raw plan: %s",
                exc_retry,
                json.dumps(raw_retry, ensure_ascii=False, default=str),
            )
            return self._free_text_fallback(retry_candidate, aliases, constraint_links)

    @staticmethod
    def _extract_raw_plans(raw_set: dict) -> List[dict]:
        """Pull the list of raw plan dicts out of a batch response.

        Tolerates a model that ignores the ``plans`` wrapper and returns a bare
        list, or even a single flat plan dict (treated as a batch of one).
        """
        if not raw_set:
            return []
        plans = raw_set.get("plans")
        if isinstance(plans, list):
            return [p for p in plans if isinstance(p, dict)]
        if isinstance(raw_set, list):
            return [p for p in raw_set if isinstance(p, dict)]
        if isinstance(raw_set, dict) and ("steps" in raw_set or "question" in raw_set):
            return [raw_set]
        return []

    def _request_plan_batch(self, prompt: str) -> tuple[dict, dict]:
        if self._batch_client is None:
            self._batch_client = QueryPlanBatchClient(self.config_path, self.kind)
        result = self._batch_client.request_plan_batch(prompt, max_tokens=GENERATION_MAX_TOKENS)
        if isinstance(result, tuple):
            plan_set, usage = result
        else:
            plan_set, usage = result, {}
        return (plan_set or {}), (usage or {})

    # ------------------------------------------------------------------
    # Plan validation (Requirements 5.3, 5.4, 5.5)
    # ------------------------------------------------------------------

    def validate_plan(
        self,
        plan: QueryPlan,
        aliases: dict,
        known_columns: dict,
    ) -> QueryPlan:
        """Validate and normalise a structured plan.

        Assigns contiguous ``1..N`` ``order`` values to the steps (Requirement
        5.3), then verifies that every table referenced by a step is a known
        alias (Requirement 5.4) and that every column referenced by a step
        exists in one of the tables that step references (Requirement 5.5).

        Also verifies ``plan.tables`` (the ``Table`` entries carrying each
        table's planning-time justification): every known alias must appear
        exactly once, with no unknown/invented aliases and no empty
        ``reason`` — table usage is decided here, exactly once, and this is
        the structural floor beneath the plan judge's qualitative review of
        each justification's substance.

        Args:
            plan: The assembled candidate plan to validate.
            aliases: Mapping of alias -> dataset name (the set of known aliases).
            known_columns: Mapping of alias -> set/collection of column names.

        Returns:
            The same ``plan`` with its step orders normalised to ``1..N``.

        Raises:
            PlanValidationError: If the plan is empty or references an unknown
                alias or a column absent from every referenced table.
        """
        if not plan.steps:
            raise PlanValidationError("plan has no steps")

        # Question-quality enforcement: a raw column identifier leaking into
        # the question is a hard rejection, not a style nit the plan judge is
        # left to catch on its own. Applies to BOTH plan kinds — SQL
        # questions can leak column names exactly the same way Pandas ones
        # can.
        for text_field in ("question", "translated_question"):
            text = getattr(plan, text_field, "") or ""
            reason = _question_leaks_implementation(text)
            if reason:
                raise PlanValidationError(
                    f"plan.{text_field} {reason}: {text!r}. Rewrite it as an "
                    "average, non-technical user would ask it — no column "
                    "names, only plain words for what they want to know."
                )

        # Requirement 5.3: assign a contiguous 1..N order sequence.
        for idx, step in enumerate(plan.steps, start=1):
            step.order = idx

        known_aliases = set(aliases.keys()) if aliases else set()

        # Table coverage: `plan.tables` is where table usage is decided (and
        # later judged by PlanJudgment.table_check) exactly once — so a
        # missing, duplicated, unknown, or unjustified alias must fail fast
        # here and trigger the standard re-request, rather than reach a judge
        # (or code generation, which requires every table the plan carries)
        # with a plan that under- or over-covers the tables it was given.
        plan_table_names = [t.name for t in (plan.tables or [])]
        plan_table_set = set(plan_table_names)
        if len(plan_table_names) != len(plan_table_set):
            dupes = sorted({n for n in plan_table_names if plan_table_names.count(n) > 1})
            raise PlanValidationError(
                f"plan.tables lists duplicate alias(es) {dupes}; each table "
                "alias must appear exactly once"
            )
        missing_tables = known_aliases - plan_table_set
        if missing_tables:
            raise PlanValidationError(
                f"plan.tables is missing an entry for alias(es) "
                f"{sorted(missing_tables)}; every provided table alias must "
                "appear in `tables` with a `reason`"
            )
        unknown_tables = plan_table_set - known_aliases
        if unknown_tables:
            raise PlanValidationError(
                f"plan.tables references unknown alias(es) {sorted(unknown_tables)}; "
                f"known aliases: {sorted(known_aliases)}"
            )
        for t in plan.tables:
            if not (t.reason or "").strip():
                raise PlanValidationError(
                    f"plan.tables entry for {t.name!r} has an empty `reason` "
                    "— every table needs a well-articulated justification"
                )

        # Normalise known_columns values into sets for membership checks.
        columns_by_alias = {
            alias: set(cols) for alias, cols in (known_columns or {}).items()
        }

        # Whether any earlier step could have produced new columns (aggregate/
        # derive, or explicitly declared outputs in params). Once true, an
        # unresolved column reference is downgraded to a warning: `params` is
        # free-form, so output harvesting is best-effort, and a false
        # rejection costs a full planner re-request that tends to yield a
        # WORSE plan (derived names pushed into prose where no validator —
        # or generator — can see them). The statement generator reads the
        # whole plan including descriptions, so an unresolved derived name is
        # a soft signal, not a broken plan.
        derivation_seen = False

        for step in plan.steps:
            # Requirement 5.4: every referenced table must be a known alias.
            for table in step.tables:
                if table not in known_aliases:
                    raise PlanValidationError(
                        f"step {step.order} ({step.op}) references unknown table "
                        f"alias {table!r}; known aliases: {sorted(known_aliases)}"
                    )

            # Structural checks for correlate/limit/rank's fixed `params`
            # shape (see _STEP_PARAMS_DESCRIPTION) — kind-aware, since SQL
            # plans are restricted to a narrower method vocabulary than
            # Pandas plans for both `correlate` and `rank`.
            new_op_error = self._validate_new_op_params(step)
            if new_op_error:
                raise PlanValidationError(new_op_error)

            # Register outputs this step declares in params (derive's
            # new_column, aggregate's output_column / aggregations keys,
            # correlate/rank's output_column) BEFORE the column-existence
            # check below, so a step that lists its own forward-declared
            # output inside its own `columns` (the established `derive`
            # convention — "output_column must also appear in the step's
            # columns") is never rejected just because it's the first
            # producing step in the plan and _COLUMN_PRODUCING_OPS doesn't
            # cover its op (correlate/rank deliberately don't, since their
            # SOURCE columns must always already exist — only their
            # declared output is new).
            declared_outputs = self._declared_output_columns(step)

            # Requirement 5.5: every referenced column must exist in at least
            # one of the tables the step references — physical schema columns
            # plus outputs registered by earlier steps (or this step's own,
            # per above). Exceptions:
            #   * ops that legitimately produce NEW columns (`aggregate`,
            #     `derive`): an unrecognized name there is the step's own
            #     output, registered for later steps rather than rejected;
            #   * unresolved names AFTER some step could have produced columns:
            #     warned and registered (see derivation_seen above).
            # A hallucinated physical column in a plan with no derivation
            # steps is still rejected — that's the expensive failure this
            # check exists to catch early.
            allowed_columns: set = set()
            for table in step.tables:
                allowed_columns |= columns_by_alias.get(table, set())
            allowed_columns |= declared_outputs

            produces_columns = step.op in _COLUMN_PRODUCING_OPS
            for column in step.columns:
                if column in allowed_columns:
                    continue
                if not produces_columns and not derivation_seen:
                    raise PlanValidationError(
                        f"step {step.order} ({step.op}) references column "
                        f"{column!r} not present in any referenced table "
                        f"{step.tables}"
                    )
                if not produces_columns:
                    logger.warning(
                        "plan step %s (%s) references column %r that no "
                        "referenced table or earlier step declares — assuming "
                        "it is derived upstream and continuing",
                        step.order, step.op, column,
                    )
                for table in step.tables:
                    columns_by_alias.setdefault(table, set()).add(column)

            # `correlate`/`limit`/`rank` name columns inside `params.by`/
            # `params.group_by` rather than `step.columns` — those are
            # otherwise invisible to the loop above, so check their
            # existence separately here (using the now-finalised
            # allowed_columns, which already includes this step's own
            # declared output).
            if step.op in ("correlate", "limit", "rank"):
                params = step.params or {}
                referenced = list(params.get("by") or []) + list(params.get("group_by") or [])
                for column in referenced:
                    if column not in allowed_columns:
                        raise PlanValidationError(
                            f"step {step.order} ({step.op}) params references "
                            f"unknown column {column!r}; known columns for "
                            f"{step.tables}: {sorted(allowed_columns)}"
                        )

            # Register this step's declared outputs for LATER steps.
            for table in step.tables:
                columns_by_alias.setdefault(table, set()).update(declared_outputs)

            if produces_columns or declared_outputs:
                derivation_seen = True

            # A `clean` step's drop_column actions remove the column from
            # what later steps may reference — enforced by simply subtracting
            # it from the known-columns set, so it trips the SAME
            # column-existence check above for whichever later step
            # references it, rather than a parallel special-cased error path.
            dropped_columns = self._clean_step_dropped_columns(step)
            for table in step.tables:
                columns_by_alias.get(table, set()).difference_update(dropped_columns)

        return plan

    @staticmethod
    def _known_columns(stats: Sequence[TableStats]) -> dict:
        """Map each alias to the set of its column names from the statistics."""
        return {table.alias: {c.column for c in table.columns} for table in stats}

    def _validate_new_op_params(self, step) -> Optional[str]:
        """Structural checks for `correlate`/`limit`/`rank`'s fixed `params`
        shape (see `_STEP_PARAMS_DESCRIPTION`). Returns an error string, or
        ``None`` when the step is fine (or not one of these three ops).

        Kind-aware (``self.kind``/``self._is_pandas``) for the two SQL/Pandas
        asymmetries this schema deliberately carries: DuckDB's `corr()` is
        Pearson-only (no Spearman/Kendall), and DuckDB has no native
        average/max tie-handling for ranks (only `RANK()`/`DENSE_RANK()`/
        `ROW_NUMBER()`, i.e. `"min"`/`"dense"`/`"first"`).
        """
        params = step.params or {}
        if step.op == "correlate":
            if len(set(step.columns)) < 2:
                return (
                    f"step {step.order} (correlate) needs 2+ distinct columns "
                    f"in `columns`; got {step.columns!r}"
                )
            method = params.get("method", "pearson")
            allowed = {"pearson", "spearman", "kendall"} if self._is_pandas else {"pearson"}
            if method not in allowed:
                return (
                    f"step {step.order} (correlate) params.method={method!r} "
                    f"invalid for {self.kind} plans; allowed: {sorted(allowed)}"
                )
        elif step.op == "limit":
            n = params.get("n")
            if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
                return f"step {step.order} (limit) params.n must be a positive int; got {n!r}"
            how = params.get("how", "head")
            if how not in {"head", "largest", "smallest"}:
                return f"step {step.order} (limit) params.how={how!r} invalid"
            if how in {"largest", "smallest"}:
                by = params.get("by")
                if not isinstance(by, list) or not by or not all(
                    isinstance(c, str) and c for c in by
                ):
                    return (
                        f"step {step.order} (limit) how={how!r} requires a "
                        "non-empty params.by list"
                    )
        elif step.op == "rank":
            output_column = params.get("output_column")
            if not isinstance(output_column, str) or not output_column:
                return f"step {step.order} (rank) requires params.output_column"
            by = params.get("by")
            if not isinstance(by, list) or not by or not all(
                isinstance(c, str) and c for c in by
            ):
                return f"step {step.order} (rank) requires a non-empty params.by list"
            default_method = "average" if self._is_pandas else "min"
            method = params.get("method", default_method)
            allowed = (
                {"average", "min", "max", "first", "dense"}
                if self._is_pandas
                else {"min", "dense", "first"}
            )
            if method not in allowed:
                return (
                    f"step {step.order} (rank) params.method={method!r} "
                    f"invalid for {self.kind} plans; allowed: {sorted(allowed)}"
                )
        return None

    @staticmethod
    def _declared_output_columns(step) -> set:
        """Output column names a step declares in its free-form ``params``.

        Well-structured plans put a step's INPUT columns in ``step.columns``
        and its OUTPUT names in ``params`` — ``derive`` declares
        ``params.new_column``, ``aggregate`` declares ``params.output_column``
        or the keys of ``params.aggregations``, and ``rank``/``correlate``
        declare (the latter optionally) ``params.output_column`` the same
        way. These outputs are legitimate references for later steps (a
        ``sort`` on an aggregate's result), so validate_plan registers them
        instead of rejecting the reference. ``params`` is free-form, so this
        harvest is best-effort by convention.
        """
        params = step.params or {}
        if not isinstance(params, dict):
            return set()
        out: set = set()
        for key in ("new_column", "output_column"):
            value = params.get(key)
            if isinstance(value, str) and value:
                out.add(value)
        for key in ("new_columns", "output_columns"):
            value = params.get(key)
            if isinstance(value, (list, tuple)):
                out.update(v for v in value if isinstance(v, str))
        aggregations = params.get("aggregations")
        if isinstance(aggregations, dict):
            out.update(k for k in aggregations if isinstance(k, str))
        return out

    @staticmethod
    def _clean_step_dropped_columns(step) -> set:
        """Columns a `clean` step's ``params.actions`` marks ``drop_column``.

        Best-effort by convention, mirroring ``_declared_output_columns`` —
        malformed/missing ``actions`` yields an empty set rather than
        raising, since ``params`` is free-form. Used to remove those columns
        from what later steps are allowed to reference (see
        ``_CLEAN_STEP_PARAMS_DESCRIPTION``: a dropped column may not be
        referenced by any later step).
        """
        if step.op != "clean":
            return set()
        params = step.params or {}
        if not isinstance(params, dict):
            return set()
        actions = params.get("actions")
        if not isinstance(actions, list):
            return set()
        return {
            a["column"] for a in actions
            if isinstance(a, dict) and a.get("action") == "drop_column"
            and isinstance(a.get("column"), str) and a["column"]
        }

    @staticmethod
    def _retry_prompt(prompt: str, error: str) -> str:
        """Append validation feedback so the re-request can self-correct."""
        return (
            f"{prompt}\n\n### VALIDATION FEEDBACK\n"
            "Your previous plan failed validation with the following error:\n"
            f"{error}\n"
            "Return a corrected plan that references only the provided table "
            "aliases and only columns that exist in those tables."
        )

    def _empty_plan(self, constraint_links: List[QueryLink]) -> QueryPlan:
        """A schema-valid, step-less plan used only as a fallback seed."""
        if self._is_pandas:
            return PandasQueryPlan(
                question="", question_keywords=[], plan_keywords=[], steps=[],
                table_links=constraint_links,
            )
        return SQLQueryPlan(
            question="", question_keywords=[], plan_keywords=[], steps=[],
            table_links=constraint_links,
        )

    def _free_text_fallback(
        self,
        plan: QueryPlan,
        aliases: dict,
        constraint_links: List[QueryLink],
    ) -> QueryPlan:
        """Build a schema-valid free-text fallback plan (Requirement 5.6).

        When both the initial request and the single re-request fail validation,
        produce a plan that still satisfies the schema and preserves the
        mandatory ``table_links``: a single free-text ``select`` step over the
        available tables with no specific column references (so it can never
        fail column validation).
        """
        alias_list = list(aliases.keys()) if aliases else []
        description = (
            "Free-text fallback: the structured plan failed validation after "
            "one re-request. Answer the question directly over the available "
            "tables without a validated step decomposition."
        )
        # Every alias still needs a `tables` entry — validate_plan's coverage
        # check is the invariant the rest of the pipeline relies on (a query's
        # `tables` is copied straight from the plan, never re-derived), and
        # this fallback plan is never re-validated after construction, so it
        # must satisfy that invariant itself rather than leaning on a
        # downstream reconciliation step. The reason is honestly generic
        # (there is no validated per-table role to report — that's exactly
        # what failed) rather than fabricating a specific-sounding one.
        fallback_tables = [
            Table(
                name=alias,
                reason=(
                    f"{alias} is one of the tables this question requires; the "
                    "structured plan failed schema validation twice, so its "
                    "specific role was never decomposed into steps. This "
                    "free-text fallback answers the question directly over "
                    "all provided tables without a per-table breakdown."
                ),
                columns_involved=[],
            )
            for alias in alias_list
        ]

        # Carry over whatever question-level metadata the failed candidate
        # already picked up from the LLM before validation rejected its steps
        # — the metadata itself was never the problem, only the steps were.
        metadata = {
            "query_plan": plan.query_plan,
            "translated_question": plan.translated_question,
            "translated_question_keywords": plan.translated_question_keywords,
            "detected_language": plan.detected_language,
            "topic": plan.topic,
            "story": plan.story,
            "difficulty": plan.difficulty,
            "expected_result_type": plan.expected_result_type,
            "expected_result_description": plan.expected_result_description,
        }

        if self._is_pandas:
            fallback_step = PandasPlanStep(
                order=1, op="select", description=description,
                tables=alias_list, columns=[],
            )
            return PandasQueryPlan(
                question=plan.question,
                question_keywords=plan.question_keywords,
                plan_keywords=plan.plan_keywords,
                steps=[fallback_step],
                tables=fallback_tables,
                table_links=constraint_links,
                **metadata,
            )

        fallback_step = SQLPlanStep(
            order=1, op="select", description=description,
            tables=alias_list, columns=[],
        )
        return SQLQueryPlan(
            question=plan.question,
            question_keywords=plan.question_keywords,
            plan_keywords=plan.plan_keywords,
            steps=[fallback_step],
            tables=fallback_tables,
            table_links=constraint_links,
            **metadata,
        )

    # ------------------------------------------------------------------
    # Constraint links (match / involved_cols preserved unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_constraint_links(
        match: Any,
        involved_cols: Optional[dict],
        aliases: dict,
    ) -> List[QueryLink]:
        """Derive the mandatory ``table_links`` from the upstream constraints.

        The links are deterministic so they can be preserved unchanged in the
        produced plan and asserted against in tests.

        * When ``match`` is already a structured list of links (dicts or
          :class:`QueryLink`), those links are coerced and returned verbatim —
          this is the normal path (see
          ``statement_generation._get_match_for_planner``): one link per
          verified pairwise relationship, each carrying its own
          table-attributed ``key_columns``.
        * Otherwise (legacy string-only match formats) a single ``join`` link
          is synthesised across the tables that carry involved columns, with
          the ``match`` text as the description; ``key_columns`` lists each
          involved column tagged with its own table but makes no claim about
          which columns on different tables correspond to each other, since
          that information isn't available in this fallback.
        * A single-table request (fewer than two linked tables) yields no link.
        """
        # Case 1: match is already a structured list of links -> preserve as-is.
        if isinstance(match, (list, tuple)):
            links: List[QueryLink] = []
            for item in match:
                if isinstance(item, QueryLink):
                    links.append(item.model_copy(deep=True))
                elif isinstance(item, dict):
                    links.append(QueryLink.model_validate(item))
            return links

        involved_cols = involved_cols or {}

        # Tables that participate in the mandatory relationship, in alias order.
        alias_order = list(aliases.keys()) if aliases else list(involved_cols.keys())
        linked_tables = [
            alias for alias in alias_order if involved_cols.get(alias)
        ]

        # A relationship needs at least two tables to link. Single-table runs
        # (single mode) carry no mandatory table link.
        if len(linked_tables) < 2:
            return []

        # One {alias: column} entry per involved column, so each column stays
        # attributed to its own table — this fallback (legacy string-only
        # match formats) has no record of which column on one table actually
        # pairs with which column on another, so it deliberately does NOT
        # claim a cross-table correspondence the way the structured
        # per-relationship path (see statement_generation._relationship_to_link)
        # does.
        key_columns: List[dict] = [
            {alias: col}
            for alias in linked_tables
            for col in involved_cols.get(alias, [])
        ]

        description = (
            match.strip()
            if isinstance(match, str) and match.strip()
            else "Verified relationship provided upstream; combine these tables only through it."
        )

        return [
            QueryLink(
                type="join",
                tables=linked_tables,
                description=description,
                key_columns=key_columns,
            )
        ]

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        analyses: Sequence[dict],
        aliases: dict,
        constraint_links: Sequence[QueryLink],
        stats: Sequence[TableStats],
        languages: Sequence[str],
        num_plans: int = 1,
        dfs: Optional[Sequence[Any]] = None,
        retrievable_keywords: Optional[list[str]] = None,
    ) -> str:
        if num_plans <= 1:
            task_statement = (
                "You are an expert data engineer. Produce an ordered, step-by-step "
                "query plan that decomposes a single business question into concrete "
                "operations over the provided tables. Return only valid JSON matching "
                "the required schema.\n\n"
            )
            batch_note = ""
        else:
            task_statement = (
                "You are an expert data engineer. Produce "
                f"{num_plans} INDEPENDENT, step-by-step query plans over the "
                "provided tables, each decomposing a DIFFERENT business question. "
                "Return only valid JSON matching the required schema, as a "
                f"`plans` list of exactly {num_plans} plan objects.\n\n"
            )
            if num_plans == 3:
                difficulty_note = (
                    "- Assign exactly one EASY, one MEDIUM, and one HARD plan across "
                    "these 3 — one plan per tier, no repeats — per the DIFFICULTY "
                    "rubric above. The tiers must come from genuinely different step "
                    "structures, not from labeling three similarly-simple plans "
                    "easy/medium/hard: if two plans would earn the same tier under "
                    "the rubric applied honestly, restructure one of them (add a "
                    "join/groupby/extra step, or simplify one down) until the three "
                    "are actually distinct in complexity, not just in label.\n"
                )
            else:
                difficulty_note = (
                    f"- Spread these {num_plans} plans across the DIFFICULTY rubric "
                    "above as evenly as possible, cycling easy -> medium -> hard -> "
                    "easy -> ... so every tier is represented — never label multiple "
                    "plans the same tier while leaving another tier completely "
                    "unused when you have enough plans to cover all three. Each "
                    "label must come from that plan's own step structure, honestly "
                    "applied, not assigned to hit the target mix and then rationalized.\n"
                )
            batch_note = (
                "MULTI-PLAN REQUIREMENTS:\n"
                f"- Each of the {num_plans} plans is fully self-contained: its own "
                "`question` and its own `steps`. Plans do NOT share steps.\n"
                "- Do not repeat the same question across plans.\n"
                f"{difficulty_note}\n"
            )

        ops_statement = (
            "- Every step's `op` MUST be exactly one of these values — no "
            "others are valid: filter, join, union, group, aggregate, sort, "
            "select, derive, clean, correlate, limit, rank.\n"
            "- Use `correlate` for a Pearson/Spearman/Kendall correlation "
            "between 2+ numeric columns — SQL plans: `pearson` only (DuckDB's "
            "built-in `corr()` has no native Spearman/Kendall). Never express "
            "a correlation as `derive`/`aggregate` instead.\n"
            "- Use `limit` to cap the number of result rows (`params.n`), "
            "typically right after a `sort` or a `group`+`aggregate` — never "
            "encode a row cap inside another step's `params`.\n"
            "- Use `rank` only when the RANK POSITION ITSELF must appear as a "
            "value in the answer (e.g. \"what rank is Brooklyn in complaint "
            "volume\"), materializing it as a new column via `params."
            "output_column`. If you only need the top-N rows and never need "
            "the rank number as a value, use `sort`+`limit` instead — never "
            "`rank`.\n"
            "- A `join-correlation` link's `correlated_columns` (see VERIFIED "
            "TABLE RELATIONSHIPS below) names pre-vetted columns to feed "
            "directly into a `correlate` step's `columns` after that join.\n"
        )

        return QueryPlannerPrompt().update(
            task_statement=task_statement,
            ops_statement=ops_statement,
            batch_note=batch_note,
            time_context=self._render_time_context(),
            table_links=self._render_links(constraint_links),
            table_aliases=json.dumps(list(aliases.keys()), indent=2, ensure_ascii=False),
            table_analysis=json.dumps(
                {"tables": list(analyses)}, indent=2, ensure_ascii=False, default=str
            ),
            table_sample=self._render_table_sample(dfs, aliases),
            column_statistics=self._render_statistics(stats),
            detected_languages=json.dumps(list(languages), ensure_ascii=False),
            retrievable_keywords=self._render_retrievable_keywords(retrievable_keywords),
        )

    @staticmethod
    def _render_retrievable_keywords(retrievable_keywords: Optional[list[str]]) -> str:
        """The "### RETRIEVABLE KEYWORDS" prompt section — empty (section
        omitted entirely, not printed as "none") when no pre-verified set is
        available, so older behavior (retrievability guidance alone, no
        anchor) is unchanged for portals with no reverse index configured.
        """
        if not retrievable_keywords:
            return ""
        terms = ", ".join(f"`{kw}`" for kw in retrievable_keywords)
        return (
            "\n### RETRIEVABLE KEYWORDS (pre-verified — do not skip this)\n"
            f"These exact terms — {terms} — were EMPIRICALLY VERIFIED just now "
            "against the real reverse index: searching with them surfaces "
            "EVERY table in TABLE ALIASES within the retrievability check's "
            "own top-K window. This is not a guess or a suggestion, it is a "
            "proven-working anchor. Every plan's `question` MUST weave ALL of "
            "these terms into its own natural prose (not just list them in "
            "`question_keywords` while the question text stays silent on "
            "them — see the question-writing rule above: `question_keywords` "
            "must literally appear in or be directly implied by the question "
            "text). You may freely add other natural topical words alongside "
            "them; you may NOT drop, paraphrase, or merge any of these terms "
            "into a different word — a paraphrase (e.g. one merged word "
            "instead of a real multi-word term above) is exactly what breaks "
            "retrieval even when it means the same thing to a person.\n"
        )

    @staticmethod
    def _render_time_context() -> str:
        """Render the "### TIME CONTEXT" block of the planner prompt.

        Gives the planner a concrete "now" so a question about a fixed-period
        table (see the TEMPORAL SCOPE guidance) is phrased relative to what
        the data actually covers, never implicitly as if it were current.
        """
        # Date-level granularity on purpose: a full timestamp changes on every
        # call and would invalidate the provider's cached prompt prefix for
        # every token that follows it; the date is stable across a whole day.
        return f"- Current date: {datetime.now().date().isoformat()}\n"

    @staticmethod
    def _render_table_sample(dfs: Optional[Sequence[Any]], aliases: dict) -> str:
        """Render up to 10 real sample rows per table.

        Grounds the planner's questions — especially a question naming a
        concrete value (a real place, category, or period) — in values that
        actually occur in the data, rather than the model inventing a
        plausible-sounding but nonexistent one. Uses ``DataFrame.to_json``
        (not a raw ``.to_dict()``) so NaN/NaT/
        Timestamp/numpy scalar values round-trip into plain JSON-safe types
        the same way ``StatementOrchestrator._serialize_query_output`` (in ``agent.py``) already does
        for executed query results.
        """
        if not dfs:
            return "(no table sample available)"
        alias_names = list(aliases.keys()) if aliases else [
            f"Table_{i}" for i in range(len(dfs))
        ]
        payload = []
        for alias, df in zip(alias_names, dfs):
            try:
                shielded = shield_dataframe_for_prompt(df.head(10))
                rows = json.loads(shielded.to_json(orient="records", date_format="iso"))
            except (TypeError, ValueError):
                rows = []
            payload.append({"alias": alias, "rows": rows})
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @staticmethod
    def _render_links(links: Sequence[QueryLink]) -> str:
        if not links:
            return "(no mandatory links — single table)"
        payload = []
        for link in links:
            entry = {
                "type": link.type,
                "tables": link.tables,
                "key_columns": link.key_columns,
                "description": link.description,
            }
            if link.correlated_columns:
                entry["correlated_columns"] = link.correlated_columns
            payload.append(entry)
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @staticmethod
    def _render_statistics(stats: Sequence[TableStats]) -> str:
        if not stats:
            return "(no column statistics available)"
        payload = []
        for table in stats:
            payload.append(
                {
                    "alias": table.alias,
                    "num_rows": table.num_rows,
                    "columns": [
                        {
                            "column": c.column,
                            "dtype": c.dtype,
                            "cardinality": c.cardinality,
                            "null_ratio": round(c.null_ratio, 4),
                            "nan_count": c.nan_count,
                            "bad_token_counts": c.bad_token_counts,
                            "numeric_parseable_ratio": (
                                round(c.numeric_parseable_ratio, 4)
                                if c.numeric_parseable_ratio is not None
                                else None
                            ),
                            "numeric_min": c.numeric_min,
                            "numeric_max": c.numeric_max,
                            "numeric_mean": c.numeric_mean,
                            "numeric_outliers": c.numeric_outliers,
                            "numeric_pinned_extreme": c.numeric_pinned_extreme,
                            "top_values": c.top_values,
                            "minority_value_groups": c.minority_value_groups,
                        }
                        for c in table.columns
                    ],
                }
            )
        return json.dumps(payload, indent=2, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------
    # Client + assembly
    # ------------------------------------------------------------------

    def _request_plan(self, prompt: str) -> tuple[dict, dict]:
        if self._client is None:
            self._client = QueryPlannerClient(self.config_path, self.kind)
        result = self._client.request_plan(prompt, max_tokens=GENERATION_MAX_TOKENS)
        # Support both (dict, usage) and bare-dict clients for flexibility.
        if isinstance(result, tuple):
            plan_dict, usage = result
        else:
            plan_dict, usage = result, {}
        return (plan_dict or {}), (usage or {})

    def _assemble_plan(
        self,
        raw_plan: dict,
        constraint_links: List[QueryLink],
    ) -> QueryPlan:
        """Build a kind-appropriate query plan from the raw LLM response.

        The ``table_links`` are always the mandatory constraint links — the
        model's own ``table_links`` output is discarded so the upstream
        relationships are preserved unchanged (Requirement 6.2).
        """
        raw_plan = raw_plan or {}
        metadata = self._extract_metadata_fields(raw_plan)

        tables = self._coerce_tables(raw_plan.get("tables", []))

        if self._is_pandas:
            steps = self._coerce_steps(raw_plan.get("steps", []), PandasPlanStep)
            return PandasQueryPlan(
                question=str(raw_plan.get("question", "")),
                question_keywords=self._limit_keywords(raw_plan.get("question_keywords")),
                plan_keywords=self._limit_keywords(raw_plan.get("plan_keywords")),
                steps=steps,
                tables=tables,
                table_links=constraint_links,
                **metadata,
            )

        steps = self._coerce_steps(raw_plan.get("steps", []), SQLPlanStep)
        return SQLQueryPlan(
            question=str(raw_plan.get("question", "")),
            question_keywords=self._limit_keywords(raw_plan.get("question_keywords")),
            plan_keywords=self._limit_keywords(raw_plan.get("plan_keywords")),
            steps=steps,
            tables=tables,
            table_links=constraint_links,
            **metadata,
        )

    def _extract_metadata_fields(self, raw_plan: dict) -> dict:
        """Read the planner-owned question-level metadata off a raw LLM plan.

        These fields (moved here from the generation-time ``Query`` model —
        see ``StatementClient.complete``'s plan-fields merge) are decided once
        during planning and copied onto every query the plan later generates.

        ``expected_result_type``/``expected_result_description`` MUST be
        harvested here too: _assemble_plan reconstructs the plan model
        field-by-field from the raw response, so any field not read here is
        silently replaced by its schema default — which once made every
        revision round "stubbornly" revert the result type to "table" no
        matter how clearly the judges' feedback asked for "number" (the
        reviser was fixing it; assembly was throwing the fix away).
        """
        # Normalise before Pydantic sees it: an invalid/missing type must
        # fall back to the schema default, not blow up assembly with a
        # ValidationError no retry path catches.
        expected_type = str(raw_plan.get("expected_result_type", "")).strip().lower()
        if expected_type not in get_args(_RESULT_TYPES):
            expected_type = "table"
        difficulty = str(raw_plan.get("difficulty", "")).strip().lower()
        if difficulty not in get_args(_DIFFICULTY_LEVELS):
            difficulty = "easy"
        return {
            "query_plan": str(raw_plan.get("query_plan", "")),
            "translated_question": str(raw_plan.get("translated_question", "")),
            "translated_question_keywords": self._limit_keywords(
                raw_plan.get("translated_question_keywords")
            ),
            "detected_language": str(raw_plan.get("detected_language", "")),
            "topic": str(raw_plan.get("topic", "")),
            "story": str(raw_plan.get("story", "")),
            "difficulty": difficulty,
            "expected_result_type": expected_type,
            "expected_result_description": str(
                raw_plan.get("expected_result_description", "")
            ),
        }

    @staticmethod
    def _coerce_steps(raw_steps: Any, step_model) -> List[Any]:
        """Coerce raw step dicts into ``step_model`` (:class:`SQLPlanStep` or
        :class:`PandasPlanStep`) objects.

        A positional ``order`` (1-based) is filled in when the model omits it so
        the step can be constructed; contiguous-order validation remains a
        separate concern handled by ``validate_plan`` (task 5.4).
        """
        steps: List[Any] = []
        if not isinstance(raw_steps, list):
            return steps
        for idx, raw_step in enumerate(raw_steps, start=1):
            if isinstance(raw_step, step_model):
                steps.append(raw_step)
                continue
            if not isinstance(raw_step, dict):
                logger.warning("Skipping non-dict plan step: %r", raw_step)
                continue
            step = dict(raw_step)
            step.setdefault("order", idx)
            try:
                steps.append(step_model.model_validate(step))
            except ValidationError as exc:
                logger.warning("Skipping invalid plan step %d: %s", idx, exc)
        return steps

    @staticmethod
    def _coerce_tables(raw_tables: Any) -> List[Table]:
        """Coerce the raw LLM ``tables`` output into :class:`Table` objects.

        Mirrors :meth:`_coerce_steps`: a table entry that fails schema
        validation (e.g. missing ``name``) is dropped with a warning rather
        than aborting the whole plan — ``validate_plan``'s coverage check is
        what actually catches the resulting gap and drives the re-request.
        """
        tables: List[Table] = []
        if not isinstance(raw_tables, list):
            return tables
        for idx, raw_table in enumerate(raw_tables, start=1):
            if isinstance(raw_table, Table):
                tables.append(raw_table)
                continue
            if not isinstance(raw_table, dict):
                logger.warning("Skipping non-dict plan table entry: %r", raw_table)
                continue
            try:
                tables.append(Table.model_validate(raw_table))
            except ValidationError as exc:
                logger.warning("Skipping invalid plan table entry %d: %s", idx, exc)
        return tables

    @staticmethod
    def _limit_keywords(value: Any, limit: int = 10) -> List[str]:
        if not isinstance(value, (list, tuple)):
            return []
        seen: List[str] = []
        for kw in value:
            kw_str = str(kw)
            if kw_str not in seen:
                seen.append(kw_str)
            if len(seen) >= limit:
                break
        return seen
