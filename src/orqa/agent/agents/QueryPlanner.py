"""Structured query planner (task 5.3).

:class:`QueryPlanner` turns per-table analyses plus the upstream ``match`` /
``involved_cols`` relationship constraints into a structured query plan (see
:mod:`orqa.agent.prompting.models`): an ordered list of plan-step decomposition
steps and the ``table_links`` carried over from the mandatory upstream
constraints. The planner is **kind-aware**: a ``"SQL"`` planner produces
:class:`~orqa.agent.prompting.models.SQLQueryPlan` (no ``task_types`` — a SQL
plan can never require an ML skill), while a ``"PANDAS"`` planner produces
:class:`~orqa.agent.prompting.models.PandasQueryPlan` (``task_types`` drawn
from the closed set of 5 ML task types, derived from the plan's steps).

Design constraints implemented here (Requirements 5.1, 5.2, 6.1, 6.2):

* The provided ``match`` / ``involved_cols`` links are the only **verified**
  relationships. The planner prompt states tables may only be combined through
  them (in any composition shape — chained or independent branches), and the
  produced plan preserves them **unchanged** in ``table_links`` regardless of
  what the language model returns for that field.
* For Pandas plans, ``task_types`` is the distinct set of skill-relevant
  operations found in the plan steps (classification, regression, timeseries,
  causal). SQL plans have no ``task_types`` field at all.
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
from ..utility.structured_outputs import QueryLink, Table

logger = logging.getLogger(__name__)

# Operations that require a skill (the ML task types). ``task_types`` on a
# produced Pandas plan is the distinct set of these ops appearing in the plan
# steps. SQL plans never carry these ops at all (schema-enforced).
SKILL_TASK_TYPES: tuple[str, ...] = (
    "classification",
    "regression",
    "timeseries",
    "causal",
)

# Ops whose entire purpose is to produce a NEW column rather than read an
# existing one (a groupby's count/sum output, a derive's computed column).
# validate_plan's column-existence check (Requirement 5.5) exempts these ops:
# an unrecognized column name here is the step's own output, not a bad
# reference.
_COLUMN_PRODUCING_OPS = frozenset({"aggregate", "derive"})

# Name tokens that mark a column as identifier-like: a label for a record or a
# place, not a measured quantity. Such columns are never valid ML prediction
# targets ("predicting" a zip code or an ID is meaningless even though the
# column is numeric) — validate_plan rejects ML steps that declare one as
# their target/outcome so the retry can pick a real outcome column.
_IDENTIFIER_TOKENS = frozenset({
    "zip", "zipcode", "postal", "postcode", "id", "ids", "uid", "uuid", "guid",
    "code", "codes", "dbn", "phone", "tel", "fax", "url", "web", "website",
    "email", "address", "adress", "street", "name", "names", "lat", "lon",
    "lng", "latitude", "longitude", "ssn", "license", "licence",
})

_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")

# Matches a `filter` step description that pins a column to a single value
# ("street equals 'Broadway'", "borough == 'Bronx'", "status is exactly
# 'Closed'") as opposed to a range/set condition ("between", "at least").
# Used to catch a specific ML-plan defect: reusing such a column as a
# predictive feature is a no-op, since the filter already fixed it to one
# value across the whole training population (see validate_plan).
_EQUALITY_FILTER_RE = re.compile(r"\bequal(?:s|ing)?\b|\bis exactly\b|==", re.IGNORECASE)

# An underscore between two letters in a QUESTION is a near-certain sign a
# raw column identifier got pasted in verbatim (natural language never uses
# underscores) — e.g. "the increase in Pre-K seats (grade_pk_half_day_full_day)"
# or "the store_and_fwd_flag is 'N'". Zero known false positives in practice,
# so this alone is grounds for rejection — see validate_plan.
_UNDERSCORE_IN_QUESTION_RE = re.compile(r"[a-zA-Z]_[a-zA-Z]")

# Phrases that name the ML METHOD rather than what the user wants to know —
# the generation prompt already tells the model never to use this vocabulary
# in `question` (see query_planner.md); this is the deterministic backstop
# that actually rejects a plan that slips one through instead of hoping a
# judge notices later. Multi-word PHRASES only (never bare words like
# "predict"/"classify"/"model"), since those are legitimate domain nouns in
# plenty of real questions (e.g. "what incident classification is most
# likely", "what model type does it have") — the phrase requires an actual
# verb-level method reference to trip.
_JARGON_PHRASES: tuple = (
    "can we predict", "would predict", "predict whether", "help predict",
    "forecast predicts", "using a machine learning model", "using machine learning",
    "training data", "held-out", "held out", "hold-out data",
    "confounding variable", "confounder",
    "regression model", "run a regression",
    "classifier model", "classification model",
    "correlation coefficient", "impute missing",
)


def _question_leaks_implementation(question: str) -> Optional[str]:
    """Human-readable reason ``question`` leaks a raw column name or ML
    method jargon, or ``None`` when it's clean. See ``validate_plan``."""
    text = question or ""
    if _UNDERSCORE_IN_QUESTION_RE.search(text):
        return (
            "contains an underscore, which almost always means a raw column "
            "identifier was pasted into the question text verbatim"
        )
    low = text.lower()
    for phrase in _JARGON_PHRASES:
        if phrase in low:
            return (
                f"contains the ML-method phrase {phrase!r} — describe what "
                "the user wants to know, not the technique"
            )
    return None


def _is_identifier_like(column: str) -> bool:
    """True when a column name reads as an identifier/label, not a quantity.

    Tokenizes on non-alphanumerics (``school_code`` -> ``school``, ``code``)
    and matches whole tokens only, so ``grid`` or ``valid`` never trip the
    ``id`` token.
    """
    tokens = [t for t in _TOKEN_SPLIT_RE.split(str(column).lower()) if t]
    return any(t in _IDENTIFIER_TOKENS for t in tokens)

QueryPlan = Union[SQLQueryPlan, PandasQueryPlan]
PlanStep = Union[SQLPlanStep, PandasPlanStep]


class PlanValidationError(ValueError):
    """Raised when a structured query plan fails structural validation.

    Signals that a plan references an unknown table alias, references a column
    that does not exist in any referenced table, has no steps, or (Pandas
    plans only) has a ``task_types`` set inconsistent with its steps
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
    query plans, each with its own skill needs). Kept as a separate client
    (rather than toggling ``response_model`` on the same instance) so a single
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
        skills_available: bool = True,
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
                a real up-to-10-row sample per table so questions — especially
                hypothetical-scenario ones — ground their concrete values in
                actually observed data rather than invented ones.
            skills_available: Whether ML skills (classification, regression,
                timeseries, causal) may be offered to the planner at all this
                call. ``False`` (set by the caller when the account is near
                its daily TabPFN usage quota — see agent.py's
                ``TABPFN_USAGE_GATE_RATIO``) omits every ML instruction from
                the prompt AND is enforced in ``validate_plan``, so a plan
                with any skill op is rejected regardless of what the model
                does.

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
            skills_available=skills_available,
        )

        # Known columns per alias are derived from the column statistics so plan
        # validation (Requirement 5.5) can check every referenced column exists.
        known_columns = self._known_columns(stats)

        raw_plan, _usage = self._request_plan(prompt)
        candidate = self._assemble_plan(raw_plan, constraint_links)

        # First validation attempt. On success the normalised plan (contiguous
        # 1..N orders) is returned directly (Requirement 5.3).
        try:
            return self.validate_plan(
                candidate, aliases, known_columns, skills_available=skills_available
            )
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
            return self.validate_plan(
                retry_candidate, aliases, known_columns, skills_available=skills_available
            )
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
        skills_available: bool = True,
    ) -> List[QueryPlan]:
        """Produce several independent query plans in a single LLM call.

        Unlike :meth:`plan` (one plan, one question), this asks the model for
        ``num_plans`` *distinct* business questions over the same tables, each
        with its own ``question``/``question_keywords`` and its own ordered
        ``steps`` — and, for Pandas plans, its own ``task_types``. This is what
        lets downstream skill injection (``GenerationCoordinator``) pick different
        skills per plan/question instead of one skill for an entire run: e.g.
        one plan may be a plain aggregation (no skill) while another needs
        ``classification`` or ``causal``.

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
                a real up-to-10-row sample per table so questions — especially
                hypothetical-scenario ones — ground their concrete values in
                actually observed data rather than invented ones.
            skills_available: Whether ML skills may be offered to the planner
                at all this call (see :meth:`plan`). ``False`` disables ML
                instructions in the prompt for every plan in the batch and is
                enforced in ``validate_plan``.

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
            num_plans=num_plans, dfs=dfs, skills_available=skills_available,
        )
        raw_set, _usage = self._request_plan_batch(prompt)
        raw_plans = self._extract_raw_plans(raw_set)

        plans: List[QueryPlan] = [
            self._validate_or_retry_one(
                raw_plan, prompt, constraint_links, aliases, known_columns,
                skills_available=skills_available,
            )
            for raw_plan in raw_plans
        ]

        if not plans:
            # The model returned nothing usable at all — fall back to a single
            # free-text plan so the caller always gets at least one plan back.
            empty = self._empty_plan(constraint_links)
            plans = [self._free_text_fallback(empty, aliases, constraint_links)]

        return plans

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
        skills_available: bool = True,
    ) -> tuple[Optional[QueryPlan], dict]:
        """Re-request ONE plan corrected against reviewer feedback.

        Used by the plan judge panel's correction loop: when the panel rejects
        a plan, its aggregated feedback/suggestions are handed back here so
        the planner can rewrite the question and/or steps. The revised plan
        goes through the same structural validation as any other plan, with
        one validation-error retry; the mandatory ``table_links`` constraints
        are preserved unchanged, exactly as in :meth:`plan`.

        Args:
            skills_available: Must match whatever the plan's ORIGINAL
                generation call used (see :meth:`plan`) — a plan denied ML
                skills for quota reasons should not regain them mid-run just
                because a correction round happens to fall after the quota
                check.

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
            skills_available=skills_available,
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
            "the top level. Do NOT wrap it in a `plans` list or any other key.\n\n"
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
        try:
            return self.validate_plan(
                candidate, aliases, known_columns, skills_available=skills_available
            ), usage_total
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
        try:
            return self.validate_plan(
                retry_candidate, aliases, known_columns, skills_available=skills_available
            ), usage_total
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
        skills_available: bool = True,
    ) -> QueryPlan:
        """Validate a single raw plan from a batch, re-requesting just that one.

        Mirrors the retry-once-then-fallback shape of :meth:`plan`, but scoped
        to a single plan within the batch so one bad plan never forces a
        re-request (and re-validation) of the whole set.
        """
        candidate = self._assemble_plan(raw_plan, constraint_links)
        try:
            return self.validate_plan(
                candidate, aliases, known_columns, skills_available=skills_available
            )
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
            return self.validate_plan(
                retry_candidate, aliases, known_columns, skills_available=skills_available
            )
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
        result = self._batch_client.request_plan_batch(prompt)
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
        skills_available: bool = True,
    ) -> QueryPlan:
        """Validate and normalise a structured plan.

        Assigns contiguous ``1..N`` ``order`` values to the steps (Requirement
        5.3), then verifies that every table referenced by a step is a known
        alias (Requirement 5.4) and that every column referenced by a step
        exists in one of the tables that step references (Requirement 5.5).
        For Pandas plans, also asserts ``task_types`` is exactly the distinct
        set of skill-relevant ops in the steps (design pseudocode). SQL plans
        have no ``task_types`` to check.

        When ``skills_available`` is ``False``, ANY step whose ``op`` is an ML
        skill (``SKILL_TASK_TYPES``) is rejected outright — this is the
        enforcement half of the quota gate (see ``_build_prompt``'s
        ``offer_skills``): the prompt not mentioning ML ops is only a nudge,
        this is what actually guarantees a throttled run cannot produce one,
        even if the model reaches for a skill unprompted.

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

        # Question-quality enforcement: a raw column identifier or ML-method
        # phrase leaking into the question is a hard rejection, not a style
        # nit the plan judge is left to catch on its own. Applies to BOTH
        # plan kinds — SQL questions can leak jargon/column names exactly
        # the same way Pandas ones can.
        for text_field in ("question", "translated_question"):
            text = getattr(plan, text_field, "") or ""
            reason = _question_leaks_implementation(text)
            if reason:
                raise PlanValidationError(
                    f"plan.{text_field} {reason}: {text!r}. Rewrite it as an "
                    "average, non-technical user would ask it — no column "
                    "names, no ML/analytics vocabulary, only plain words for "
                    "what they want to know."
                )

        # Quota gate enforcement: independent of everything else below, a
        # throttled request (skills_available=False) may not produce ANY
        # skill step, regardless of what the prompt did or didn't say.
        if not skills_available:
            skill_ops_used = sorted({s.op for s in plan.steps if s.op in SKILL_TASK_TYPES})
            if skill_ops_used:
                raise PlanValidationError(
                    f"plan uses ML skill op(s) {skill_ops_used}, but ML skills "
                    "are not available for this batch (the account is near "
                    "its daily usage quota) — rewrite every step as a plain "
                    "filter/join/union/group/aggregate/sort/select/derive "
                    "operation, with no task_types"
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
        # derive, an ML skill step's predictions, or explicitly declared
        # outputs in params). Once true, an unresolved column reference is
        # downgraded to a warning: `params` is free-form, so output harvesting
        # is best-effort, and a false rejection costs a full planner
        # re-request that tends to yield a WORSE plan (derived names pushed
        # into prose where no validator — or generator — can see them). The
        # statement generator reads the whole plan including descriptions, so
        # an unresolved derived name is a soft signal, not a broken plan.
        derivation_seen = False

        # Columns an earlier `filter` step pinned to a single value via an
        # equality condition, per table alias — reusing one of these as an ML
        # feature later in the same plan is checked below (validate_plan).
        pinned_columns_by_alias: dict = {}

        for step in plan.steps:
            # Requirement 5.4: every referenced table must be a known alias.
            for table in step.tables:
                if table not in known_aliases:
                    raise PlanValidationError(
                        f"step {step.order} ({step.op}) references unknown table "
                        f"alias {table!r}; known aliases: {sorted(known_aliases)}"
                    )

            # Requirement 5.5: every referenced column must exist in at least
            # one of the tables the step references — physical schema columns
            # plus outputs registered by earlier steps. Exceptions:
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

            # Track columns a `filter` step pins to a single value (see
            # _EQUALITY_FILTER_RE) so a later ML step reusing one as a
            # feature can be caught below — after such a filter the column
            # is constant across the training population.
            if step.op == "filter" and _EQUALITY_FILTER_RE.search(step.description or ""):
                for table in step.tables:
                    pinned_columns_by_alias.setdefault(table, set()).update(step.columns)

            # Register outputs this step declares in params (derive's
            # new_column, aggregate's output_column / aggregations keys) so
            # later steps can reference them without tripping the check.
            declared_outputs = self._declared_output_columns(step)
            for table in step.tables:
                columns_by_alias.setdefault(table, set()).update(declared_outputs)

            if produces_columns or declared_outputs or step.op in SKILL_TASK_TYPES:
                derivation_seen = True

            # ML target sanity: a classification/regression step must declare
            # its target, and no ML step may point its target/outcome at an
            # identifier-like column ("predicting" a zip code or an ID is
            # meaningless). Raising here feeds the reason back through the
            # standard re-request loop so the retry picks a real outcome.
            if step.op in SKILL_TASK_TYPES:
                roles = step.columns_role or {}
                declared = []
                for role in ("target", "outcome"):
                    value = roles.get(role)
                    if isinstance(value, str):
                        declared.append(value)
                    elif isinstance(value, (list, tuple)):
                        declared.extend(str(v) for v in value)
                if step.op in ("classification", "regression") and not declared:
                    raise PlanValidationError(
                        f"step {step.order} ({step.op}) does not declare its "
                        "prediction target under `columns_role.target` — add "
                        "the real outcome column it predicts"
                    )
                for column in declared:
                    if _is_identifier_like(column):
                        raise PlanValidationError(
                            f"step {step.order} ({step.op}) uses identifier-like "
                            f"column {column!r} as its prediction target. "
                            "Identifiers (zip/postal codes, IDs, codes, phone "
                            "numbers, URLs, addresses, names, coordinates) are "
                            "labels, not quantities — predicting them is "
                            "meaningless. Choose a genuinely measured or "
                            "categorical outcome column instead, or drop the "
                            "ML step if the table has none"
                        )

                # Zero-variance feature sanity: a feature column an earlier
                # `filter` step already pinned to a single value (per table)
                # carries no signal for this step to learn from — the
                # "prediction" collapses to the target's mode/mean among the
                # filtered rows, i.e. exactly the plain aggregate this ML
                # step is dressing up as a model.
                feature_columns = [c for c in step.columns if c not in declared]
                for table in step.tables:
                    pinned = pinned_columns_by_alias.get(table, set())
                    reused = sorted(set(feature_columns) & pinned)
                    if reused:
                        raise PlanValidationError(
                            f"step {step.order} ({step.op}) uses "
                            f"{reused} as predictive feature(s) on table "
                            f"{table!r}, but an earlier `filter` step already "
                            "pinned that column to a single value there. "
                            "After that filter the column is constant across "
                            "the training population, so the model has no "
                            "variance to learn from in it. Either drop the "
                            "column as a feature, or drop the ML step and "
                            "answer with a plain aggregate over the filtered "
                            "rows instead"
                        )

        # Design pseudocode: task_types must equal the distinct skill ops in
        # steps. Only Pandas plans carry task_types at all.
        if isinstance(plan, PandasQueryPlan):
            expected_task_types = self._derive_task_types(plan.steps)
            if plan.task_types != expected_task_types:
                raise PlanValidationError(
                    f"task_types {plan.task_types} do not match the skill ops "
                    f"derived from steps {expected_task_types}"
                )

        return plan

    @staticmethod
    def _known_columns(stats: Sequence[TableStats]) -> dict:
        """Map each alias to the set of its column names from the statistics."""
        return {table.alias: {c.column for c in table.columns} for table in stats}

    @staticmethod
    def _declared_output_columns(step) -> set:
        """Output column names a step declares in its free-form ``params``.

        Well-structured plans put a step's INPUT columns in ``step.columns``
        and its OUTPUT names in ``params`` — ``derive`` declares
        ``params.new_column``, ``aggregate`` declares ``params.output_column``
        or the keys of ``params.aggregations``. These outputs are legitimate
        references for later steps (a ``sort`` on an aggregate's result), so
        validate_plan registers them instead of rejecting the reference.
        ``params`` is free-form, so this harvest is best-effort by convention.
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
                task_types=[], table_links=constraint_links,
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
        fail column validation), and (for Pandas plans) an empty ``task_types``
        set (no skill).
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
                task_types=[],
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
          :class:`QueryLink`), those links are coerced and returned verbatim.
        * Otherwise a single ``join`` link is synthesised across the tables that
          carry involved columns, using the ordered union of those columns and
          the ``match`` text (when a string) as the description.
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

        # Ordered union of the involved columns across the linked tables.
        columns: List[str] = []
        for alias in linked_tables:
            for col in involved_cols.get(alias, []):
                if col not in columns:
                    columns.append(col)

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
                columns=columns,
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
        skills_available: bool = True,
    ) -> str:
        # Whether the planner is even told ML skills exist this call. Gated
        # OFF (independently of the question/tables) when the caller reports
        # the account is near its daily TabPFN usage quota (see
        # agent.py's TABPFN_USAGE_GATE_RATIO) — a throttled Pandas run gets
        # the exact same plain-operations-only wording a SQL run always gets.
        offer_skills = self._is_pandas and skills_available
        skill_ops = ", ".join(SKILL_TASK_TYPES)
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
            if offer_skills:
                batch_note = (
                    "MULTI-PLAN REQUIREMENTS:\n"
                    f"- Each of the {num_plans} plans is fully self-contained: its own "
                    "`question`, its own `steps`, and therefore its own `task_types` "
                    "(derived from its own steps). Plans do NOT share steps.\n"
                    "- Each plan's `task_types` should follow from whatever is the "
                    "most genuinely interesting, well-justified question that plan "
                    "asks — do NOT aim for a fixed mix of plain vs. ML plans across "
                    f"the batch. A plain plan (only filter/join/group/aggregate/sort/"
                    "select/derive, no `task_types`) is just as valid an outcome as "
                    f"one using a machine-learning operation ({skill_ops}); never "
                    "force an ML operation onto a plan merely to have one, or "
                    "because the data happens to allow it — only when the question "
                    "it produces is one a curious non-expert would actually ask.\n"
                    "- Consider EVERY ML operation before choosing, so you're not "
                    "defaulting to the same one or two every time — but 'consider' "
                    "means check whether it's the right fit, not manufacture a plan "
                    "for it. `causal` in particular tends to get overlooked: when "
                    "the tables contain a treatment-like column (an intervention, "
                    "program, policy, or group membership), an outcome column, and "
                    "other covariates that could confound their relationship, weigh "
                    "whether a causal framing is genuinely the more interesting "
                    "question there — not plain stratified aggregation wearing "
                    "`causal` framing. Only build that plan when it clears that "
                    "bar; the mere presence of the triple is not, by itself, "
                    "sufficient reason to build one.\n"
                    "- Do not repeat the same question across plans.\n\n"
                )
            else:
                batch_note = (
                    "MULTI-PLAN REQUIREMENTS:\n"
                    f"- Each of the {num_plans} plans is fully self-contained: its own "
                    "`question` and its own `steps`. Plans do NOT share steps.\n"
                    "- Do not repeat the same question across plans.\n\n"
                )

        if offer_skills:
            ops_statement = (
                "- Every step's `op` MUST be exactly one of these values — no others "
                "are valid, even if a step's intent (e.g. correlating two columns) "
                "isn't literally one of these words: filter, join, union, group, "
                "aggregate, sort, select, derive, classification, regression, "
                "timeseries, causal. There is no `correlation` op — "
                "express a correlation/relationship-strength analysis as `derive` "
                "(compute the statistic, e.g. `.corr()`) or `aggregate`, not as its "
                "own op.\n"
                f"- Machine-learning operations are: {skill_ops}. Use them only when "
                "the question genuinely requires prediction/inference. A "
                "`classification` or `regression` step MUST record its target "
                "column under `columns_role.target` — this is validated.\n"
                "- THE CORE TEST for whether ANY of these four ops genuinely "
                "belongs: is the target's value for the case in question "
                "genuinely UNCERTAIN given the inputs, or is it already "
                "recoverable by a LOOKUP or by ARITHMETIC? If a human could "
                "get the same answer by finding matching rows and reading "
                "off the result (a lookup — e.g. zip code -> borough is a "
                "near-fixed mapping), or by combining the given numbers with "
                "+, -, *, / (arithmetic — e.g. total_enrollment is literally "
                "the sum of the grade-level columns), NO ML step belongs "
                "there at all, no matter how the question is phrased — plan "
                "it as `derive`/`aggregate` instead. Reach for classification/"
                "regression/timeseries/causal only when the relationship is "
                "genuinely probabilistic: real-world cases with similar "
                "inputs still land on different real outcomes. Every rule "
                "below is a specific consequence of this one test.\n"
                "- For any `classification` step, the prediction target must be "
                "a low-cardinality column: check its `cardinality` in COLUMN "
                "STATISTICS and never target a column with more than 50 "
                "distinct values — the classifier hard-fails above 160 classes, "
                "and targets anywhere near that are unanswerable. Pick a "
                "genuinely categorical column instead (status, category, "
                "type, ...), or add an explicit earlier step that collapses "
                "the target to its top ~10 most frequent values plus \"other\" "
                "and note that collapse in the step's `description`.\n"
                "- An ML target must be a genuinely measured quantity or a "
                "meaningful category. Identifier-like columns are LABELS, not "
                "quantities, and are NEVER valid targets (and rarely useful "
                "features): zip/postal codes, IDs and record codes, phone "
                "numbers, URLs, emails, street addresses, entity names, and "
                "latitude/longitude. \"Predicting\" a zip code or an ID is "
                "meaningless even if the column is numeric. If a table's only "
                "numeric columns are identifiers, do NOT propose an ML plan "
                "for it — plain aggregation plans are the right choice there.\n"
                "- Never use a column as an ML feature if an earlier step in "
                "the SAME plan already filtered it down to one specific value "
                "(e.g. `filter` where street equals 'Broadway', then "
                "`classification` using street as a feature). After that "
                "filter the column is constant across the rows the model "
                "would train on — it carries no signal, and the \"prediction\" "
                "is really just the target's mode/mean among the filtered "
                "rows. That's a plain `aggregate` on the filtered subset, not "
                "an ML step — plan it as such.\n"
            )
            if len(aliases or {}) >= 2:
                ops_statement += (
                    "- An ML plan MAY combine tables first: when joining/unioning "
                    "the provided tables surfaces a column that itself makes a "
                    "genuinely good ML feature or target — e.g. join two tables, "
                    "then run `regression` on a column contributed by the joined "
                    "table — plan it as one plan combining tables (`join`/`union`) "
                    "AND an ML step over the newly-combined columns. This is an "
                    "option, not a quota: do NOT manufacture a join/union chain "
                    "just to have somewhere to attach an ML step, and do not force "
                    "it when the joined columns aren't actually useful features/"
                    "targets for the ML step — a single-table ML plan, or no ML "
                    "plan at all, is the right call when that's what the tables "
                    "support.\n"
                )
            ops_statement += (
                "- A `classification` or `regression` step's `question` should "
                "read as a specific what-if insight question a curious "
                "non-expert would ask — PREFER this hypothetical framing; fall "
                "back to a general-reliability framing only when the data "
                "offers no natural concrete scenario:\n"
                "  - Hypothetical framing MUST name concrete values for the "
                "scenario's predictors IN THE QUESTION TEXT ITSELF — a question "
                "that only names the predictor COLUMNS, with no committed values, "
                "is not a hypothetical scenario, it's an unstated evaluation "
                "question wearing hypothetical phrasing. "
                "Wrong: \"Can we predict the number of students in a class based "
                "on the grade level and program type?\" (names the columns, "
                "commits to no scenario). "
                "Right: \"What would be the number of students in a class of "
                "grade 5 and program type STEM?\" (the scenario's own values are "
                "stated, and it asks for one specific insight). \n"
                "  - The fallback reliability framing asks about general "
                "predictive power without committing to specific input values, "
                "phrased in everyday words (e.g. \"how well do grade level and "
                "program type tell us how big a class will be?\") — never in "
                "model-evaluation vocabulary (\"accuracy\", \"held-out\", "
                "\"predictive model\"). Use it only when the question is "
                "genuinely about reliability, not one scenario.\n"
                "  - Make the step's `description` state which framing was chosen "
                "and, for hypothetical framing, the concrete scenario values "
                "chosen — the code-generation step needs both to build the right "
                "input row.\n"
                "- Use a `timeseries` op ONLY when a table carries a genuine "
                "ordered time dimension with enough history to learn from: a "
                "date/datetime/period column (or many sequential period "
                "columns) covering roughly 8+ observed points per series. "
                "Comparing or differencing a handful of wide-format year "
                "columns (e.g. `rate_2012` vs `rate_2013`) is NOT a timeseries "
                "operation — it needs no forecasting model; plan it as "
                "`derive`/`aggregate` instead.\n"
                "- A `timeseries` step's `question` should ask about the "
                "DIRECTION/TENDENCY of the series (e.g. is it expected to "
                "increase, decrease, or stay stable) rather than demanding an "
                "exact future value, and must STATE ITS AS-OF ANCHOR IN THE "
                "QUESTION TEXT ITSELF — the last observed date/period of the "
                "data (see the Time Context section), e.g. \"As of <last "
                "observed period>, is ... heading up or down?\". The projection "
                "target is always relative to that stated anchor: within the "
                "observed range or the period(s) immediately following it — "
                "never a date implied by the current datetime when the data "
                "ends earlier.\n"
                "- A `causal` step's `question` should be phrased as the "
                "natural disentangling question a non-expert would ask: "
                "explicitly contrast the suspected driver against the "
                "alternative explanation in plain words (\"Is it <the "
                "treatment/choice> itself that leads to <the outcome>, or is "
                "it rather <the confounding factor> that explains it?\") — "
                "with both candidate explanations named from the data. It must "
                "motivate why a simple association is not enough (a suspected "
                "confound, or an intervention whose effect needs isolating) — "
                "not a generic \"what's associated with X\" framing that "
                "`derive`/`aggregate` could already answer.\n"
            )
        elif self._is_pandas:
            # skills_available is False: the account is near its daily TabPFN
            # usage quota (see agent.py's TABPFN_USAGE_GATE_RATIO). The model
            # isn't even told ML ops exist this call — same wording SQL
            # always gets — and validate_plan additionally rejects any skill
            # op that slips through regardless of this prompt (belt and
            # braces, not just a hopeful nudge).
            ops_statement = (
                "- Steps must use only plain relational/aggregation operations "
                "(filter, join, union, group, aggregate, sort, select, derive) — "
                "machine-learning operations are not available for this batch "
                "(the account is near its daily ML-skill usage quota).\n"
            )
        else:
            ops_statement = (
                "- Steps must use only plain relational/aggregation operations "
                "(filter, join, union, group, aggregate, sort, select, derive) — "
                "machine-learning operations are not available for SQL generation.\n"
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
        )

    @staticmethod
    def _render_time_context() -> str:
        """Render the "### TIME CONTEXT" block of the planner prompt.

        Gives the planner a concrete "now" plus the rule that matters for
        `timeseries` plans: the tables' data usually ends well BEFORE the
        current datetime, so a projection must be anchored to the data's own
        last observed timestamp/period (visible in the table sample and column
        statistics), never to the present day.
        """
        # Date-level granularity on purpose: a full timestamp changes on every
        # call and would invalidate the provider's cached prompt prefix for
        # every token that follows it; the date is stable across a whole day.
        return (
            f"- Current date: {datetime.now().date().isoformat()}\n"
            "- The tables' observed data may end well before this datetime. "
            "For any `timeseries` step, first identify the LAST "
            "timestamp/period actually observed in the table sample and "
            "column statistics, then frame the question's projection target "
            "relative to THAT point: within the observed range or the "
            "period(s) immediately following it. Never phrase a forecast "
            "relative to the current datetime (e.g. \"next year\", \"today\", "
            "\"upcoming months\") when the data ends earlier than it."
        )

    @staticmethod
    def _render_table_sample(dfs: Optional[Sequence[Any]], aliases: dict) -> str:
        """Render up to 10 real sample rows per table.

        Grounds the planner's questions — especially a hypothetical scenario's
        concrete predictor values — in values that actually occur in the data,
        rather than the model inventing a plausible-sounding but nonexistent
        one. Uses ``DataFrame.to_json`` (not a raw ``.to_dict()``) so NaN/NaT/
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
                rows = json.loads(df.head(10).to_json(orient="records", date_format="iso"))
            except (TypeError, ValueError):
                rows = []
            payload.append({"alias": alias, "rows": rows})
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @staticmethod
    def _render_links(links: Sequence[QueryLink]) -> str:
        if not links:
            return "(no mandatory links — single table)"
        payload = [
            {
                "type": link.type,
                "tables": link.tables,
                "columns": link.columns,
                "description": link.description,
            }
            for link in links
        ]
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
                            "numeric_min": c.numeric_min,
                            "numeric_max": c.numeric_max,
                            "numeric_mean": c.numeric_mean,
                            "top_values": c.top_values,
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
        result = self._client.request_plan(prompt)
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
        relationships are preserved unchanged (Requirement 6.2). For Pandas
        plans, ``task_types`` is derived from the ordered steps (Requirement
        5.2); SQL plans have no ``task_types`` field.
        """
        raw_plan = raw_plan or {}
        metadata = self._extract_metadata_fields(raw_plan)

        tables = self._coerce_tables(raw_plan.get("tables", []))

        if self._is_pandas:
            steps = self._coerce_steps(raw_plan.get("steps", []), PandasPlanStep)
            task_types = self._derive_task_types(steps)
            logger.info("Selected task_types: %s", task_types if task_types else "(none)")
            return PandasQueryPlan(
                question=str(raw_plan.get("question", "")),
                question_keywords=self._limit_keywords(raw_plan.get("question_keywords")),
                plan_keywords=self._limit_keywords(raw_plan.get("plan_keywords")),
                steps=steps,
                task_types=task_types,
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
        return {
            "query_plan": str(raw_plan.get("query_plan", "")),
            "translated_question": str(raw_plan.get("translated_question", "")),
            "translated_question_keywords": self._limit_keywords(
                raw_plan.get("translated_question_keywords")
            ),
            "detected_language": str(raw_plan.get("detected_language", "")),
            "topic": str(raw_plan.get("topic", "")),
            "story": str(raw_plan.get("story", "")),
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
    def _derive_task_types(steps: Sequence[Any]) -> List[str]:
        """Distinct skill-relevant ops, in first-appearance order."""
        seen: List[str] = []
        for step in steps:
            if step.op in SKILL_TASK_TYPES and step.op not in seen:
                seen.append(step.op)
        return seen

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
