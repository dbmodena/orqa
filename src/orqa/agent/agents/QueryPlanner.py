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
``1..N`` step orders and checks table/column references, and
:meth:`QueryPlanner.plan_batch` re-requests a plan once on failure before
falling back to a schema-valid free-text plan (Requirements 5.3, 5.4, 5.5,
5.6).

The planner never writes a question: every plan answers one question that was
written and approved upstream (see :mod:`orqa.agent.agents.QuestionStage`),
which is pinned onto the plan and cannot change in a revision.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Union, get_args

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
from ..utility.column_provenance import compose_feedback, resolve_plan_columns
from ..utility.structured_outputs import QueryLink, Table
from ...utils import shield_dataframe_for_prompt
from .table_analyzer import render_table_facts

logger = logging.getLogger(__name__)

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


_HEADER_SEPARATORS = r"[\s_\-/]+"
_CAMEL_CASE_RE = re.compile(r"[a-z][A-Z]")


def _question_copies_column_header(question: str, headers: Iterable[str]) -> Optional[str]:
    """Human-readable reason ``question`` reproduces a column header as it is
    written in the table, or ``None``. The counterpart of the underscore check
    above for the headers a question can copy WITHOUT an underscore.

    Conservative on purpose — an ordinary word that merely coincides with a
    header ("year", "borough", "amount") is just language and never flagged.
    A header counts as copied when it is:

    * a single identifier-like token (camel case, ALL CAPS such as ``DBN``, or
      letters mixed with digits) that appears in the question exactly as
      written; or
    * two words written with capitals or joined by ``-``/``/``/``_`` ("Boro
      Cd") that appear together exactly as written — a plain lowercase
      two-word header is skipped, since the same two words are natural prose;
    * three or more words that appear together, in any casing — nobody
      says a three-word spreadsheet label by accident.
    """
    text = question or ""
    for raw in headers:
        header = str(raw).strip()
        if not header or header.lower().startswith("unnamed"):
            continue
        words = [w for w in re.split(_HEADER_SEPARATORS, header) if w]
        if not words:
            continue
        if len(words) == 1:
            word = words[0]
            identifier_like = (
                bool(_CAMEL_CASE_RE.search(word))
                or (word.isalpha() and word.isupper() and len(word) >= 2)
                or (any(c.isdigit() for c in word) and any(c.isalpha() for c in word))
            )
            if identifier_like and re.search(
                rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])", text
            ):
                return f"copies the column header {header!r} as written"
            continue
        pattern = r"(?<![A-Za-z0-9])" + _HEADER_SEPARATORS.join(map(re.escape, words)) + r"(?![A-Za-z0-9])"
        if len(words) >= 3:
            found = re.search(pattern, text, re.IGNORECASE)
        elif header != header.lower() or re.search(r"[_\-/]", header):
            found = re.search(pattern, text)
        else:
            found = None
        if found:
            return f"copies the column header {header!r} as written"
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

    def plan_batch(
        self,
        analyses: Sequence[dict],
        aliases: dict,
        match: Any,
        involved_cols: Optional[dict],
        stats: Sequence[TableStats],
        languages: Optional[Sequence[str]] = None,
        dfs: Optional[Sequence[Any]] = None,
        table_metadata: Optional[dict] = None,
        *,
        questions: Sequence[dict],
    ) -> List[QueryPlan]:
        """Plan every approved question in a single LLM call.

        The planner never writes a question: each plan answers one question
        already written AND approved upstream (see
        ``orqa.agent.agents.QuestionStage``), and is asked for its ordered
        ``steps``, each table's role, and the result declaration.

        Each returned plan preserves the mandatory ``table_links`` from the
        upstream ``match``/``involved_cols`` constraints (never altered), and
        each plan is independently validated: a plan that fails validation is
        re-requested once on its own (not the whole batch), then falls back to
        a schema-valid free-text plan if the retry also fails.

        Args:
            analyses: Per-table analysis dicts (one per alias).
            aliases: Mapping of alias -> dataset name.
            match: The upstream match constraint.
            involved_cols: Mapping of alias -> mandatory relationship columns.
            stats: Per-table column statistics injected into the prompt.
            languages: Detected languages for the plans' metadata fields.
            dfs: The tables in scope (one per alias, same order), used to inject
                a real up-to-10-row sample per table so the steps ground their
                concrete values in actually observed data rather than invented
                ones.
            table_metadata: Mapping of alias -> the table's portal metadata
                (see ``table_analyzer.portal_metadata``); also forwarded to
                every revision this batch triggers.
            questions: The approved questions, ``[{"question": str,
                "question_keywords": [str, ...], "difficulty": "easy" |
                "medium" | "hard"}, ...]``, at least one. Plan ``i`` is the
                plan for ``questions[i]``, whose text, keywords and
                ``difficulty`` are pinned onto it after every request (never
                trusted to the model). The difficulty is the slot's target
                tier, judged on the QUESTION upstream: nothing on the plan side
                estimates, checks or revises against a tier, so plans in a
                batch need not have distinct tiers. No free-text placeholder
                plan is invented when the model returns nothing: the result
                may then be empty, or shorter than ``questions`` (an unplanned
                question is dropped, never guessed at).

        Returns:
            A list of kind-appropriate query plans, each independently
            schema-valid, each preserving the mandatory ``table_links``; empty
            when nothing could be planned.
        """
        if not questions:
            raise ValueError("plan_batch needs at least one approved question to plan")
        languages = list(languages or [])
        fixed = [dict(q) for q in questions]

        constraint_links = self._build_constraint_links(match, involved_cols, aliases)
        known_columns = self._known_columns(stats)

        prompt = self._build_prompt(
            analyses, aliases, constraint_links, stats, languages,
            dfs=dfs,
            table_metadata=table_metadata,
            fixed_questions=fixed,
        )
        raw_set, _usage = self._request_plan_batch(prompt)
        raw_plans = self._extract_raw_plans(raw_set)

        if len(raw_plans) != len(fixed):
            logger.warning(
                "Fixed-question planning returned %d plan(s) for %d "
                "question(s); pairing by position — an unmatched "
                "question is dropped.", len(raw_plans), len(fixed),
            )
        paired = list(zip(raw_plans, fixed))
        raw_plans = [self._pin_question(rp, fq) for rp, fq in paired]
        fixed_by_index = [fq for _, fq in paired]

        plans: List[QueryPlan] = [
            self._validate_or_retry_one(
                raw_plan, prompt, constraint_links, aliases, known_columns,
                fixed_question=fixed_question,
            )
            for raw_plan, fixed_question in zip(raw_plans, fixed_by_index)
        ]

        if not plans:
            return []

        plans = [self._pin_table_descriptions(p, analyses) for p in plans]

        return plans

    @staticmethod
    def _revision_pins(plan: QueryPlan) -> dict:
        """Fields a revision may never change: the question with its keywords
        and its difficulty (both settled upstream, on the question)."""
        return {
            "difficulty": plan.difficulty,
            "question": plan.question,
            "question_keywords": list(plan.question_keywords),
        }

    @staticmethod
    def _pin_question(raw_plan: dict, fixed_question: dict) -> dict:
        """A copy of a raw plan dict with the upstream-approved question and
        keywords — and its difficulty tier — forced over whatever the model
        returned; deterministic, like ``_pin_table_descriptions``: the prompt
        already asks the model to copy them verbatim, but a request is not a
        guarantee. A question carrying no valid tier leaves the model's own."""
        pinned = {
            **raw_plan,
            "question": fixed_question["question"],
            "question_keywords": list(fixed_question.get("question_keywords") or []),
        }
        tier = str(fixed_question.get("difficulty") or "").strip().lower()
        if tier in get_args(_DIFFICULTY_LEVELS):
            pinned["difficulty"] = tier
        return pinned

    def _pin_table_descriptions(
        self, plan: QueryPlan, analyses: Sequence[dict]
    ) -> QueryPlan:
        """Overwrite each `tables[].description`/`.keywords` with the cached
        table-analysis value for that alias — deterministic, no judge
        involved.

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
        table_metadata: Optional[dict] = None,
    ) -> tuple[Optional[QueryPlan], dict]:
        """Re-request ONE plan corrected against reviewer feedback.

        The plan's question was approved upstream, so a revision may change
        only the steps / table roles / the result declaration: the prompt says
        so and the question and its keywords are pinned back onto the result,
        exactly like ``difficulty`` is (see :meth:`_revision_pins`).

        Used by the plan judge panel's correction loop: when the panel rejects
        a plan, its aggregated feedback/suggestions are handed back here so
        the planner can rewrite the steps. The revised plan goes through the
        same structural validation as any other plan, with one
        validation-error retry; the mandatory ``table_links`` constraints are
        preserved unchanged, exactly as in :meth:`plan_batch`.

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
            table_metadata=table_metadata,
            fixed_questions=[
                {"question": plan.question, "question_keywords": list(plan.question_keywords)}
            ],
        )
        rewrite_scope = (
            "fix the steps, the tables' roles and the result declaration "
            "as needed. The QUESTION is frozen — it was approved before "
            "planning: keep `question` and `question_keywords` exactly "
            "as given (anything you return for them is ignored), and "
            "never try to fix a problem by rewording it."
        )
        correction_prompt = (
            f"{base_prompt}\n\n"
            "### PLAN CORRECTION REQUEST\n"
            "A review panel rejected the plan below. Revise it so every point "
            f"of the review feedback is addressed — {rewrite_scope} "
            "Return a SINGLE flat JSON object matching the "
            "plan schema directly (`question`, `steps`, `table_links`, ...) at "
            "the top level. Do NOT wrap it in a `plans` list or any other key. "
            "The `difficulty` is fixed for this plan and is not a reason to "
            "add or remove steps (any `difficulty` you return here is "
            "ignored).\n\n"
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
        # The question, its keywords and its difficulty are fixed for a
        # correction round. Force-override whatever the model returned,
        # mirroring how table_links is already force-preserved above (see
        # _assemble_plan's docstring).
        candidate = candidate.model_copy(update=self._revision_pins(plan))
        try:
            validated = self.validate_plan(candidate, aliases, known_columns)
            validated = self._pin_table_descriptions(validated, analyses)
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
        retry_candidate = retry_candidate.model_copy(
            update=self._revision_pins(plan)
        )
        try:
            validated_retry = self.validate_plan(retry_candidate, aliases, known_columns)
            validated_retry = self._pin_table_descriptions(validated_retry, analyses)
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
        fixed_question: Optional[dict] = None,
    ) -> QueryPlan:
        """Validate a single raw plan from a batch, re-requesting just that one.

        ``fixed_question`` (see ``plan_batch``'s ``questions``): re-pinned onto
        the retry's raw plan too, since a re-request is a fresh model output.

        Retries once, then falls back to a free-text plan, scoped to a single
        plan within the batch so one bad plan never forces a re-request (and
        re-validation) of the whole set.
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

        # NOTE: base_prompt is the BATCH prompt (one plan per fixed question)
        # and may still contain its "produce N plans, return a `plans` list"
        # instructions. But this retry is single-plan: _request_plan
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
        if fixed_question is not None:
            raw_retry = self._pin_question(raw_retry, fixed_question)
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
        resolves (Requirement 5.5) — either to a real column of a table the
        step names, or to a derived column an earlier step declared in its
        ``produces``, whose own sources transitively root in real columns.

        Column resolution lives in
        :mod:`orqa.agent.utility.column_provenance`. It reports EVERY column
        problem at once rather than the first, because there is only one
        re-request before :meth:`_free_text_fallback` degrades the plan.

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
                "alias must appear exactly once\n"
                "→ merge the duplicate entries into one."
            )
        missing_tables = known_aliases - plan_table_set
        if missing_tables:
            raise PlanValidationError(
                f"plan.tables is missing an entry for alias(es) "
                f"{sorted(missing_tables)}; every provided table alias must "
                "appear in `tables` with a `reason`\n"
                "→ add one entry per missing alias, each with a justification "
                "specific to that table."
            )
        unknown_tables = plan_table_set - known_aliases
        if unknown_tables:
            raise PlanValidationError(
                f"plan.tables references unknown alias(es) {sorted(unknown_tables)}; "
                f"known aliases: {sorted(known_aliases)}\n"
                "→ use the aliases exactly as given in TABLE ALIASES."
            )
        for t in plan.tables:
            if not (t.reason or "").strip():
                raise PlanValidationError(
                    f"plan.tables entry for {t.name!r} has an empty `reason` "
                    "— every table needs a well-articulated justification\n"
                    "→ state which step(s) it feeds, the specific columns the "
                    "answer depends on it for, and why the question could not "
                    "be answered without it."
                )

        # Normalise known_columns into sets. This is the RAW schema and the
        # resolver treats it as read-only — derived names are never folded in
        # here. Keeping the two namespaces apart is precisely what the old
        # single-set approach could not do, and why a hallucinated name became
        # indistinguishable from a real one a step later.
        raw_columns_by_alias = {
            alias: set(cols) for alias, cols in (known_columns or {}).items()
        }

        # Structural per-step checks that must still fail fast: an unknown
        # alias or a malformed correlate/limit/rank `params` shape leaves the
        # column resolver nothing meaningful to resolve against, so there is
        # no point collecting further column violations on top of them.
        for step in plan.steps:
            # Requirement 5.4: every referenced table must be a known alias.
            for table in step.tables:
                if table not in known_aliases:
                    raise PlanValidationError(
                        f"step {step.order} ({step.op}) references unknown table "
                        f"alias {table!r}; known aliases: {sorted(known_aliases)}\n"
                        "→ use one of the aliases exactly as given in TABLE ALIASES."
                    )
            new_op_error = self._validate_new_op_params(step)
            if new_op_error:
                raise PlanValidationError(new_op_error)

        # Requirement 5.5: every column a step names must resolve to a real
        # schema column or to a derived column an earlier step declared in its
        # `produces` (see orqa.agent.utility.column_provenance). ALL violations
        # are collected and reported together: the planner gets exactly one
        # re-request before degrading to _free_text_fallback, so surfacing one
        # problem per round-trip would spend that retry on a plan with two
        # mistakes.
        violations, _derived = resolve_plan_columns(
            plan.steps,
            raw_columns_by_alias,
            known_aliases=known_aliases,
            declared_output_columns=self._declared_output_columns,
            clean_dropped_columns=self._clean_step_dropped_columns,
        )
        if violations:
            raise PlanValidationError(
                compose_feedback(
                    violations,
                    example_source=self._example_source(raw_columns_by_alias),
                )
            )

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
    def _example_source(raw_columns_by_alias: dict) -> str:
        """A real ``Alias.column`` pair to use in the retry message's example.

        The feedback shows one inline ``produces`` snippet; building it from
        THIS plan's own tables makes it copy-pasteable instead of something
        the model has to translate to its own aliases first.
        """
        for alias in sorted(raw_columns_by_alias):
            columns = sorted(raw_columns_by_alias[alias] or [])
            if columns:
                return f"{alias}.{columns[0]}"
        return "Table_0.column"

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
        """Append validation feedback so the re-request can self-correct.

        The feedback goes at the TAIL, after the unchanged prompt, so the
        static planning prefix stays cacheable across the retry.

        No generic closing instruction is added: ``error`` is already a
        self-contained message that states the rule it broke and the fix for
        each problem (see
        :func:`orqa.agent.utility.column_provenance.compose_feedback`). The
        boilerplate that used to live here ("return a corrected plan...")
        both repeated that preamble and worked against its closing line by
        implying the whole plan should be rewritten, which tends to lose
        question quality the earlier checks already cleared.
        """
        return f"{prompt}\n\n### VALIDATION FEEDBACK\n{error}"

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
        dfs: Optional[Sequence[Any]] = None,
        table_metadata: Optional[dict] = None,
        *,
        fixed_questions: Sequence[dict],
    ) -> str:
        n_fixed = len(fixed_questions)
        if n_fixed == 1:
            task_statement = (
                "You are an expert data engineer. Produce an ordered, "
                "step-by-step query plan over the provided tables that "
                "answers the FIXED QUESTION below. Return only valid JSON "
                "matching the required schema.\n\n"
            )
        else:
            task_statement = (
                "You are an expert data engineer. Produce "
                f"{n_fixed} step-by-step query plans over the provided "
                "tables, one for EACH of the FIXED QUESTIONS below, in "
                "the same order. Return only valid JSON matching the "
                f"required schema, as a `plans` list of exactly {n_fixed} "
                "plan objects.\n\n"
            )
        batch_note = (
            "FIXED-QUESTION REQUIREMENTS:\n"
            "- The questions below are already approved. Plan i answers "
            "question i: copy its text EXACTLY into that plan's "
            "`question` — never reword, shorten or extend it — and copy "
            "its listed terms into `question_keywords`.\n"
            "- Design the steps that answer each question as naturally as "
            "the tables allow. Every provided table must still be "
            "genuinely needed by the steps.\n"
            "- Copy each question's `difficulty` into that plan's "
            "`difficulty`: it is fixed. Never shape, pad or trim the steps "
            "to reach a tier.\n"
            "- If a step would need a hardcoded literal for the current "
            "date, avoid it and use a column value instead: the question "
            "cannot be changed to state a reference point.\n\n"
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
            fixed_questions=self._render_fixed_questions(fixed_questions),
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
            table_metadata=self._render_table_metadata(table_metadata, aliases),
        )

    def question_context(
        self,
        analyses: Sequence[dict],
        aliases: dict,
        match: Any,
        involved_cols: Optional[dict],
        stats: Sequence[TableStats],
        languages: Optional[Sequence[str]] = None,
        dfs: Optional[Sequence[Any]] = None,
        table_metadata: Optional[dict] = None,
    ) -> dict:
        """Prompt blocks for the question generator (see
        ``orqa.agent.agents.QuestionGenerator``), rendered with this planner's
        own renderers so the question agent sees the tables exactly as the
        planner would. Column statistics are left out on purpose: a question
        needs what the tables are about, which columns exist and real values —
        not the cleaning signals the plan's steps will act on. The table FACTS
        are in: computed over every row, they say which columns a question can
        group or compare by, which is what a question's difficulty depends on.
        """
        links = self._build_constraint_links(match, involved_cols, aliases)
        columns = {
            table.alias: [f"{c.column} ({c.dtype})" for c in table.columns]
            for table in stats
        }
        facts_block = "\n\n".join(
            f"{alias}\n{render_table_facts(df)}"
            for alias, df in zip(aliases, dfs or [])
        ) or "(not available)"
        tables_block = "\n".join([
            "### TABLE ALIASES",
            json.dumps(list(aliases.keys()), indent=2, ensure_ascii=False),
            "",
            "### TABLE-LEVEL ANALYSIS",
            json.dumps({"tables": list(analyses)}, indent=2, ensure_ascii=False, default=str),
            "",
            "### TABLE METADATA (from the open-data portal)",
            self._render_table_metadata(table_metadata, aliases),
            "",
            "### TABLE FACTS (computed over every row — a question can only group or compare by a column listed under Breakdowns)",
            facts_block,
            "",
            "### COLUMNS (name and type — for your understanding only: never write a column name in the question)",
            json.dumps(columns, indent=2, ensure_ascii=False),
            "",
            "### TABLE SAMPLE (real rows, up to 10 per table)",
            self._render_table_sample(dfs, aliases),
        ])
        return {
            "languages": ", ".join(languages or []) or "English",
            "time_context": self._render_time_context(),
            "links_block": self._render_links(links),
            "tables_block": tables_block,
        }

    @staticmethod
    def _render_fixed_questions(fixed_questions: Optional[Sequence[dict]]) -> str:
        """The "### FIXED QUESTIONS" section — empty (omitted) unless the
        questions were approved upstream. Ends with a blank line so it
        slots in front of the next section header."""
        if not fixed_questions:
            return ""
        lines = ["### FIXED QUESTIONS"]
        for i, item in enumerate(fixed_questions, start=1):
            lines.append(f"{i}. {item['question']}")
            keywords = item.get("question_keywords") or []
            if keywords:
                lines.append(f"   question_keywords: {', '.join(keywords)}")
            if item.get("difficulty"):
                lines.append(f"   difficulty: {item['difficulty']}")
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _render_table_metadata(table_metadata: Optional[dict], aliases: dict) -> str:
        """The "### TABLE METADATA" section body: each alias's portal metadata."""
        if not table_metadata or not any(table_metadata.values()):
            return "(no portal metadata available)"
        return json.dumps(
            {alias: table_metadata.get(alias) or {} for alias in aliases},
            indent=2, ensure_ascii=False, default=str,
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
