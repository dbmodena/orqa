"""Structured query planner (task 5.3).

:class:`QueryPlanner` turns per-table analyses plus the upstream ``match`` /
``involved_cols`` relationship constraints into a :class:`StructuredQueryPlan`
(see :mod:`orqa.agent.prompting.models`): an ordered list of :class:`PlanStep`
decomposition steps, the distinct skill-relevant ``task_types`` derived from
those steps, and the ``table_links`` carried over from the mandatory upstream
constraints.

Design constraints implemented here (Requirements 5.1, 5.2, 6.1, 6.2):

* The provided ``match`` / ``involved_cols`` links are treated as **mandatory**
  constraints. The planner prompt states they must not be altered, and the
  produced plan preserves them **unchanged** in ``table_links`` regardless of
  what the language model returns for that field.
* ``task_types`` is the distinct set of skill-relevant operations found in the
  plan steps (classification, regression, timeseries, imputation, causal).
* Column statistics (``TableStats``) are injected into the planner prompt.

Plan *validation* and the re-request / free-text fallback (task 5.4) are
implemented here too: :meth:`QueryPlanner.validate_plan` assigns contiguous
``1..N`` step orders and checks table/column references, and :meth:`plan`
re-requests the plan once on failure before falling back to a schema-valid
free-text plan (Requirements 5.3, 5.4, 5.5, 5.6).
"""

import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence

from pydantic import BaseModel, ValidationError

from ..LLMClientStructured import LLMClientStructured
from ..prompting.models import PlanStep, StructuredQueryPlan, TableStats
from ..structured_outputs import QueryLink

logger = logging.getLogger(__name__)

# Operations that require a skill (the ML task types). ``task_types`` on a
# produced plan is the distinct set of these ops appearing in the plan steps.
SKILL_TASK_TYPES: tuple[str, ...] = (
    "classification",
    "regression",
    "timeseries",
    "imputation",
    "causal",
)

# Statement rendered into the planner prompt so the model never re-derives or
# alters the mandatory upstream relationships.
MANDATORY_LINKS_STATEMENT = (
    "The following table links are mandatory and must not be altered."
)


class PlanValidationError(ValueError):
    """Raised when a :class:`StructuredQueryPlan` fails structural validation.

    Signals that a plan references an unknown table alias, references a column
    that does not exist in any referenced table, has no steps, or has a
    ``task_types`` set inconsistent with its steps (Requirements 5.3-5.5).
    """


class QueryPlannerClient(LLMClientStructured):
    """Structured LLM client that returns a :class:`StructuredQueryPlan`.

    Reuses the shared JSON-repair / retry pipeline from
    :class:`LLMClientStructured` but pins the response model to
    :class:`StructuredQueryPlan` rather than the legacy flat ``QueryPlan``.
    """

    def __init__(self, config_path: Path):
        # ``query_planner`` is a valid config key (legacy ``QueryPlan``); load it
        # to satisfy the base constructor, then override with the structured model.
        super().__init__(config_path, response_model="query_planner")
        self.response_model = StructuredQueryPlan

    def request_plan(self, prompt: str, **kwargs) -> tuple[dict, dict]:
        """Return ``(plan_dict, usage)``; ``plan_dict`` is ``{}`` on failure."""
        return self.complete(prompt, root_key=None, **kwargs)


class QueryPlanner:
    """Produces a :class:`StructuredQueryPlan` from analyses and constraints."""

    def __init__(self, config_path: Path, client: Optional[Any] = None):
        """Create a planner.

        Args:
            config_path: Path to the LLM YAML configuration.
            client: An object exposing ``request_plan(prompt) -> (dict, usage)``.
                Injected for testing; when omitted a :class:`QueryPlannerClient`
                is constructed lazily on first use.
        """
        self.config_path = config_path
        self._client = client

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
    ) -> StructuredQueryPlan:
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

        Returns:
            A :class:`StructuredQueryPlan` whose ``table_links`` preserve the
            provided constraints unchanged and whose ``task_types`` are derived
            from the ordered plan steps.
        """
        languages = list(languages or [])

        # Build the mandatory relationship constraints ONCE. These are preserved
        # verbatim in the produced plan (Requirements 6.1, 6.2).
        constraint_links = self._build_constraint_links(match, involved_cols, aliases)

        prompt = self._build_prompt(
            analyses, aliases, constraint_links, stats, languages
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
                "Structured plan validation failed (%s); re-requesting once.",
                first_error,
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
                "falling back to a free-text plan.",
                exc_retry,
            )
            return self._free_text_fallback(retry_candidate, aliases, constraint_links)

    # ------------------------------------------------------------------
    # Plan validation (Requirements 5.3, 5.4, 5.5)
    # ------------------------------------------------------------------

    def validate_plan(
        self,
        plan: StructuredQueryPlan,
        aliases: dict,
        known_columns: dict,
    ) -> StructuredQueryPlan:
        """Validate and normalise a structured plan.

        Assigns contiguous ``1..N`` ``order`` values to the steps (Requirement
        5.3), then verifies that every table referenced by a step is a known
        alias (Requirement 5.4) and that every column referenced by a step
        exists in one of the tables that step references (Requirement 5.5).
        Also asserts ``task_types`` is exactly the distinct set of skill-relevant
        ops in the steps (design pseudocode).

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

        # Requirement 5.3: assign a contiguous 1..N order sequence.
        for idx, step in enumerate(plan.steps, start=1):
            step.order = idx

        known_aliases = set(aliases.keys()) if aliases else set()
        # Normalise known_columns values into sets for membership checks.
        columns_by_alias = {
            alias: set(cols) for alias, cols in (known_columns or {}).items()
        }

        for step in plan.steps:
            # Requirement 5.4: every referenced table must be a known alias.
            for table in step.tables:
                if table not in known_aliases:
                    raise PlanValidationError(
                        f"step {step.order} ({step.op}) references unknown table "
                        f"alias {table!r}; known aliases: {sorted(known_aliases)}"
                    )

            # Requirement 5.5: every referenced column must exist in at least one
            # of the tables the step references.
            allowed_columns: set = set()
            for table in step.tables:
                allowed_columns |= columns_by_alias.get(table, set())
            for column in step.columns:
                if column not in allowed_columns:
                    raise PlanValidationError(
                        f"step {step.order} ({step.op}) references column "
                        f"{column!r} not present in any referenced table "
                        f"{step.tables}"
                    )

        # Design pseudocode: task_types must equal the distinct skill ops in steps.
        expected_task_types = self._derive_task_types(plan.steps)
        if plan.task_types != expected_task_types:
            raise PlanValidationError(
                f"task_types {plan.task_types} do not match the skill ops derived "
                f"from steps {expected_task_types}"
            )

        return plan

    @staticmethod
    def _known_columns(stats: Sequence[TableStats]) -> dict:
        """Map each alias to the set of its column names from the statistics."""
        return {table.alias: {c.column for c in table.columns} for table in stats}

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

    def _free_text_fallback(
        self,
        plan: StructuredQueryPlan,
        aliases: dict,
        constraint_links: List[QueryLink],
    ) -> StructuredQueryPlan:
        """Build a schema-valid free-text fallback plan (Requirement 5.6).

        When both the initial request and the single re-request fail validation,
        produce a plan that still satisfies the schema and preserves the
        mandatory ``table_links``: a single free-text ``select`` step over the
        available tables with no specific column references (so it can never
        fail column validation), and an empty ``task_types`` set (no skill).
        """
        alias_list = list(aliases.keys()) if aliases else []
        fallback_step = PlanStep(
            order=1,
            op="select",
            description=(
                "Free-text fallback: the structured plan failed validation after "
                "one re-request. Answer the question directly over the available "
                "tables without a validated step decomposition."
            ),
            tables=alias_list,
            columns=[],
        )
        return StructuredQueryPlan(
            question=plan.question,
            question_keywords=plan.question_keywords,
            plan_keywords=plan.plan_keywords,
            steps=[fallback_step],
            task_types=[],
            table_links=constraint_links,
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
            else "Mandatory relationship provided upstream; must not be altered."
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
    ) -> str:
        skill_ops = ", ".join(SKILL_TASK_TYPES)
        return (
            "You are an expert data engineer. Produce an ordered, step-by-step "
            "query plan that decomposes a single business question into concrete "
            "operations over the provided tables. Return only valid JSON matching "
            "the required schema.\n\n"
            "PLAN REQUIREMENTS:\n"
            "- Provide `steps`: an ordered list where each step has an `op`, a "
            "`description`, the `tables` (aliases) it touches, and the concrete "
            "`columns` it reads or writes.\n"
            "- Use only the provided table aliases and columns that exist in those "
            "tables.\n"
            f"- Machine-learning operations are: {skill_ops}. Use them only when the "
            "question genuinely requires prediction/inference; for a "
            "classification step, record the target column under "
            "`columns_role.target`.\n"
            "- Also provide a business `question`, `question_keywords` (max 10), and "
            "`plan_keywords` (max 10).\n\n"
            f"### MANDATORY TABLE LINKS\n{MANDATORY_LINKS_STATEMENT}\n"
            f"{self._render_links(constraint_links)}\n\n"
            f"### TABLE ALIASES\n"
            f"{json.dumps(list(aliases.keys()), indent=2, ensure_ascii=False)}\n\n"
            f"### TABLE-LEVEL ANALYSIS\n"
            f"{json.dumps({'tables': list(analyses)}, indent=2, ensure_ascii=False, default=str)}\n\n"
            f"### COLUMN STATISTICS\n{self._render_statistics(stats)}\n\n"
            f"### DETECTED LANGUAGES\n"
            f"{json.dumps(list(languages), ensure_ascii=False)}\n"
        )

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
            self._client = QueryPlannerClient(self.config_path)
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
    ) -> StructuredQueryPlan:
        """Build a :class:`StructuredQueryPlan` from the raw LLM response.

        The ``table_links`` are always the mandatory constraint links — the
        model's own ``table_links`` output is discarded so the upstream
        relationships are preserved unchanged (Requirement 6.2). ``task_types``
        is derived from the ordered steps (Requirement 5.2).
        """
        raw_plan = raw_plan or {}

        steps = self._coerce_steps(raw_plan.get("steps", []))
        task_types = self._derive_task_types(steps)

        return StructuredQueryPlan(
            question=str(raw_plan.get("question", "")),
            question_keywords=self._limit_keywords(raw_plan.get("question_keywords")),
            plan_keywords=self._limit_keywords(raw_plan.get("plan_keywords")),
            steps=steps,
            task_types=task_types,
            table_links=constraint_links,
        )

    @staticmethod
    def _coerce_steps(raw_steps: Any) -> List[PlanStep]:
        """Coerce raw step dicts into :class:`PlanStep` objects.

        A positional ``order`` (1-based) is filled in when the model omits it so
        the step can be constructed; contiguous-order validation remains a
        separate concern handled by ``validate_plan`` (task 5.4).
        """
        steps: List[PlanStep] = []
        if not isinstance(raw_steps, list):
            return steps
        for idx, raw_step in enumerate(raw_steps, start=1):
            if isinstance(raw_step, PlanStep):
                steps.append(raw_step)
                continue
            if not isinstance(raw_step, dict):
                logger.warning("Skipping non-dict plan step: %r", raw_step)
                continue
            step = dict(raw_step)
            step.setdefault("order", idx)
            try:
                steps.append(PlanStep.model_validate(step))
            except ValidationError as exc:
                logger.warning("Skipping invalid plan step %d: %s", idx, exc)
        return steps

    @staticmethod
    def _derive_task_types(steps: Sequence[PlanStep]) -> List[str]:
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
