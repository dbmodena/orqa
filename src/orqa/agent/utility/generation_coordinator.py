"""Skill-aware code generator and the opaque ``client_id`` contract.

:class:`CodeGenerator` owns the *initial generation* phase of the pipeline. Per
the design (see the component table and the ``client_id`` contract in the
requirements, Requirement 16), it:

* Builds the final generation prompt (skill-injected, plan-grounded — task 8.2)
  and
* Enforces the **opaque ``client_id`` echo contract**: every generated query is
  assigned a unique, opaque short identifier, and the model is instructed to echo
  that identifier back unchanged on each query it later returns (correction and
  judging).

This module implements **tasks 8.1 and 8.2**: the ``client_id`` assignment and
echo instruction (8.1) plus the skill-injected, plan-grounded prompt body — the
ordered plan steps, the column-statistics block, and the selected skill card
section (8.2), plus the ``client_id``-keyed corrected-code merge (task 8.3):
mapping corrected code back onto source queries by ``client_id`` with the
documented unknown-id / missing-id fallbacks (Requirements 16.3-16.5).

Why assign ids programmatically rather than trust the model to mint them: the
``client_id`` is the key the validator and judge use to map results back to the
source query without trusting positional order. Minting the ids here guarantees
they are unique and non-empty regardless of model behaviour (Requirement 16.1);
the prompt then documents the echo contract (Requirement 16.2) so downstream
correction/judging returns preserve them (Requirements 16.3-16.6, tasks 8.3/8.4).
"""

import json
import logging
from typing import Any, Callable, List, Optional, Sequence, Union
from uuid import uuid4

from ..prompting.models import StructuredQueryPlan, TableStats
from ..prompting.skill_registry import SkillCard, SkillSelection

logger = logging.getLogger(__name__)

# Length of an assigned client_id token. Eight hex characters of a uuid4 give an
# opaque, collision-resistant, human-copyable short id.
_CLIENT_ID_LENGTH = 8

# Instruction rendered into the generation prompt documenting the echo contract
# (Requirement 16.2). It is also suitable for reuse by the correction and judge
# prompts (task 8.4). The phrasing is deliberately explicit that the id is opaque
# and must be preserved verbatim.
CLIENT_ID_ECHO_INSTRUCTION = (
    "### CLIENT ID CONTRACT\n"
    "Every query carries an opaque `client_id` — a short, meaningless token used "
    "only to track the query across later steps. You MUST echo each query's "
    "`client_id` back UNCHANGED on every query you return, now and in any later "
    "correction or judging step. Do not invent new ids, do not reorder them, do "
    "not modify, translate, or drop them. Treat the `client_id` as an opaque "
    "string and copy it verbatim."
)

# Instruction rendered into the judge prompts (Requirements 20.5, 16.6). The
# judge sees each query's opaque tracking token in the `id` field. That id must
# be echoed verbatim so judgments map back to the source query by id rather than
# by position (Requirement 15.2); it is an opaque token, never a sequence number
# to renumber or reorder. Phrasing mirrors CLIENT_ID_ECHO_INSTRUCTION for
# consistency across generation, correction, and judging prompts.
JUDGE_ID_ECHO_INSTRUCTION = (
    "### ID CONTRACT\n"
    "Each query carries an opaque `id` — a short, meaningless tracking token, "
    "NOT a sequence number. Copy each query's `id` into your judgment for that "
    "query UNCHANGED and verbatim. Treat it as an opaque string: do not renumber, "
    "reorder, invent, modify, translate, or drop ids. Judgments are matched back "
    "to queries by this `id`, not by their position in the list."
)

# Instruction appended to the injected skill section (Requirement 20.3). It tells
# the model to use the skill only when the plan actually calls for an ML-style
# operation, and otherwise to fall back to plain pandas (design §6, skill section).
_SKILL_USAGE_INSTRUCTION = (
    "If the plan contains an ML/predict/impute/timeseries/causal step, use this "
    "skill's documented pattern; otherwise use plain pandas."
)


def _render_plan_steps(plan: StructuredQueryPlan) -> str:
    """Render the ordered plan steps as a numbered, human-readable block.

    Renders each :class:`~orqa.agent.prompting.models.PlanStep` in execution
    order (Requirement 20.1) so the model generates code that follows the
    validated decomposition step-by-step. Only fields that carry information are
    emitted per step to keep the prompt compact.
    """
    lines: List[str] = []
    if plan.question:
        lines.append(f"Question: {plan.question}")
    if plan.task_types:
        lines.append(f"Task types: {', '.join(plan.task_types)}")
    lines.append("Steps:")
    for step in plan.steps:
        lines.append(f"{step.order}. [{step.op}] {step.description}")
        if step.tables:
            lines.append(f"   - tables: {', '.join(step.tables)}")
        if step.columns:
            lines.append(f"   - columns: {', '.join(step.columns)}")
        if step.columns_role:
            lines.append(
                f"   - columns_role: {json.dumps(step.columns_role, ensure_ascii=False)}"
            )
        if step.params:
            lines.append(
                f"   - params: {json.dumps(step.params, ensure_ascii=False)}"
            )
    return "\n".join(lines)


def _render_column_statistics(stats: Sequence[TableStats]) -> str:
    """Render the per-table column-statistics block (Requirement 20.2).

    Mirrors the rendering style of
    :meth:`orqa.agent.utility.query_planner.QueryPlanner._render_statistics` so the plan
    and generation prompts present statistics consistently, without coupling the
    two components (QueryPlanner keeps its own copy untouched).
    """
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


def _select_skill_card(
    skill_selection: Union[SkillSelection, SkillCard, None],
) -> Optional[SkillCard]:
    """Resolve the single skill card to inject, or ``None`` when none is selected.

    Accepts either a :class:`SkillSelection` (using its first card, if any) or a
    bare :class:`SkillCard`. Returns ``None`` when nothing is selected so the
    caller omits the skill section entirely (Requirements 20.4, 9.6).
    """
    if skill_selection is None:
        return None
    if isinstance(skill_selection, SkillCard):
        return skill_selection
    if isinstance(skill_selection, SkillSelection):
        return skill_selection.cards[0] if skill_selection.cards else None
    logger.warning("Unrecognized skill selection type: %r", type(skill_selection))
    return None


class CodeGenerator:
    """Builds the generation prompt and enforces the ``client_id`` contract.

    Args:
        id_factory: Optional zero-argument callable returning a fresh id string.
            Injected for deterministic testing; defaults to an 8-char uuid4 hex
            token. The generator guarantees uniqueness across a single
            assignment regardless of the factory (regenerating on collision).
    """

    def __init__(self, id_factory: Optional[Callable[[], str]] = None):
        self._id_factory = id_factory or self._default_id_factory

    # ------------------------------------------------------------------
    # client_id assignment (Requirement 16.1)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_id_factory() -> str:
        """Return a fresh opaque short id (8 hex chars of a uuid4)."""
        return uuid4().hex[:_CLIENT_ID_LENGTH]

    def _new_client_id(self, taken: set) -> str:
        """Return a fresh id not already present in ``taken``.

        Regenerates on the (astronomically unlikely) event of a collision so the
        assigned ids are always unique within a single query set. Falls back to a
        full uuid4 hex if the factory keeps colliding (e.g. a constant test stub).
        """
        for _ in range(8):
            candidate = self._id_factory()
            if candidate and candidate not in taken:
                return candidate
        # Factory is degenerate (constant/empty) — guarantee uniqueness ourselves.
        return uuid4().hex

    def assign_client_ids(self, queries: List[dict]) -> List[dict]:
        """Assign a unique, non-empty ``client_id`` to each query.

        Each query dict is given a fresh opaque ``client_id`` (Requirement 16.1).
        Any ``client_id`` already present on an incoming query is overwritten so
        the generator remains the single source of truth for the identifier — the
        model is never trusted to mint ids, only to echo them.

        Args:
            queries: The generated query dicts (each mutated in place).

        Returns:
            The same list, with every query carrying a unique ``client_id``.
        """
        assigned: set = set()
        for query in queries:
            if not isinstance(query, dict):
                logger.warning("Skipping client_id assignment for non-dict query: %r", query)
                continue
            client_id = self._new_client_id(assigned)
            query["client_id"] = client_id
            assigned.add(client_id)
        return queries

    # ------------------------------------------------------------------
    # Prompt construction (Requirement 16.2)
    # ------------------------------------------------------------------

    def build_generation_prompt(
        self,
        base_prompt: str,
        plan: Optional[StructuredQueryPlan] = None,
        stats: Optional[Sequence[TableStats]] = None,
        skill_selection: Union[SkillSelection, SkillCard, None] = None,
    ) -> str:
        """Build the skill-injected, plan-grounded generation prompt.

        Assembles the generation prompt from the caller-provided ``base_prompt``
        plus, in order, the optional plan-grounding sections and the mandatory
        ``client_id`` echo contract:

        * ``### Structured Plan`` — the ordered plan steps of ``plan`` (Requirement
          20.1), rendered only when a plan is supplied.
        * ``### Column Statistics`` — the per-table column-statistics block from
          ``stats`` (Requirement 20.2), rendered only when statistics are supplied.
        * ``### Available Skill: {name}`` — the selected skill card's ``body``
          together with usage guidance (Requirement 20.3). This section is
          injected only WHERE a skill is selected; WHERE no skill is selected it
          is omitted entirely so plain code is generated (Requirements 20.4, 9.6).
        * The ``client_id`` echo instruction (Requirement 16.2) is always appended
          last so the contract from task 8.1 stays intact regardless of which
          plan/skill sections are present.

        All plan/skill/statistics parameters are optional and default to ``None``;
        when absent the corresponding section renders nothing, preserving backward
        compatibility with task 8.1 callers that pass only ``base_prompt``.

        Args:
            base_prompt: The base generation prompt supplied by the caller.
            plan: The validated structured plan whose ordered steps ground the
                generation. Omitted section when ``None``.
            stats: Per-table column statistics rendered into the prompt. Omitted
                section when ``None`` or empty.
            skill_selection: The selected skill (a :class:`SkillSelection` or a
                bare :class:`SkillCard`). When it resolves to no card the skill
                section is omitted entirely.

        Returns:
            The fully-assembled generation prompt string.
        """
        sections: List[str] = [base_prompt]

        if plan is not None:
            sections.append(f"### Structured Plan\n{_render_plan_steps(plan)}")

        if stats:
            sections.append(
                f"### Column Statistics\n{_render_column_statistics(stats)}"
            )

        skill_card = _select_skill_card(skill_selection)
        if skill_card is not None:
            sections.append(
                f"### Available Skill: {skill_card.name}\n"
                f"{skill_card.body}\n\n"
                f"{_SKILL_USAGE_INSTRUCTION}"
            )

        # The client_id echo contract (Requirement 16.2) is always appended last.
        sections.append(CLIENT_ID_ECHO_INSTRUCTION)

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Generation orchestration
    # ------------------------------------------------------------------

    def generate(
        self,
        base_prompt: str,
        generate_fn: Callable[[str], Any],
        plan: Optional[StructuredQueryPlan] = None,
        stats: Optional[Sequence[TableStats]] = None,
        skill_selection: Union[SkillSelection, SkillCard, None] = None,
    ) -> tuple[dict, dict, list, str]:
        """Generate queries and assign each a unique ``client_id``.

        Args:
            base_prompt: The base generation prompt.
            generate_fn: A callable that takes the fully-built prompt and returns
                the model's query set. It may return either a bare
                ``{"queries": [...]}`` dict, or the richer
                ``(queryset_dict, usage, errors, model)`` tuple produced by
                :class:`LLMClientStatementGenerator.complete`. Injected so the LLM
                client can be mocked in tests.
            plan: Optional structured plan whose ordered steps ground generation
                (Requirement 20.1).
            stats: Optional per-table column statistics rendered into the prompt
                (Requirement 20.2).
            skill_selection: Optional selected skill injected into the prompt
                (Requirement 20.3); omitted when it resolves to no card
                (Requirements 20.4, 9.6).

        Returns:
            ``(query_set, usage, errors, model)`` where every query in
            ``query_set["queries"]`` carries a unique, non-empty ``client_id``.
        """
        prompt = self.build_generation_prompt(base_prompt, plan, stats, skill_selection)
        raw = generate_fn(prompt)
        query_set, usage, errors, model = self._normalize_result(raw)

        queries = query_set.get("queries", [])
        if not isinstance(queries, list):
            logger.warning("Generation returned a non-list 'queries'; coercing to empty list.")
            queries = []
        query_set["queries"] = self.assign_client_ids(queries)
        return query_set, usage, errors, model

    @staticmethod
    def _normalize_result(raw: Any) -> tuple[dict, dict, list, str]:
        """Normalise a generate_fn return into ``(query_set, usage, errors, model)``.

        Accepts either the 4-tuple returned by
        :meth:`LLMClientStatementGenerator.complete` or a bare query-set dict.
        """
        usage: dict = {}
        errors: list = []
        model: str = ""

        if isinstance(raw, tuple):
            query_set = raw[0] if len(raw) > 0 else {}
            usage = raw[1] if len(raw) > 1 and isinstance(raw[1], dict) else {}
            errors = raw[2] if len(raw) > 2 and isinstance(raw[2], list) else []
            model = raw[3] if len(raw) > 3 and isinstance(raw[3], str) else ""
        else:
            query_set = raw

        if not isinstance(query_set, dict):
            logger.warning("Generation returned a non-dict query set; coercing to empty.")
            query_set = {"queries": []}
        return query_set, usage, errors, model

    # ------------------------------------------------------------------
    # client_id-keyed corrected-code merge (task 8.3, Requirements 16.3-16.5)
    # ------------------------------------------------------------------

    def merge_corrected_code(
        self,
        source_queries: List[dict],
        corrected: List[dict],
    ) -> List[dict]:
        """Merge corrected code back onto the source queries by ``client_id``.

        The correction step returns, for each query it re-emitted, an item that
        echoes the query's opaque ``client_id`` (per the contract from task 8.1)
        and carries the corrected ``code``. This method maps each corrected item
        back to its originating source query by that ``client_id`` and updates the
        source query's ``code`` (Requirement 16.3). The ``client_id`` itself is
        always preserved unchanged (Requirement 16.6).

        Fallback behaviour when the model does not honour the contract:

        * **Unknown id** — a corrected item whose ``client_id`` does not match any
          source query is treated as a stray correction and is discarded; the
          affected source query keeps its original code (Requirement 16.4).
        * **Missing id (single item)** — WHERE the model returns a single
          correction with no/empty ``client_id`` and there is exactly one source
          query to merge, the correction is applied positionally and a warning is
          logged (Requirement 16.5).
        * **Missing id (multiple items)** — WHERE ids are missing but there is
          more than one item to merge, positional guessing is unsafe (the order is
          not trustworthy), so every affected source query retains its original
          code and a warning is logged. This deliberately does not guess.

        Source query dicts are not mutated; a shallow copy of each is returned so
        callers keep their originals intact.

        Args:
            source_queries: The originating query dicts, each carrying a
                ``client_id`` and the current ``code``.
            corrected: The correction responses, each ideally echoing a
                ``client_id`` and carrying corrected ``code``.

        Returns:
            A new list of source-query dicts (shallow copies) in the original
            order, with ``code`` updated wherever a correction was matched and
            ``client_id`` preserved throughout.
        """
        source_queries = source_queries or []
        corrected = corrected or []

        # Work on shallow copies so caller-provided dicts are never mutated.
        merged: List[dict] = [
            dict(q) if isinstance(q, dict) else q for q in source_queries
        ]

        # Partition the corrections into keyed (echoed a non-empty client_id) and
        # unkeyed (the model dropped the id).
        keyed: dict = {}
        unkeyed: List[dict] = []
        for item in corrected:
            if not isinstance(item, dict):
                logger.warning("Ignoring non-dict correction item: %r", item)
                continue
            client_id = item.get("client_id")
            if client_id:
                # Last write wins on the (unexpected) event of duplicate ids.
                keyed[client_id] = item
            else:
                unkeyed.append(item)

        # ----- id-keyed merge (Requirements 16.3, 16.4) -----
        matched_ids: set = set()
        for query in merged:
            if not isinstance(query, dict):
                continue
            client_id = query.get("client_id")
            if client_id and client_id in keyed:
                self._apply_corrected_code(query, keyed[client_id])
                matched_ids.add(client_id)

        # Corrected items whose client_id matched no source query are stray
        # (unknown id) — the original query is retained for that item (16.4).
        for client_id in keyed:
            if client_id not in matched_ids:
                logger.warning(
                    "Discarding corrected code for unknown client_id %r; "
                    "retaining original query.",
                    client_id,
                )

        # ----- missing-id fallback (Requirement 16.5) -----
        if unkeyed:
            if len(unkeyed) == 1 and len(merged) == 1 and isinstance(merged[0], dict):
                # Exactly one item on each side: safe to merge positionally.
                logger.warning(
                    "Corrected code is missing a client_id; falling back to "
                    "positional merge for the single item."
                )
                self._apply_corrected_code(merged[0], unkeyed[0])
            else:
                # Ambiguous: multiple items with missing ids. Do not guess order.
                logger.warning(
                    "%d corrected item(s) are missing a client_id and cannot be "
                    "matched unambiguously; retaining original queries (no "
                    "positional guess for multiple items).",
                    len(unkeyed),
                )

        return merged

    @staticmethod
    def _apply_corrected_code(query: dict, correction: dict) -> None:
        """Copy the corrected ``code`` onto ``query`` in place, preserving its id.

        Only the ``code`` field is transferred; the source query's ``client_id``
        (and every other field) is left untouched (Requirement 16.6). When the
        correction carries no ``code`` the source query is left unchanged.
        """
        if "code" in correction:
            query["code"] = correction["code"]
        else:
            logger.warning(
                "Correction for client_id %r carries no 'code'; retaining "
                "original code.",
                query.get("client_id"),
            )
