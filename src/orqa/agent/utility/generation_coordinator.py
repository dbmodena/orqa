"""Generation orchestration and the opaque ``client_id`` bookkeeping key.

:class:`GenerationCoordinator` owns the *initial generation* phase of the pipeline. Per
the design (see the component table in the requirements, Requirement 16), it:

* Builds the final generation prompt by delegating to
  :func:`orqa.agent.prompting.build_generation_prompt` (plan-grounded —
  task 8.2), calls the injected ``generate_fn`` with it, and
* Assigns every generated query a unique, opaque ``client_id`` used purely as
  an **internal** bookkeeping key (``plan_by_client_id``, judge-result maps,
  ...) — never something the LLM is asked to declare or echo back. Every
  downstream match (correction, judging) is positional, since generation,
  correction, and judging each process queries one at a time or in strict
  input order; nothing ever reads an id back out of a model response.

This module implements **tasks 8.1 and 8.2**: the ``client_id`` assignment
(8.1) plus orchestrating the plan-grounded prompt build and the generation
call itself (8.2). The prompt-assembly logic (rendering plan steps and
column statistics) lives in :mod:`orqa.agent.prompting.prompts` — it is
shared with ``StatementValidator``'s correction prompts, so it belongs with
the rest of the prompting module rather than here.

Why assign ids programmatically rather than trust the model to mint them: the
``client_id`` is the key the validator and judge use to map results back to
the source query without trusting positional order *within this module's own
bookkeeping* — but that mapping is built and read entirely by our own code
(see ``plan_by_client_id`` and ``all_judgments_by_client_id`` in ``agent.py``),
never by asking the model to produce or preserve an id in
its output.
"""

import logging
from typing import Any, Callable, List, Optional, Sequence, Union
from uuid import uuid4

from ..prompting import build_generation_prompt
from ..prompting.models import PandasQueryPlan, SQLQueryPlan, TableStats

logger = logging.getLogger(__name__)

QueryPlan = Union[SQLQueryPlan, PandasQueryPlan]

# Length of an assigned client_id token. Eight hex characters of a uuid4 give an
# opaque, collision-resistant, human-copyable short id.
_CLIENT_ID_LENGTH = 8


class GenerationCoordinator:
    """Orchestrates generation and enforces the ``client_id`` contract.

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
        the generator remains the single source of truth for the identifier —
        the model is never trusted to mint one, and never asked to.

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
    # Generation orchestration
    # ------------------------------------------------------------------

    def generate(
        self,
        base_prompt: str,
        generate_fn: Callable[[str], Any],
        plan: Optional[QueryPlan] = None,
        stats: Optional[Sequence[TableStats]] = None,
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
                (Requirement 20.1). Passed through to
                :func:`orqa.agent.prompting.build_generation_prompt`.
            stats: Optional per-table column statistics rendered into the prompt
                (Requirement 20.2).

        Returns:
            ``(query_set, usage, errors, model)`` where every query in
            ``query_set["queries"]`` carries a unique, non-empty ``client_id``.
        """
        prompt = build_generation_prompt(base_prompt, plan, stats)
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
