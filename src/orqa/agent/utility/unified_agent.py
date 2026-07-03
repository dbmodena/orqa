"""Unified, mode-aware statement-generation agent.

This module hosts the ``UnifiedStatementGenerationAgent`` (added in task 11.3)
and the pure, total ``resolve_mode`` helper (task 11.1).

``resolve_mode`` replaces the fragile ``isinstance(dataset_paths, Path)`` mode
sniffing of the legacy agents with a deterministic decision based on an explicit
hint, the match constraint, and the number of aliases.

The ``UnifiedStatementGenerationAgent._run`` core (task 11.3) implements the
five-phase pipeline once for both modes:

    1. data preparation + cheap column statistics,
    2. batched table analysis (ONE LLM call) + structured planning + skill
       selection + skill-injected code generation, and
    3. the validator-to-judge loop, then
    4. final assembly.

The generation prompt is built exactly ONCE after the per-dataset loop (rather
than accumulated across it via the legacy mutable ``prompt.update()`` pattern),
and the mode divergence points from the design are honoured: single mode sets the
judge ``single_table`` flag and constrains the assembled tables to the single
alias, while ``proposed_columns`` is included only in multi mode.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import yaml

from ... import utils
from ...queries.query_execution import QueryExecutor
from .budget_guard import BudgetGuard
from .code_generator import CodeGenerator
from ..PipelineLogger import PipelineLogger
from ..prompting import (
    ColumnStatistics,
    DatasetDescription,
    LightDatasetDescription,
    PandasStatementGenerationPrompt,
    SingleTablePandasPrompt,
    SingleTableSQLPrompt,
    SkillGateContext,
    SkillRegistry,
    SQLStatementGenerationPrompt,
)
from ..prompting.models import (
    ExecutionTrace,
    JudgeTrace,
    PlanningTrace,
    SkillTrace,
    StructuredQueryPlan,
    TableTrace,
    TraceableQuery,
    TraceableQuerySet,
    UsageTrace,
    ValidationTrace,
)
from .query_planner import QueryPlanner
from ..StatementClient import LLMClientStatementGenerator
from ..StatementValidator import LLMStatementValidator
from .table_analyzer import TableAnalyzer

# NOTE: ``JudgementResponseAgent`` is imported from ``.agent`` at the BOTTOM of
# this module (after all class definitions) rather than here. ``agent.py``
# re-exports the ``StatementGenerationAgent`` /
# ``SingleTableStatementGenerationAgent`` named subclasses defined below, so a
# top-level ``from .agent import ...`` here would create an import cycle that
# fails depending on which module is imported first. Deferring this single
# import to the end of the file breaks the cycle in both directions while
# keeping ``JudgementResponseAgent`` a module-level name (so existing
# ``patch("orqa.agent.unified_agent.JudgementResponseAgent", ...)`` calls in the
# tests continue to resolve).

logger = logging.getLogger(__name__)

# Environment variable holding the hosted TabPFN API key. Presence (not value) is
# all the skill gate needs; the value itself is never read here nor logged.
TABPFN_API_KEY_ENV = "TABPFN_API_KEY"

# The two run modes the pipeline supports.
MULTI = "multi"
SINGLE = "single"


def _is_nonempty_match(match: Any) -> bool:
    """Return True when ``match`` represents a non-empty constraint.

    The ``match`` constraint may arrive in several shapes:

    - ``None``            -> empty
    - ``""``              -> empty (empty string)
    - ``[]`` / ``{}``     -> empty (empty collection)
    - ``"a=b"``           -> non-empty (non-empty string)
    - ``["a", "b"]``      -> non-empty (non-empty collection)

    Emptiness is defined by ordinary truthiness: ``None``, empty strings, and
    empty containers are all falsy in Python, while any populated string or
    container is truthy. This treats "any truthy non-empty match" as non-empty
    exactly as the design specifies.
    """
    return bool(match)


def resolve_mode(
    mode_hint: Optional[str],
    aliases: Any,
    match: Any,
) -> str:
    """Deterministically resolve the run mode.

    Implements the design pseudocode exactly and is a *total* function: it
    returns a defined mode (``"multi"`` or ``"single"``, or the caller-provided
    hint) for every possible combination of inputs.

    Args:
        mode_hint: An explicit mode override. When not ``None`` it is returned
            unchanged (the caller is trusted to pass a valid mode).
        aliases: The alias collection. May be a ``dict`` (alias -> name) or a
            ``list``; only ``len(aliases)`` is consulted, which works for both.
        match: The match/link constraint. May be ``None``, ``""``, a non-empty
            string, or a non-empty list. Any truthy non-empty value counts as a
            present constraint (see :func:`_is_nonempty_match`).

    Returns:
        The resolved mode string.
    """
    # 1. An explicit hint always wins.
    if mode_hint is not None:
        return mode_hint
    # 2. A non-empty match constraint implies a multi-table run.
    if _is_nonempty_match(match):
        return MULTI
    # 3. More than one alias implies a multi-table run.
    if len(aliases) > 1:
        return MULTI
    # 4. Otherwise (empty match + single alias) it is a single-table run.
    return SINGLE


def _read_allow_tabpfn(config_path: Any) -> bool:
    """Defensively read the ``allow_tabpfn`` gate from a workflow config.

    The config field is formalised in the ``StatementGeneration`` dataclass and
    each ``conf/workflow/*.yaml`` (task 14.1). This reader remains as a robust,
    self-contained fallback that parses ``tasks.query_generation.allow_tabpfn``
    (or a top-level ``query_generation`` block) directly from the yaml at
    ``config_path``, defaulting to ``False`` when the key, section, or file is
    absent or unreadable. Never raises.
    """
    try:
        with open(config_path, "r") as file:
            parsed = yaml.safe_load(file) or {}
    except (OSError, yaml.YAMLError, TypeError):
        return False
    if not isinstance(parsed, dict):
        return False
    tasks = parsed.get("tasks")
    section = None
    if isinstance(tasks, dict) and isinstance(tasks.get("query_generation"), dict):
        section = tasks["query_generation"]
    elif isinstance(parsed.get("query_generation"), dict):
        section = parsed["query_generation"]
    if not isinstance(section, dict):
        return False
    return bool(section.get("allow_tabpfn", False))


class UnifiedStatementGenerationAgent:
    """Single, mode-aware statement-generation agent (task 11.3).

    Owns the five-phase pipeline for both ``single`` and ``multi`` modes through
    a single private core, :meth:`_run`. The backward-compatible public adapter
    methods (:meth:`generate_statements` and :meth:`generate_statements_single`,
    task 11.4) preserve the two legacy positional signatures and map them onto
    ``_run`` without swapping the match/metadata arguments. The named subclasses
    ``StatementGenerationAgent`` / ``SingleTableStatementGenerationAgent`` at the
    end of this module preserve the legacy import names so
    ``statement_generation.py`` needs no edits.

    All heavy collaborators are constructed in ``__init__`` but may be injected
    for testing: the :class:`SkillRegistry`, :class:`BudgetGuard`, the skill
    gate context, the :class:`QueryPlanner`, the :class:`TableAnalyzer`, the
    :class:`CodeGenerator`, the LLM generation client, and the
    :class:`LLMStatementValidator`.
    """

    def __init__(
        self,
        config_path: Path,
        kind: str,
        bad_tokens: list,
        max_judge_iterations: int = 3,
        languages: Optional[list] = None,
        gate_ctx: Optional[SkillGateContext] = None,
        budget: Optional[BudgetGuard] = None,
        skill_registry: Optional[SkillRegistry] = None,
        client: Optional[Any] = None,
        analyzer: Optional[TableAnalyzer] = None,
        planner: Optional[QueryPlanner] = None,
        generator: Optional[CodeGenerator] = None,
        validator: Optional[LLMStatementValidator] = None,
    ):
        self.config_path = config_path
        self.kind = kind
        self.bad_tokens = bad_tokens
        self.max_judge_iterations = max_judge_iterations
        self.languages = languages if languages is not None else ["English"]

        # Overall budget ceiling across the (multiplicative) retry loops (Req 17).
        self._budget = budget or BudgetGuard.from_config(config_path)

        # Skill gate context: the yaml ``allow_tabpfn`` gate (formalised in the
        # StatementGeneration config, read here directly from the workflow yaml
        # as a self-contained fallback) and the .env API-key gate.
        # The key value itself is never read here — only its presence.
        self._gate_ctx = gate_ctx or SkillGateContext(
            allow_tabpfn=_read_allow_tabpfn(config_path),
            tabpfn_api_key_present=bool(os.environ.get(TABPFN_API_KEY_ENV)),
        )

        # Skill registry (loads skill cards; may query live TabPFN limits once
        # when a key is present — see SkillRegistry.load).
        self._skill_registry = skill_registry or SkillRegistry.load(
            gate_ctx=self._gate_ctx
        )

        # LLM generation client + the components that reuse it.
        self._client = client or LLMClientStatementGenerator(config_path)
        self._analyzer = analyzer or TableAnalyzer(self._client)
        self._planner = planner or QueryPlanner(config_path)
        self._generator = generator or CodeGenerator()
        self._validator = validator or LLMStatementValidator(config_path, kind)

        self._log = PipelineLogger()

    # ------------------------------------------------------------------
    # Backward-compatible adapter methods (task 11.4)
    # ------------------------------------------------------------------

    def generate_statements(
        self,
        dataset_paths,
        aliases,
        kind,
        match,
        involved_cols,
        metadatas,
        max_cols: int = 10,
        sample_size: int = 5,
    ):
        """Multi-table positional adapter (Requirements 2.1, 2.3).

        Preserves the exact legacy multi-table positional signature used by the
        ``statement_generation.py`` call site::

            generate_statements(dataset_paths, aliases, kind, match,
                                involved_cols, metadatas, max_cols, sample_size)

        The mapping onto :meth:`_run` is explicit and keyword-based so the fourth
        positional argument (``match``) reaches the match constraint and the
        sixth (``metadatas``) reaches ``metadata`` — the two are NOT swapped
        (Requirement 2.3).
        """
        return self._run(
            mode=MULTI,
            dataset_paths=dataset_paths,
            aliases=aliases,
            kind=kind,
            match=match,
            involved_cols=involved_cols,
            metadata=metadatas,
            max_cols=max_cols,
            sample_size=sample_size,
        )

    def generate_statements_single(
        self,
        dataset_path,
        alias,
        kind,
        metadata,
        max_cols: int = 10,
        sample_size: int = 5,
    ):
        """Single-table positional adapter (Requirements 2.2, 2.4).

        Preserves the exact legacy single-table positional signature::

            generate_statements_single(dataset_path, alias, kind, metadata,
                                        max_cols, sample_size)

        The mapping onto :meth:`_run` routes the fourth positional argument
        (``metadata``) to ``metadata`` and treats the match constraint as empty
        (Requirement 2.4). The lone dataset path is wrapped into a single-element
        list and ``involved_cols`` is seeded with the sole alias so the ``_run``
        data-prep loop resolves ``Table_0`` cleanly.
        """
        first_alias = list(alias.keys())[0] if alias else "Table_0"
        return self._run(
            mode=SINGLE,
            dataset_paths=[dataset_path],
            aliases=alias,
            kind=kind,
            match=None,
            involved_cols={first_alias: []},
            metadata=[metadata] if not isinstance(metadata, list) else metadata,
            max_cols=max_cols,
            sample_size=sample_size,
        )

    # ------------------------------------------------------------------
    # Private five-phase core (task 11.3)
    # ------------------------------------------------------------------

    def _run(
        self,
        mode: Optional[str],
        dataset_paths,
        aliases: dict,
        kind: str,
        match,
        involved_cols: Optional[dict],
        metadata,
        max_cols: int = 10,
        sample_size: int = 5,
    ) -> dict | None:
        """Execute the unified five-phase pipeline for one generation run.

        Phases (design §1 ``_run`` pseudocode):

        1. **Data prep + statistics** — prepare each dataset and compute cheap
           per-table :class:`TableStats`.
        2. **Initial generation** — build the generation prompt ONCE after the
           dataset loop, run ONE batched table analysis, produce a structured
           plan, select skills, and generate queries under the ``client_id``
           contract.
        3. **Validator-to-judge loop** — validate/correct and judge queries,
           bounded by the budget guard and ``max_judge_iterations``.
        4. **Assembly** — build the result dict (single mode constrains tables to
           the one alias; multi mode includes ``proposed_columns``).

        Args:
            mode: Explicit mode hint (``"single"``/``"multi"``) or ``None`` to
                resolve deterministically from ``aliases``/``match``.
            dataset_paths: Ordered dataset paths (one per ``Table_i`` alias).
            aliases: Mapping of ``Table_i`` alias -> dataset name.
            kind: Generation kind (``"PANDAS"`` or ``"SQL"``).
            match: The relationship/link constraint (empty in single mode).
            involved_cols: Mapping of alias -> mandatory relationship columns.
            metadata: Per-table metadata (list aligned with aliases, or dict).
            max_cols: Column cap forwarded to ``prepare_dataset``.
            sample_size: Sample-row count forwarded to ``prepare_dataset``.

        Returns:
            The assembled result dict (same shape as the legacy agents; task 13.2
            upgrades this to a full ``TraceableQuerySet``).
        """
        mode = resolve_mode(mode, aliases, match)
        involved_cols = involved_cols or {}

        try:
            # Requirement 17: begin the overall wall-clock budget.
            self._budget.start()

            dataset_names = [Path(p).name for p in dataset_paths]
            self._log.section(
                f"{'Single' if mode == SINGLE else 'Multi'}-table generation — "
                f"{', '.join(dataset_names)}  [{kind}]"
            )

            # ── Phase 1: data preparation + column statistics ──────────────
            dfs: list = []
            infos: list = []
            stats: list = []
            columns_by_table: list = []
            total_columns = 0

            for idx, dataset_path in enumerate(dataset_paths):
                cols = involved_cols.get(f"Table_{idx}", [])
                df, dataset_info = utils.prepare_dataset(
                    dataset_path,
                    cols,
                    max_cols,
                    sample_size,
                    self.bad_tokens,
                )
                dfs.append(df)
                infos.append(dataset_info)
                # Cheap, LLM-free per-table statistics (Requirement 7).
                stats.append(ColumnStatistics.compute(df, alias=f"Table_{idx}"))
                total_columns += len(df.columns)
                columns_by_table.append(
                    {
                        "table_alias": f"Table_{idx}",
                        "columns_provided": df.columns.tolist(),
                    }
                )

            avg_cols = total_columns / len(dataset_paths) if dataset_paths else 0

            # ── Build the generation prompt ONCE after the loop ────────────
            # (Fixes the prompt-statefulness issue: descriptions are gathered
            # into local lists and formatted in a single call rather than
            # accumulated across the loop via mutable prompt state / reset().)
            metadata_list = self._normalize_metadata_list(metadata, len(infos))
            base_prompt, table_schemas = self._build_generation_prompt(
                mode, kind, infos, aliases, match, dfs, metadata_list
            )

            # ── Phase 2a: batched table analysis (ONE LLM call) ────────────
            _t = time.perf_counter()
            analyses = self._analyzer.analyze_batch(
                dfs, aliases, metadata, self.languages
            )
            analysis_ms = (time.perf_counter() - _t) * 1000.0
            self._budget.add_tokens(getattr(self._analyzer, "last_usage", None))

            # ── Phase 2b: structured planning ──────────────────────────────
            _t = time.perf_counter()
            plan = self._planner.plan(
                analyses, aliases, match, involved_cols, stats, self.languages
            )
            planning_ms = (time.perf_counter() - _t) * 1000.0

            # ── Phase 2c: skill selection + gating ─────────────────────────
            skills = self._skill_registry.select(plan, kind, stats, self._gate_ctx)

            # ── Phase 2d: skill-injected code generation (client_id contract)
            # NOTE (seam for the full generator migration): the CodeGenerator
            # augments ``base_prompt`` with the plan/statistics/skill sections and
            # the client_id contract, then delegates the actual completion to the
            # legacy generation client. The client still performs its own internal
            # analysis/planning today; migrating generation fully off that path is
            # tracked with the traceable-assembly work (task 13.2) and is out of
            # scope for 11.3. The unified pipeline itself performs the single
            # batched analysis and structured planning above.
            def _generate_fn(built_prompt: str):
                return self._client.complete(
                    built_prompt,
                    dfs,
                    aliases,
                    typology=kind,
                    involved_cols=involved_cols,
                    matches=match,
                    metadata=metadata,
                    languages=self.languages,
                )

            start = time.perf_counter()
            query_set, tokens, errors, model = self._generator.generate(
                base_prompt,
                _generate_fn,
                plan=plan,
                stats=stats,
                skill_selection=skills,
            )
            total_time = time.perf_counter() - start
            generation_ms = total_time * 1000.0

            # Per-phase timings accumulated across the run (Requirement 21.3).
            timings_ms: dict = {
                "analysis": analysis_ms,
                "planning": planning_ms,
                "generation": generation_ms,
                "validation": 0.0,
                "judging": 0.0,
            }

            all_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            self._accumulate_tokens(all_tokens, tokens)
            self._budget.add_tokens(tokens)

            all_errors: list = list(errors or [])
            last_model = model or ""

            pending_queries: list = query_set.get("queries", []) or []
            self._log.step1_generated(pending_queries)

            if not pending_queries:
                return self._assemble_result(
                    mode=mode,
                    kind=kind,
                    aliases=aliases,
                    all_approved_executed=[],
                    all_approved_query_dicts={},
                    judge=None,
                    original_ids=set(),
                    columns_by_table=columns_by_table,
                    all_tokens=all_tokens,
                    all_errors=all_errors,
                    last_model=last_model,
                    total_time=total_time,
                    avg_cols=avg_cols,
                    plan=plan,
                    skills=skills,
                    timings_ms=timings_ms,
                )

            original_ids = {str(q.get("id")) for q in pending_queries}

            # ── Phase 3: validator-to-judge loop ───────────────────────────
            executor = QueryExecutor(
                datasets_path=Path(dataset_paths[0]).parent,
                bad_tokens=self.bad_tokens,
            )
            entry = {"tables": aliases}
            # Divergence: set the judge single_table flag in single mode (Req 1.5).
            judge = JudgementResponseAgent(
                self.config_path,
                kind,
                executor,
                entry,
                single_table=(mode == SINGLE),
            )

            (
                all_approved_executed,
                all_approved_query_dicts,
                loop_time,
                loop_timings,
            ) = self._validator_judge_loop(
                pending_queries=pending_queries,
                dfs=dfs,
                aliases=aliases,
                table_schemas=table_schemas,
                judge=judge,
                all_tokens=all_tokens,
                all_errors=all_errors,
            )
            total_time += loop_time
            timings_ms["validation"] = loop_timings.get("validation", 0.0)
            timings_ms["judging"] = loop_timings.get("judging", 0.0)

            # ── Phase 4: final assembly ────────────────────────────────────
            return self._assemble_result(
                mode=mode,
                kind=kind,
                aliases=aliases,
                all_approved_executed=all_approved_executed,
                all_approved_query_dicts=all_approved_query_dicts,
                judge=judge,
                original_ids=original_ids,
                columns_by_table=columns_by_table,
                all_tokens=all_tokens,
                all_errors=all_errors,
                last_model=last_model,
                total_time=total_time,
                avg_cols=avg_cols,
                plan=plan,
                skills=skills,
                timings_ms=timings_ms,
            )

        except FileNotFoundError as exc:
            logger.error("Dataset not found: %s", exc)
            raise
        except Exception:
            logger.exception("Unified statement generation failed")
            raise

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_metadata_list(metadata: Any, n: int) -> list:
        """Coerce ``metadata`` into a per-table list of length >= ``n``.

        Accepts a list (already per-table), ``None``, or a single mapping to be
        broadcast. Missing entries default to an empty dict so per-index access
        stays safe when building the prompt.
        """
        if metadata is None:
            return [{} for _ in range(n)]
        if isinstance(metadata, list):
            out = list(metadata)
        else:
            # A single metadata mapping (single-table run) — broadcast it.
            out = [metadata for _ in range(n)]
        if len(out) < n:
            out = out + [{} for _ in range(n - len(out))]
        return out

    def _build_generation_prompt(
        self,
        mode: str,
        kind: str,
        infos: list,
        aliases: dict,
        match,
        dfs: list,
        metadata_list: list,
    ) -> tuple[str, str]:
        """Build the base generation prompt exactly once, after the dataset loop.

        Rather than calling the legacy ``prompt.update()`` inside the per-dataset
        loop (which mutates and accumulates prompt state across iterations), this
        gathers each table's description blocks into local lists and formats the
        prompt in a single ``_update`` call. This removes the reliance on
        ``prompt.reset()`` and mutable accumulation flagged in the design.

        Returns:
            ``(base_prompt, table_schemas)`` where ``table_schemas`` is the light
            per-table schema block the validator consumes (matching the legacy
            ``_light_datasets_descriptions`` it was passed).
        """
        is_single = mode == SINGLE
        if kind == "PANDAS":
            prompt = SingleTablePandasPrompt() if is_single else PandasStatementGenerationPrompt()
        else:
            prompt = SingleTableSQLPrompt() if is_single else SQLStatementGenerationPrompt()

        full_blocks: list = []
        light_blocks: list = []
        secondary_blocks: list = []

        for idx, info in enumerate(infos):
            df = dfs[idx]
            alias_key = f"Table_{idx}"
            columns_for_prompt = "".join(
                f"\n- {col} ({df[col].dtype})" for col in df.columns
            )
            metadata_i = metadata_list[idx] if idx < len(metadata_list) else {}

            full_blocks.append(
                DatasetDescription().update(
                    info["dataset_name"],
                    info["num_rows"],
                    info["num_columns"],
                    metadata_i,
                    info["columns_details"],
                    info["sample_data"],
                )
            )
            # Single mode carries no match constraint in the light description.
            light_blocks.append(
                LightDatasetDescription().update(
                    alias_key,
                    columns_for_prompt,
                    info["sample_data"],
                    "" if is_single else match,
                )
            )
            secondary_blocks.append(
                DatasetDescription().update(
                    alias_key,
                    info["num_rows"],
                    info["num_columns"],
                    metadata_i,
                    info["columns_details"],
                    info["sample_data"],
                )
            )

        # Assign the accumulated descriptions ONCE (no per-iteration mutation).
        prompt._datasets_descriptions = "".join(f"\n{b}" for b in full_blocks)
        prompt._light_datasets_descriptions = "".join(f"\n{b}" for b in light_blocks)
        prompt._secondary_datasets_descriptions = "".join(
            f"\n{b}" for b in secondary_blocks
        )

        if is_single:
            base_prompt = prompt._update(
                table=prompt._datasets_descriptions,
                languages=self.languages,
                alias=json.dumps(aliases, indent=2),
            )
        else:
            base_prompt = prompt._update(
                table=prompt._datasets_descriptions,
                matches=match,
                languages=self.languages,
                aliases=json.dumps(aliases, indent=2),
            )

        return base_prompt, prompt._light_datasets_descriptions

    def _validator_judge_loop(
        self,
        pending_queries: list,
        dfs: list,
        aliases: dict,
        table_schemas: str,
        judge: JudgementResponseAgent,
        all_tokens: dict,
        all_errors: list,
    ) -> tuple[list, dict, float, dict]:
        """Run the budget-bounded validator-to-judge loop.

        Mirrors the legacy agents' loop: at each iteration boundary the budget
        guard is consulted (Requirement 17.2/17.3/17.5) and the loop runs at most
        ``max_judge_iterations`` times (17.4). Approved queries are accumulated
        across iterations and never discarded when the budget stops the loop.

        Returns:
            ``(all_approved_executed, all_approved_query_dicts, elapsed_seconds,
            phase_times_ms)`` where ``phase_times_ms`` splits the wall-clock time
            into ``{"validation": ms, "judging": ms}`` for the per-phase timings
            recorded on each assembled query (Requirement 21.3).
        """
        all_approved_executed: list = []
        all_approved_query_dicts: dict = {}
        elapsed = 0.0
        validation_ms = 0.0
        judging_ms = 0.0
        structured_judge_feedback: list | None = None

        for iteration in range(self.max_judge_iterations):
            if self._budget.exceeded():
                self._log.warning(
                    f"Budget exceeded before iteration {iteration + 1} — "
                    "stopping loop and returning approved queries so far."
                )
                break

            self._log.step2_start(iteration + 1)

            # --- 2a. Validate + correct ---
            queries_before_validation = list(pending_queries)
            start = time.perf_counter()
            pending_queries, val_tokens, val_errors = self._validator.validate_and_correct(
                pending_queries,
                dfs,
                aliases,
                table_schemas=table_schemas,
                judge_feedback=structured_judge_feedback,
            )
            _val_dt = time.perf_counter() - start
            elapsed += _val_dt
            validation_ms += _val_dt * 1000.0

            self._log.validator_result(queries_before_validation, pending_queries)
            self._accumulate_tokens(all_tokens, val_tokens)
            self._budget.add_tokens(val_tokens)

            tagged_val_errors = [
                {**e, "query_ids": [str(q.get("id")) for q in queries_before_validation]}
                if isinstance(e, dict) and "query_ids" not in e
                else e
                for e in val_errors
            ]
            all_errors.extend(tagged_val_errors)

            if not pending_queries:
                self._log.warning(
                    f"No queries after validation on iteration {iteration + 1} — stopping."
                )
                break

            # --- 2b. Judge ---
            _t = time.perf_counter()
            evaluation = judge.evaluate(pending_queries)
            _judge_dt = time.perf_counter() - _t
            elapsed += _judge_dt
            judging_ms += _judge_dt * 1000.0

            for er in evaluation["approved"]:
                qid = str(er["id"])
                orig = er.get("_original_query") or next(
                    (q for q in pending_queries if str(q.get("id")) == qid), {}
                )
                all_approved_query_dicts[qid] = orig
                self._log.query_approved(er["id"], er.get("question", ""))

            for er in evaluation["rejected"]:
                j = judge.all_judgments_by_id.get(str(er["id"]), {})
                self._log.query_rejected(
                    er["id"],
                    er.get("question", ""),
                    feedback=j.get("feedback", ""),
                    suggestions=j.get("suggestions", ""),
                    attempt=judge._rejection_counts.get(str(er["id"]), 1),
                )

            for er in evaluation["permanently_rejected"]:
                self._log.query_permanent_reject(er["id"], er.get("question", ""))

            for f in evaluation["execution_failures"]:
                self._log.query_execution_failure(f["id"], f["error"])

            self._log.judge_result(
                evaluation["approved"],
                evaluation["rejected"],
                evaluation["permanently_rejected"],
                evaluation["execution_failures"],
            )

            all_approved_executed.extend(evaluation["approved"])

            # --- 2c. Prepare next iteration ---
            structured_judge_feedback = evaluation.get("structured_feedback", [])

            pending_queries = [
                er.get("_original_query") or er for er in evaluation["rejected"]
            ]

            for er in evaluation["permanently_rejected"]:
                qid = str(er["id"])
                orig = er.get("_original_query") or next(
                    (q for q in pending_queries if str(q.get("id")) == qid), {}
                )
                pending_queries.append(orig)
                structured_judge_feedback.append(
                    {
                        "id": er["id"],
                        "query": orig,
                        "error": (
                            f"Permanently rejected after "
                            f"{judge._rejection_counts.get(qid, 2)} attempt(s). "
                            f"History: {'; '.join(judge._accumulated_feedback.get(qid, []))}"
                        ),
                    }
                )

            for f in evaluation["execution_failures"]:
                pending_queries.append(f["query"])
                structured_judge_feedback.append(
                    {
                        "id": f["id"],
                        "query": f["query"],
                        "error": f"Execution failed: {f['error']}",
                    }
                )

            if not pending_queries:
                self._log.iteration_done()
                break

        return (
            all_approved_executed,
            all_approved_query_dicts,
            elapsed,
            {"validation": validation_ms, "judging": judging_ms},
        )

    def _assemble_result(
        self,
        mode: str,
        kind: str,
        aliases: dict,
        all_approved_executed: list,
        all_approved_query_dicts: dict,
        judge: Optional[JudgementResponseAgent],
        original_ids: set,
        columns_by_table: list,
        all_tokens: dict,
        all_errors: list,
        last_model: str,
        total_time: float,
        avg_cols: float,
        plan: Optional[StructuredQueryPlan] = None,
        skills: Optional[Any] = None,
        timings_ms: Optional[dict] = None,
    ) -> dict:
        """Assemble the final result dict.

        Backward compatibility is preserved by keeping the exact legacy result
        shape — ``result["queries"]`` remains a list of flat query dicts carrying
        every field the pipeline caller (``statement_generation.py``) and existing
        consumers read (``question``, ``code``, ``tables``, ``response``,
        ``judge_feedback``, ``keyword_count``, ...), and the top-level keys
        (``token_usage``, ``errors``, ``model``, ``time_elapsed``, ``avg_cols``,
        ``executed_results``, ``proposed_columns`` in multi mode) are unchanged.

        The traceable output (Requirement 21) is added **additively** under the
        ``traceable`` key as a :class:`TraceableQuerySet` dump: each approved
        query is assembled into a :class:`TraceableQuery` grouped by planning /
        skill / validation / judging / execution / usage phases (21.1), retaining
        every legacy field (21.2), including the structured plan, the selected
        skill identity, and the per-phase timings (21.3), with a distinct
        non-negative integer id (21.4) and an execution status that is exactly
        one of ``success`` / ``execution_failure`` / ``empty`` / ``not_run``
        (21.5). This choice keeps the flat queries as the drop-in
        backward-compatible payload while exposing the superseding traceable view
        alongside it.

        Mode divergences honoured here:

        * multi mode includes ``proposed_columns`` (Requirement 1.3); single mode
          omits it,
        * single mode constrains each assembled query's ``tables`` to the single
          provided alias (Requirement 1.4).
        """
        judgments_by_id = judge.all_judgments_by_id if judge is not None else {}
        approved_query_ids = {str(er["id"]) for er in all_approved_executed}
        approved_executed_by_id = {str(er["id"]): er for er in all_approved_executed}
        expected_alias = list(aliases.keys())[0] if aliases else None
        timings_ms = dict(timings_ms or {})

        next_id = (
            max(
                (int(i) for i in original_ids if str(i).lstrip("-").isdigit()),
                default=0,
            )
            + 1
        )

        # Deterministic assembly order: sort ids so the integer-id assignment and
        # the resulting traceable set are stable across runs.
        def _sort_key(qid: str):
            s = str(qid)
            return (0, int(s)) if s.lstrip("-").isdigit() else (1, s)

        ordered_ids = sorted(
            (qid for qid in approved_query_ids if all_approved_query_dicts.get(qid)),
            key=_sort_key,
        )

        used_ids: set = set()
        approved_queries: list = []
        traceable_queries: list = []
        for qid in ordered_ids:
            q = all_approved_query_dicts.get(qid)
            if not q:
                continue
            q_copy = dict(q)

            # Enforce a distinct, non-negative integer id (Requirement 21.4),
            # preferring the post-normalisation id when it is already a clean
            # non-negative integer.
            er = approved_executed_by_id.get(qid)
            raw_id = er.get("id") if er else None
            clean_id: Optional[int] = None
            if isinstance(raw_id, bool):
                clean_id = None
            elif isinstance(raw_id, int) and raw_id >= 0:
                clean_id = raw_id
            elif (
                isinstance(raw_id, str)
                and raw_id.isdigit()
            ):
                clean_id = int(raw_id)
            if clean_id is None or clean_id in used_ids:
                while next_id in used_ids:
                    next_id += 1
                clean_id = next_id
                next_id += 1
            used_ids.add(clean_id)
            q_copy["id"] = clean_id

            # Divergence (Requirement 1.4): single mode constrains the assembled
            # tables to the single provided alias.
            if mode == SINGLE and expected_alias is not None:
                tables_field = q_copy.get("tables", [])
                if (
                    len(tables_field) != 1
                    or tables_field[0].get("name") != expected_alias
                ):
                    first = tables_field[0] if tables_field else {}
                    q_copy["tables"] = [
                        {
                            "name": expected_alias,
                            "reason": first.get("reason", ""),
                            "columns_involved": first.get("columns_involved", []),
                        }
                    ]

            judgment = judgments_by_id.get(qid, {})
            keyword_count = utils.count_keywords(q_copy.get("code"), kind)
            approved_queries.append(
                {
                    **q_copy,
                    "response": judgment.get("response", ""),
                    "translated_response": judgment.get("translated_response", ""),
                    "judge_feedback": judgment.get("feedback", ""),
                    "keyword_count": keyword_count,
                }
            )

            # Requirement 21: assemble the fully-traceable, phase-grouped query.
            traceable_queries.append(
                self._build_traceable_query(
                    q_copy=q_copy,
                    judgment=judgment,
                    executed=er,
                    plan=plan,
                    skills=skills,
                    timings_ms=timings_ms,
                    all_tokens=all_tokens,
                    last_model=last_model,
                    keyword_count=keyword_count,
                )
            )

        self._log.step3_summary(approved_queries, elapsed=total_time)

        traceable_set = TraceableQuerySet(queries=traceable_queries)

        result: dict = {
            "result": {"queries": approved_queries},
            "traceable": traceable_set.model_dump(),
            "token_usage": all_tokens,
            "errors": all_errors,
            "model": last_model,
            "time_elapsed": total_time,
            "avg_cols": avg_cols,
            "executed_results": all_approved_executed,
        }
        # Divergence (Requirement 1.3): proposed_columns only in multi mode.
        if mode == MULTI:
            result["proposed_columns"] = columns_by_table
        return result

    # ------------------------------------------------------------------
    # Traceable assembly helpers (task 13.2)
    # ------------------------------------------------------------------

    def _build_traceable_query(
        self,
        q_copy: dict,
        judgment: dict,
        executed: Optional[dict],
        plan: Optional[StructuredQueryPlan],
        skills: Optional[Any],
        timings_ms: dict,
        all_tokens: dict,
        last_model: str,
        keyword_count: int,
    ) -> TraceableQuery:
        """Assemble one approved query into a phase-grouped :class:`TraceableQuery`.

        Every legacy field is mapped into exactly one phase sub-object so no field
        is lost (Requirement 21.2): the question/keywords/translated variants,
        difficulty, topic, story, free-text query plan and per-table
        reason/columns/keywords go into :class:`PlanningTrace`; the judge's
        feedback/suggestions/violated criteria/vagueness and requirements checks
        plus the response/translated response go into :class:`JudgeTrace`; the
        code, token usage and model go into the top-level fields and
        :class:`UsageTrace`. The structured plan, the selected skill identity and
        the per-phase timings are attached too (Requirement 21.3), and the
        execution status is derived as exactly one of the four allowed literals
        (Requirement 21.5).
        """
        # ── Planning trace ────────────────────────────────────────────────
        tables = [
            TableTrace(
                name=t.get("name", ""),
                reason=t.get("reason", ""),
                columns_involved=list(t.get("columns_involved", []) or []),
                description=t.get("description", "") or "",
                keywords=list(t.get("keywords", []) or []),
                translated_keywords=list(t.get("translated_keywords", []) or []),
            )
            for t in (q_copy.get("tables") or [])
        ]

        planning = PlanningTrace(
            question=q_copy.get("question", "") or "",
            question_keywords=list(q_copy.get("question_keywords", []) or []),
            translated_question=q_copy.get("translated_question", "") or "",
            translated_question_keywords=list(
                q_copy.get("translated_question_keywords", []) or []
            ),
            difficulty=q_copy.get("difficulty", "") or "",
            topic=q_copy.get("topic", "") or "",
            story=q_copy.get("story", "") or "",
            detected_language=q_copy.get("detected_language", "") or "",
            query_plan=q_copy.get("query_plan", "") or "",
            structured_plan=plan if plan is not None else self._empty_plan(q_copy),
            tables=tables,
        )

        # ── Skill trace: the selected skill identity (if any) ─────────────
        skill = self._build_skill_trace(skills)

        # ── Validation trace ──────────────────────────────────────────────
        # Approved queries have passed validation; per-query static-correction
        # audit is not tracked separately at assembly, so the final code is
        # recorded with a passed outcome and an empty (never-dropped) error list.
        validation = ValidationTrace(
            passed=True,
            correction_cycles=0,
            static_errors=[],
            final_code=q_copy.get("code", "") or "",
        )

        # ── Judge trace ───────────────────────────────────────────────────
        judging = JudgeTrace(
            approved=bool(judgment.get("approved", True)),
            feedback=judgment.get("feedback", "") or "",
            suggestions=judgment.get("suggestions", "") or "",
            violated_criteria=[
                str(c) for c in (judgment.get("violated_criteria", []) or [])
            ],
            vagueness_check=judgment.get("vagueness_check", "") or "",
            requirements_check=judgment.get("requirements_check", "") or "",
            response=judgment.get("response", "") or "",
            translated_response=judgment.get("translated_response", "") or "",
        )

        # ── Execution trace (Requirement 21.5) ────────────────────────────
        execution = self._derive_execution_trace(executed)

        # ── Usage trace (Requirement 21.3 timings) ────────────────────────
        usage = UsageTrace(
            prompt_tokens=int(all_tokens.get("prompt_tokens", 0) or 0),
            completion_tokens=int(all_tokens.get("completion_tokens", 0) or 0),
            total_tokens=int(all_tokens.get("total_tokens", 0) or 0),
            model=last_model or "",
            timings_ms=dict(timings_ms or {}),
        )

        return TraceableQuery(
            id=int(q_copy.get("id", 0)),
            client_id=str(q_copy.get("client_id", "") or ""),
            code=q_copy.get("code", "") or "",
            keyword_count=int(keyword_count or 0),
            planning=planning,
            skill=skill,
            validation=validation,
            judging=judging,
            execution=execution,
            usage=usage,
        )

    @staticmethod
    def _build_skill_trace(skills: Optional[Any]) -> SkillTrace:
        """Extract the selected skill identity into a :class:`SkillTrace`.

        ``skills`` is a ``SkillSelection`` (``.cards`` list of ``SkillCard``).
        When one or more skills were selected the first card's name/version are
        recorded as the query's selected skill; otherwise the trace records no
        skill (plain-Python generation).
        """
        cards = list(getattr(skills, "cards", []) or []) if skills is not None else []
        if not cards:
            return SkillTrace(skill_used=None, skill_version=None, reason="")
        card = cards[0]
        return SkillTrace(
            skill_used=getattr(card, "name", None),
            skill_version=getattr(card, "version", None),
            reason=f"Selected skill '{getattr(card, 'name', '')}' for the plan task types.",
        )

    @staticmethod
    def _derive_execution_trace(executed: Optional[dict]) -> ExecutionTrace:
        """Derive the execution status of an approved query (Requirement 21.5).

        Approved queries reached judging, which executes them. The status is one
        of exactly four literals:

        * ``success`` — executed and returned at least one row,
        * ``empty`` — executed but returned zero rows,
        * ``execution_failure`` — execution raised (not expected for an approved
          query, but derivable defensively),
        * ``not_run`` — never executed (no execution record available).
        """
        if not executed:
            return ExecutionTrace(status="not_run")

        if executed.get("error"):
            return ExecutionTrace(status="execution_failure", error=str(executed["error"]))

        df = executed.get("dataframe")
        if df is None:
            # Approved but no captured frame — it did execute (judge ran it), so
            # treat as a successful run with an unknown row count.
            return ExecutionTrace(status="success", row_count=None)

        try:
            row_count = int(len(df))
        except (TypeError, ValueError):
            return ExecutionTrace(status="success", row_count=None)

        status = "success" if row_count > 0 else "empty"
        return ExecutionTrace(status=status, row_count=row_count)

    @staticmethod
    def _empty_plan(q_copy: dict) -> StructuredQueryPlan:
        """Build a minimal schema-valid plan when none was threaded in.

        The traceable schema requires a ``structured_plan``; this defensive
        fallback (used only if assembly is ever called without a plan) records the
        query's question with no decomposition steps rather than dropping the
        field.
        """
        return StructuredQueryPlan(
            question=q_copy.get("question", "") or "",
            question_keywords=[],
            plan_keywords=[],
            steps=[],
            task_types=[],
            table_links=[],
        )

    @staticmethod
    def _accumulate_tokens(total: dict, partial: Any) -> None:
        """Add a usage dict into ``total`` in place (missing keys contribute 0)."""
        if not isinstance(partial, dict):
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total[key] += partial.get(key, 0)


# ----------------------------------------------------------------------------
# Backward-compatible named subclasses (task 11.4, design §1 Option A)
# ----------------------------------------------------------------------------
#
# These preserve the exact import names the legacy call sites in
# ``src/orqa/statement_generation.py`` rely on:
#
#     from .agent.agent import StatementGenerationAgent, SingleTableStatementGenerationAgent
#
# ``agent.py`` re-exports both names from here (see the bottom of ``agent.py``),
# so ``statement_generation.py`` needs no edits. Each subclass simply pins the
# public ``generate_statements`` method to the matching positional signature of
# the legacy agent it replaces.


class StatementGenerationAgent(UnifiedStatementGenerationAgent):
    """Multi-table drop-in replacement for the legacy agent of the same name.

    Its ``generate_statements`` keeps the multi-table positional signature and
    delegates to the unified multi-table adapter (``_run(mode="multi", ...)``).
    """

    def generate_statements(
        self,
        dataset_paths,
        aliases,
        kind,
        match,
        involved_cols,
        metadatas,
        max_cols: int = 10,
        sample_size: int = 5,
    ):
        return self._run(
            mode=MULTI,
            dataset_paths=dataset_paths,
            aliases=aliases,
            kind=kind,
            match=match,
            involved_cols=involved_cols,
            metadata=metadatas,
            max_cols=max_cols,
            sample_size=sample_size,
        )


class SingleTableStatementGenerationAgent(UnifiedStatementGenerationAgent):
    """Single-table drop-in replacement for the legacy agent of the same name.

    IMPORTANT: the single-table call site invokes
    ``generate_statements(csv_path, aliases, kind, metadata, max_cols, sample_size)``,
    so this subclass's ``generate_statements`` MUST accept the SINGLE-table
    positional signature ``(dataset_path, alias, kind, metadata, ...)`` — NOT the
    multi-table one — and forward to the unified single-table adapter
    (``generate_statements_single`` -> ``_run(mode="single", ...)``).
    """

    def generate_statements(
        self,
        dataset_path,
        alias,
        kind,
        metadata,
        max_cols: int = 10,
        sample_size: int = 5,
    ):
        return self.generate_statements_single(
            dataset_path,
            alias,
            kind,
            metadata,
            max_cols=max_cols,
            sample_size=sample_size,
        )


# ----------------------------------------------------------------------------
# Deferred cross-module import (breaks the agent <-> unified_agent cycle)
# ----------------------------------------------------------------------------
# Imported here, at the end of the module, so that when ``unified_agent`` is the
# first of the two modules to load, ``agent.py``'s own bottom re-export of the
# subclasses above can resolve them (they are defined by now). ``_run``
# references ``JudgementResponseAgent`` at call time only, so a bottom-of-module
# binding is sufficient, and the name stays module-level for test patching.
from ..agent import JudgementResponseAgent  # noqa: E402
