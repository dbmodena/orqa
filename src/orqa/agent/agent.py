from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union
from .. import utils
import json
import uuid
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from ..queries.query_execution import QueryExecutor
from .utility.difficulty_estimator import estimate_plan_tier
from .utility.generation_coordinator import GenerationCoordinator
from .utility.keyword_searchability import check_keyword_searchability
from .utility.plan_code_alignment import alignment_warning
from .utility.keyword_suggestion import suggest_retrievable_keywords
from .agents.TaskProposer import PairTaskSelectorLLMClient, TaskProposerLLMClient
from .agents.StatementJudge import LLMStatementJudge
from .agents.JudgePanel import JudgePanel
from .agents.QueryPlanner import QueryPlanner
from .agents.StatementClient import LLMClientStatementGenerator
from .agents.StatementValidator import LLMStatementValidator
from .agents.table_analyzer import TableAnalyzer
from ..utils.pipeline_logger import PipelineLogger
from .prompting import (
    CandidatesDiscoveryPrompt,
    ColumnStatistics,
    DatasetDescription,
    LightDatasetDescription,
    PairTaskSelectionPrompt,
    JudgementResponseGenerationPrompt,
    PandasStatementGenerationPrompt,
    PlanJudgementPrompt,
    SingleTableJudgementResponseGenerationPrompt,
    SingleTablePandasPrompt,
    SingleTableSQLPrompt,
    SQLStatementGenerationPrompt,
)
from .prompting.models import (
    ExecutionTrace,
    JudgeTrace,
    PandasQueryPlan,
    PlanningTrace,
    SkillTrace,
    SQLQueryPlan,
    TableTrace,
    TraceableQuery,
    TraceableQuerySet,
    UsageTrace,
    ValidationTrace,
)
import copy
import re


logger = logging.getLogger(__name__)

QueryPlan = Union[SQLQueryPlan, PandasQueryPlan]

# The two run modes the statement-generation orchestrator supports.
MULTI = "multi"
SINGLE = "single"

# Default bound on the number of concurrent judging calls (Requirement 15.1).
# Kept small so the LLM backend is not overwhelmed while still removing the
# serial one-at-a-time bottleneck. Configurable per JudgementResponseAgent.
JUDGE_CONCURRENCY = 6

# How many correction retries a rejected artifact gets before it is given up
# on: a query rejected by the code panel stays actionable (fed back to the
# validator's correction path with the judge's feedback) until it has been
# rejected more than this many times (the validator-judge loop runs
# 1 + MAX_QUERY_CORRECTIONS iterations so every retry can actually happen);
# a plan rejected by the plan panel is revised (QueryPlanner.revise_plan) and
# re-judged up to MAX_PLAN_CORRECTIONS times.
MAX_QUERY_CORRECTIONS = 3
MAX_PLAN_CORRECTIONS = 3

# Output-token cap for every judge call (code panel, plan panel, single-judge
# fallback). Must leave headroom for REASONING models (gemini-2.5-flash,
# gpt-oss): their hidden thinking tokens count against max_tokens, and a judge
# that exhausts the cap while still thinking returns a choice with
# finish_reason=max_tokens and NO message — which litellm's OCI transformer
# rejects, so the judge produces no verdict at all. 2000 was routinely consumed
# by thinking alone (~1997 reasoning tokens observed) before any JSON came out.
JUDGE_MAX_TOKENS = 4096

# Keys on an executed-result record that are INTERNAL bookkeeping and must not
# reach the code judge's payload. `_original_query` is a full deep copy of the
# query (see _execute_queries) that the validator-to-judge loop reads to map
# verdicts back to their source — indispensable on the record, but rendering it
# into the payload sent the judge the same question/code/tables a second time.
# `client_id` is an opaque internal key the judge has no instruction to read.
_JUDGE_PAYLOAD_EXCLUDED = ("_original_query", "client_id")

# Minimum rows a prepared table must have for a statement-generation run to
# proceed. A table that comes out of ``prepare_dataset`` with (nearly) no rows
# cannot ground question generation, and every query validated against it
# fails spuriously ("Empty result", "x_train is empty", ...) — fail the
# entry fast and explicitly instead. Mirrors the candidates_discovery
# ``min_dataset_height`` config default.
MIN_PREPARED_ROWS = 10


class CandidatesDiscoveryAgent:
    """Classical (BLEND) discovery agent: proposes union/join/JC tasks for a
    SINGLE dataset, blind to partner tables — the BLEND index then searches
    for candidates matching the proposed columns."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.prompt = CandidatesDiscoveryPrompt()
        self._client = TaskProposerLLMClient(self.config_path)

    def propose_tasks(
        self,
        dataset_path: Path,
        dataset_format: str,
        metadata: dict,
        polars_opts: dict,
        min_dataset_height: int = 10,
        limit_to_n_columns: int = 20,
        sample_size: int = 5,
        seed: int = 0,
    ) -> dict | None:
        try:
            dataset_info, column_typings = utils.load_dataset_info(
                dataset_path,
                polars_opts,
                limit_to_n_columns,
                sample_size,
                seed,
            )

            # Skip tables that are unusable as a whole: too short, no
            # columns, or WIDER than limit_to_n_columns. A wide table is
            # dropped outright (not vertically sliced) so the agent only
            # ever works with integral tables — mirrors the up-front filter
            # the semantic pipeline applies in build_embedding_texts.
            if (
                dataset_info["num_rows"] < min_dataset_height
                or dataset_info["num_columns"] == 0
                or dataset_info["num_columns_raw"] > limit_to_n_columns
            ):
                return

            prompt_str = self.prompt.update(
                dataset_info["dataset_name"],
                dataset_info["num_rows"],
                dataset_info["num_columns"],
                metadata,
                dataset_info["columns_details"],
                dataset_info["sample_data"],
            )

            result, tokens = self._client.complete(
                prompt_str,
                schema=dataset_info["columns"],
                column_typings=column_typings,
            )

            return {
                "dataset": dataset_info["dataset_name"],
                "tasks": result,
                "token_usage": tokens,
            }
        except FileNotFoundError as e:
            logger.error("Dataset not found: %s", e)
        except Exception as e:
            logger.error("Analysis failed: %s", e)
            raise e


class PairTaskSelectionAgent:
    """Selects discovery operations (union/join/join-correlation) for one
    concrete (Q, R) candidate pair, given both schemas and the schema-matching
    evidence gathered by the discovery pipeline."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.prompt = PairTaskSelectionPrompt()
        self._client = PairTaskSelectorLLMClient(self.config_path)

    @staticmethod
    def _describe(dataset_info: dict, metadata: dict) -> str:
        return DatasetDescription().update(
            dataset_info["dataset_name"],
            dataset_info["num_rows"],
            dataset_info["num_columns"],
            metadata,
            dataset_info["columns_details"],
            dataset_info["sample_data"],
        )

    def select_tasks(
        self,
        q_info: dict,
        q_column_typings: dict,
        q_metadata: dict,
        r_info: dict,
        r_column_typings: dict,
        r_metadata: dict,
        valentine_matches: list,
        cosine_sim: float,
        top_n_matches: int = 15,
    ) -> dict | None:
        try:
            prompt_str = self.prompt.update(
                self._describe(q_info, q_metadata),
                self._describe(r_info, r_metadata),
                cosine_sim,
                valentine_matches[:top_n_matches],
            )

            result, tokens = self._client.complete(
                prompt_str,
                q_schema=q_info["columns"],
                r_schema=r_info["columns"],
                q_column_typings=q_column_typings,
                r_column_typings=r_column_typings,
            )

            return {
                "pair": [q_info["dataset_name"], r_info["dataset_name"]],
                "tasks": result,
                "token_usage": tokens,
            }
        except Exception as e:
            logger.error("Pair task selection failed: %s", e)
            raise e


class TableAnalysisAgent:
    """Standalone pre-generation table analysis stage.

    Runs BEFORE ``create_statements``: takes every table involved in the
    upcoming generation run (all cross-table candidate datasets plus the
    deterministically sampled single tables), analyses them in batches with
    :class:`TableAnalyzer`, and persists description + keywords in the shared
    analysis cache (``table id -> model -> {description, keywords}``).

    The generation pipeline's own analyzer then serves every table straight
    from that cache, so table-description generation is fully decoupled from
    ``StatementOrchestrator``: no description/keyword LLM call happens inside the
    generation loop (the in-loop analyzer only remains as a read-through
    fallback for a table this stage somehow missed).

    Tables already cached for the configured model are skipped WITHOUT even
    loading their CSV, so re-runs and sibling workflows over the same portal
    cost nothing.
    """

    def __init__(
        self,
        config_path: Path,
        cache_path: Path,
        languages: list | None = None,
        bad_tokens: list | None = None,
        max_cols: int = 10,
        sample_size: int = 5,
        max_rows: int | None = None,
        batch_size: int = 8,
        seed: int = 0,
        analyzer: TableAnalyzer | None = None,
    ):
        self._analyzer = analyzer or TableAnalyzer(config_path, cache_path=cache_path)
        self.languages = languages if languages is not None else ["English"]
        self.bad_tokens = bad_tokens or []
        self.max_cols = max_cols
        self.sample_size = sample_size
        self.max_rows = max_rows
        # Tables sent per batched analysis call — bounds the prompt size the
        # same way the generation pipeline's own per-run batches did.
        self.batch_size = max(1, int(batch_size))
        self.seed = seed
        self._log = PipelineLogger()

    def analyze_tables(
        self,
        dataset_paths: list,
        datasets_metadata: dict | None = None,
        involved_columns_by_dataset: dict[str, list] | None = None,
    ) -> dict:
        """Analyse (and cache) every distinct table in ``dataset_paths``.

        Args:
            dataset_paths: Paths of every table involved in the upcoming run;
                duplicates are fine (deduplicated by dataset name here).
            datasets_metadata: Mapping of dataset name -> normalized metadata,
                injected into the analysis prompt when available.
            involved_columns_by_dataset: Mapping of dataset name -> the
                join/union/correlation columns that table contributes to any
                candidate in this run (union across candidates). Forced into
                the analysis column budget (see ``utils.select_columns``)
                exactly as per-candidate generation does, so the description/
                keywords are built from the same columns the planner will
                actually work on — not just the first ``max_cols`` in file
                order.

        Returns:
            A summary dict: ``{"total", "cached", "analyzed", "failed"}``.
        """
        datasets_metadata = datasets_metadata or {}
        involved_columns_by_dataset = involved_columns_by_dataset or {}

        # Dedupe by dataset name (the cache's table id), preserving order.
        unique: dict[str, Path] = {}
        for p in dataset_paths:
            path = Path(p)
            unique.setdefault(path.stem, path)

        summary = {"total": len(unique), "cached": 0, "analyzed": 0, "failed": 0}

        # Partition first so cached tables never pay a CSV load.
        pending: list[tuple[str, Path]] = []
        for name, path in unique.items():
            if self._analyzer.is_cached(name):
                summary["cached"] += 1
            else:
                pending.append((name, path))

        self._log.analysis_start(summary["total"], summary["cached"], len(pending))

        total_batches = -(-len(pending) // self.batch_size) if pending else 0
        for batch_idx, start in enumerate(
            range(0, len(pending), self.batch_size), start=1
        ):
            chunk = pending[start : start + self.batch_size]
            self._log.analysis_batch(batch_idx, total_batches, [name for name, _ in chunk])
            dfs: list = []
            aliases: dict = {}
            metadata: list = []
            for name, path in chunk:
                try:
                    df, _info = utils.prepare_dataset(
                        path,
                        list(involved_columns_by_dataset.get(name, []) or []),
                        self.max_cols,
                        self.sample_size,
                        limit_to_n_rows=self.max_rows,
                        seed=self.seed,
                    )
                except Exception as exc:
                    self._log.table_analysis_failed(name, f"could not load: {exc}")
                    summary["failed"] += 1
                    continue
                aliases[f"Table_{len(dfs)}"] = name
                dfs.append(df)
                metadata.append(datasets_metadata.get(name) or {})

            if not dfs:
                continue

            try:
                # analyze_batch writes each fresh (non-empty) analysis to the
                # cache itself; an analysis the model failed to produce comes
                # back as an empty default and is deliberately NOT cached, so
                # it stays retryable.
                analyses = self._analyzer.analyze_batch(
                    dfs, aliases, metadata, self.languages
                )
            except Exception as exc:
                self._log.error(
                    f"Batch analysis failed for {list(aliases.values())}: {exc}"
                )
                summary["failed"] += len(dfs)
                continue

            for entry in analyses:
                table_id = aliases.get(entry.get("alias"), entry.get("alias"))
                description = entry.get("table_description", "")
                keywords = entry.get("table_keywords", []) or []
                if description or keywords:
                    summary["analyzed"] += 1
                    self._log.table_analyzed(table_id, description, keywords)
                else:
                    summary["failed"] += 1
                    self._log.table_analysis_failed(
                        table_id, "model returned no description/keywords"
                    )

        self._log.analysis_summary(summary)
        return summary


def _execution_row_count(executed: Optional[dict]) -> Optional[int]:
    """Row count of an executed query's captured DataFrame.

    Returns ``None`` when unknown (no execution record, an error, or no
    captured frame) and an ``int`` otherwise — including ``0`` for a
    genuinely empty result. Module-level (not a method) so both
    ``JudgementResponseAgent.evaluate`` (the empty-result routing gate) and
    ``StatementOrchestrator`` (:meth:`_derive_execution_trace`'s final-trace
    classification, :meth:`_retry_empty_results`'s escalation gate) share
    the exact same definition of "empty": zero rows, never a falsy scalar
    (a `number` result of ``0`` or a `boolean` of ``False`` is still a
    1-row frame).
    """
    if not executed or executed.get("error"):
        return None
    df = executed.get("dataframe")
    if df is None:
        return None
    try:
        return int(len(df))
    except (TypeError, ValueError):
        return None


class JudgementResponseAgent:
    def __init__(
        self,
        config_path: Path,
        kind: str,
        executor,
        entry: dict,
        max_tokens: int = JUDGE_MAX_TOKENS,
        single_table: bool = False,
        judge_concurrency: int = JUDGE_CONCURRENCY,
        dataframes: dict | None = None,
        max_corrections: int = MAX_QUERY_CORRECTIONS,
        code_judge_count: Optional[int] = None,
    ):
        self.config_path = config_path
        self.single_table = single_table
        # Prompt factory (Requirement 15.3): each concurrent judging call builds
        # its OWN prompt instance rather than sharing one mutable prompt object,
        # so no mutable prompt/message state is shared across worker threads.
        self._prompt_factory = (
            SingleTableJudgementResponseGenerationPrompt
            if single_table
            else JudgementResponseGenerationPrompt
        )
        # Kept for backward compatibility with callers that read `self.prompt`.
        self.prompt = self._prompt_factory()
        self._client = LLMStatementJudge(config_path)
        # Majority-vote panel for code judging (judge_profiles.code in the
        # LLM yaml): N small models — different from the generation model, so
        # no judge grades its own family's output. Two independent layers
        # (see Judgment/JudgePanel._aggregate): plan_compliance_approval
        # (does the code correctly implement the plan) and
        # present_result_approval (is the executed result actually there and
        # meaningful) — each majority-aggregated on its own, so a query whose
        # code genuinely complies but whose result is empty is
        # distinguishable from one whose code is wrong (see evaluate()'s
        # empty-result routing below). Falls back to the single primary-model
        # judge (self._client) when the yaml has no code panel configured.
        self._code_panel = JudgePanel(
            config_path, "code", response_model="statement_judge", root_key="queries",
            vote_fields=["plan_compliance_approval", "present_result_approval"],
            judge_count=code_judge_count,
        )
        self.max_tokens = max_tokens
        self.executor = executor
        self.entry = entry
        self.kind = kind
        # Bound the worker pool (Requirement 15.1). Never below 1.
        self.judge_concurrency = max(1, int(judge_concurrency))
        # A rejected query stays correctable until it exceeds this many
        # rejections; after that it is reported as permanently rejected.
        self.max_corrections = max(0, int(max_corrections))
        self._rejection_counts: dict[str, int] = {}
        self._accumulated_feedback: dict[str, list[str]] = {}
        self.all_judgments_by_id: dict[str, dict] = {}
        # Judgments keyed by the opaque echoed client_id (Requirement 15.2).
        self.all_judgments_by_client_id: dict[str, dict] = {}
        # The SAME already-prepared {alias: DataFrame} the rest of the run
        # (analysis, planning, generation) was built from — e.g.
        # StatementOrchestrator's Phase-1 `dfs`. Injected so execution never reloads/re-derives its own
        # copy from disk: the schema the LLM saw and the schema code executes
        # against must be identical, or a column beyond what the LLM was shown
        # could silently pass execution while never having been visible during
        # generation (or vice versa). Falls back to loading from disk via
        # ``self.executor`` only if a caller genuinely doesn't have prepared
        # dataframes to inject (see ``_get_dataframes``).
        self._dataframes: dict | None = dataframes
        # Cache of the rendered judge instructions, so concurrent calls don't
        # re-read/re-format the same markdown on every call.
        self._judge_instructions_cache: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, pending_queries: list) -> dict:
        # ── Guarantee every query has a unique, non-sentinel string id ──
        pending_queries = self._normalise_ids(pending_queries)

        executed, execution_failures = self._execute_queries(pending_queries)

        if not executed:
            return {
                "approved": [],
                "rejected": [],
                "permanently_rejected": [],
                "empty_result": [],
                "execution_failures": execution_failures,
                "feedback_messages": None,
                "structured_feedback": [],
                "all_done": False,
            }

        # ── Judge concurrently, keyed by the opaque client_id ──
        # (Requirement 15.1) Judging runs on a bounded worker pool instead of
        # serially, removing the one-at-a-time bottleneck.
        # (Requirement 15.2) Each judgment is mapped back to its query by the
        # query's client_id — never by positional/completion order, so results
        # that complete out-of-order are still attributed to the right query.
        # (Requirement 15.3) Every judging call rebuilds its own messages via a
        # fresh prompt instance; no mutable prompt/message state is shared.
        self.all_judgments_by_client_id = self._judge_all_concurrent(executed)

        # Re-key by the (normalised, unique) query id for downstream consumers
        # that still look judgments up by id.
        for er in executed:
            self.all_judgments_by_id[str(er["id"])] = (
                self.all_judgments_by_client_id.get(self._client_id_of(er), {})
            )

        approved = [
            er for er in executed
            if self.all_judgments_by_id.get(str(er["id"]), {}).get("approved", False)
        ]
        not_approved = [
            er for er in executed
            if not self.all_judgments_by_id.get(str(er["id"]), {}).get("approved", False)
        ]

        # A query whose code the panel majority found PLAN-COMPLIANT but whose
        # executed result is genuinely empty (0 rows) is a plan-level symptom
        # (an over-restrictive filter, a mismatched join), not a code bug —
        # the normal code-correction loop can never fix it, so it must never
        # consume a correction cycle. Pulled out here, before rejection
        # counting, and routed separately (see empty_result below); the
        # caller (StatementOrchestrator._validator_judge_loop) escalates it
        # to a one-shot plan-level retry instead of feeding it back as
        # `rejected` (see _retry_empty_results).
        empty_result_pending = [
            er for er in not_approved
            if self.all_judgments_by_id.get(str(er["id"]), {}).get("plan_compliance_approval")
            and not self.all_judgments_by_id.get(str(er["id"]), {}).get("present_result_approval")
            and _execution_row_count(er) == 0
        ]
        empty_result_ids = {str(er["id"]) for er in empty_result_pending}
        rejected = [er for er in not_approved if str(er["id"]) not in empty_result_ids]

        for er in rejected:
            qid = str(er["id"])
            self._rejection_counts[qid] = self._rejection_counts.get(qid, 0) + 1
            judgment = self.all_judgments_by_id.get(qid, {})
            feedback = (
                f"Feedback: {judgment.get('feedback', 'no feedback')}\n"
                f"Suggestions: {judgment.get('suggestions', 'none')}"
            )
            self._accumulated_feedback.setdefault(qid, []).append(feedback)

        still_actionable = [
            er for er in rejected
            if self._rejection_counts.get(str(er["id"]), 0) <= self.max_corrections
        ]
        permanently_rejected = [
            er for er in rejected
            if self._rejection_counts.get(str(er["id"]), 0) > self.max_corrections
        ]

        feedback_messages = self._build_feedback_messages(still_actionable, pending_queries)
        structured_feedback = self._build_structured_feedback(still_actionable, pending_queries)

        return {
            "approved": approved,
            "rejected": still_actionable,
            "permanently_rejected": permanently_rejected,
            "empty_result": empty_result_pending,
            "execution_failures": execution_failures,
            "feedback_messages": feedback_messages,
            "structured_feedback": structured_feedback,
            "all_done": not still_actionable and not empty_result_pending and not execution_failures,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_ids(queries: list) -> list:
        """
        Return a new list where every query dict has a unique, non-sentinel id.

        A query is considered id-less when:
          - the "id" key is absent
          - the value is None
          - the string representation is "" or "?"

        Such queries receive a fresh short UUID.  Any duplicate ids (two queries
        that happen to share the same value) also get a fresh id so they are
        never merged in downstream dicts.
        """
        seen_ids: set[str] = set()
        normalised: list[dict] = []
        for q in queries:
            q = dict(q)  # shallow copy — don't mutate the caller's objects
            qid = str(q.get("id", "")).strip()
            if not qid or qid == "?" or qid in seen_ids:
                qid = f"auto_{uuid.uuid4().hex[:8]}"
                q["id"] = qid
            seen_ids.add(qid)
            normalised.append(q)
        return normalised

    def _get_dataframes(self) -> dict:
        """Return the prepared ``{alias: DataFrame}`` used for every query and
        every validator-to-judge iteration in this run.

        Normally this is exactly the ``dataframes`` injected at construction —
        the same objects analysis/planning/generation already saw — so
        execution never reloads or re-derives its own copy from disk. Falls
        back to a one-time ``self.executor.load_tables(...)`` read, cached for
        the rest of the run, only if the caller genuinely didn't inject
        prepared dataframes (e.g. a legacy/test caller) — that fallback loads
        the full, uncapped table rather than the run's prepared schema, so it
        is not equivalent, just a safety net against a hard crash.
        """
        if self._dataframes is None:
            logger.warning(
                "JudgementResponseAgent: no prepared dataframes were injected; "
                "falling back to loading tables from disk for entry %r.",
                self.entry.get("tables", {}),
            )
            self._dataframes = self.executor.load_tables(self.entry.get("tables", {}))
        return self._dataframes

    def _execute_queries(self, queries: list) -> tuple[list, list]:
        # Build a set of dataset names so we can guard against them leaking
        # into columns_involved or code as column references.
        alias_map: dict = self.entry.get("tables", {})   # {"Table_0": "dataset_name", …}
        dataset_names: set = set(alias_map.values())

        try:
            dataframes = self._get_dataframes()
        except Exception as exc:
            return [], [
                {"id": q.get("id", "?"), "error": str(exc), "query": copy.deepcopy(q)}
                for q in queries
            ]

        executed, failures = [], []
        for query in queries:
            qid = query.get("id", "?")

            # Deep-copy and sanitize before execution.
            # The correction LLM occasionally fixes columns_involved in JSON but
            # leaves the dataset name as a column reference in the code.
            # We apply alias replacement here as a last-resort safety net.
            sanitized = copy.deepcopy(query)

            # 1. Sanitize columns_involved: strip dataset names mistaken for columns.
            for table in sanitized.get("tables", []):
                cols = table.get("columns_involved") or []
                table["columns_involved"] = [c for c in cols if c not in dataset_names]

            # 2. Apply alias replacement to the code as a safety net.
            code = sanitized.get("code", "")
            if code and dataset_names:
                for table_name, dataset_name in alias_map.items():
                    code = code.replace(f'"{dataset_name}"', table_name)
                    code = code.replace(f"'{dataset_name}'", table_name)
                    code = re.sub(rf'\b{re.escape(dataset_name)}\b', table_name, code)
                sanitized["code"] = code

            # 3. Build the tables list for the executor/judge payload. Execution
            # itself only reads code (QueryExecutor.execute_prepared never
            # touches `tables`), so this list exists purely to give the code
            # judge context. `columns_involved` is dropped (not needed
            # there). `reason` (the plan's own, panel-judged justification —
            # see PlanJudgment.table_check) is deliberately KEPT: the code
            # judge reads it as CONTEXT for why each table is in the query
            # (grounding `requirements_check`/`response`), but table choice
            # itself is decided and judged once, during PLANNING — judge.md
            # explicitly forbids re-judging table choice, so seeing the
            # justification must never be read as an invitation to
            # re-litigate it.
            tables = copy.deepcopy(sanitized.get("tables", []))
            for table in tables:
                table.pop("columns_involved", None)

            try:
                df_result = self.executor.execute_prepared(
                    sanitized, self.kind, dataframes
                ).head(8)
                executed.append({
                    "id": qid,
                    # Preserve the opaque echoed client_id (task 8) so judging can
                    # map judgments back by client_id rather than position (15.2).
                    "client_id": query.get("client_id"),
                    "code": sanitized.get("code", ""),
                    "question": sanitized.get("question", ""),
                    "translated_question": sanitized.get("translated_question", ""),
                    "detected_language": sanitized.get("detected_language", ""),
                    "dataframe": df_result,
                    "tables": tables,
                    # FIX 2: Preserve the original unstripped query so subsequent
                    # validator iterations still have columns_involved intact.
                    "_original_query": copy.deepcopy(query),
                })
            except (Exception, SystemExit) as exc:
                # SystemExit too: judge-time execution runs the generated code
                # with a plain in-process exec (QueryExecutor._execute_pandas),
                # so a generated `raise SystemExit` would otherwise terminate
                # the whole pipeline instead of failing this one query.
                failures.append({
                    "id": qid,
                    "error": str(exc) or type(exc).__name__,
                    # FIX 2: Also deep-copy here so pending_queries rebuild in
                    # Step 2c always gets a fresh, unstripped original.
                    "query": copy.deepcopy(query),
                })
        return executed, failures

    def _build_feedback_messages(
        self, rejected_executed: list, pending_queries: list
    ) -> list:
        rejected_ids = {str(er["id"]) for er in rejected_executed}
        rejected_queries = [
            q for q in pending_queries if str(q.get("id")) in rejected_ids
        ]

        feedback_lines = [
            (
                f"- id={er['id']} (failed {self._rejection_counts.get(str(er['id']), 1)} time/s):\n"
                f"  History: {'; '.join(self._accumulated_feedback.get(str(er['id']), []))}"
            )
            for er in rejected_executed
        ]
        feedback_text = (
            "The following queries were rejected. Fix ONLY these queries and return them "
            "with the same IDs. Do NOT modify or return the approved queries.\n\n"
            "Rejected queries:\n" + "\n".join(feedback_lines)
        )
        return [
            {
                "role": "assistant",
                "content": json.dumps({"queries": rejected_queries}, indent=2),
            },
            {"role": "user", "content": feedback_text},
        ]

    def _build_structured_feedback(
        self, rejected_executed: list, pending_queries: list
    ) -> list:
        rejected_ids = {str(er["id"]) for er in rejected_executed}
        rejected_query_map = {
            str(q.get("id")): q
            for q in pending_queries
            if str(q.get("id")) in rejected_ids
        }
        feedback = []
        for er in rejected_executed:
            qid = str(er["id"])
            count = self._rejection_counts.get(qid, 1)
            history = "; ".join(self._accumulated_feedback.get(qid, []))

            # FIX 1: Pull the actual judgment fields from all_judgments_by_id
            # instead of only surfacing the pre-serialised accumulated history
            # string (which previously contained "no feedback" because it was
            # built before the judgment data was available).
            judgment = self.all_judgments_by_id.get(qid, {})
            # The current judgment was already appended to _accumulated_feedback
            # in evaluate() before this method is called.  Using history alone
            # (which already contains it) avoids printing the same feedback twice
            # — once as explicit fields and once inside "History:".
            history_entries = self._accumulated_feedback.get(qid, [])
            if history_entries:
                # Only show the most recent two entries to prevent unbounded growth.
                history_excerpt = "; ".join(history_entries[-2:])
                detailed_error = f"Failed {count} time(s). History: {history_excerpt}"
            else:
                detailed_error = (
                    f"Failed {count} time(s). "
                    f"Feedback: {judgment.get('feedback', 'no feedback')}. "
                    f"Suggestions: {judgment.get('suggestions', 'none')}."
                )

            feedback.append({
                "id": er["id"],
                "query": rejected_query_map.get(qid, {}),
                "error": detailed_error,
            })
        return feedback

    def _client_id_of(self, executed_result: dict) -> str:
        """Return the opaque client_id used to key a query's judgment.

        Falls back to the (already-normalised, unique) query id when a query has
        no ``client_id`` — e.g. legacy callers that never ran the task-8
        generator. Because ids are made unique by ``_normalise_ids``, this
        fallback still yields a stable, collision-free key (Requirement 15.2).
        """
        cid = executed_result.get("client_id")
        if cid is not None and str(cid).strip():
            return str(cid)
        return str(executed_result.get("id"))

    def _judge_all_concurrent(self, executed: list) -> dict:
        """Judge executed queries concurrently, returning ``{client_id: judgment}``.

        Requirement 15.1: judging runs on a bounded ``ThreadPoolExecutor`` whose
        worker count never exceeds ``self.judge_concurrency`` (nor the number of
        queries). Requirement 15.2: each future is tagged with its query's
        ``client_id`` up front, so the returned mapping is keyed by client_id and
        is unaffected by the order in which futures complete. Requirement 15.3:
        each task calls ``_judge_one``, which rebuilds its own prompt and message
        array — no mutable prompt/message state is shared across workers.
        """
        judgments_by_client_id: dict = {}
        if not executed:
            return judgments_by_client_id

        max_workers = max(1, min(self.judge_concurrency, len(executed)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_client_id: dict = {}
            for er in executed:
                client_id = self._client_id_of(er)
                future = pool.submit(self._judge_one, er)
                future_to_client_id[future] = client_id

            for future in as_completed(future_to_client_id):
                client_id = future_to_client_id[future]
                try:
                    judgments_by_client_id[client_id] = future.result()
                except Exception as exc:  # noqa: BLE001 — isolate per-query failures
                    logger.warning(
                        "Judging failed for client_id=%s: %s", client_id, exc
                    )
                    judgments_by_client_id[client_id] = {}
        return judgments_by_client_id

    def _judge_one(self, executed_result: dict) -> dict:
        """
        Judge a single executed query in isolation.

        The Judgment Pydantic model requires an integer ``id`` field. To avoid
        any mismatch between the LLM-assigned integer and the real query id
        (which may be an auto_* UUID string), we always send ``id=1`` to the
        LLM and extract ``queries[0]`` from the response, then store it under
        the real query's client_id in ``all_judgments_by_client_id``.

        A fresh :class:`Prompt` instance is built the first time this is
        called (Requirement 15.3: no *mutable* prompt object is ever shared
        across concurrent judging workers), and its rendered text is cached
        in ``_judge_instructions_cache`` so every subsequent call reuses the
        same prefix, same principle as the plan judge panel.

        Returns the judgment dict for the query, or an empty dict on failure.
        """
        # Temporarily override id to 1 so the LLM produces a valid integer id.
        # Internal bookkeeping keys are dropped rather than copied through —
        # see _JUDGE_PAYLOAD_EXCLUDED. They stay on `executed_result` itself,
        # which the validator-to-judge loop still reads.
        single = {
            k: v for k, v in executed_result.items()
            if k not in _JUDGE_PAYLOAD_EXCLUDED
        }
        single["id"] = 1
        # Shield the executed result's dataframe before it is embedded (via
        # repr, both below) into the judge payload — an oversized cell (e.g.
        # a full-precision WKT geometry) would otherwise blow up the prompt.
        # Pandas' own default repr already truncates long strings, but that
        # is an incidental side effect of a global display option, not a
        # guarantee; shield explicitly instead of relying on it.
        if isinstance(single.get("dataframe"), pd.DataFrame):
            single["dataframe"] = utils.shield_dataframe_for_prompt(single["dataframe"])

        if "instructions" not in self._judge_instructions_cache:
            self._judge_instructions_cache["instructions"] = self._prompt_factory().update()
        prompt_str = self._judge_instructions_cache["instructions"]

        # Panel path: N judges vote on this query in parallel; the returned
        # judgment is the majority verdict and carries every per-judge vote
        # under its "panel" key (surfaced downstream as `code_feedback`).
        # The payload is built exactly like LLMStatementJudge.complete's so
        # both paths show the judges the identical evaluation content.
        if self._code_panel.is_configured:
            # Deterministic, code-computed signal (see plan_code_alignment.py)
            # folded into the payload text itself — cheaper and more visible
            # to the judge than a separate field it has no instruction to
            # read, same "surface it as a given fact" spirit as the
            # difficulty estimator's feedback into the plan-revision prompt.
            warning = alignment_warning(
                executed_result.get("tables", []), executed_result.get("code", "")
            )
            payload = (
                f"Queries:\n{[single]}\n\n"
                "Evaluate the queries above following the instructions and "
                f"return only the JSON verdict.{warning}"
            )
            judgment, _ = self._code_panel.evaluate(
                prompt_str, payload, max_tokens=self.max_tokens
            )
            return judgment

        result, _ = self._client.complete(
            prompt_str, data=[single], max_tokens=self.max_tokens
        )
        judgments = result.get("queries", [])
        return judgments[0] if judgments else {}



# ----------------------------------------------------------------------------
# Statement-generation orchestrator
# ----------------------------------------------------------------------------
# resolve_mode/_is_nonempty_match and StatementOrchestrator implement the
# five-phase, mode-aware statement-generation pipeline:
#
#     1. data preparation + cheap column statistics,
#     2. batched table analysis (ONE LLM call) + structured planning
#        (kind-aware: SQL plans never carry ML task types) + per-task-type
#        skill-injected code generation, and
#     3. the validator-to-judge loop (JudgementResponseAgent, above), then
#     4. final assembly.
#
# The generation prompt is built exactly ONCE after the per-dataset loop
# (rather than accumulated across it via a mutable prompt.update()/reset()
# pattern), and the mode divergence: single mode sets the judge
# single_table flag, while proposed_columns is included only in multi mode.

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

class StatementOrchestrator:
    """Single, mode-aware statement-generation orchestrator.

    Owns the five-phase pipeline for both ``single`` and ``multi`` modes through
    a single private core, :meth:`_run`. The public adapter methods
    (:meth:`generate_statements` and :meth:`generate_statements_single`)
    preserve the two legacy positional signatures and map them onto ``_run``
    without swapping the match/metadata arguments. The mode-specific entry
    points ``StatementAgent`` / ``SingleStatementAgent``
    (``orqa.agent.agents.StatementAgent`` / ``orqa.agent.agents.SingleStatementAgent``)
    subclass this orchestrator and each pin ``generate_statements`` to the
    positional signature ``statement_generation.py`` calls.

    All heavy collaborators are constructed in ``__init__`` but may be injected
    for testing: the :class:`QueryPlanner`, the :class:`TableAnalyzer`, the
    :class:`GenerationCoordinator`, the LLM generation client, and the
    :class:`LLMStatementValidator`.
    """

    def __init__(
        self,
        config_path: Path,
        kind: str,
        bad_tokens: list,
        # 1 initial judge pass + MAX_QUERY_CORRECTIONS correction rounds: a
        # rejected query gets up to 3 corrected re-submissions before the
        # loop gives up on it.
        max_judge_iterations: int = 1 + MAX_QUERY_CORRECTIONS,
        num_query_plans: int = 3,
        languages: Optional[list] = None,
        seed: int = 0,
        client: Optional[Any] = None,
        analyzer: Optional[TableAnalyzer] = None,
        planner: Optional[QueryPlanner] = None,
        generator: Optional[GenerationCoordinator] = None,
        validator: Optional[LLMStatementValidator] = None,
        analysis_cache_path: Optional[Path] = None,
        search_index: Optional[Any] = None,
        keyword_search_top_k_coefficient: float = 5.0,
        gate_unretrievable_groups: bool = False,
        retrieval_gate_enabled: bool = True,
        plan_judge_count: Optional[int] = None,
        code_judge_count: Optional[int] = None,
    ):
        self.config_path = config_path
        self.kind = kind
        self.bad_tokens = bad_tokens
        # Reverse index (DatasetIndex/ESDatasetIndex, or None when
        # unavailable) the plan judge's keyword-searchability check
        # searches to verify a plan's tables are actually retrievable from
        # its question's keywords — see
        # orqa.agent.utility.keyword_searchability.check_keyword_searchability
        # and its use in _judge_plans. `retrieval_gate_enabled=False`
        # (tasks.mcp_search.retrieval_gate_enabled in the workflow yaml) is
        # the master off switch for the retrieval gate as a whole — it's
        # implemented by simply treating the index as unset here, since both
        # the pre-planning keyword-suggestion check (below, in _run) and
        # check_keyword_searchability already no-op to an automatic pass
        # when the index is None, so nothing else needs to branch on this.
        self._search_index = search_index if retrieval_gate_enabled else None
        # K for that check is adaptive, not fixed: each plan's own K is
        # round(len(plan_tables) * this coefficient) (see _judge_plans), so a
        # 3-table plan is held to a wider net than a 1-table one.
        self._keyword_search_top_k_coefficient = keyword_search_top_k_coefficient
        # Whether the pre-planning retrievability check (see _run) may
        # ABORT a table group as "unretrievable_group" when it can't find a
        # keyword combination that works (tasks.mcp_search.
        # gate_unretrievable_groups in the workflow yaml). False: the check
        # still runs and still hands the planner a verified anchor when one
        # is found — only the abort-on-failure half is gated.
        self._gate_unretrievable_groups = gate_unretrievable_groups
        # code_judge_count is only STORED here — the code panel itself is
        # built later, per-run, by JudgementResponseAgent (see _run's Phase
        # 3), so it's threaded through there rather than constructed here.
        # plan_judge_count is used immediately below.
        self._code_judge_count = code_judge_count
        self.max_judge_iterations = max_judge_iterations
        # How many independent query plans (each with its own question) are
        # requested per run. See ``QueryPlanner.plan_batch``.
        self.num_query_plans = max(1, int(num_query_plans))
        self.languages = languages if languages is not None else ["English"]
        # Workflow-wide seed: sample rows shown to the LLM are drawn with it,
        # so sibling workflows sharing a yaml seed see the same rows.
        self.seed = seed

        # LLM generation client + the components that reuse it.
        self._client = client or LLMClientStatementGenerator(config_path)
        # TableAnalyzer owns its own dedicated client (config_path) rather than
        # sharing self._client — see table_analyzer.py's module docstring for why.
        # ``analysis_cache_path`` points at the per-portal JSON cache of table
        # descriptions/keywords (table id -> model -> analysis), so a table
        # already analysed by the configured model is served from disk.
        self._analyzer = analyzer or TableAnalyzer(
            config_path, cache_path=analysis_cache_path
        )
        self._planner = planner or QueryPlanner(config_path, kind)
        self._generator = generator or GenerationCoordinator()
        self._validator = validator or LLMStatementValidator(config_path, kind)
        # Plan judge panel (judge_profiles.plan in the LLM yaml): N small
        # models vote each structured plan BEFORE code generation, in six
        # independent layers — question quality, step alignment, table
        # usage, result coherence, metric-combination soundness, and topic
        # linkage — each majority-aggregated on its own; the plan passes
        # only when EVERY layer majority approves (see
        # JudgePanel._aggregate). Unconfigured (older yamls) -> plans flow
        # through unjudged, as before.
        #
        # Two layers were deliberately retired: DIFFICULTY (it is EFFORT,
        # computed deterministically by difficulty_estimator and reconciled
        # before judging — never an opinion) and CONVERGENCE (subsumed by
        # result coherence: branches left uncombined are exactly a declared
        # result that fails to account for every analysis).
        self._plan_panel = JudgePanel(
            config_path,
            "plan",
            response_model="plan_judge",
            vote_fields=[
                "question_approval",
                "plan_approval",
                "table_usage_approval",
                "expected_result_approval",
                "metric_combination_approval",
                "topic_linkage_approval",
            ],
            judge_count=plan_judge_count,
        )

        self._log = PipelineLogger()

    def _adaptive_top_k(self, plan_tables: list) -> int:
        """The retrievability target window for a plan touching these tables.

        Single-table plans target literal RANK 1, not a window — with only
        one target, "near the top" has no reason to settle for less, and a
        greedy search over a table's own real vocabulary reaching rank 1
        costs no more than reaching rank 9 (see keyword_suggestion.py's
        module docstring on why this fitness search is correct by
        construction, not a heuristic estimate).

        Multi-table plans keep the coefficient-scaled window
        (``round(len(plan_tables) * keyword_search_top_k_coefficient)``):
        two or more tables cannot all occupy rank 1 of the SAME search
        simultaneously, so widening the window with table count is the
        actual generalization of "as high as possible" once more than one
        target is in play, not a looser standard.

        Shared by both the pre-planning suggestion (_run) and the plan
        judge's reactive keyword-searchability check (_judge_plans) so the
        two never compute a different target for the same plan.
        """
        if len(plan_tables) <= 1:
            return 1
        return max(1, round(len(plan_tables) * self._keyword_search_top_k_coefficient))

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
        max_rows: Optional[int] = None,
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
           bounded by ``max_judge_iterations``.
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
                    limit_to_n_rows=max_rows,
                    seed=self.seed,
                )
                dfs.append(df)
                infos.append(dataset_info)
                # Cheap, LLM-free per-table statistics (Requirement 7), including
                # raw missing-value/bad-token signal for the planner's `clean` step.
                stats.append(
                    ColumnStatistics.compute(
                        df, alias=f"Table_{idx}", bad_tokens=self.bad_tokens
                    )
                )
                total_columns += len(df.columns)
                columns_by_table.append(
                    {
                        "table_alias": f"Table_{idx}",
                        "columns_provided": df.columns.tolist(),
                    }
                )

            avg_cols = total_columns / len(dataset_paths) if dataset_paths else 0

            # ── Insufficient-data guard ────────────────────────────────────
            # Runs before any LLM call: generating/validating against an
            # (almost) empty table only burns tokens on doomed queries.
            underfilled = [
                f"Table_{idx} ({Path(dataset_paths[idx]).stem}): "
                f"{len(df_)} row(s) after preparation"
                for idx, df_ in enumerate(dfs)
                if len(df_) < MIN_PREPARED_ROWS
            ]
            if underfilled:
                msg = (
                    "Insufficient data after preparation (minimum "
                    f"{MIN_PREPARED_ROWS} rows per table): "
                    + "; ".join(underfilled)
                )
                self._log.error(msg)
                return self._assemble_result(
                    mode=mode,
                    kind=kind,
                    all_approved_executed=[],
                    all_approved_query_dicts={},
                    judge=None,
                    original_ids=set(),
                    columns_by_table=columns_by_table,
                    all_tokens={
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    all_errors=[msg],
                    last_model="",
                    total_time=0.0,
                    avg_cols=avg_cols,
                    status_override="insufficient_rows",
                )

            # ── Pre-planning retrievability check ───────────────────────────
            # Runs before ANY LLM call (table analysis, planning): given the
            # exact table group this run is fixed to (see
            # orqa.agent.utility.keyword_suggestion), empirically search for
            # a keyword set that surfaces every one of these tables within
            # the SAME adaptive top-K the plan judge's keyword-searchability
            # gate will later check (see _judge_plans) — not a guess, a
            # verified result from the real reverse index. Two outcomes:
            #   - Found: handed to the planner as a proven-retrievable
            #     anchor (see QueryPlanner._render_retrievable_keywords), so
            #     the run starts correct instead of discovering
            #     retrievability by trial and error across correction
            #     rounds.
            #   - Not found even after an exhaustive search: BM25 scoring is
            #     a deterministic function of literal term overlap with a
            #     table's own indexed text, so if no combination of the
            #     tables' own real title/tags/columns/publisher vocabulary
            #     can win, no natural-language question could either — this
            #     group aborts HERE, before spending a single planning/
            #     judging token on a run doomed the same way the reactive
            #     gate would eventually catch it, just far more cheaply.
            retrievable_keywords: Optional[list[str]] = None
            if self._search_index is not None:
                plan_tables = [
                    {"alias": alias, "resource_id": aliases.get(alias, alias)}
                    for alias in aliases
                ]
                adaptive_top_k = self._adaptive_top_k(plan_tables)
                suggestion = suggest_retrievable_keywords(
                    plan_tables, self._search_index, adaptive_top_k
                )
                if suggestion["achieved"]:
                    retrievable_keywords = suggestion["keywords"]
                    self._log.info(
                        "Pre-planning retrievability: verified keywords "
                        f"{suggestion['keywords']} surface every table "
                        f"within top {adaptive_top_k} at ranks "
                        f"{suggestion['ranks']}"
                        + (
                            " (needed column names too, not just title/tags/publisher)"
                            if suggestion["used_fallback_fields"] else ""
                        ) + "."
                    )
                else:
                    msg = (
                        f"No keyword combination (searched {suggestion['iterations_used']} "
                        f"candidates over title/tags/columns/publisher) surfaces "
                        f"{', '.join(suggestion['missing_tables'])} within top "
                        f"{adaptive_top_k} — this table group cannot be jointly "
                        "retrieved by any natural-language question, regardless "
                        "of phrasing."
                    )
                    # Whether this actually ABORTS the group is a config
                    # choice (tasks.mcp_search.gate_unretrievable_groups) —
                    # the finding itself (no working keyword combination
                    # exists) is unconditional and always logged; only the
                    # decision to skip planning over it is gated. When not
                    # gated, planning proceeds WITHOUT a verified anchor
                    # (retrievable_keywords stays None) — the pre-existing
                    # behavior from before this check existed.
                    if self._gate_unretrievable_groups:
                        self._log.error(msg + " Aborting (gate_unretrievable_groups: true).")
                        return self._assemble_result(
                            mode=mode,
                            kind=kind,
                            all_approved_executed=[],
                            all_approved_query_dicts={},
                            judge=None,
                            original_ids=set(),
                            columns_by_table=columns_by_table,
                            all_tokens={
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0,
                            },
                            all_errors=[msg],
                            last_model="",
                            total_time=0.0,
                            avg_cols=avg_cols,
                            status_override="unretrievable_group",
                        )
                    self._log.warning(
                        msg + " Proceeding without a verified keyword anchor "
                        "(gate_unretrievable_groups: false) — the reactive "
                        "keyword-searchability gate may still reject "
                        "individual plans later."
                    )

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

            # Fold the table-analysis enrichment into the base prompt ONCE,
            # before the per-plan loop: it is identical for every plan in the
            # run, so keeping it in this shared prefix (instead of appending
            # it after each call's unique plan section, as the client used to)
            # lets the provider's prompt cache serve it on every generation
            # call after the first.
            base_prompt = self._client.enrich_prompt_with_table_analysis(
                base_prompt, analyses
            )

            # ── Phase 2b: structured planning — SEVERAL independent plans ──
            # Rather than one plan with one question, ``plan_batch`` asks for
            # ``num_query_plans`` distinct plans in one LLM call. Each plan has
            # its own question/question_keywords/steps.
            _t = time.perf_counter()
            plans = self._planner.plan_batch(
                analyses,
                aliases,
                match,
                involved_cols,
                stats,
                self.languages,
                num_plans=self.num_query_plans,
                dfs=dfs,
                retrievable_keywords=retrievable_keywords,
            )
            planning_ms = (time.perf_counter() - _t) * 1000.0

            # Log every produced plan (question, ordered steps) before the
            # judge loop touches them.
            for plan_item in plans:
                self._log.query_plan(plan_item.model_dump())

            all_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            all_errors: list = []
            last_model = ""
            generation_ms = 0.0
            pending_queries: list = []

            # ── Phase 2b': plan judge loop (majority voting + correction) ──
            # Each plan is voted on by the configured small-model panel before
            # any code is generated for it. A rejected plan is revised by the
            # planner against the panel's feedback and re-judged, up to
            # MAX_PLAN_CORRECTIONS rounds; only then is it given up on. The
            # full attempt history is preserved under the run's
            # `plan_feedback`, so generation effort is only spent on plans the
            # panel approved.
            plan_feedback: list = []
            plan_judging_ms = 0.0
            plan_abort_status: Optional[str] = None
            # Every judged plan's own attempts, keyed by id(plan object) —
            # see _judge_plans's Returns doc. Threaded down to _assemble_result
            # so each approved/failed query can carry the SPECIFIC plan
            # judging history that produced it, alongside plan_by_client_id.
            plan_attempts_by_id: dict = {}
            if plans and self._plan_panel.is_configured:
                _t = time.perf_counter()
                plans, plan_feedback, panel_usage, plan_abort_status, plan_attempts_by_id = (
                    self._judge_plans(
                        plans, analyses, aliases, stats,
                        match=match, involved_cols=involved_cols, dfs=dfs,
                        retrievable_keywords=retrievable_keywords,
                    )
                )
                plan_judging_ms = (time.perf_counter() - _t) * 1000.0
                self._accumulate_tokens(all_tokens, panel_usage)
            # Per-query lookup so assembly (Phase 4) can attach the SPECIFIC
            # plan that produced each query, rather than one plan for the
            # whole run. Keyed by the opaque client_id the generator assigns
            # (survives the validator/judge loop and stays on the query dict
            # through to the approved result).
            plan_by_client_id: dict = {}

            for plan_idx, plan_item in enumerate(plans, start=1):
                # ── Phase 2c: code generation for this plan (client_id
                # contract). NOTE (seam for the full generator migration):
                # GenerationCoordinator augments ``base_prompt`` (via
                # build_generation_prompt) with this plan's steps/statistics,
                # enforces the client_id contract, then delegates the actual
                # completion to the legacy generation client. The client still performs its
                # own internal analysis/planning today; migrating generation
                # fully off that path is tracked separately and is out of
                # scope here. The unified pipeline performs the single batched
                # analysis and the multi-plan planning above.
                def _generate_fn(built_prompt: str, _plan=plan_item):
                    return self._client.complete(
                        built_prompt,
                        dfs,
                        aliases,
                        typology=kind,
                        involved_cols=involved_cols,
                        matches=match,
                        metadata=metadata,
                        languages=self.languages,
                        # Reuse THIS plan (not a fresh one) — skips the legacy
                        # client's own redundant query-planner LLM call, which
                        # would otherwise run once per plan in the batch and
                        # inject a second, competing plan into the prompt
                        # alongside the one GenerationCoordinator already added.
                        precomputed_plan=_plan.model_dump(),
                    )

                start = time.perf_counter()
                query_set, tokens, errors, model = self._generator.generate(
                    base_prompt,
                    _generate_fn,
                    plan=plan_item,
                    stats=stats,
                )
                generation_ms += (time.perf_counter() - start) * 1000.0

                self._accumulate_tokens(all_tokens, tokens)
                all_errors.extend(errors or [])
                last_model = model or last_model

                plan_queries = query_set.get("queries", []) or []
                if not plan_queries:
                    reason = "; ".join(errors) if errors else "no errors reported by the generation client"
                    self._log.warning(
                        f"Plan {plan_idx}/{len(plans)} (\"{plan_item.question}\") "
                        f"produced 0 queries — dropped silently otherwise. Reason: {reason}"
                    )
                for q in plan_queries:
                    client_id = q.get("client_id") if isinstance(q, dict) else None
                    if client_id:
                        plan_by_client_id[client_id] = plan_item
                pending_queries.extend(plan_queries)

            total_time = generation_ms / 1000.0

            # Per-phase timings accumulated across the run (Requirement 21.3).
            timings_ms: dict = {
                "analysis": analysis_ms,
                "planning": planning_ms,
                "plan_judging": plan_judging_ms,
                "generation": generation_ms,
                "validation": 0.0,
                "judging": 0.0,
            }

            self._log.step1_generated(pending_queries)

            if not pending_queries:
                return self._assemble_result(
                    mode=mode,
                    kind=kind,
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
                    plan=plans[0] if plans else None,
                    plan_by_client_id=plan_by_client_id,
                    plan_attempts_by_id=plan_attempts_by_id,
                    timings_ms=timings_ms,
                    plan_feedback=plan_feedback,
                    status_override=plan_abort_status,
                )

            original_ids = {str(q.get("id")) for q in pending_queries}

            # ── Phase 3: validator-to-judge loop ───────────────────────────
            executor = QueryExecutor(
                datasets_path=Path(dataset_paths[0]).parent,
                extension=Path(dataset_paths[0]).suffix.lstrip("."),
            )
            entry = {"tables": aliases}
            # Code execution — static validation AND the code judge — runs
            # against every ROW of each table (only utils.clean_columns
            # applied), so an aggregate/join is never approved on a truncated
            # row slice and then found wrong at benchmark time.
            #
            # COLUMNS are clamped to the per-table allow list: exactly the
            # columns the Phase-1 view exposed to analysis / statistics /
            # planning / generation (``dfs[idx].columns`` — the involved
            # columns plus the file-order fill up to ``max_cols``). This is a
            # SUPERSET of each query's declared ``columns_involved``, so it
            # does not reintroduce the "execute on a columns_involved-sliced
            # frame" bug QueryExecutor's docstring warns about — a query may
            # still freely use any column it was shown. What it can no longer
            # do is reach a real-but-UNSHOWN column: a hallucinated / confused
            # name now lands as a KeyError -> correction instead of silently
            # resolving to unvetted data. Derived columns are created by the
            # generated code at runtime, not loaded, so they are unaffected.
            column_allow_list = {
                alias: list(view_df.columns)
                for alias, view_df in zip(aliases, dfs)
            }
            execution_dataframes = {
                alias: df[[c for c in column_allow_list.get(alias, df.columns) if c in df.columns]]
                for alias, df in executor.load_tables(aliases).items()
            }
            prepared_dataframes = execution_dataframes
            execution_dfs = [execution_dataframes[a] for a in aliases]
            # Divergence: set the judge single_table flag in single mode (Req 1.5).
            judge = JudgementResponseAgent(
                self.config_path,
                kind,
                executor,
                entry,
                single_table=(mode == SINGLE),
                dataframes=prepared_dataframes,
                code_judge_count=self._code_judge_count,
            )

            (
                all_approved_executed,
                all_approved_query_dicts,
                loop_time,
                loop_timings,
                failed_queries,
                empty_result_pending,
            ) = self._validator_judge_loop(
                pending_queries=pending_queries,
                dfs=execution_dfs,
                aliases=aliases,
                table_schemas=table_schemas,
                judge=judge,
                all_tokens=all_tokens,
                all_errors=all_errors,
                plan_by_client_id=plan_by_client_id,
            )
            total_time += loop_time
            timings_ms["validation"] = loop_timings.get("validation", 0.0)
            timings_ms["judging"] = loop_timings.get("judging", 0.0)

            # ── Phase 3b: empty-result plan-level retry ────────────────────
            # A query the code panel found PLAN-COMPLIANT but that executed
            # to an EMPTY result (0 rows) is usually a plan-level symptom,
            # not a code bug — escalate it back to plan revision instead of
            # the normal code-correction loop. `empty_result_pending` is the
            # explicit, vote-driven signal from _validator_judge_loop
            # (plan_compliance_approval true, present_result_approval false);
            # _retry_empty_results also defensively re-scans
            # all_approved_executed for any 0-row result that still slipped
            # through as fully approved (belt-and-suspenders, see its
            # docstring). One retry per query; still empty (or the revised
            # plan can't be re-approved) -> permanently failed.
            all_approved_executed, all_approved_query_dicts, failed_queries = (
                self._retry_empty_results(
                    all_approved_executed, all_approved_query_dicts, failed_queries,
                    empty_result_pending,
                    plan_by_client_id, plan_feedback, plan_attempts_by_id,
                    analyses, aliases, stats, match, involved_cols, dfs,
                    retrievable_keywords, base_prompt, kind,
                    metadata, judge, all_tokens, all_errors,
                )
            )

            # ── Phase 4: final assembly ────────────────────────────────────
            return self._assemble_result(
                mode=mode,
                kind=kind,
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
                plan=plans[0] if plans else None,
                plan_by_client_id=plan_by_client_id,
                plan_attempts_by_id=plan_attempts_by_id,
                timings_ms=timings_ms,
                failed_queries=failed_queries,
                plan_feedback=plan_feedback,
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

        # Assign the accumulated descriptions ONCE (no per-iteration mutation).
        prompt._datasets_descriptions = "".join(f"\n{b}" for b in full_blocks)
        prompt._light_datasets_descriptions = "".join(f"\n{b}" for b in light_blocks)

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

    # ------------------------------------------------------------------
    # Plan judge panel (majority voting)
    # ------------------------------------------------------------------

    def _judge_plans(
        self,
        plans: list,
        analyses: list,
        aliases: dict,
        stats: list,
        match: Any = None,
        involved_cols: Optional[dict] = None,
        dfs: Optional[list] = None,
        retrievable_keywords: Optional[list[str]] = None,
    ) -> tuple[list, list, dict, Optional[str], dict]:
        """Plan judge loop: layered majority vote, then correct-and-re-judge.

        ``retrievable_keywords``: the pre-verified anchor computed once in
        ``_run`` before any plan existed (see
        ``orqa.agent.utility.keyword_suggestion``) — re-passed into every
        ``revise_plan`` correction call so a revision never drifts away from
        it, even though this loop's OWN deterministic keyword-searchability
        layer below (a per-plan check, since each plan can propose its own
        ``question_keywords``) is a separate, independent verification of
        the plan's ACTUAL final keywords, not a re-check of this anchor.

        Each plan is sent (with the table analyses and per-table columns as
        context) to every judge in ``judge_profiles.plan``. Every judge casts
        SIX independent votes — ``question_approval`` (realistic,
        average-user, keyword-retrievable question), ``plan_approval`` (the
        steps produce exactly what the question asks), ``table_usage_approval``
        (every table justified by the question), ``expected_result_approval``
        (the declared result is the natural conclusion of the steps and
        accounts for every analysis the plan performs),
        ``metric_combination_approval``, and ``topic_linkage_approval`` (the
        question names the table's specific program/initiative when its
        identity hinges on one) — and each layer is majority-aggregated
        on its own; the
        plan passes only when EVERY layer majority approves (see
        ``JudgePanel._aggregate``). A rejected plan is NOT dropped immediately: the
        panel's aggregated feedback/suggestions are handed back to the planner
        (:meth:`QueryPlanner.revise_plan`), and the revised plan is re-judged —
        up to ``MAX_PLAN_CORRECTIONS`` correction rounds per plan. Only a plan
        still rejected after its last round is given up on.

        Every attempt (question as judged, verdict, per-judge votes) is
        recorded under the run's ``plan_feedback`` so the whole correction
        history stays traceable.

        A run is normally never left with zero plans: if every plan exhausts
        its corrections, the one whose final round drew the most approve votes
        is kept (flagged ``kept_as_fallback``) — EXCLUDING any plan whose final
        rejection was keyword-searchability-driven (see below); that gate is
        a real query against the reverse index, not a judge opinion, so a
        plan that failed it is provably unretrievable and must never be the
        fallback kept just to avoid an empty run. There are two dead-end
        exceptions where zero plans are kept on purpose:
        - TABLE-DRIVEN: when every plan's final table-usage layer was
          majority-rejected (the panel found a table no question can
          justify), no amount of code generation can succeed — the validator
          forces every table into the code while the judges reject the
          forced join — so the run aborts with ``abort_status =
          "unjustifiable_group"``.
        - KEYWORD-DRIVEN: when every plan's final round failed the keyword-
          searchability gate, no candidate is safe to ship even as a
          fallback — the run aborts with ``abort_status =
          "unretrievable_group"``.

        Returns:
            ``(approved_plans, plan_feedback, usage_total, abort_status,
            plan_attempts_by_id)`` where ``abort_status`` is ``None``
            normally, ``"unjustifiable_group"`` on the table-driven dead end,
            and ``"unretrievable_group"`` on the keyword-driven dead end
            above, and ``plan_attempts_by_id`` maps ``id(plan_object)``
            (for every plan object that ends up in ``approved_plans``,
            including the fallback-kept one) to that plan's own ``attempts``
            list — so a caller holding a plan object (e.g. via
            ``plan_by_client_id``) can look up exactly which judging rounds
            produced it, keyed by object identity rather than a positional
            index that shifts whenever a plan is dropped.
        """
        # Built once — identical for every plan/attempt in this run.
        plan_judge_instructions = PlanJudgementPrompt().update()

        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        plan_feedback: list = []
        approved_plans: list = []
        scored: list = []
        # Every plan's own attempts, keyed by id(plan object) — see Returns.
        plan_attempts_by_id: dict = {}
        # Whether each plan's FINAL rejection was table-driven (see docstring).
        table_driven_rejections: list = []
        # Whether each plan's FINAL rejection was keyword-searchability-driven
        # (see docstring and the fallback-selection guard below).
        keyword_driven_rejections: list = []

        for plan_idx, plan in enumerate(plans, start=1):
            current = plan
            attempts: list = []
            approved = False
            last_panel: dict = {}
            last_judgment: dict = {}

            for attempt in range(1 + MAX_PLAN_CORRECTIONS):
                self._log.plan_judge_attempt(
                    plan_idx, len(plans), attempt + 1, 1 + MAX_PLAN_CORRECTIONS,
                    getattr(current, "question", ""),
                )
                payload = self._build_plan_judge_payload(
                    current, analyses, aliases, stats
                )
                # max_tokens mirrors the code panel's cap: without it the plan
                # judges run on the provider's default, and a reasoning model
                # (gemini-2.5-flash) that thinks past that cap returns no
                # message at all (see JUDGE_MAX_TOKENS in agent.py).
                judgment, usage = self._plan_panel.evaluate(
                    plan_judge_instructions, payload, max_tokens=JUDGE_MAX_TOKENS
                )
                self._accumulate_tokens(usage_total, usage)

                # Deterministic 7th layer, computed once (not voted by the
                # LLM panel — it has a ground-truth answer): would keyword-
                # searching this question's `question_keywords` against the
                # portal's real reverse index actually surface every table
                # this plan uses? Folds straight into `judgment["approved"]`
                # since it's not part of JudgePanel's own aggregation.
                plan_tables = [
                    {
                        "alias": t.name,
                        "resource_id": aliases.get(t.name, t.name),
                        # The table's OWN analysis keywords — not used for
                        # matching, only so a miss's feedback can name the
                        # actual vocabulary to draw from instead of leaving
                        # the planner to guess synonyms blind (see
                        # check_keyword_searchability's docstring).
                        "keywords": list(getattr(t, "keywords", None) or []),
                    }
                    for t in (getattr(current, "tables", None) or [])
                ]
                # Adaptive K: a plan joining more tables needs a wider net for
                # ALL of them to plausibly surface together, so K scales with
                # THIS plan's own table count rather than being one fixed
                # constant regardless of how many tables it uses.
                adaptive_top_k = self._adaptive_top_k(plan_tables)
                kw_result = check_keyword_searchability(
                    getattr(current, "question_keywords", None) or [],
                    plan_tables,
                    self._search_index,
                    adaptive_top_k,
                )
                judgment["keyword_searchability_approval"] = kw_result["approved"]
                self._log.info(
                    f"Keyword-search retrieval: "
                    f"{'OK' if kw_result['approved'] else 'MISS ' + ', '.join(kw_result['missing_tables'])}"
                )
                # The EXACT text this gate sends the planner on rejection —
                # built ONCE so both the stored attempt (the web UI's
                # "Retrieval Gate" card in the judge panel) and the actual
                # revision prompt below carry IDENTICALLY the same feedback,
                # never a UI-only paraphrase of what the planner really saw.
                kw_planner_feedback = (
                    kw_result["feedback"] + (
                        " Reword the QUESTION ITSELF — weave the missed "
                        "table's REAL indexed title/tags terms (and/or the "
                        "competing titles') naturally into the question's "
                        "own prose, not just the `question_keywords` list "
                        "— using the exact wording the reverse index "
                        "actually matches on, not a paraphrase of it (a "
                        "topic worded differently, e.g. as one merged word "
                        "instead of the indexed multi-word tag, will not "
                        "match even if it means the same thing). Editing "
                        "`question_keywords` alone while leaving the "
                        "question's own wording unchanged does not fix "
                        "this: `question_keywords` must stay a faithful "
                        "set of terms that literally appear in (or are "
                        "directly implied by) the question text, not a "
                        "wishlist bolted on separately from it. Do not "
                        "drop or swap out the affected table(s)."
                    )
                ) if not kw_result["approved"] else ""
                # The panel's OWN majority verdict, captured BEFORE the gate
                # can override it — kept separate so the correction feedback
                # built below can tell "the judges themselves rejected this"
                # apart from "the judges liked it but the gate didn't", and
                # never blend the panel's positive "looks sound" text into a
                # rejection the panel itself never voted for.
                panel_approved = bool(judgment.get("approved", False))
                judgment["approved"] = panel_approved and kw_result["approved"]
                if not kw_result["approved"]:
                    for field in ("response", "translated_response"):
                        if field in judgment:
                            judgment[field] = ""

                last_panel = judgment.get("panel", {})
                last_judgment = judgment
                approved = bool(judgment.get("approved", False))
                # Named layers a human reviewer can see at a glance without
                # opening every judge's full check text (see
                # _summarize_judge_outcome).
                failed_layers = [
                    label
                    for field, label in (
                        ("question_approval", "question"),
                        ("plan_approval", "plan"),
                        ("table_usage_approval", "tables"),
                        ("expected_result_approval", "expected-result"),
                        ("metric_combination_approval", "combination"),
                        ("topic_linkage_approval", "topic-linkage"),
                        ("keyword_searchability_approval", "keyword-search"),
                    )
                    if judgment.get(field) is False
                ]
                attempts.append(
                    {
                        # Scannable first: what happened, in one line, before
                        # the full plan/panel detail below.
                        "attempt": attempt + 1,
                        "approved": approved,
                        "summary": self._summarize_judge_outcome(
                            "approved" if approved else "rejected",
                            last_panel, failed_layers=failed_layers,
                        ),
                        "question": getattr(current, "question", ""),
                        "feedback": judgment.get("feedback", ""),
                        "suggestions": judgment.get("suggestions", ""),
                        # Per-layer majority results (see JudgePanel vote_fields).
                        "question_approval": judgment.get("question_approval"),
                        "plan_approval": judgment.get("plan_approval"),
                        "table_usage_approval": judgment.get("table_usage_approval"),
                        "expected_result_approval": judgment.get("expected_result_approval"),
                        "metric_combination_approval": judgment.get("metric_combination_approval"),
                        "topic_linkage_approval": judgment.get("topic_linkage_approval"),
                        "keyword_searchability_approval": judgment.get("keyword_searchability_approval"),
                        # Which of THIS round's tables missed the retrieval
                        # (empty on approval) — the "Retrieval Gate" card in
                        # the web UI's judge panel reads these directly
                        # rather than re-deriving them client-side.
                        "keyword_searchability_missing_tables": kw_result["missing_tables"],
                        # The constructed feedback (see kw_planner_feedback
                        # above) — empty on approval, otherwise EXACTLY the
                        # text this round actually sent (or would send) to
                        # the planner, not just the raw retrieval-index miss.
                        "keyword_searchability_feedback": kw_planner_feedback,
                        # The exact plan version judged THIS round — captured
                        # before any revision below reassigns `current`, so a
                        # rejected round's proposal is preserved even after
                        # the planner rewrites it for the next attempt.
                        "proposed_plan": (
                            current.model_dump() if hasattr(current, "model_dump")
                            else current
                        ),
                        "panel": last_panel,
                    }
                )

                # Every judge's vote, feedback and suggestion for this attempt.
                self._log.panel_votes(last_panel, level=1)
                self._log.plan_verdict(approved, getattr(current, "question", ""))

                if approved or attempt >= MAX_PLAN_CORRECTIONS:
                    break

                # Hand the panel's feedback back to the planner for a
                # revision. When the panel itself rejected the plan, start
                # from ITS feedback (as before); when the panel actually
                # approved and only the deterministic keyword-search gate
                # disagreed, the panel has nothing useful to say — starting
                # from its "looks sound" text would only confuse the
                # revision prompt, so leave it out and let the gate's own
                # measured miss (appended below) be the entire feedback.
                if panel_approved:
                    feedback_text = ""
                else:
                    feedback_text = judgment.get("feedback", "")
                    suggestions = judgment.get("suggestions", "")
                    if suggestions:
                        feedback_text = f"{feedback_text}\nSuggestions: {suggestions}"
                    # Per-layer revision directives: each failed layer names
                    # the exact artifact the planner must change, so the
                    # revision doesn't churn the parts the panel already
                    # accepted.
                    if judgment.get("question_approval") is False:
                        feedback_text += (
                            "\nThe question itself was rejected: rewrite it as an "
                            "average, non-technical user with no knowledge of the "
                            "underlying data would phrase it, while weaving in the "
                            "tables' distinctive keyword vocabulary (topic, entity "
                            "type, place, period) so it stays retrievable."
                        )
                    # A failed table-usage layer has exactly one legal fix —
                    # the question, never the table set (the generated code
                    # is required to use every table). Make that explicit so
                    # the planner reframes instead of trying to drop the table.
                    if judgment.get("table_usage_approval") is False:
                        flagged = ", ".join(judgment.get("unjustified_tables") or [])
                        feedback_text += (
                            "\nTable usage was rejected"
                            + (f" for: {flagged}" if flagged else "")
                            + ". Reframe the QUESTION so every provided table "
                            "becomes genuinely necessary to answer it — do NOT "
                            "drop or ignore any table."
                        )
                    # A failed result layer means the declared result is not the
                    # coherent conclusion of the steps — either the shape
                    # contradicts the question/steps, or the steps compute
                    # something the declaration never accounts for (the case the
                    # retired convergence layer used to own).
                    if judgment.get("expected_result_approval") is False:
                        feedback_text += (
                            "\nThe declared result was rejected. Either: (a) set "
                            "expected_result_type to the shape the question "
                            "actually asks for (number/boolean/text/list/table) "
                            "and make expected_result_description concretely "
                            "describe that result — the declared type is "
                            "mechanically enforced against the executed result "
                            "downstream, so it must be right; or (b) the steps "
                            "compute a result the declaration never accounts "
                            "for — every analysis the plan performs must land in "
                            "the declared result. Fix that by folding it in (a "
                            "join/union on a shared key, a correlation, a "
                            "comparison or ranking across the branches, or a "
                            "final step reporting on both together — no "
                            "mechanism is mandatory), by widening the "
                            "declaration to honestly cover both, or by dropping "
                            "the step that earns nothing. Never drop a branch, "
                            "and never reword the question instead of fixing the "
                            "steps or the declaration."
                        )
                    # NOTE: there is deliberately no difficulty directive here.
                    # Difficulty is EFFORT, computed deterministically by
                    # `difficulty_estimator.estimate_plan_tier` and reconciled
                    # (and, failing that, stamped) by
                    # `QueryPlanner._reconcile_difficulty` BEFORE the panel ever
                    # sees the plan. The judge no longer votes on it, so
                    # `build_reconciliation_feedback` is the single voice that
                    # tells the planner about difficulty.
                    # NOTE: no convergence directive here either. Branches left
                    # uncombined ARE a declared result that fails to account for
                    # every analysis, so the expected-result directive above
                    # owns that case — see PlanJudgment's docstring.
                    # A failed metric-combination layer means a step blends
                    # figures from 2+ tables into one value in a way that
                    # isn't dimensionally sound — the fix is in HOW the
                    # figures combine (separate columns, or a ratio/rate),
                    # never in dropping a table (that's table_usage's job).
                    if judgment.get("metric_combination_approval") is False:
                        feedback_text += (
                            "\nMetric combination was rejected: a step sums or "
                            "otherwise arithmetically blends raw figures from "
                            "different tables that are on incommensurate units/"
                            "scales, or folds different time periods' figures "
                            "into one undifferentiated total. Report the "
                            "components as separate columns instead of summing "
                            "them, or replace the additive combination with a "
                            "dimensionally-sound rate/ratio/normalized index. "
                            "Comparing figures across time periods is fine when "
                            "that comparison is the point — only collapsing them "
                            "into one opaque sum is not."
                        )
                    # A failed topic-linkage layer means the question only
                    # describes the generic activity the table falls under,
                    # never the specific named program/initiative that
                    # actually distinguishes the table — a DIFFERENT flaw
                    # from a failed question_approval (which is about
                    # phrasing/retrievability, not topic identity), so the
                    # fix is always in the QUESTION's own prose, never in
                    # question_keywords alone.
                    if judgment.get("topic_linkage_approval") is False:
                        feedback_text += (
                            "\nTopic linkage was rejected: the question only "
                            "describes the generic activity this table happens "
                            "to record, never the specific named program/"
                            "initiative/agency that actually distinguishes it — "
                            "so a reader could just as plausibly think it's "
                            "about a different, unrelated program. Reword the "
                            "QUESTION itself (not just question_keywords) to "
                            "name that specific program/initiative, drawn from "
                            "the table's own description/keywords, while still "
                            "reading like an average, non-technical user's "
                            "question."
                        )
                # A failed keyword-searchability layer is a real, measured
                # miss — not a guess — against the portal's reverse index:
                # this question's keywords do not surface every table the
                # plan uses. Only the question/its keywords can fix this
                # (the table set stays forced, same as table_usage_approval).
                # Appended regardless of panel_approved above — this is the
                # ONLY feedback line when the panel approved and the gate was
                # the sole rejection reason. Reuses kw_planner_feedback
                # verbatim (built above) so the stored attempt's
                # keyword_searchability_feedback is never out of sync with
                # what the planner is actually handed here.
                if judgment.get("keyword_searchability_approval") is False:
                    feedback_text += ("\n" if feedback_text else "") + kw_planner_feedback
                revised, rev_usage = self._planner.revise_plan(
                    current,
                    feedback_text,
                    analyses,
                    aliases,
                    match,
                    involved_cols,
                    stats,
                    self.languages,
                    dfs=dfs,
                    retrievable_keywords=retrievable_keywords,
                )
                self._accumulate_tokens(usage_total, rev_usage)
                if revised is None:
                    self._log.warning(
                        f"Plan {plan_idx}: revision failed structural validation "
                        "twice — stopping the correction loop for this plan."
                    )
                    break
                self._log.plan_revised(
                    getattr(current, "question", ""),
                    getattr(revised, "question", ""),
                )
                self._log.query_plan(revised.model_dump())
                current = revised

            # Table-driven when the table-usage layer's MAJORITY failed on
            # the final attempt (authoritative, from the layered vote); the
            # votes-based fallback only covers a panel without vote_fields.
            if "table_usage_approval" in last_judgment:
                table_driven = (
                    not approved
                    and last_judgment.get("table_usage_approval") is False
                )
            else:
                table_driven = (
                    not approved and self._rejection_is_table_driven(last_panel)
                )
            table_driven_rejections.append(table_driven)
            # Keyword-searchability is NOT a judge opinion — it's a real
            # query against the reverse index (see check_keyword_searchability
            # above), so a plan whose FINAL round still failed it is provably
            # unretrievable by its own question's keywords, not merely
            # suspected to be. Tracked separately from table_driven so the
            # "keep the least-bad rejected plan" fallback below can never
            # resurrect one (that was the bug: a table missing from the
            # top-K still shipped because its plan happened to have the best
            # vote count among all-rejected plans).
            keyword_driven = (
                not approved
                and last_judgment.get("keyword_searchability_approval") is False
            )
            keyword_driven_rejections.append(keyword_driven)

            entry = {
                "plan_index": plan_idx,
                "question": getattr(current, "question", ""),
                "approved": approved,
                "attempts": attempts,
            }
            if table_driven:
                entry["rejected_for_unjustified_tables"] = True
            if keyword_driven:
                entry["rejected_for_unretrievable_tables"] = True
            plan_feedback.append(entry)
            scored.append((last_panel.get("approve_votes", 0), current, entry))
            # Keyed by identity so it survives regardless of whether `current`
            # is later kept outright, kept as the fallback, or dropped.
            plan_attempts_by_id[id(current)] = attempts
            if approved:
                approved_plans.append(current)

        if not approved_plans and scored:
            if table_driven_rejections and all(table_driven_rejections):
                # Table-driven dead end: the panel says at least one table can
                # never be justified by a question over this group, and the
                # validator will force that table into every query anyway.
                # Generating code would only feed the doomed
                # validator-vs-judge loop — abort the entry explicitly.
                self._log.error(
                    "The plan panel rejected every plan for unjustified "
                    "tables after all corrections — the table group cannot "
                    "ground a question that needs all of its tables. "
                    "Aborting the entry as 'unjustifiable_group'."
                )
                return [], plan_feedback, usage_total, "unjustifiable_group", plan_attempts_by_id

            # Never let the "keep the least-bad rejected plan" fallback
            # resurrect a plan that failed the deterministic keyword-search
            # gate — shipping it would silently defeat the whole point of
            # the gate (a query whose own table provably never surfaces in
            # the reverse index would reach the saved output anyway).
            retrievable = [
                s for s, kw in zip(scored, keyword_driven_rejections) if not kw
            ]
            if not retrievable:
                self._log.error(
                    "Every plan's final round failed the keyword-"
                    "searchability gate — at least one required table never "
                    "surfaced in the reverse index for any question phrasing "
                    "tried. Aborting the entry as 'unretrievable_group' "
                    "rather than shipping an unretrievable query."
                )
                return [], plan_feedback, usage_total, "unretrievable_group", plan_attempts_by_id

            best_votes, best_plan, best_entry = max(retrievable, key=lambda t: t[0])
            best_entry["kept_as_fallback"] = True
            approved_plans = [best_plan]
            self._log.warning(
                "The plan panel rejected every plan after all corrections — "
                f"keeping the least-rejected one ({best_votes} approve vote/s) "
                f"so the run isn't empty: \"{best_entry['question']}\""
            )

        return approved_plans, plan_feedback, usage_total, None, plan_attempts_by_id

    @staticmethod
    def _rejection_is_table_driven(panel: dict) -> bool:
        """True when a majority of the panel's valid votes flagged at least
        one unjustified table — i.e. the plan failed because the table group
        itself cannot be justified by the question, not because of a fixable
        question/step flaw."""
        votes = [v for v in (panel.get("votes") or []) if "error" not in v]
        flagging = [v for v in votes if v.get("unjustified_tables")]
        return bool(votes) and len(flagging) * 2 > len(votes)

    @staticmethod
    def _summarize_judge_outcome(
        outcome: str,
        panel: Optional[dict] = None,
        failed_layers: Optional[list] = None,
        detail: str = "",
    ) -> str:
        """One-line, human-scannable summary of a judge/validation event.

        Combines the outcome with the panel's vote tally when one is
        available (e.g. "REJECTED — 1/3 judges approved") and, for a layered
        (plan) panel, which named layer(s) failed — so a reviewer can skim
        `attempt_history`/plan `attempts` top-to-bottom without opening every
        judge's full text first. Falls back to a short excerpt of `detail`
        when there's no panel to tally (a validation drop or an execution
        failure never reaches a judge).
        """
        label = outcome.replace("_", " ").upper()
        votes = (panel or {}).get("votes") or []
        if votes:
            total = len(votes)
            approve = (panel or {}).get("approve_votes", 0)
            tally = f"{approve}/{total} judges approved"
            if failed_layers:
                return f"{label} — {tally}; failed: {', '.join(failed_layers)}"
            return f"{label} — {tally}"
        short_detail = (detail or "").strip().splitlines()[0][:140] if detail else ""
        return f"{label} — {short_detail}" if short_detail else label

    @staticmethod
    def _build_plan_judge_payload(
        plan: Any, analyses: list, aliases: dict, stats: list
    ) -> str:
        """Serialize one plan plus its table context for the plan judges.

        Context is intentionally compact (analyses + column names/dtypes, not
        full samples/statistics): plan panels run on small models, and the
        judged questions are feasibility and coherence, not value-level checks.
        """
        columns_by_alias = {
            table.alias: [f"{c.column} ({c.dtype})" for c in table.columns]
            for table in stats
        }
        # Deterministic structural/data-engineering facts (see
        # orqa.agent.utility.difficulty_estimator) — computed here, not
        # judge-voted, so Check 5 spends its vote only on the two things a
        # counter can't decide (pattern distinctness, bucket-logic realism)
        # instead of re-deriving step/branch counts the plan already fixes.
        estimate = estimate_plan_tier(plan)
        payload = {
            "plan": plan.model_dump() if hasattr(plan, "model_dump") else plan,
            "computed_difficulty": {
                "tier": estimate.tier,
                "structural_tier": estimate.structural_tier,
                "data_engineering_tier": estimate.data_engineering_tier,
                "dq_hard_pending_your_confirmation": estimate.dq_hard_pending_judge,
                "breakdown": estimate.explanation,
            },
            "tables": {
                "aliases": aliases,
                "analyses": list(analyses),
                "columns": columns_by_alias,
            },
        }
        return (
            "Plan to evaluate:\n"
            + json.dumps(payload, indent=2, ensure_ascii=False, default=str)
            + "\n\nEvaluate the plan above following the instructions and "
            "return only the JSON verdict."
        )

    def _validator_judge_loop(
        self,
        pending_queries: list,
        dfs: list,
        aliases: dict,
        table_schemas: str,
        judge: JudgementResponseAgent,
        all_tokens: dict,
        all_errors: list,
        plan_by_client_id: Optional[dict] = None,
    ) -> tuple[list, dict, float, dict, list, list]:
        """Run the validator-to-judge loop.

        Mirrors the legacy agents' loop: it runs at most ``max_judge_iterations``
        times. Approved queries are accumulated across iterations.

        ``plan_by_client_id`` (the same map built in Phase 2, keyed by the
        opaque ``client_id`` each query carries) is threaded into the validator
        so a query's originating skill (if any) can be re-injected into its
        correction prompt — otherwise a static-validation or judge-feedback
        correction cycle has no idea the query was ever bound to a skill and
        silently drops it (Requirement: correction mode must not lose skill
        context).

        Returns:
            ``(all_approved_executed, all_approved_query_dicts, elapsed_seconds,
            phase_times_ms, failed_queries, empty_result_pending)`` where
            ``phase_times_ms`` splits the
            wall-clock time into ``{"validation": ms, "judging": ms}`` for the
            per-phase timings recorded on each assembled query (Requirement 21.3),
            ``failed_queries`` is the list of queries that were NEVER approved
            by the time the loop ended — each a dict carrying the query's latest
            ``question``/``code``, a query-level ``status``
            (``validation_failed`` / ``judge_rejected`` /
            ``judge_rejected_permanent`` / ``execution_failure``) and the last
            error/feedback message observed for it, so failures stay traceable in
            the final saved JSON instead of being dropped silently, and
            ``empty_result_pending`` is the list of executed-result dicts the
            code panel found PLAN-COMPLIANT but whose result is genuinely
            empty (see ``JudgementResponseAgent.evaluate``'s
            ``empty_result`` bucket) — pulled out of the loop immediately
            (never re-fed as a normal code-correction retry, never counted
            against ``max_corrections``) for the caller to escalate via
            :meth:`_retry_empty_results`, exactly once per query.

        Both approved and failed query dicts additionally carry
        ``attempt_history``: the numbered trail of EVERY verdict the query
        received across the loop (validation drops, judge rejections, the
        final approval/rejection), each event as ``{attempt, outcome,
        summary, iteration, stage, detail, proposed_code, panel}`` — scannable
        fields first (``summary`` is a one-line human-readable digest, e.g.
        "REJECTED — 1/3 judges approved", built by
        ``_summarize_judge_outcome``), heavier detail after — so the saved
        JSON shows e.g. "attempt 1: judge rejected — <feedback>; attempt 2:
        approved", not just the last outcome; ``proposed_code`` is the exact
        code that earned THAT round's outcome (before whatever correction
        followed it); and ``panel`` on a judge event carries every judge's
        individual verdict for that round (same shape as the top-level
        ``code_feedback`` field), not only the blaming judges' flattened
        ``detail`` text.
        """
        all_approved_executed: list = []
        all_approved_query_dicts: dict = {}
        empty_result_pending: list = []
        elapsed = 0.0
        validation_ms = 0.0
        judging_ms = 0.0
        structured_judge_feedback: list | None = None

        # Latest failure state per query, keyed by client_id (falling back to
        # id). A query rejected on iteration 1 but approved on iteration 2 is
        # popped back out; whatever is still here when the loop ends is a
        # genuine, final failure.
        failures_by_key: dict = {}

        def _failure_key(q: dict) -> str:
            return str(q.get("client_id") or q.get("id"))

        def _record_failure(
            q: dict, status: str, error: str, panel: Optional[dict] = None
        ) -> None:
            failures_by_key[_failure_key(q)] = {
                "id": q.get("id"),
                "client_id": q.get("client_id", ""),
                "status": status,
                "question": q.get("question", ""),
                "code": q.get("code", ""),
                "error": error,
                # Per-judge record of the code panel vote that rejected this
                # query (empty for non-judge failures / single-judge fallback).
                "code_feedback": dict(panel or {}),
            }

        # Full per-query outcome trail across the loop iterations, keyed like
        # failures_by_key. Every validation drop and every judge verdict
        # appends one numbered event, so the saved JSON carries the whole
        # correction story ("attempt 1: rejected — <feedback>; attempt 2:
        # approved"), not just each query's final outcome. Each event also
        # carries the full structured judge panel (same shape as the
        # top-level `code_feedback` field — every judge's individual
        # approved/feedback/requirements_check/result_check/violated_criteria/
        # suggestions) when the event came from a judge verdict, not just the
        # `detail` one-liner — so a rejected-then-approved query's saved JSON
        # shows exactly what every judge said on every round, not only the
        # final one.
        history_by_key: dict = {}

        def _record_event(
            q: dict,
            iteration: int,
            stage: str,
            outcome: str,
            detail: str = "",
            panel: Optional[dict] = None,
        ) -> None:
            events = history_by_key.setdefault(_failure_key(q), [])
            events.append(
                {
                    # Scannable first: what happened, in one line, before the
                    # full code/panel detail below.
                    "attempt": len(events) + 1,
                    "outcome": outcome,
                    "summary": self._summarize_judge_outcome(
                        outcome, panel, detail=detail,
                    ),
                    "iteration": iteration + 1,
                    "stage": stage,
                    "detail": (detail or "").strip(),
                    # The exact code proposed/evaluated THIS round — `q` is
                    # always the query dict as it stood going into this
                    # round's validation or judge call, so this is the
                    # version that actually earned this round's outcome, not
                    # whatever correction produced afterward.
                    "proposed_code": q.get("code", ""),
                    "panel": dict(panel) if panel else {},
                }
            )

        for iteration in range(self.max_judge_iterations):
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

            self._log.validator_result(
                queries_before_validation, pending_queries, errors=val_errors
            )
            self._accumulate_tokens(all_tokens, val_tokens)

            tagged_val_errors = [
                {**e, "query_ids": [str(q.get("id")) for q in queries_before_validation]}
                if isinstance(e, dict) and "query_ids" not in e
                else e
                for e in val_errors
            ]
            all_errors.extend(tagged_val_errors)

            # Queries the validator dropped (still failing static validation
            # after every correction cycle) never reach the judge — record them
            # as failures with the validator's drop message(s).
            surviving_keys = {_failure_key(q) for q in pending_queries}
            drop_messages = [
                e for e in (val_errors or [])
                if isinstance(e, str) and "Dropped query" in e
            ]
            for q in queries_before_validation:
                if _failure_key(q) not in surviving_keys:
                    drop_error = (
                        "\n".join(drop_messages)
                        or "Dropped during static validation (no specific error captured)."
                    )
                    _record_failure(q, "validation_failed", drop_error)
                    _record_event(
                        q, iteration, "validation", "validation_failed", drop_error
                    )

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
                # A query rejected/failed on an earlier iteration but approved
                # now is no longer a failure.
                failures_by_key.pop(_failure_key(orig), None)
                _record_event(
                    orig, iteration, "judge", "approved",
                    judge.all_judgments_by_id.get(qid, {}).get("feedback", ""),
                    panel=judge.all_judgments_by_id.get(qid, {}).get("panel"),
                )
                # The full trail (earlier rejections included) rides on the
                # approved query dict into the saved JSON.
                orig["attempt_history"] = history_by_key.get(_failure_key(orig), [])
                self._log.query_approved(er["id"], er.get("question", ""))
                self._log.panel_votes(
                    judge.all_judgments_by_id.get(qid, {}).get("panel", {}), level=2
                )

            for er in evaluation["rejected"]:
                j = judge.all_judgments_by_id.get(str(er["id"]), {})
                feedback = j.get("feedback", "")
                suggestions = j.get("suggestions", "")
                rejection_error = (
                    f"{feedback}"
                    + (f"\nSuggestions: {suggestions}" if suggestions else "")
                )
                _record_failure(
                    er.get("_original_query") or er,
                    "judge_rejected",
                    rejection_error,
                    panel=j.get("panel"),
                )
                _record_event(
                    er.get("_original_query") or er,
                    iteration, "judge", "rejected", rejection_error,
                    panel=j.get("panel"),
                )
                self._log.query_rejected(
                    er["id"],
                    er.get("question", ""),
                    feedback=feedback,
                    suggestions=suggestions,
                    attempt=judge._rejection_counts.get(str(er["id"]), 1),
                )
                self._log.panel_votes(j.get("panel", {}), level=2)

            for er in evaluation["permanently_rejected"]:
                qid = str(er["id"])
                history = "; ".join(judge._accumulated_feedback.get(qid, []))
                _record_failure(
                    er.get("_original_query") or er,
                    "judge_rejected_permanent",
                    f"Permanently rejected after "
                    f"{judge._rejection_counts.get(qid, 2)} attempt(s). "
                    f"History: {history}",
                    panel=judge.all_judgments_by_id.get(qid, {}).get("panel"),
                )
                _record_event(
                    er.get("_original_query") or er,
                    iteration, "judge", "permanently_rejected",
                    judge.all_judgments_by_id.get(qid, {}).get("feedback", "")
                    or history,
                    panel=judge.all_judgments_by_id.get(qid, {}).get("panel"),
                )
                self._log.query_permanent_reject(er["id"], er.get("question", ""))

            for f in evaluation["execution_failures"]:
                _record_failure(f["query"], "execution_failure", str(f["error"]))
                _record_event(
                    f["query"], iteration, "execution", "execution_failure",
                    str(f["error"]),
                )
                self._log.query_execution_failure(f["id"], f["error"])

            # PLAN-COMPLIANT code, genuinely empty result: pulled out of the
            # loop immediately — never fed back as a code-correction retry
            # (see evaluate()'s empty_result bucket) — for the caller to
            # escalate via _retry_empty_results, exactly once per query.
            for er in evaluation["empty_result"]:
                qid = str(er["id"])
                self._log.empty_result_escalation(qid, er.get("question", ""))
                _record_event(
                    er.get("_original_query") or er,
                    iteration, "judge", "empty_result",
                    judge.all_judgments_by_id.get(qid, {}).get("feedback", ""),
                    panel=judge.all_judgments_by_id.get(qid, {}).get("panel"),
                )
            empty_result_pending.extend(evaluation["empty_result"])

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

        # Attach each failed query's full outcome trail (every earlier
        # validation drop / judge rejection, numbered), mirroring the
        # `attempt_history` approved queries carry.
        failed_queries_out: list = []
        for key, failure in failures_by_key.items():
            failure["attempt_history"] = history_by_key.get(key, [])
            failed_queries_out.append(failure)

        return (
            all_approved_executed,
            all_approved_query_dicts,
            elapsed,
            {"validation": validation_ms, "judging": judging_ms},
            failed_queries_out,
            empty_result_pending,
        )

    def _assemble_result(
        self,
        mode: str,
        kind: str,
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
        plan: Optional[QueryPlan] = None,
        plan_by_client_id: Optional[dict] = None,
        plan_attempts_by_id: Optional[dict] = None,
        timings_ms: Optional[dict] = None,
        failed_queries: Optional[list] = None,
        plan_feedback: Optional[list] = None,
        status_override: Optional[str] = None,
    ) -> dict:
        """Assemble the final result dict.

        Query-level traceability: every approved query dict carries
        ``status: "success"``, and the queries that never made it out of the
        validator-judge loop are saved alongside them under
        ``result["failed_queries"]`` — each with its own ``status``
        (``validation_failed`` / ``judge_rejected`` / ``judge_rejected_permanent``
        / ``execution_failure``), the query's ``question`` and ``code``, and the
        last error/feedback message observed for it — instead of being dropped
        from the saved JSON.

        ``plan`` is the run-level fallback (e.g. the first plan, used only when
        a query's own ``client_id`` isn't found in ``plan_by_client_id`` below —
        this should not normally happen for approved queries, but keeps assembly
        total). ``plan_by_client_id`` carries the SPECIFIC plan that produced
        each query (several plans are generated per run, each potentially
        needing a different skill — see ``QueryPlanner.plan_batch``), so every
        assembled query's trace reflects the plan (and, via its ``task_types``,
        the skill(s)) that actually generated it rather than a single plan for
        the whole run. ``plan_attempts_by_id`` (see ``_judge_plans``'s Returns
        doc — keyed by ``id(plan_object)``) is looked up the same way, via
        ``query_plan``/``f_plan``, so every approved AND failed query also
        carries its own plan's full judging trail under
        ``plan_attempt_history`` — the plan-side analogue of ``attempt_history``
        — plus the FINAL plan itself (post-revision, the version that actually
        produced the query) under ``structured_plan``.

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
          omits it.

        (Requirement 1.4 — single mode's assembled query carries exactly the
        single provided alias in ``tables`` — is guaranteed upstream now:
        ``QueryPlanner.validate_plan`` requires every plan's ``tables`` to
        cover every known alias exactly once, and every query's ``tables``
        is copied straight from its plan, so there is nothing left to
        reconcile here.)
        """
        judgments_by_id = judge.all_judgments_by_id if judge is not None else {}
        approved_query_ids = {str(er["id"]) for er in all_approved_executed}
        approved_executed_by_id = {str(er["id"]): er for er in all_approved_executed}
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

            judgment = judgments_by_id.get(qid, {})
            keyword_count = utils.count_keywords(q_copy.get("code"), kind)

            # This query's OWN plan (it may differ from every other query's,
            # since several plans are generated per run — see
            # ``QueryPlanner.plan_batch``). Fall back to the run-level default
            # only if the client_id somehow isn't in the map.
            client_id = q_copy.get("client_id")
            query_plan = (plan_by_client_id or {}).get(client_id, plan)
            skills_used = list(self._plan_task_types(query_plan))
            # This query's OWN plan's full judging history — every attempt
            # (question as judged, per-layer votes, per-judge panel), keyed
            # by the plan object's identity (see _judge_plans's Returns doc)
            # so it survives regardless of plan-list reshuffling.
            plan_attempt_history = (plan_attempts_by_id or {}).get(id(query_plan), [])
            # The FINAL structured plan (steps, tables, task_types, links) —
            # the approved (and possibly revised) version that actually
            # produced this query, not just its free-text `query_plan` summary.
            structured_plan = (
                query_plan.model_dump() if hasattr(query_plan, "model_dump")
                else query_plan
            )

            approved_queries.append(
                {
                    **q_copy,
                    "status": "success",
                    "response": judgment.get("response", ""),
                    "translated_response": judgment.get("translated_response", ""),
                    "judge_feedback": judgment.get("feedback", ""),
                    # Full per-judge record of the code panel's majority vote
                    # on this query (empty when the single-judge fallback ran).
                    "code_feedback": judgment.get("panel", {}),
                    # Full trail of the PLAN judge loop that approved the plan
                    # this query was generated from — same shape as
                    # attempt_history (attempt/approved/panel per round).
                    "plan_attempt_history": plan_attempt_history,
                    "structured_plan": structured_plan,
                    "keyword_count": keyword_count,
                    "skills_used": skills_used,
                    "query_result": self._serialize_query_output(er),
                }
            )

            # Requirement 21: assemble the fully-traceable, phase-grouped query.
            traceable_queries.append(
                self._build_traceable_query(
                    q_copy=q_copy,
                    judgment=judgment,
                    executed=er,
                    plan=query_plan,
                    timings_ms=timings_ms,
                    all_tokens=all_tokens,
                    last_model=last_model,
                    keyword_count=keyword_count,
                )
            )

        self._log.step3_summary(approved_queries, elapsed=total_time)

        # Failed queries stay in the saved JSON (query-level traceability):
        # enrich each failure record with the skill(s) its own plan injected,
        # and its plan's own judging history, so a skill-bound or
        # plan-judging-adjacent failure is recognisable as such.
        failed_out: list = []
        for f in failed_queries or []:
            f_plan = (plan_by_client_id or {}).get(f.get("client_id"), None)
            failed_out.append(
                {
                    **f,
                    "skills_used": list(self._plan_task_types(f_plan)),
                    "plan_attempt_history": (plan_attempts_by_id or {}).get(id(f_plan), []),
                    "structured_plan": (
                        f_plan.model_dump() if hasattr(f_plan, "model_dump") else f_plan
                    ),
                }
            )
        if failed_out:
            self._log.warning(
                f"{len(failed_out)} failing query/queries saved to the result "
                f"under 'failed_queries' for traceability."
            )

        traceable_set = TraceableQuerySet(queries=traceable_queries)

        result: dict = {
            "result": {
                "queries": approved_queries,
                "failed_queries": failed_out,
                # Per-plan majority-vote record of the plan judge panel
                # (empty when judge_profiles.plan is not configured).
                "plan_feedback": list(plan_feedback or []),
            },
            "traceable": traceable_set.model_dump(),
            "token_usage": all_tokens,
            "errors": all_errors,
            "model": last_model,
            "time_elapsed": total_time,
            "avg_cols": avg_cols,
            "executed_results": all_approved_executed,
            # Union of every ML task-type skill actually used across all
            # generated plans (not just one run-level plan) — see
            # plan_by_client_id above.
            "skills": self._summarize_skills_by_client_id(plan_by_client_id, plan),
        }
        # A structural, pre-generation failure (e.g. "insufficient_rows",
        # "unjustifiable_group") — _store_generation surfaces it as the
        # entry's _meta status instead of the generic "failure".
        if status_override:
            result["result"]["status"] = status_override
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
        plan: Optional[QueryPlan],
        timings_ms: dict,
        all_tokens: dict,
        last_model: str,
        keyword_count: Any,
    ) -> TraceableQuery:
        """Assemble one approved query into a phase-grouped :class:`TraceableQuery`.

        Every legacy field is mapped into exactly one phase sub-object so no field
        is lost (Requirement 21.2): the question/keywords/translated variants,
        difficulty, topic, story, free-text query plan and per-table
        reason/columns/keywords go into :class:`PlanningTrace`; the judge's
        feedback/suggestions/violated criteria/result and requirements checks
        plus the response/translated response go into :class:`JudgeTrace`; the
        code, token usage and model go into the top-level fields and
        :class:`UsageTrace`. The structured plan, the injected skill identity
        (derived from the plan's ``task_types``) and the per-phase timings are
        attached too (Requirement 21.3), and the execution status is derived as
        exactly one of the four allowed literals (Requirement 21.5).
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

        # ── Skill trace: the injected skill identity (if any) ─────────────
        skill = self._build_skill_trace(plan)

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
            result_check=judgment.get("result_check", "") or "",
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
            keyword_count=self._coerce_keyword_count(keyword_count),
            planning=planning,
            skill=skill,
            validation=validation,
            judging=judging,
            execution=execution,
            usage=usage,
        )

    @staticmethod
    def _coerce_keyword_count(keyword_count: Any) -> int:
        """Reduce a keyword-count value to a single integer total.

        ``utils.count_keywords`` returns a ``dict[str, int]`` mapping each
        matched keyword to its number of occurrences, but the traceable model
        stores a single integer. Sum the occurrences when a mapping is given,
        otherwise fall back to a plain integer coercion.
        """
        if isinstance(keyword_count, dict):
            return sum(int(v or 0) for v in keyword_count.values())
        try:
            return int(keyword_count or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _plan_task_types(plan: Optional[QueryPlan]) -> list:
        """A plan's ``task_types``, or ``[]`` for a SQL plan / ``None`` plan.

        SQL plans have no ``task_types`` field at all (see
        ``prompting.models.SQLQueryPlan``), so this is a plain attribute read
        everywhere a plan's injected skill(s) matter, without every call site
        repeating the ``getattr(..., "task_types", None) or []`` fallback.
        """
        return getattr(plan, "task_types", None) or []

    @staticmethod
    def _summarize_skills(plan: Optional[QueryPlan]) -> list:
        """Summarise the task-type skill(s) injected for one plan.

        Returns a list of ``{"name": task_type}`` entries (empty for a SQL plan,
        or a Pandas plan with no ML steps — plain-Python generation).
        """
        task_types = StatementOrchestrator._plan_task_types(plan)
        return [{"name": task_type} for task_type in task_types]

    @staticmethod
    def _summarize_skills_by_client_id(
        plan_by_client_id: Optional[dict], fallback: Optional[QueryPlan] = None
    ) -> list:
        """Union of every distinct task-type skill injected across all plans in the run.

        Several plans are generated per run (``QueryPlanner.plan_batch``) and
        each can inject different skills (via its own ``task_types``), so the
        run-level ``skills`` summary must be the union across all of them
        (deduplicated by name) rather than a single plan's. Falls back to
        summarising ``fallback`` alone when the map is empty (e.g. no queries
        were ever generated).
        """
        seen: set = set()
        summary: list = []
        plans = list((plan_by_client_id or {}).values()) or (
            [fallback] if fallback is not None else []
        )
        for plan in plans:
            for entry in StatementOrchestrator._summarize_skills(plan):
                key = entry.get("name")
                if key not in seen:
                    seen.add(key)
                    summary.append(entry)
        return summary

    @staticmethod
    def _build_skill_trace(plan: Optional[QueryPlan]) -> SkillTrace:
        """Extract the injected skill identity into a :class:`SkillTrace`.

        When the plan's ``task_types`` is non-empty, the (comma-joined) task
        types are recorded as the query's injected skill(s); otherwise the
        trace records no skill (plain-Python generation, or a SQL plan).
        """
        task_types = StatementOrchestrator._plan_task_types(plan)
        if not task_types:
            return SkillTrace(skill_used=None, skill_version=None, reason="")
        return SkillTrace(
            skill_used=",".join(task_types),
            skill_version=None,
            reason=f"Injected skill markdown(s) for task type(s): {', '.join(task_types)}.",
        )

    @staticmethod
    def _serialize_query_output(executed: Optional[dict]) -> list:
        """Convert an approved query's executed result into JSON-safe records.

        ``executed["dataframe"]`` (set by ``JudgementResponseAgent._execute_queries``,
        already capped to ``.head(8)`` rows) is a pandas DataFrame — not directly
        JSON-serialisable, since NaN/NaT/Timestamp/numpy scalar values aren't
        native ``json.dump`` types. Round-tripping through ``DataFrame.to_json``
        (pandas' own encoder, which handles all of those) and back through
        ``json.loads`` yields plain Python objects safe for ``utils.save_json``.
        """
        if not executed:
            return []
        df = executed.get("dataframe")
        if df is None:
            return []
        try:
            safe = StatementOrchestrator._jsonify_dataframe(df)
            return json.loads(safe.to_json(orient="records", date_format="iso"))
        except Exception as exc:
            # Broad on purpose: pandas' ujson encoder raises OverflowError
            # ("Maximum recursion level reached") — not TypeError — on values
            # it cannot encode, and a serialisation failure must never abort a
            # run whose queries were already generated and approved.
            logger.warning("Failed to serialise query result to JSON: %s", exc)
            return []

    @staticmethod
    def _jsonify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce columns pandas' ujson encoder cannot handle into strings.

        ``DataFrame.to_json`` handles datetimes/NaN/numpy scalars, but chokes
        on extension dtypes like ``period[M]`` (e.g. from ``.dt.to_period()``)
        or ``interval`` — and on arbitrary Python objects inside ``object``
        columns — with a misleading ``OverflowError: Maximum recursion level
        reached``. Convert those to their string form (``"2025-01"``) so the
        result survives serialisation instead of being dropped.
        """
        native = (str, int, float, bool)

        def _coerce(value: Any) -> Any:
            if value is None or isinstance(value, native):
                return value
            if value is pd.NaT or value is pd.NA:
                return None
            return str(value)

        safe = df.copy()
        # Positional access so duplicate column labels can't select a frame.
        for i in range(safe.shape[1]):
            col = safe.iloc[:, i]
            if isinstance(col.dtype, pd.CategoricalDtype):
                col = col.astype(object)
            if isinstance(col.dtype, (pd.PeriodDtype, pd.IntervalDtype)):
                # Via _coerce (not .astype(str)) so NaT becomes null, not "NaT".
                safe.isetitem(i, col.astype(object).map(_coerce))
            elif col.dtype == object:
                safe.isetitem(i, col.map(_coerce))
        return safe

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

        reliability = executed.get("reliability")

        if executed.get("error"):
            return ExecutionTrace(
                status="execution_failure", error=str(executed["error"]), reliability=reliability
            )

        row_count = _execution_row_count(executed)
        if row_count is None:
            # Approved but no captured frame — it did execute (judge ran it), so
            # treat as a successful run with an unknown row count.
            return ExecutionTrace(status="success", row_count=None, reliability=reliability)

        status = "success" if row_count > 0 else "empty"
        return ExecutionTrace(status=status, row_count=row_count, reliability=reliability)

    def _retry_empty_results(
        self,
        all_approved_executed: list,
        all_approved_query_dicts: dict,
        failed_queries: list,
        empty_result_pending: list,
        plan_by_client_id: dict,
        plan_feedback: list,
        plan_attempts_by_id: dict,
        analyses: list,
        aliases: dict,
        stats: list,
        match: Any,
        involved_cols: Optional[dict],
        dfs: list,
        retrievable_keywords: Optional[list],
        base_prompt: str,
        kind: str,
        metadata: Any,
        judge: "JudgementResponseAgent",
        all_tokens: dict,
        all_errors: list,
    ) -> tuple[list, dict, list]:
        """Escalate a PLAN-COMPLIANT query whose executed result is empty
        (0 rows) back to PLAN-level revision, exactly once per query.

        Two sources feed this, merged into one ``still_empty`` list:
        ``empty_result_pending`` is the explicit, vote-driven signal from
        ``_validator_judge_loop`` (the code panel majority voted
        ``plan_compliance_approval`` true and ``present_result_approval``
        false — see ``Judgment``'s two independent layers) — the PRIMARY
        path, since it fires immediately on the iteration the code panel
        actually sees the empty result, without wasting any
        code-correction cycles first. The defensive re-scan of
        ``all_approved_executed`` catches the residual case where a
        query's OWN present-result vote was wrong (voted true despite 0
        actual rows) and it slipped through as fully ``approved`` anyway —
        belt-and-suspenders, so a genuinely empty result is never silently
        kept regardless of what the judge claimed.

        An empty result is usually a plan-level symptom (an over-restrictive
        filter combination, a mismatched join) rather than a code bug — so
        instead of the normal
        code-correction retry, this revises the PLAN (preserving its
        difficulty tier, via ``QueryPlanner.revise_plan``'s existing
        force-override), re-enters the FULL plan judge panel, regenerates
        code for the resulting plan (reusing the exact single-plan
        generation call Phase 2c already uses), and re-judges/re-executes
        it ONE more time via the same ``judge`` instance already in scope.
        Still empty (or rejected, or the plan can't be revised/re-approved
        at all) -> permanently failed with status ``"empty_result"``,
        recorded in ``failed_queries`` like any other terminal failure —
        never silently dropped.
        """
        approved_but_empty = [
            er for er in all_approved_executed
            if _execution_row_count(er) == 0
        ]
        pending_ids = {str(er.get("id")) for er in empty_result_pending}
        still_empty = list(empty_result_pending) + [
            er for er in approved_but_empty
            if str(er.get("id")) not in pending_ids
        ]
        if not still_empty:
            return all_approved_executed, all_approved_query_dicts, failed_queries

        empty_ids = {str(er.get("id")) for er in still_empty}
        kept = [
            er for er in all_approved_executed
            if str(er.get("id")) not in empty_ids
        ]

        def _fail(er: dict, error: str, history: Optional[list] = None) -> None:
            qid = str(er.get("id"))
            all_approved_query_dicts.pop(qid, None)
            self._log.empty_result_permanent_fail(qid, er.get("question", ""), error)
            failed_queries.append({
                "id": er.get("id"),
                "client_id": er.get("client_id", ""),
                "status": "empty_result",
                "question": er.get("question", ""),
                "code": er.get("code", ""),
                "error": error,
                "code_feedback": {},
                "attempt_history": history or [],
            })

        for er in still_empty:
            self._log.empty_result_escalation(str(er.get("id")), er.get("question", ""))
            client_id = er.get("client_id")
            plan = plan_by_client_id.get(client_id)

            # Round-1 CODE JUDGE attempt-history entry: the panel verdict
            # that flagged this query as plan-compliant-but-empty in the
            # first place. `judge` is the SAME JudgementResponseAgent
            # instance used by `_validator_judge_loop`, so its
            # `all_judgments_by_id` still carries that verdict here — but
            # `_validator_judge_loop`'s own `history_by_key` (which would
            # normally record it) is local to that call and never threaded
            # through to this method. Without rebuilding it here, a
            # recovered query's `attempt_history` stayed permanently empty
            # even after this method approved reworked code for it, so the
            # web UI's Code Judge Loop / Judge Responses sections showed
            # nothing at all for it, as if it had never been judged.
            empty_judgment = judge.all_judgments_by_id.get(str(er.get("id")), {})
            empty_panel = empty_judgment.get("panel")
            round_1 = {
                "attempt": 1,
                "outcome": "empty_result",
                "summary": self._summarize_judge_outcome("empty_result", empty_panel),
                "iteration": 0,
                "stage": "judge",
                "detail": empty_judgment.get("feedback", ""),
                "proposed_code": er.get("code", ""),
                "panel": dict(empty_panel) if empty_panel else {},
            }

            if plan is None:
                _fail(
                    er,
                    "Query result was empty, but its originating plan could "
                    "not be found for a plan-level retry.",
                    history=[round_1],
                )
                continue

            feedback = (
                "This plan's query executed successfully but returned an "
                "empty result. Revise the plan so it produces a non-empty "
                "result — loosen an over-restrictive filter, fix a "
                "mismatched join or predictor/target choice, or reconsider "
                "the steps — while keeping the plan genuinely at its "
                "assigned difficulty tier."
            )
            revised, usage = self._planner.revise_plan(
                plan, feedback, analyses, aliases, match, involved_cols, stats,
                self.languages, dfs=dfs,
                retrievable_keywords=retrievable_keywords,
            )
            self._accumulate_tokens(all_tokens, usage)
            if revised is None:
                _fail(
                    er,
                    "Empty result: the plan could not be revised (failed "
                    "structural validation twice). One retry already "
                    "exhausted.",
                    history=[round_1],
                )
                continue

            (
                approved_plans, retry_plan_feedback, panel_usage,
                _abort_status, retry_attempts_by_id,
            ) = self._judge_plans(
                [revised], analyses, aliases, stats, match=match,
                involved_cols=involved_cols, dfs=dfs,
                retrievable_keywords=retrievable_keywords,
            )
            self._accumulate_tokens(all_tokens, panel_usage)
            # Tagged so the saved run distinguishes an ordinary planning-phase
            # judging round from this later, empty-result-driven escalation —
            # never affects the run-level status_override, which stays
            # scoped to the original planning phase's own abort conditions.
            for fb_entry in retry_plan_feedback:
                fb_entry["empty_result_retry"] = True
            plan_feedback.extend(retry_plan_feedback)
            plan_attempts_by_id.update(retry_attempts_by_id)

            if not approved_plans:
                _fail(
                    er,
                    "Empty result: the revised plan was rejected by the "
                    "plan judge panel. One retry already exhausted.",
                    history=[round_1],
                )
                continue

            new_plan = approved_plans[0]
            self._log.empty_result_code_regen(str(er.get("id")), er.get("question", ""))

            def _generate_fn(built_prompt: str, _plan=new_plan) -> Any:
                return self._client.complete(
                    built_prompt,
                    dfs,
                    aliases,
                    typology=kind,
                    involved_cols=involved_cols,
                    matches=match,
                    metadata=metadata,
                    languages=self.languages,
                    precomputed_plan=_plan.model_dump(),
                )

            query_set, tokens, errors, _model = self._generator.generate(
                base_prompt, _generate_fn, plan=new_plan, stats=stats,
            )
            self._accumulate_tokens(all_tokens, tokens)
            all_errors.extend(errors or [])

            new_queries = query_set.get("queries", []) or []
            if not new_queries:
                _fail(
                    er,
                    "Empty result: code regeneration from the revised plan "
                    "produced no query. One retry already exhausted.",
                    history=[round_1],
                )
                continue

            for q in new_queries:
                new_client_id = q.get("client_id") if isinstance(q, dict) else None
                if new_client_id:
                    plan_by_client_id[new_client_id] = new_plan

            result = judge.evaluate(new_queries)
            newly_approved = result.get("approved") or []

            # Surface the CODE JUDGE panel for this one-shot round exactly
            # like the main _validator_judge_loop does (query_approved /
            # query_rejected + panel_votes) — evaluate() itself never logs,
            # so without this the console goes straight from the PLAN JUDGE
            # verdict to the final recovered/exhausted line with no visible
            # trace of the code that was actually written and judged.
            approved_ids = {str(a.get("id")) for a in newly_approved}
            exec_failures_by_id = {
                str(f.get("id")): f for f in result.get("execution_failures", [])
            }
            for nq in new_queries:
                nqid = str(nq.get("id"))
                if nqid in exec_failures_by_id:
                    self._log.query_execution_failure(
                        nqid, exec_failures_by_id[nqid].get("error", "")
                    )
                    continue
                j = judge.all_judgments_by_id.get(nqid, {})
                if nqid in approved_ids:
                    self._log.query_approved(nqid, nq.get("question", ""))
                else:
                    self._log.query_rejected(
                        nqid, nq.get("question", ""),
                        feedback=j.get("feedback", ""),
                        suggestions=j.get("suggestions", ""),
                    )
                self._log.panel_votes(j.get("panel", {}), level=2)

            success = next(
                (
                    new_er for new_er in newly_approved
                    if _execution_row_count(new_er) != 0
                ),
                None,
            )

            if success is None:
                # Best-effort round-2 entry from whichever regenerated query
                # the judge actually ruled on, so even the give-up path keeps
                # a trace of the second code judge round instead of just
                # round_1 — same rationale as round_1 above.
                first_new = new_queries[0] if new_queries else {}
                first_new_id = str(first_new.get("id"))
                if first_new_id in exec_failures_by_id:
                    round_2_outcome = "execution_failure"
                elif first_new_id in approved_ids:
                    round_2_outcome = "empty_result"  # approved but still 0 rows
                else:
                    round_2_outcome = "rejected"
                round_2_judgment = judge.all_judgments_by_id.get(first_new_id, {})
                round_2_panel = round_2_judgment.get("panel")
                round_2 = {
                    "attempt": 2,
                    "outcome": round_2_outcome,
                    "summary": self._summarize_judge_outcome(
                        round_2_outcome, round_2_panel,
                        detail=exec_failures_by_id.get(first_new_id, {}).get("error", ""),
                    ),
                    "iteration": 0,
                    "stage": "judge",
                    "detail": round_2_judgment.get("feedback", "")
                    or exec_failures_by_id.get(first_new_id, {}).get("error", ""),
                    "proposed_code": first_new.get("code", ""),
                    "panel": dict(round_2_panel) if round_2_panel else {},
                }
                _fail(
                    er,
                    "Empty result: the retried plan's query was still empty "
                    "(or rejected) on regeneration. One retry already "
                    "exhausted.",
                    history=[round_1, round_2],
                )
                continue

            qid = str(success.get("id"))
            orig = success.get("_original_query") or next(
                (q for q in new_queries if str(q.get("id")) == qid), {}
            )
            success_judgment = judge.all_judgments_by_id.get(qid, {})
            success_panel = success_judgment.get("panel")
            round_2 = {
                "attempt": 2,
                "outcome": "approved",
                "summary": self._summarize_judge_outcome("approved", success_panel),
                "iteration": 0,
                "stage": "judge",
                "detail": success_judgment.get("feedback", ""),
                "proposed_code": orig.get("code", ""),
                "panel": dict(success_panel) if success_panel else {},
            }
            orig["attempt_history"] = [round_1, round_2]
            all_approved_query_dicts[qid] = orig
            kept.append(success)
            self._log.empty_result_recovered(qid, success.get("question", ""))

        return kept, all_approved_query_dicts, failed_queries

    def _empty_plan(self, q_copy: dict) -> QueryPlan:
        """Build a minimal schema-valid plan when none was threaded in.

        The traceable schema requires a ``structured_plan``; this defensive
        fallback (used only if assembly is ever called without a plan) records the
        query's question with no decomposition steps rather than dropping the
        field.
        """
        if self.kind == "PANDAS":
            return PandasQueryPlan(
                question=q_copy.get("question", "") or "",
                question_keywords=[],
                plan_keywords=[],
                steps=[],
                table_links=[],
            )
        return SQLQueryPlan(
            question=q_copy.get("question", "") or "",
            question_keywords=[],
            plan_keywords=[],
            steps=[],
            table_links=[],
        )

    @staticmethod
    def _accumulate_tokens(total: dict, partial: Any) -> None:
        """Add a usage dict into ``total`` in place (missing keys contribute 0)."""
        if not isinstance(partial, dict):
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total[key] += partial.get(key, 0)
