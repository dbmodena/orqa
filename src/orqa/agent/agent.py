from pathlib import Path
from .. import utils
import time
import json
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from .TaskProposer import TaskProposerLLMClient
from .StatementClient import LLMClientStatementGenerator
from .StatementValidator import LLMStatementValidator
from .StatementJudge import LLMStatementJudge
from .LLMClient import LLMClient
from .utility.budget_guard import BudgetGuard
from .prompting import (
    CandidatesDiscoveryPrompt,
    JudgementResponseGenerationPrompt,
    SingleTableJudgementResponseGenerationPrompt,
    PandasStatementGenerationPrompt,
    SQLStatementGenerationPrompt,
    ResponseGenerationPrompt,
    SingleTablePandasPrompt,
    SingleTableSQLPrompt,
)
from ..queries.query_execution import QueryExecutor
from .PipelineLogger import PipelineLogger
import pandas as pd
import copy
import re


logger = logging.getLogger(__name__)

# Default bound on the number of concurrent judging calls (Requirement 15.1).
# Kept small so the LLM backend is not overwhelmed while still removing the
# serial one-at-a-time bottleneck. Configurable per JudgementResponseAgent.
JUDGE_CONCURRENCY = 6


class CandidatesDiscoveryAgent:
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

            if (
                dataset_info["num_rows"] < min_dataset_height
                or dataset_info["num_columns"] == 0
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


class JudgementResponseAgent:
    def __init__(
        self,
        config_path: Path,
        kind: str,
        executor,
        entry: dict,
        max_tokens: int = 2000,
        single_table: bool = False,
        judge_concurrency: int = JUDGE_CONCURRENCY,
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
        self.max_tokens = max_tokens
        self.executor = executor
        self.entry = entry
        self.kind = kind
        # Bound the worker pool (Requirement 15.1). Never below 1.
        self.judge_concurrency = max(1, int(judge_concurrency))
        self._rejection_counts: dict[str, int] = {}
        self._accumulated_feedback: dict[str, list[str]] = {}
        self.all_judgments_by_id: dict[str, dict] = {}
        # Judgments keyed by the opaque echoed client_id (Requirement 15.2).
        self.all_judgments_by_client_id: dict[str, dict] = {}

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
        rejected = [
            er for er in executed
            if not self.all_judgments_by_id.get(str(er["id"]), {}).get("approved", False)
        ]

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
            if self._rejection_counts.get(str(er["id"]), 0) < 2
        ]
        permanently_rejected = [
            er for er in rejected
            if self._rejection_counts.get(str(er["id"]), 0) >= 2
        ]

        feedback_messages = self._build_feedback_messages(still_actionable, pending_queries)
        structured_feedback = self._build_structured_feedback(still_actionable, pending_queries)

        return {
            "approved": approved,
            "rejected": still_actionable,
            "permanently_rejected": permanently_rejected,
            "execution_failures": execution_failures,
            "feedback_messages": feedback_messages,
            "structured_feedback": structured_feedback,
            "all_done": not still_actionable and not execution_failures,
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

    def _execute_queries(self, queries: list) -> tuple[list, list]:
        # Build a set of dataset names so we can guard against them leaking
        # into columns_involved or code as column references.
        alias_map: dict = self.entry.get("tables", {})   # {"Table_0": "dataset_name", …}
        dataset_names: set = set(alias_map.values())

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

            # 3. Build the tables list for the executor (no columns_involved needed).
            tables = copy.deepcopy(sanitized.get("tables", []))
            for table in tables:
                table.pop("columns_involved", None)

            try:
                df_result = self.executor.execute(self.entry, sanitized, self.kind).head(8)
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
            except Exception as exc:
                failures.append({
                    "id": qid,
                    "error": str(exc),
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

        A fresh prompt instance is built per call (Requirement 15.3) so no
        mutable prompt state is shared across concurrent judging workers.

        Returns the judgment dict for the query, or an empty dict on failure.
        """
        # Temporarily override id to 1 so the LLM produces a valid integer id.
        single = dict(executed_result)
        single["id"] = 1
        # Build an independent prompt per call — no shared mutable prompt state.
        prompt = self._prompt_factory()
        prompt_str = prompt.update([single])
        result, _ = self._client.complete(prompt_str, max_tokens=self.max_tokens)
        judgments = result.get("queries", [])
        return judgments[0] if judgments else {}


class LegacyStatementGenerationAgent:
    """
    Orchestrates the full multi-table query-generation pipeline:

        StatementClient (one-shot generation)
            ↓
        Loop:
            StatementValidator (static validation + LLM correction + judge feedback)
            ↓
            StatementJudge
            ↓  (rejected queries fed back to Validator)
            StatementValidator → StatementJudge → … until all_done or max_iterations
    """

    def __init__(
        self,
        config_path: Path,
        kind: str,
        bad_tokens: list,
        max_judge_iterations: int = 3,languages:list=["English"],
        budget: "BudgetGuard | None" = None,
    ):
        self.config_path = config_path
        self.kind = kind
        self.bad_tokens = bad_tokens
        self.max_judge_iterations = max_judge_iterations
        # Overall budget ceiling across the (multiplicative) retry loops
        # (Requirement 17). A guard may be injected for testing; otherwise one is
        # built from the query_generation config. Never raises — see BudgetGuard.
        self._budget = budget or BudgetGuard.from_config(self.config_path)

        if kind == "PANDAS":
            self.prompt = PandasStatementGenerationPrompt()
        else:
            self.prompt = SQLStatementGenerationPrompt()

        self._client = LLMClientStatementGenerator(self.config_path)
        self._validator = LLMStatementValidator(self.config_path, kind)
        self._log = PipelineLogger()
        self.languages=languages

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
    ) -> dict | None:
        columns = 0
        columns_by_table = []
        try:
            tables = []
            base_prompt = ""
            self.prompt.reset()
            dataset_names = [p.name for p in dataset_paths]
            self._log.section(f"Multi-table generation — {', '.join(dataset_names)}  [{kind}]")
            inverted_aliases = {v: k for k, v in aliases.items()}
            for idx, dataset_path in enumerate(dataset_paths):
                df, dataset_info = utils.prepare_dataset(
                    dataset_path,
                    involved_cols[f"Table_{idx}"],
                    max_cols,
                    sample_size,
                    self.bad_tokens,
                )
                columns += len(df.columns)
                columns_by_table.append({
                    "table_alias": f"Table_{idx}",
                    "columns_provided": df.columns.tolist(),
                })
                tables.append(df)
                columns_for_prompt=""
                for col in df.columns:
                    dtype = df[col].dtype
                    columns_for_prompt+=f"\n- {col} ({dtype})"
                base_prompt = self.prompt.update(
                    dataset_info["dataset_name"],
                    dataset_info["num_rows"],
                    dataset_info["num_columns"],
                    metadatas[idx],
                    dataset_info["columns_details"],
                    dataset_info["sample_data"],
                    json.dumps(aliases, indent=2),
                    match,
                    self.languages,columns_for_prompt,inverted_aliases[dataset_info["dataset_name"]]
                )

            table_schemas = self.prompt.datasets_descriptions
            secondary_table_schemas = self.prompt._light_datasets_descriptions
            datasets_path = dataset_paths[0].parent
            executor = QueryExecutor(datasets_path=datasets_path, bad_tokens=self.bad_tokens)
            entry = {"tables": aliases}
            avg_cols = columns / len(dataset_paths)

            judge = JudgementResponseAgent(self.config_path, kind, executor, entry)

            all_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            all_errors: list = []
            all_approved_executed: list = []
            all_approved_query_dicts: dict[str, dict] = {}
            last_model = ""
            total_time = 0.0
            original_ids: set[str] = set()

            # ----------------------------------------------------------------
            # Step 1 — One-shot initial generation
            # ----------------------------------------------------------------
            start = time.perf_counter()
            self._budget.start()  # Requirement 17: begin the overall wall-clock budget
            result, tokens, errors, model = self._client.complete(
                base_prompt,
                tables,
                aliases,
                typology=kind,
                involved_cols=involved_cols,
                matches=match,
                metadata=metadatas,
                languages=self.languages,
            )
            total_time += time.perf_counter() - start

            for k in all_tokens:
                all_tokens[k] += tokens.get(k, 0)
            # Feed generation tokens into the overall budget (Requirement 17).
            self._budget.add_tokens(tokens)
            all_errors.extend(errors)
            last_model = model

            pending_queries: list = result.get("queries", [])
            self._log.step1_generated(pending_queries)

            if not pending_queries:
                return self._empty_result(all_tokens, all_errors, last_model, total_time, avg_cols, columns_by_table)

            original_ids = {str(q.get("id")) for q in pending_queries}

            # ----------------------------------------------------------------
            # Step 2 — Validator ↔ Judge loop
            # ----------------------------------------------------------------
            structured_judge_feedback: list | None = None

            for iteration in range(self.max_judge_iterations):
                # Requirement 17.2/17.3/17.5: at each iteration boundary, stop the
                # validator↔judge loop early when the overall budget is exceeded,
                # returning the queries approved so far without raising or
                # discarding them (Step 3 assembles from all_approved_executed).
                if self._budget.exceeded():
                    self._log.warning(
                        f"Budget exceeded before iteration {iteration + 1} — "
                        f"stopping loop and returning approved queries so far."
                    )
                    break

                self._log.step2_start(iteration + 1)

                # --- 2a. Validate + correct ---
                queries_before_validation = list(pending_queries)
                start = time.perf_counter()
                pending_queries, val_tokens, val_errors = self._validator.validate_and_correct(
                    pending_queries,
                    tables,
                    aliases,
                    table_schemas=secondary_table_schemas,
                    judge_feedback=structured_judge_feedback,
                )
                total_time += time.perf_counter() - start

                self._log.validator_result(queries_before_validation, pending_queries)

                for k in all_tokens:
                    all_tokens[k] += val_tokens.get(k, 0)
                # Feed validation/correction tokens into the overall budget (Req 17).
                self._budget.add_tokens(val_tokens)
                # Tag validator errors with the query ids they belong to so errors
                # remain associated with their source query, not lost in a flat pile.
                tagged_val_errors = [
                    {**e, "query_ids": [str(q.get("id")) for q in queries_before_validation]}
                    if isinstance(e, dict) and "query_ids" not in e
                    else e
                    for e in val_errors
                ]
                all_errors.extend(tagged_val_errors)

                if not pending_queries:
                    self._log.warning(f"No queries after validation on iteration {iteration + 1} — stopping.")
                    break

                # --- 2b. Judge ---
                evaluation = judge.evaluate(pending_queries)

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

                # FIX 3: Use _original_query (preserved by _execute_queries) so
                # the next validator pass still has columns_involved intact.
                # Previously, pending_queries was rebuilt from executed result dicts
                # whose "tables" had already had columns_involved stripped, causing
                # the validator to see empty DataFrames on iteration 2+.
                pending_queries = [
                    er.get("_original_query") or er
                    for er in evaluation["rejected"]
                ]

                for er in evaluation["permanently_rejected"]:
                    qid = str(er["id"])
                    orig = er.get("_original_query") or next(
                        (q for q in pending_queries if str(q.get("id")) == qid), {}
                    )
                    pending_queries.append(orig)
                    structured_judge_feedback.append({
                        "id": er["id"],
                        "query": orig,
                        "error": (
                            f"Permanently rejected after "
                            f"{judge._rejection_counts.get(qid, 2)} attempt(s). "
                            f"History: {'; '.join(judge._accumulated_feedback.get(qid, []))}"
                        ),
                    })

                for f in evaluation["execution_failures"]:
                    # f["query"] is now a deep copy of the original (Fix 2)
                    pending_queries.append(f["query"])
                    structured_judge_feedback.append({
                        "id": f["id"],
                        "query": f["query"],
                        "error": f"Execution failed: {f['error']}",
                    })

                if not pending_queries:
                    self._log.iteration_done()
                    break

            # ----------------------------------------------------------------
            # Step 3 — Build final approved output
            # ----------------------------------------------------------------
            approved_query_ids = {str(er["id"]) for er in all_approved_executed}

            # Build a lookup from normalised qid → executed result so we can
            # recover the post-normalisation id (guaranteed clean by _normalise_ids).
            approved_executed_by_id = {str(er["id"]): er for er in all_approved_executed}

            next_id = (
                max(
                    (int(i) for i in original_ids if str(i).lstrip("-").isdigit()),
                    default=0,
                )
                + 1
            )

            approved_queries = []
            for qid in approved_query_ids:
                q = all_approved_query_dicts.get(qid)
                if not q:
                    continue
                q_copy = dict(q)

                # Always enforce a clean integer id. The original query dict
                # (_original_query) is captured before _normalise_ids runs, so it
                # can still carry a null/sentinel id. We prefer the post-normalisation
                # id stored on the executed result; fall back to a fresh incremental id.
                er = approved_executed_by_id.get(qid)
                raw_id = er.get("id") if er else None
                if isinstance(raw_id, int):
                    q_copy["id"] = raw_id
                elif isinstance(raw_id, str) and raw_id.lstrip("-").isdigit():
                    q_copy["id"] = int(raw_id)
                else:
                    q_copy["id"] = next_id
                    next_id += 1

                approved_queries.append({
                    **q_copy,
                    "response": judge.all_judgments_by_id.get(qid, {}).get("response", ""),
                    "translated_response": judge.all_judgments_by_id.get(qid, {}).get("translated_response", ""),
                    "judge_feedback": judge.all_judgments_by_id.get(qid, {}).get("feedback", ""),
                    "keyword_count": utils.count_keywords(q_copy.get("code"), kind),
                })

            self._log.step3_summary(approved_queries, elapsed=total_time)

            return {
                "result": {"queries": approved_queries},
                "token_usage": all_tokens,
                "errors": all_errors,
                "proposed_columns": columns_by_table,
                "model": last_model,
                "time_elapsed": total_time,
                "avg_cols": avg_cols,
                "executed_results": all_approved_executed,
            }

        except FileNotFoundError as e:
            logger.error("Dataset not found: %s", e)
        except Exception as e:
            raise e

    @staticmethod
    def _empty_result(tokens, errors, model, elapsed, avg_cols, columns_by_table):
        return {
            "result": {"queries": []},
            "token_usage": tokens,
            "errors": errors,
            "proposed_columns": columns_by_table,
            "model": model,
            "time_elapsed": elapsed,
            "avg_cols": avg_cols,
            "executed_results": [],
        }


class LegacySingleTableStatementGenerationAgent:
    """
    Single-table variant of StatementGenerationAgent.
    Uses the same Validator ↔ Judge loop architecture.
    """

    def __init__(
        self,
        config_path: Path,
        kind: str,
        bad_tokens: list,
        max_judge_iterations: int = 3,languages:list=["English"],
        budget: "BudgetGuard | None" = None,
    ):
        self.config_path = config_path
        self.kind = kind
        self.bad_tokens = bad_tokens
        self.max_judge_iterations = max_judge_iterations
        # Overall budget ceiling across the (multiplicative) retry loops
        # (Requirement 17). A guard may be injected for testing; otherwise one is
        # built from the query_generation config. Never raises — see BudgetGuard.
        self._budget = budget or BudgetGuard.from_config(self.config_path)

        if kind == "PANDAS":
            self.prompt = SingleTablePandasPrompt()
        else:
            self.prompt = SingleTableSQLPrompt()

        self._client = LLMClientStatementGenerator(self.config_path)
        self._validator = LLMStatementValidator(self.config_path, kind)
        self._log = PipelineLogger()
        self.languages=languages

    def generate_statements(
        self,
        dataset_path,
        alias: dict,
        kind: str,
        metadata: dict,
        max_cols: int = 10,
        sample_size: int = 5,
    ) -> dict | None:
        try:
            df, dataset_info = utils.prepare_dataset(
                dataset_path,
                [],
                max_cols,
                sample_size,
                self.bad_tokens,
            )

            self._log.section(f"Single-table generation — {dataset_path.name}  [{kind}]")

            # Prevent prompt bloat: reset accumulated dataset descriptions before
            # each run so descriptions don't append across repeated calls.
            self.prompt.reset()

            base_prompt = self.prompt.update(
                dataset_info["dataset_name"],
                dataset_info["num_rows"],
                dataset_info["num_columns"],
                metadata,
                dataset_info["columns_details"],
                dataset_info["sample_data"],
                json.dumps(alias, indent=2),self.languages,df.columns.values,alias
            )

            table_schemas = self.prompt.datasets_descriptions
            secondary_table_schemas = self.prompt._light_datasets_descriptions
            avg_cols = len(df.columns)

            datasets_path = dataset_path.parent
            executor = QueryExecutor(datasets_path=datasets_path, bad_tokens=self.bad_tokens)
            entry = {"tables": alias}

            judge = JudgementResponseAgent(
                self.config_path, kind, executor, entry, single_table=True
            )

            all_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            all_errors: list = []
            all_approved_executed: list = []
            all_approved_query_dicts: dict[str, dict] = {}
            last_model = ""
            total_time = 0.0
            original_ids: set[str] = set()

            # ----------------------------------------------------------------
            # Step 1 — One-shot initial generation
            # ----------------------------------------------------------------
            start = time.perf_counter()
            self._budget.start()  # Requirement 17: begin the overall wall-clock budget
            result, tokens, errors, model = self._client.complete(
                base_prompt,
                [df],
                alias,
                typology=kind,
                metadata=metadata,
                languages=self.languages,
            )
            total_time += time.perf_counter() - start

            for k in all_tokens:
                all_tokens[k] += tokens.get(k, 0)
            # Feed generation tokens into the overall budget (Requirement 17).
            self._budget.add_tokens(tokens)
            all_errors.extend(errors)
            last_model = model

            pending_queries: list = result.get("queries", [])
            self._log.step1_generated(pending_queries)

            if not pending_queries:
                return self._empty_result(all_tokens, all_errors, last_model, total_time, avg_cols)

            original_ids = {str(q.get("id")) for q in pending_queries}

            # ----------------------------------------------------------------
            # Step 2 — Validator ↔ Judge loop
            # ----------------------------------------------------------------
            structured_judge_feedback: list | None = None

            for iteration in range(self.max_judge_iterations):
                # Requirement 17.2/17.3/17.5: at each iteration boundary, stop the
                # validator↔judge loop early when the overall budget is exceeded,
                # returning the queries approved so far without raising or
                # discarding them (Step 3 assembles from all_approved_executed).
                if self._budget.exceeded():
                    self._log.warning(
                        f"Budget exceeded before iteration {iteration + 1} — "
                        f"stopping loop and returning approved queries so far."
                    )
                    break

                self._log.step2_start(iteration + 1)

                # --- 2a. Validate + correct ---
                queries_before_validation = list(pending_queries)
                start = time.perf_counter()
                pending_queries, val_tokens, val_errors = self._validator.validate_and_correct(
                    pending_queries,
                    [df],
                    alias,
                    table_schemas=secondary_table_schemas,
                    judge_feedback=structured_judge_feedback,
                )
                total_time += time.perf_counter() - start

                self._log.validator_result(queries_before_validation, pending_queries)

                for k in all_tokens:
                    all_tokens[k] += val_tokens.get(k, 0)
                # Feed validation/correction tokens into the overall budget (Req 17).
                self._budget.add_tokens(val_tokens)
                # Tag validator errors with the query ids they belong to so errors
                # remain associated with their source query, not lost in a flat pile.
                tagged_val_errors = [
                    {**e, "query_ids": [str(q.get("id")) for q in queries_before_validation]}
                    if isinstance(e, dict) and "query_ids" not in e
                    else e
                    for e in val_errors
                ]
                all_errors.extend(tagged_val_errors)

                if not pending_queries:
                    self._log.warning(f"No queries after validation on iteration {iteration + 1} — stopping.")
                    break

                # --- 2b. Judge ---
                evaluation = judge.evaluate(pending_queries)

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

                # FIX 3: Use _original_query (preserved by _execute_queries) so
                # the next validator pass still has columns_involved intact.
                # Previously, pending_queries was rebuilt from executed result dicts
                # whose "tables" had already had columns_involved stripped, causing
                # the validator to see empty DataFrames on iteration 2+.
                pending_queries = [
                    er.get("_original_query") or er
                    for er in evaluation["rejected"]
                ]

                for er in evaluation["permanently_rejected"]:
                    qid = str(er["id"])
                    orig = er.get("_original_query") or next(
                        (q for q in pending_queries if str(q.get("id")) == qid), {}
                    )
                    pending_queries.append(orig)
                    structured_judge_feedback.append({
                        "id": er["id"],
                        "query": orig,
                        "error": (
                            f"Permanently rejected after "
                            f"{judge._rejection_counts.get(qid, 2)} attempt(s). "
                            f"History: {'; '.join(judge._accumulated_feedback.get(qid, []))}"
                        ),
                    })

                for f in evaluation["execution_failures"]:
                    # f["query"] is now a deep copy of the original (Fix 2)
                    pending_queries.append(f["query"])
                    structured_judge_feedback.append({
                        "id": f["id"],
                        "query": f["query"],
                        "error": f"Execution failed: {f['error']}",
                    })

                if not pending_queries:
                    self._log.iteration_done()
                    break

            # ----------------------------------------------------------------
            # Step 3 — Build final approved output (single-table enforcement)
            # ----------------------------------------------------------------
            approved_query_ids = {str(er["id"]) for er in all_approved_executed}
            expected_alias = list(alias.keys())[0]

            # Build a lookup from normalised qid → executed result so we can
            # recover the post-normalisation id (guaranteed clean by _normalise_ids).
            approved_executed_by_id = {str(er["id"]): er for er in all_approved_executed}

            next_id = (
                max(
                    (int(i) for i in original_ids if str(i).lstrip("-").isdigit()),
                    default=0,
                )
                + 1
            )

            approved_queries = []
            for qid in approved_query_ids:
                q = all_approved_query_dicts.get(qid)
                if not q:
                    continue
                q_copy = dict(q)

                # Always enforce a clean integer id. The original query dict
                # (_original_query) is captured before _normalise_ids runs, so it
                # can still carry a null/sentinel id. We prefer the post-normalisation
                # id stored on the executed result; fall back to a fresh incremental id.
                er = approved_executed_by_id.get(qid)
                raw_id = er.get("id") if er else None
                if isinstance(raw_id, int):
                    q_copy["id"] = raw_id
                elif isinstance(raw_id, str) and raw_id.lstrip("-").isdigit():
                    q_copy["id"] = int(raw_id)
                else:
                    q_copy["id"] = next_id
                    next_id += 1

                tables_field = q_copy.get("tables", [])
                if len(tables_field) != 1 or tables_field[0].get("name") != expected_alias:
                    q_copy["tables"] = [{
                        "name": expected_alias,
                        "reason": tables_field[0].get("reason", "") if tables_field else "",
                        "columns_involved": tables_field[0].get("columns_involved", []) if tables_field else [],
                    }]
                approved_queries.append({
                    **q_copy,
                    "response": judge.all_judgments_by_id.get(qid, {}).get("response", ""),
                    "translated_response": judge.all_judgments_by_id.get(qid, {}).get("translated_response", ""),
                    "judge_feedback": judge.all_judgments_by_id.get(qid, {}).get("feedback", ""),
                    "keyword_count": utils.count_keywords(q_copy.get("code"), kind),
                })

            self._log.step3_summary(approved_queries, elapsed=total_time)

            return {
                "result": {"queries": approved_queries},
                "token_usage": all_tokens,
                "errors": all_errors,
                "model": last_model,
                "time_elapsed": total_time,
                "avg_cols": avg_cols,
                "executed_results": all_approved_executed,
            }

        except FileNotFoundError as e:
            logger.error("Dataset not found: %s", e)
        except Exception as e:
            raise e

    @staticmethod
    def _empty_result(tokens, errors, model, elapsed, avg_cols):
        return {
            "result": {"queries": []},
            "token_usage": tokens,
            "errors": errors,
            "model": model,
            "time_elapsed": elapsed,
            "avg_cols": avg_cols,
            "executed_results": [],
        }


# ----------------------------------------------------------------------------
# Backward-compatible re-export of the unified-backed agents (task 11.4)
# ----------------------------------------------------------------------------
#
# ``src/orqa/statement_generation.py`` imports the two agent names from this
# module:
#
#     from .agent.agent import StatementGenerationAgent, SingleTableStatementGenerationAgent
#
# To route those call sites onto the unified, mode-aware pipeline WITHOUT editing
# ``statement_generation.py``, we bind those names to the unified-backed named
# subclasses defined in ``unified_agent.py``. The original implementations are
# preserved (not deleted) in this file as ``LegacyStatementGenerationAgent`` and
# ``LegacySingleTableStatementGenerationAgent`` so the legacy budget-wiring test
# can still exercise them.
#
# This import sits at the BOTTOM of the module, after ``JudgementResponseAgent``
# (which ``unified_agent`` imports back from here) is defined, so the
# ``agent`` <-> ``unified_agent`` cycle resolves regardless of which module the
# interpreter loads first.
from .utility.unified_agent import (  # noqa: E402
    StatementGenerationAgent,
    SingleTableStatementGenerationAgent,
)
