from pathlib import Path
from .. import utils
import time
import json
from .TaskProposer import TaskProposerLLMClient
from .StatementClient import LLMClientStatementGenerator
from .StatementJudge import LLMStatementJudge
from .LLMClient import LLMClient
from .prompting import CandidatesDiscoveryPrompt,JudgementResponseGenerationPrompt,PandasStatementGenerationPrompt,SQLStatementGenerationPrompt,ResponseGenerationPrompt
from ..queries.query_execution import QueryExecutor
import pandas as pd
import copy

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

            #print("=" * 60)
            #print("TASKS PROPOSAL RESULTS")
            #print("=" * 60)
            #print(prompt_str)
            #print("=" * 60)
            #print(result)
            #print("=" * 60)
            #print(tokens)

            return {
                "dataset": dataset_info["dataset_name"],
                "tasks": result,
                "token_usage": tokens,
            }
        except FileNotFoundError as e:
            print(f"Error: '{e}'")
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            raise e


class JudgementResponseAgent:
    def __init__(self, config_path: Path, kind: str, executor, entry: dict, max_tokens: int = 2000):
        self.config_path = config_path
        self.prompt = JudgementResponseGenerationPrompt()
        self._client = LLMStatementJudge(config_path)
        self.max_tokens = max_tokens
        self.executor = executor
        self.entry = entry
        self.kind = kind
        # accumulated state across iterations
        self._rejection_counts: dict[str, int] = {}
        self._accumulated_feedback: dict[str, list[str]] = {}
        self.all_judgments_by_id: dict[str, dict] = {}
 
    def evaluate(self, pending_queries: list) -> dict:
        """
        Executes queries, judges results, builds feedback.
        Returns approved, rejected, permanently_rejected, execution_failures, feedback_messages, all_done.
        """
        executed, execution_failures = self._execute_queries(pending_queries)
 
        if not executed:
            return {
                "approved": [],
                "rejected": [],
                "permanently_rejected": [],
                "execution_failures": execution_failures,
                "feedback_messages": None,
                "all_done": False,
            }
 
        judgment_response = self._judge(executed)
        judgments = judgment_response.get("result", {}).get("queries", [])
        judgments_by_id = {str(j["id"]): j for j in judgments}
 
        # accumulate judgments globally so the generator can enrich approved queries at the end
        self.all_judgments_by_id.update(judgments_by_id)
 
        approved = [
            er for er in executed
            if judgments_by_id.get(str(er["id"]), {}).get("Approved", False)
        ]
        rejected = [
            er for er in executed
            if not judgments_by_id.get(str(er["id"]), {}).get("Approved", False)
        ]
 
        # update rejection counts and accumulated feedback per id
        for er in rejected:
            qid = str(er["id"])
            self._rejection_counts[qid] = self._rejection_counts.get(qid, 0) + 1
            feedback = judgments_by_id.get(qid, {}).get("Feedback", "no feedback")
            self._accumulated_feedback.setdefault(qid, []).append(feedback)
 
        # queries rejected more than twice are considered stuck — stop retrying them
        still_actionable = [
            er for er in rejected
            if self._rejection_counts.get(str(er["id"]), 0) < 2
        ]
        permanently_rejected = [
            er for er in rejected
            if self._rejection_counts.get(str(er["id"]), 0) >= 2
        ]
 
        feedback_messages = (
            self._build_feedback_messages(still_actionable, pending_queries)
            if still_actionable
            else None
        )
 
        return {
            "approved": approved,
            "rejected": still_actionable,
            "permanently_rejected": permanently_rejected,
            "execution_failures": execution_failures,
            "feedback_messages": feedback_messages,
            "all_done": not still_actionable and not execution_failures,
        }
 
    def _execute_queries(self, queries: list) -> tuple[list, list]:
        executed, failures = [], []
        for query in queries:
            qid = query.get("id", "?")
            tables = copy.deepcopy(query.get("tables", []))
            for table in tables:
                table.pop("columns_involved", None)
            try:
                df_result = self.executor.execute(self.entry, query, self.kind).head(8)
                executed.append({
                    "id": qid,
                    "code": query.get("code", ""),
                    "question": query.get("question", ""),
                    "dataframe": df_result,
                    "tables": tables,
                })
            except Exception as exc:
                print(f"⚠️  Execution failed for query #{qid}: {exc}")
                failures.append({"id": qid, "error": str(exc), "query": query})
        return executed, failures
 
    def _build_feedback_messages(self, rejected_executed: list, pending_queries: list) -> list:
        rejected_ids = {str(er["id"]) for er in rejected_executed}
        rejected_queries = [q for q in pending_queries if str(q.get("id")) in rejected_ids]
 
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
            {"role": "assistant", "content": json.dumps({"queries": rejected_queries}, indent=2)},
            {"role": "user", "content": feedback_text},
        ]
 
    def _judge(self, executed: list) -> dict:
        prompt_str = self.prompt.update(executed)
        result, tokens = self._client.complete(prompt_str, max_tokens=self.max_tokens)
        return {"result": result, "token_usage": tokens}
 
 
class StatementGenerationAgent:
    def __init__(self, config_path: Path, kind: str, bad_tokens: list, max_judge_iterations: int = 3):
        self.config_path = config_path
        if kind == "PANDAS":
            self.prompt = PandasStatementGenerationPrompt()
        else:
            self.prompt = SQLStatementGenerationPrompt()
        self._client = LLMClientStatementGenerator(self.config_path)
        self.bad_tokens = bad_tokens
        self.max_judge_iterations = max_judge_iterations
 
    def generate_statements(
        self,
        dataset_paths,
        aliases,
        kind,
        match,
        involved_cols,
        metadatas,
        max_cols: int = 20,
        sample_size: int = 5,
    ) -> dict | None:
        columns = 0
        try:
            tables = []
            base_prompt = ""
 
            for idx, dataset_path in enumerate(dataset_paths):
                df, dataset_info = utils.prepare_dataset(
                    dataset_path,
                    involved_cols[f"Table_{idx}"],
                    max_cols,
                    sample_size,
                    self.bad_tokens,
                )
                columns += len(df.columns)
                tables.append(df)
                base_prompt = self.prompt.update(
                    dataset_info["dataset_name"],
                    dataset_info["num_rows"],
                    dataset_info["num_columns"],
                    metadatas[idx],
                    dataset_info["columns_details"],
                    dataset_info["sample_data"],
                    json.dumps(aliases, indent=2),
                    match,
                )
 
            datasets_path = dataset_paths[0].parent
            executor = QueryExecutor(datasets_path=datasets_path, bad_tokens=self.bad_tokens)
            entry = {"tables": aliases}
            avg_cols = columns / len(dataset_paths)
 
            judge = JudgementResponseAgent(self.config_path, kind, executor, entry)
 
            all_approved_executed: list = []
            all_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            all_errors: list = []
            last_model = ""
            total_time = 0.0
            feedback_messages = None
            pending_queries: list = []
            expected_ids: set[str] = set()
 
            for iteration in range(self.max_judge_iterations):
                start = time.perf_counter()
                result, tokens, errors, model = self._client.complete(
                    base_prompt,
                    tables,
                    aliases,
                    typology=kind,
                    involved_cols=involved_cols,
                    feedback=feedback_messages,
                )
                total_time += time.perf_counter() - start
 
                for k in all_tokens:
                    all_tokens[k] += tokens.get(k, 0)
                all_errors.extend(errors)
                last_model = model
 
                pending_queries = result.get("queries", [])
                if not pending_queries:
                    print("⚠️  No queries returned by the statement client — stopping.")
                    break
 
                # validate returned IDs haven't shifted on correction iterations
                returned_ids = {str(q.get("id")) for q in pending_queries}
                if iteration > 0 and not returned_ids.issubset(expected_ids):
                    rogue = returned_ids - expected_ids
                    print(f"⚠️  Unexpected IDs returned on iteration {iteration + 1}: {rogue} — stopping.")
                    break
                expected_ids = returned_ids
 
                evaluation = judge.evaluate(pending_queries)
 
                all_approved_executed.extend(evaluation["approved"])
 
                # surface execution failures back to the generator alongside any judge feedback
                execution_failure_feedback = []
                if evaluation["execution_failures"]:
                    failure_lines = "\n".join(
                        f"- id={f['id']}: {f['error']}"
                        for f in evaluation["execution_failures"]
                    )
                    failure_queries = [f["query"] for f in evaluation["execution_failures"]]
                    execution_failure_feedback = [
                        {
                            "role": "assistant",
                            "content": json.dumps({"queries": failure_queries}, indent=2),
                        },
                        {
                            "role": "user",
                            "content": (
                                "The following queries failed execution entirely. "
                                "Fix ONLY these queries and return them with the same IDs:\n\n"
                                + failure_lines
                            ),
                        },
                    ]
 
                if evaluation["all_done"]:
                    break
 
                # merge judge feedback and execution failure feedback for next iteration
                feedback_messages = (evaluation["feedback_messages"] or []) + execution_failure_feedback
 
            # build final approved queries enriched with judge response and feedback
            approved_query_ids = {str(er["id"]) for er in all_approved_executed}
            approved_queries = [
                {
                    **q,
                    "response": judge.all_judgments_by_id.get(str(q.get("id")), {}).get("Response", ""),
                    "judge_feedback": judge.all_judgments_by_id.get(str(q.get("id")), {}).get("Feedback", ""),
                    "keyword_count":utils.count_keywords(q.get("code"),kind)
                }
                for q in pending_queries
                if str(q.get("id")) in approved_query_ids
            ]
 
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
            print(f"Error: '{e}'")
        except Exception as e:
            raise e

class GenerateResponseAgent:
    def __init__(self, config_path: Path,max_tokens:int=2000):
        self.config_path = config_path
        self.prompt = ResponseGenerationPrompt()
        self._client = LLMClient(self.config_path)
        self.max_tokens = max_tokens

    def generate_statements(
        self,
        question,
        data
    ) -> dict | None:
        try:
            prompt_str=""
            prompt_str = self.prompt.update(
                    question,
                    data
                )
            result, tokens = self._client.complete(prompt_str,max_tokens=self.max_tokens)
            return {
                    "result": result,
                    "token_usage": tokens,
                }
        except FileNotFoundError as e:
            print(f"Error: '{e}'")
        except Exception as e:
            #print(f"\n❌ Analysis failed: {e}")
            raise e
        

