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


class StatementGenerationAgent:
    def __init__(self, config_path: Path, kind: str, bad_tokens: list, max_judge_iterations=3):
        self.config_path = config_path
        if kind == "PANDAS":
            self.prompt = PandasStatementGenerationPrompt()
        else:
            self.prompt = SQLStatementGenerationPrompt()
        self._client = LLMClientStatementGenerator(self.config_path)
        self.bad_tokens = bad_tokens
        self.max_judge_iterations = max_judge_iterations

    def _execute_queries(self, queries, executor, entry, kind):
        executed = []
        for query in queries:
            query_id = query.get("id", "?")
            question = query.get("question", "")
            code = query.get("code", "")
            tables = copy.deepcopy(query.get("tables", []))
            for table in tables:
                table.pop("columns_involved", None)
            try:
                df_result = executor.execute(entry, query, kind).head(8)
                executed.append({"id": query_id, "code": code, "question": question, "dataframe": df_result, "tables":tables})
            except Exception as exc:
                print(code)
                print(f"⚠️  Execution failed for query #{query_id}: {exc}")
        return executed

    def _build_feedback_messages(self, rejected_executed, judgments_by_id, pending_queries):
        rejected_ids = {str(er["id"]) for er in rejected_executed}
        rejected_queries = [q for q in pending_queries if str(q.get("id")) in rejected_ids]
        feedback_lines = [
            f"- id={er['id']}: {judgments_by_id.get(str(er['id']), {}).get('Feedback', 'no feedback provided')}"
            for er in rejected_executed
        ]
        feedback_text = "The following queries were rejected, fix them:\n" + "\n".join(feedback_lines)
        print(f"Sending back {len(rejected_queries)} queries")
        return [
            {"role": "assistant", "content": json.dumps({"queries": rejected_queries}, indent=2)},
            {"role": "user", "content": feedback_text},
        ]

    def generate_statements(self, dataset_paths, aliases, kind, match, involved_cols, metadatas, max_cols=20, sample_size=5):
        columns = 0
        try:
            tables = []
            base_prompt = ""
            judge = JudgementResponseAgent(self.config_path)

            for idx, dataset_path in enumerate(dataset_paths):
                df, dataset_info = utils.prepare_dataset(
                    dataset_path, involved_cols[f"Table_{idx}"], max_cols, sample_size, self.bad_tokens,
                )
                columns += len(df.columns)
                tables.append(df)
                base_prompt = self.prompt.update(
                    dataset_info["dataset_name"], dataset_info["num_rows"], dataset_info["num_columns"],
                    metadatas[idx], dataset_info["columns_details"], dataset_info["sample_data"],
                    json.dumps(aliases, indent=2), match,
                )

            datasets_path = dataset_paths[0].parent
            executor = QueryExecutor(datasets_path=datasets_path, bad_tokens=self.bad_tokens)
            entry = {"tables": aliases}
            avg_cols = columns / len(dataset_paths)

            all_approved_executed = []
            all_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            all_errors = []
            last_model = ""
            total_time = 0.0
            feedback_messages = None
            pending_queries = None

            for iteration in range(self.max_judge_iterations):
                #print(f"\n{'#' * 60}")
                #print(f"# Statement generation — iteration {iteration + 1}/{self.max_judge_iterations}")
                #print(f"{'#' * 60}")

                start = time.perf_counter()
                result, tokens, errors, model = self._client.complete(
                    base_prompt, tables, aliases, typology=kind,
                    involved_cols=involved_cols, feedback=feedback_messages,
                )
                total_time += time.perf_counter() - start

                for k in all_tokens:
                    all_tokens[k] += tokens.get(k, 0)
                all_errors.extend(errors)
                last_model = model

                pending_queries = result.get("queries", [])
                if not pending_queries:
                    #print("⚠️  No queries returned by the statement client — stopping.")
                    break

                executed = self._execute_queries(pending_queries, executor, entry, kind)
                if not executed:
                    #print("⚠️  All queries failed execution — stopping.")
                    break

                feedback_response = judge.judge(data=executed)
                #print(feedback_response)

                judgments = feedback_response.get("result", {}).get("queries", [])
                judgments_by_id = {str(j["id"]): j for j in judgments}

                approved_executed = [er for er in executed if judgments_by_id.get(str(er["id"]), {}).get("Approved", False)]
                rejected_executed = [er for er in executed if not judgments_by_id.get(str(er["id"]), {}).get("Approved", False)]

                all_approved_executed.extend(approved_executed)
                #print(f"Iteration {iteration + 1}: {len(approved_executed)} approved, {len(rejected_executed)} rejected.")

                if not rejected_executed:
                    #print("✅ All queries approved.")
                    break

                if iteration == self.max_judge_iterations - 1:
                    #print(f"⛔ Reached max iterations ({self.max_judge_iterations}). Discarding {len(rejected_executed)} still-rejected queries.")
                    break

                feedback_messages = self._build_feedback_messages(rejected_executed, judgments_by_id, pending_queries)
                #print(feedback_messages)
            approved_query_ids = {str(er["id"]) for er in all_approved_executed}
            #approved_queries = [q for q in (pending_queries or []) if str(q.get("id")) in approved_query_ids]
            approved_query_ids = {str(er["id"]): er for er in all_approved_executed}
            approved_queries = [
                {
                    **q,
                    "response": judgments_by_id.get(str(q.get("id")), {}).get("Response", ""),
                    "judge_feedback": judgments_by_id.get(str(q.get("id")), {}).get("Feedback", ""),
                }
                for q in (pending_queries or [])
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
            #print(f"\n❌ Analysis failed: {e}")
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
        

class JudgementResponseAgent:
    def __init__(self, config_path: Path,max_tokens:int=2000):
        self.config_path = config_path
        self.prompt = JudgementResponseGenerationPrompt()
        self._client = LLMStatementJudge(self.config_path)
        self.max_tokens = max_tokens

    def judge(
        self,
        data
    ) -> dict | None:
        try:
            prompt_str=""
            prompt_str = self.prompt.update(
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