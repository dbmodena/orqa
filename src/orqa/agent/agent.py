from pathlib import Path
from .. import utils
#import utils
import json
from .TaskProposer import TaskProposerLLMClient
from .StatementClient import LLMClientStatementGenerator
from .LLMClient import LLMClient
from .prompting import CandidatesDiscoveryPrompt,PandasStatementGenerationPrompt,SQLStatementGenerationPrompt,ResponseGenerationPrompt


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

            print("=" * 60)
            print("TASKS PROPOSAL RESULTS")
            #print("=" * 60)
            #print(prompt_str)
            #print("=" * 60)
            print(result)
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
    def __init__(self, config_path: Path,kind:str,bad_tokens:list):
        self.config_path = config_path
        if kind=="PANDAS":
            self.prompt = PandasStatementGenerationPrompt()
        else:
            self.prompt = SQLStatementGenerationPrompt()
        self._client = LLMClientStatementGenerator(self.config_path)
        self.bad_tokens = bad_tokens

    def generate_statements(
        self,
        dataset_paths: list[Path],
        aliases,
        kind,
        match,
        involved_cols,
        metadatas: list[dict],
        max_cols: int = 20,
        sample_size: int = 5,
    ) -> dict | None:
        try:
            prompt_str=""
            tables = []
            #table_names = [f"Table_{i}" for i in range(len(dataset_paths))]
            for idx, dataset_path in enumerate(dataset_paths):
                df,dataset_info = utils.prepare_dataset(
                    dataset_path,
                    involved_cols[f"Table_{idx}"],
                    max_cols,
                    sample_size,
                    self.bad_tokens
                )
                #print(df.columns)
                tables.append(df)
                prompt_str = self.prompt.update(
                    dataset_info["dataset_name"],
                    dataset_info["num_rows"],
                    dataset_info["num_columns"],
                    metadatas[idx],
                    dataset_info["columns_details"],
                    dataset_info["sample_data"],
                    json.dumps(aliases, indent=2),
                    match
                )
            result, tokens = self._client.complete(prompt_str, tables, aliases, typology=kind)
            return {
                    "result": result,
                    "token_usage": tokens,
                    "keyword_counts":{}
                }
        except FileNotFoundError as e:
            print(f"Error: '{e}'")
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            raise e
        


class GenerateResponseAgent:
    def __init__(self, config_path: Path,kind:str):
        self.config_path = config_path
        self.prompt = ResponseGenerationPrompt()
        self._client = LLMClient(self.config_path)

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
            result, tokens = self._client.complete(prompt_str)
            return {
                    "result": result,
                    "token_usage": tokens,
                }
        except FileNotFoundError as e:
            print(f"Error: '{e}'")
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            raise e