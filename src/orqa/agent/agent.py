from pathlib import Path

from .. import utils
from .llm_client import LLMClient
from .prompting import CandidatesDiscoveryPrompt


class CandidatesDiscoveryAgent:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.prompt = CandidatesDiscoveryPrompt()
        self._client = LLMClient(self.config_path)

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
            print("=" * 60)
            print(prompt_str)
            print("=" * 60)
            print(result)
            print("=" * 60)
            print(tokens)

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


# if __name__ == "__main__":
#     data = {}
#     documents = [
#         r"D:\uk_small\uk_small\datasets\csv\2011-03-31-Organogram-(Senior)__c529c8da-0e24-4778-8c0a-968ff209b5b5.csv",
#         r"D:\uk_small\uk_small\datasets\csv\2011-09-30-Organogram-(Junior)__344ad9e9-a77c-4b18-a4aa-46980878a062.csv",
#         r"D:\uk_small\uk_small\datasets\csv\2011-09-30-Organogram-(Senior)__13a28011-59e7-4aa7-a41a-d3afed9552f5.csv",
#         r"D:\uk_small\uk_small\datasets\csv\2016-July-transactions__a27f90fc-e4eb-4d06-b376-f84b6b17188d.csv",
#         r"D:\uk_small\uk_small\datasets\csv\2017-03-31-Organogram-(Junior)__b08128ce-2293-4ff9-881b-57f9236b4984.csv",
#         r"D:\uk_small\uk_small\datasets\csv\2017-03-31-Organogram-(Senior)__f9c9ee5a-3adb-4670-8961-6cdd6c0ac6b5.csv",
#         r"D:\uk_small\uk_small\datasets\csv\2019-February-Return-(Forestry-Commission-England)__139700c2-fd54-4f40-bbf1-3f6e80308518.csv",
#         r"D:\uk_small\uk_small\datasets\csv\2019-February-return__5cea3557-6fb5-4998-822e-cb5cf57731fa.csv",
#         r"D:\uk_small\uk_small\datasets\csv\2019-July-Return-Forestry-England__a1c25c75-ddd4-4b3c-b4fd-e950ef148bd0.csv",
#         r"D:\uk_small\uk_small\datasets\csv\2019-Q1-transactions__ed3a0fd2-9185-43b2-89ea-c1558168355f.csv",
#         r"D:\uk_small\uk_small\datasets\csv\2020-June-Return---Forestry-England__1d790441-f671-4e38-b92e-5196c762ea45.csv",
#         r"D:\uk_small\uk_small\datasets\csv\2023-12-31-Organogram-(Senior)__86b38dcc-8047-4e0e-ac56-d80d3cf913f9.csv",
#     ]
#
#     my_agent = agent(
#         Path("prompt.md"),
#         Path("config.yaml"),
#         Path(r"D:\uk_small\uk_small\metadata\metadata.json"),
#     )
#     file_path = "results.json"
#     for document in documents:
#         id, result = my_agent.analyze(Path(document))
#         data[id] = result
#         with open(file_path, "w", encoding="utf-8") as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)
#         time.sleep(100)
