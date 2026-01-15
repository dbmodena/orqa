"""
Candidates Discovery Stage

In this stage, candidates for actual dataset generation in next steps
are proposed through an agentic step.

1 - The CandidatesDiscoveryAgent agent analyses a random subset of datasets
    drawned from the whole available collection;
        a. for each dataset, it inspects its available metadata and a sample
            of few rows;
        b. if the dataset is not recognized as valid, any

"""

import json
import os
import random
import time
from pathlib import Path

import dotenv
from tqdm import tqdm
import polars as pl

from conf import OrQAConfig
from blend import BLEND
from blend.utils import clean, remove_null_rows, remove_null_columns

from .agent import agent
from .utils import load_datasets_metadata, remove_file_extension, pl_read_dataset

# make the API key for LLM available
dotenv.load_dotenv(Path(__file__).parent.parent.joinpath(".env"))


def sample_seed_datasets(
    datasets_path: Path, n_datasets_to_sample: int, seed: int
) -> list[tuple[str, Path, str, str]]:
    datasets = os.listdir(datasets_path)

    assert len(datasets) >= n_datasets_to_sample, (
        f"Too many datasets to sample: {len(datasets)} < {n_datasets_to_sample}"
    )

    random.seed(seed)
    sample = random.sample(datasets, n_datasets_to_sample)

    # remove the filetype extension from each sample filename
    sample = [(remove_file_extension(f), datasets_path.joinpath(f)) for f in sample]

    sample = [
        (
            filename,
            filepath,
            *(filename.split("::") if "::" in filename else ("", filename)),
        )
        for filename, filepath in sample
    ]

    return sample  # ty: ignore


def generate_tasks(cfg: OrQAConfig):
    datasets_path = cfg.datasets_path

    # sample dataset seeds for the candidates discovery step
    print("Sample seed datasets...")
    sample = sample_seed_datasets(
        datasets_path, cfg.candidates_discovery.n_random_dataset_seeds, cfg.seed
    )

    # for each of these sample datasets, fetch its relative metadata
    print("Loading metadata for sampled datasets...")
    metadata = load_datasets_metadata(
        cfg.metadata_path.joinpath("metadata.json"),
        [s[3 if len(s) == 4 else 2] for s in sample],
    )

    litellm_config_path = cfg.llm_config_path.joinpath("litellm.yaml")

    # setup the Agent
    # In this case, we call this entity "agent", even if
    # it is just a wrapper of a LLM client, without any
    # needs of tool-calling or memory or other properties
    print("Loading LLM-agent")
    _agent = agent.CandidatesDiscoveryAgent(litellm_config_path)

    results = {}

    for dataset_filename, dataset_path, dataset_name, dataset_id in sample:
        _metadata = metadata[dataset_id]
        _format = dataset_path.suffix.replace(".", "")

        results[dataset_id] = _agent.propose_tasks(
            dataset_path,
            _format,
            _metadata,
            cfg.polars_opts.scan,
            cfg.candidates_discovery.min_dataset_height,
            cfg.candidates_discovery.limit_to_n_columns,
            cfg.candidates_discovery.sample_size,
            seed=cfg.seed,
        )
        time.sleep(5)

    with open(
        cfg.candidates_discovery.candidate_tasks_path, "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    return results


def execute_tasks(cfg: OrQAConfig, tasks: dict) -> list[dict]:
    datasets_path = cfg.datasets_path
    _format = cfg.datasets_format

    # instantiate the BLEND index
    index = BLEND(
        cfg.indexing.index_database_path,
        clean_args=cfg.indexing.clean_args,
        xash_size=cfg.indexing.xash_size,
    )

    top_k = cfg.candidates_discovery.candidates_per_task

    # a collection where we will store our effective candidates as dictionaries
    candidates = []

    # for each task, execute it over the index
    for dataset_id, task_set in tqdm(tasks.items(), desc="Executing tasks: "):
        if not task_set:
            continue
        dataset_filename = task_set["dataset"]
        dataset_path = datasets_path.joinpath(f"{dataset_filename}.{_format}")

        _tasks = task_set["tasks"]

        union_tasks = _tasks["union_tasks"]
        join_tasks = _tasks["join_tasks"]
        join_correlation = _tasks["join_correlation_tasks"]

        df = pl_read_dataset(dataset_path, cfg.polars_opts.read)

        for task in tqdm(union_tasks, desc="Union tasks", position=1, leave=False):
            columns = task["columns"]
            table = df.select(columns).rows()

            top_res = index.union_search(table, top_k)

            for candidate in top_res:
                cand_id = candidate[0]
                candidates.append(
                    {
                        "Q": dataset_filename,
                        "R": cand_id,
                        "task": "U",
                    }
                )

        for task in tqdm(join_tasks, desc="Join tasks", position=1, leave=False):
            columns = task["columns"]
            if len(columns) == 1:
                column = df.get_column(columns[0]).to_list()
                top_res = index.single_column_join_search(column, top_k)
            else:
                table = df.select(columns).unique().rows()

                # print(table)
                # import pandas as pd
                #
                # pd_df = pd.DataFrame(table)
                # print("\n\n>>> ", pd_df.head(), "\n\n\n")
                top_res = index.multi_column_join_search(table, top_k, verbose=False)
            for cand_id in top_res:
                cand_id = cand_id[0]

                candidates.append(
                    {
                        "Q": dataset_filename,
                        "R": cand_id,
                        "task": "J",
                    }
                )

        for task in tqdm(
            join_correlation, desc="Join-Correlation tasks", position=1, leave=False
        ):
            # get the column names
            key_column = task["join_column"]
            target_column = task["correlation_column"]

            # get the actual column
            keys = df.get_column(key_column)
            targets = df.get_column(target_column)

            continue
            # TODO: check dtype for targets
            top_res = index.join_correlation_search(
                keys, targets, top_k, hash_size=cfg.indexing.hash_size
            )

            for candidate in top_res:
                cand_id = candidate[0]
                candidates.append({"Q": dataset_filename, "R": cand_id, "task": "JC"})

    return candidates


def candidates_discovery(cfg: OrQAConfig):
    # generated_tasks = generate_tasks(cfg)
    with open(
        cfg.candidates_discovery.candidate_tasks_path, "r", encoding="utf-8"
    ) as file:
        generated_tasks = json.load(file)

    discovered_candidates = execute_tasks(cfg, generated_tasks)
    discovered_candidates = pl.from_records(discovered_candidates, orient="row")
    discovered_candidates.write_csv(cfg.candidates_discovery.candidates_results_path)
    # with open(cfg.candidates_discovery.candidates_results_path, "w") as file:
    #     json.dump(discovered_candidates, file)
