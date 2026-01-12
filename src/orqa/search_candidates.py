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

from conf import OrQAConfig

from .agent import agent
from .utils import load_datasets_metadata, remove_file_extension

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


def candidates_discovery(cfg: OrQAConfig):
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
            cfg.polars_opts.scan[_format],
            cfg.candidates_discovery.min_dataset_height,
            cfg.candidates_discovery.limit_to_n_columns,
            cfg.candidates_discovery.sample_size,
            seed=cfg.seed,
        )
        time.sleep(10)

    with open(
        cfg.candidates_discovery.candidate_tasks_path, "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
