import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml


@dataclass
class PolarsOpts:
    read: dict = field(
        default_factory=lambda: {
            "csv": {"ignore_errors": True},
            "parquet": {},
            "json": {},
        }
    )
    write: dict = field(default_factory=lambda: {"csv": {}, "parquet": {}, "json": {}})
    scan: dict = field(
        default_factory=lambda: {"csv": {"ignore_errors": True}, "parquet": {}}
    )


@dataclass
class PandasOpts:
    read: dict = field(
        default_factory=lambda: {
            "csv": {
                "sep": None,
                # "encoding": "latin-1",
                "encoding": "utf-8",  # avoid BOM in many files
                "encoding_errors": "ignore",
                "on_bad_lines": "skip",
                "engine": "python",
                "nrows": 1_000,
            },
            "parquet": {},
            "json": {"nrows": 1_000},
        }
    )

    write: dict = field(
        default_factory=lambda: {"csv": {"index": False}, "parquet": {}, "json": {}}
    )


@dataclass
class Crawling:
    """
    Configuration class for the Open Data datasets
    crawling stage.
    """

    max_datasets: int
    from_dataset_index: int
    batch_fetch_metadata: int
    accept_zip: bool
    download_format: Literal["csv", "parquet", "json"]
    engine: Literal["pandas", "polars"]
    max_resource_size: int
    max_process_workers: int
    max_thread_workers: int
    verbose: bool
    search_filters: dict = field(default_factory=lambda: {})


@dataclass
class Indexing:
    """
    Configuration class for the Indexing stage
    """

    xash_size: Literal[64, 128, 256, 512]
    index_folder_path: Path
    index_database_path: Path
    max_process_workers: int

    clean_func_args: Optional[dict]
    top_k: int
    verbose: bool


@dataclass
class CandidatesDiscovery:
    """
    Configuration class for the Candidates Discovery stage.

    An agent evaluates a randomly sampled subset of all the available
    datasets; for each of them, propose a list of tasks (join, union
    or join-correlation discovery) based on available metadata and
    a sample of the dataset's rows.

    Each proposed task is then validated, to assure whether the
    proposed columns to work on actually exists and that it returns
    at least one result.
    """

    # Where the discovered candidates are stored as
    # possible executable tasks
    candidate_tasks_path: Path

    # How many datasets we will randomly sample as
    # seeds for the discovery task
    n_random_dataset_seeds: int

    # Many datasets have just 1 or very few rows
    # and usually are not very interesting
    min_dataset_height: int

    # Some datasets have a lot of columns. We might
    # limit our tasks to a smaller subset
    limit_to_n_columns: int

    # The number of randomly sampled rows we'll pass to the LLM
    # as snapshot of a dataset
    sample_size: int


@dataclass
class OrQAConfig:
    seed: int
    crawling: Crawling
    indexing: Indexing
    candidates_discovery: CandidatesDiscovery

    # Dataframe tools configurations for read/write ops
    polars_opts: PolarsOpts = field(init=False)
    pandas_opts: PandasOpts = field(init=False)

    # The base path where OrQA data is stored
    data_path: Path

    # Where all the datasets are stored once downloaded
    datasets_path: Path = field(init=False)

    # Where the datasets metadata are stored once downloaded
    metadata_path: Path = field(init=False)

    # Where all the default prompts are stored
    # This path is not under the main data_path of OrQA,
    # but in the configuration directory already present
    # in the repository
    prompts_path: Path = field(init=False)

    # Where the LiteLLM and other LLM-related config things
    # are kept
    llm_config_path: Path = field(init=False)

    logging_path: Path = field(init=False)

    def __post_init__(self):
        self.datasets_path = Path(
            self.data_path, "datasets", self.crawling.download_format
        )

        self.metadata_path = Path(self.data_path, "metadata")
        self.logging_path = Path(self.data_path, "logging_path")
        self.prompts_path = Path(__file__).parent.parent.parent.joinpath(
            "conf", "prompts"
        )
        assert self.prompts_path.exists()

        self.llm_config_path = Path(__file__).parent.parent.parent.joinpath(
            "conf", "llm"
        )

        self.pandas_opts = PandasOpts()
        self.polars_opts = PolarsOpts()


def to_bytes(b: str) -> int:
    m = re.match(r"^(\d+)((K|M|G)B)$", b)
    assert m, f"Format of bytes invalid: {b}"

    n, e = m.group(1), m.group(2)

    match e:
        case "KB":
            return int(n) * (2**10)
        case "MB":
            return int(n) * (2**20)
        case "GB":
            return int(n) * (2**30)
        case _:
            raise ValueError(f"Invalid value: {b}")


def load_config(yaml_path: Path, data_path: Path) -> OrQAConfig:
    with open(yaml_path, "r") as file:
        parsed = yaml.safe_load(file)

    seed = int(parsed["seed"])

    crawling_task = parsed["tasks"]["crawling"]
    crawling_task["max_resource_size"] = to_bytes(crawling_task["max_resource_size"])
    crawling = Crawling(**crawling_task)

    indexing_task = parsed["tasks"]["indexing"]
    index_folder_path = data_path.joinpath("blend")
    index_database_path = index_folder_path.joinpath("index.db")
    indexing = Indexing(
        **indexing_task,
        index_folder_path=index_folder_path,
        index_database_path=index_database_path,
    )

    candidates_discovery_task = parsed["tasks"]["candidates_discovery"]
    candidate_tasks_path = data_path.joinpath("candidate_tasks.json")
    candidates_discovery = CandidatesDiscovery(
        candidate_tasks_path, **candidates_discovery_task
    )

    orqa_cfg = OrQAConfig(
        seed=seed,
        crawling=crawling,
        indexing=indexing,
        candidates_discovery=candidates_discovery,
        data_path=data_path,
    )

    for engine in ["pandas", "polars"]:
        if engine not in parsed:
            continue
        engine_cfg = parsed[engine]
        for op in ["read", "write"]:
            if op not in engine_cfg:
                continue
            for format in ["csv", "parquet", "json"]:
                match (engine, op):
                    case ("pandas", "read"):
                        orqa_cfg.pandas_opts.read[format] = orqa_cfg.pandas_opts.read[
                            format
                        ] | engine_cfg[op].pop(format, {})
                    case ("pandas", "write"):
                        orqa_cfg.pandas_opts.write[format] = orqa_cfg.pandas_opts.write[
                            format
                        ] | engine_cfg[op].pop(format, {})
                    case ("polars", "read"):
                        orqa_cfg.polars_opts.read[format] = orqa_cfg.polars_opts.read[
                            format
                        ] | engine_cfg[op].pop(format, {})
                    case ("pandas", "read"):
                        orqa_cfg.polars_opts.read[format] = orqa_cfg.polars_opts.read[
                            format
                        ] | engine_cfg[op].pop(format, {})

    return orqa_cfg
