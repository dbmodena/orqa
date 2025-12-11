import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml


@dataclass
class PolarsOpts:
    read: dict = field(default_factory=lambda: {"csv": {}, "parquet": {}, "json": {}})
    write: dict = field(default_factory=lambda: {"csv": {}, "parquet": {}, "json": {}})
    scan: dict = field(default_factory=lambda: {"csv": {}, "parquet": {}})


@dataclass
class PandasOpts:
    read: dict = field(
        default_factory=lambda: {
            "csv": {
                "sep": None,
                "encoding": "latin-1",
                "encoding_errors": "ignore",
                "on_bad_lines": "skip",
                "engine": "python",
                "nrows": 1_000,
            },
            "parquet": {},
            "json": {},
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
class OrQAConfig:
    crawling: Crawling
    indexing: Indexing
    polars_opts: PolarsOpts = field(init=False)
    pandas_opts: PandasOpts = field(init=False)
    data_path: Path
    datasets_path: Path = field(init=False)
    metadata_path: Path = field(init=False)
    logging_path: Path = field(init=False)

    def __post_init__(self):
        self.datasets_path = Path(
            self.data_path, "datasets", self.crawling.download_format
        )

        self.metadata_path = Path(self.data_path, "metadata")
        self.logging_path = Path(self.data_path, "logging_path")

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

    crawling_task = parsed["tasks"]["crawling"]
    crawling_task["max_resource_size"] = to_bytes(crawling_task["max_resource_size"])

    indexing_task = parsed["tasks"]["indexing"]

    index_folder_path = data_path.joinpath("BLEND")
    index_database_path = index_folder_path.joinpath("index.db")
    indexing = Indexing(
        **indexing_task,
        index_folder_path=index_folder_path,
        index_database_path=index_database_path,
    )

    return OrQAConfig(
        crawling=Crawling(**crawling_task), indexing=indexing, data_path=data_path
    )
