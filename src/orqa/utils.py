import copy
import json
import re
from pathlib import Path

import polars as pl
from bs4 import BeautifulSoup


def pl_read_dataset(dataset_path: Path, opts) -> pl.DataFrame:
    match dataset_path.suffix:
        case ".csv":
            return pl.read_csv(dataset_path, **opts.get("csv", {}))
        case ".parquet":
            return pl.read_parquet(dataset_path, **opts.get("parquet", {}))
        case _:
            raise ValueError(
                f"Unknown dataset format for file {dataset_path.absolute()}"
            )


def pl_scan_dataset(dataset_path: Path, opts) -> pl.LazyFrame:
    match dataset_path.suffix:
        case ".csv":
            return pl.scan_csv(dataset_path, **opts.get("csv", {}))
        case ".parquet":
            return pl.scan_parquet(dataset_path, **opts.get("parquet", {}))
        case _:
            raise ValueError(
                f"Unknown dataset format for file {dataset_path.absolute()}"
            )


def remove_file_extension(filename: str) -> str:
    # convert to Path object and stem it
    return Path(filename).stem


def clean_html_from_metadata_notes(html_text):
    """
    Clean HTML from notes field and extract plain text.
    """
    if not html_text:
        return ""

    # Parse HTML
    soup = BeautifulSoup(html_text, "html.parser")

    # Extract text and clean up whitespace
    text = soup.get_text(separator=" ", strip=True)

    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)

    # Remove &nbsp; and other HTML entities that might remain
    text = text.replace("\xa0", " ")

    return text.strip()


def load_datasets_metadata(
    metadata_path: Path, dataset_ids: str | list[str], field: str = "id"
) -> dict[str, dict]:
    """
    Load the datasets metadata from the main JSON file.
    This function is targeted for the CKAN formatted metadata.

    :param metadata_path: the path object to the metadata JSON file
    :param dataset_ids: a single dataset ID or a list of IDs
    :param field: the key field on which search for the metadata
    :return: a list of dictionaries containing the metadata, one for each identified dataset
    """
    _dataset_ids = copy.copy(dataset_ids)
    _dataset_ids = (
        {_dataset_ids} if isinstance(_dataset_ids, str) else set(_dataset_ids)
    )

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    rv = {}

    for package in metadata:
        resources = package.get("resources", [])
        for resource in resources:
            if resource.get(field) in dataset_ids:
                _dataset_ids.remove(resource.get(field))
                rv[resource.get(field)] = {
                    "package.title": package.get("title", "N/A"),
                    "resource.title": resource.get("title", "N/A"),
                    "notes": clean_html_from_metadata_notes(
                        package.get("notes", "N/A")
                    ),
                    "organization": package.get("organization", {}).get("title", "N/A"),
                    "tags": [
                        tag.get("display_name", "") for tag in package.get("tags", [])
                    ],
                    # "metadata_created": obj.get("metadata_created", "N/A"),
                    # "metadata_modified": obj.get("metadata_modified", "N/A"),
                    # "license": obj.get("license_title", "N/A"),
                    # "url": obj.get("url", "N/A"),
                }

    return rv


def load_dataset_info(
    dataset_path: Path,
    polars_opts: dict = {},
    limit_to_n_columns: int = 20,
    sample_size: int = 5,
    seed: int = 0,
) -> tuple[dict, dict]:
    """
    Load CSV and extract relevant information for the LLM.
    Returns a dict ready to be unpacked as kwargs for load_prompt.
    """
    lf = pl_scan_dataset(dataset_path, polars_opts)

    # Get first N columns (or all if fewer)
    schema = lf.collect_schema()
    df = lf.select(schema.names()[:limit_to_n_columns]).collect()

    # Build detailed column information string
    column_typings = {}
    coldetails = ""
    for col in df.columns:
        coldetails += f"\n- {col}:"
        coldetails += f"\n  Type: {df[col].dtype}"
        coldetails += f"\n  Num Unique: {df[col].n_unique()}"
        coldetails += f"\n  Null count: {df[col].null_count()}"

        column_typings[col] = df[col].dtype.is_numeric()

    sample = df.sample(min(sample_size, df.height), seed=seed)

    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_dataframe_shape=True,
        tbl_cols=limit_to_n_columns,
        # tbl_width_chars=300,
    ):
        sample = str(sample)

    # Return dictionary with keys matching load_prompt parameters
    info = {
        "id": dataset_path.stem,
        "dataset_name": dataset_path.stem,
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns_details": coldetails,
        "sample_data": sample,
        "columns": df.columns,
    }

    return info, column_typings
