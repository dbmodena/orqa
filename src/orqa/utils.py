from typing import Optional, Literal
import copy
import json
import re
from pathlib import Path
import tempfile
import polars as pl
import pandas as pd
from bs4 import BeautifulSoup
import os

SQL_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING",
    "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN", "OUTER JOIN", "FULL JOIN", "CROSS JOIN",
    "UNION", "UNION ALL", "INTERSECT", "EXCEPT","LOWER", "UPPER",
    "SUM", "AVG", "MIN", "MAX", "COUNT", "RANK", "ROW_NUMBER", "DENSE_RANK", "CORR",
    "CAST", "COALESCE", "NULLIF", "CASE", "WHEN", "THEN", "ELSE",
    "DISTINCT", "LIMIT", "OFFSET", "WITH", "AS", "ON", "AND", "OR", "NOT", "IN", "EXISTS",
}

PANDAS_KEYWORDS = {
    "merge", "join", "concat", "groupby", "agg", "aggregate","corr", "lower","upper",
    "sum", "mean", "avg", "min", "max", "count", "nunique", "rank",
    "sort_values", "sort_index", "drop_duplicates", "fillna", "dropna",
    "apply", "map", "filter", "query", "where", "assign", "pivot", "melt",
    "stack", "unstack", "explode", "resample", "rolling", "expanding",
}


def pl_read_dataset(dataset_path: Path, opts: dict = {}) -> pl.DataFrame:
    match dataset_path.suffix:
        case ".csv":
            csv_defaults = {"infer_schema_length": None}
            csv_opts = {**csv_defaults, **opts.get("csv", {})}
            return pl.read_csv(dataset_path, **csv_opts)
        case ".parquet":
            return pl.read_parquet(dataset_path, **opts.get("parquet", {}))
        case _:
            raise ValueError(
                f"Unknown dataset format for file {dataset_path.absolute()}"
            )


def pl_scan_dataset(dataset_path: Path, opts: dict = {}) -> pl.LazyFrame:
    match dataset_path.suffix:
        case ".csv":
            csv_defaults = {"infer_schema_length": None}
            csv_opts = {**csv_defaults, **opts.get("csv", {})}
            return pl.scan_csv(dataset_path, **csv_opts)
        case ".parquet":
            return pl.scan_parquet(dataset_path, **opts.get("parquet", {}))
        case _:
            raise ValueError(
                f"Unknown dataset format for file {dataset_path.absolute()}"
            )


def pl_write_dataset(df: pl.DataFrame, dataset_path: Path, opts: dict = {}):
    match dataset_path.suffix:
        case ".csv":
            df.write_csv(dataset_path, **opts.get("csv", {}))
        case ".parquet":
            df.write_parquet(dataset_path, **opts.get("parquet", {}))
        case _:
            raise ValueError(
                f"Unknown dataset format for file {dataset_path.absolute()}"
            )


def remove_null_rows(df: pl.DataFrame, *exclude_columns) -> pl.DataFrame:
    expr = pl.all() if not exclude_columns else pl.all().exclude(*exclude_columns)
    return df.filter(~pl.all_horizontal(expr.is_null()))


def remove_null_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df[[s.name for s in df if not (s.null_count() == df.height)]]


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
    metadata_path: Path,
    dataset_ids: Optional[list[str]] = None,
    field: str = "id",
    source: Literal["ckan", "socrata"] = "ckan",
) -> dict[str, dict]:
    """
    Load the datasets metadata from the main JSON file.
    This function is targeted for the CKAN formatted metadata.

    :param metadata_path: the path object to the metadata JSON file
    :param dataset_ids: a single dataset ID or a list of IDs
    :param field: the key field on which search for the metadata
    :return: a list of dictionaries containing the metadata, one for each identified dataset
    """
    if dataset_ids is not None:
        dataset_ids = copy.copy(dataset_ids)
        dataset_ids = set(dataset_ids)  # ty: ignore

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    rv = {}

    if source == "ckan":
        for package in metadata:
            resources = package.get("resources", [])
            for resource in resources:
                if dataset_ids is None or resource.get(field) in dataset_ids:
                    if dataset_ids is not None:
                        dataset_ids.remove(resource.get(field))
                    rv[resource.get(field)] = {
                        "package.title": package.get("title", "N/A"),
                        "resource.title": resource.get("title", "N/A"),
                        "notes": clean_html_from_metadata_notes(
                            package.get("notes", "N/A")
                        ),
                        "organization": package.get("organization", {}).get(
                            "title", "N/A"
                        ),
                        "tags": [
                            tag.get("display_name", "")
                            for tag in package.get("tags", [])
                        ],
                        # "metadata_created": obj.get("metadata_created", "N/A"),
                        # "metadata_modified": obj.get("metadata_modified", "N/A"),
                        # "license": obj.get("license_title", "N/A"),
                        # "url": obj.get("url", "N/A"),
                    }
    else:
        for dataset in metadata:
            resource = dataset.get("resource", {})
            cls_dict = dataset.get("classification", {})

            rv[resource["id"]] = {
                "package.title": "N/A",
                "resource.title": resource.get("name", "N/A"),
                "notes": resource.get("description", "N/A"),
                "organization": "N/A",
                "tags": [
                    tag
                    for tag in cls_dict.get("domain_tags", [])
                    + cls_dict.get("tags", [])
                    + cls_dict.get("categories", [])
                ],
            }

    return rv


def load_normalized_datasets_metadata(metadata_path: Path) -> dict[str, dict]:
    """Load normalized metadata and index it by resource identifier."""
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, list):
        raise ValueError(
            f"Expected normalized metadata to be a JSON list, got {type(metadata)}"
        )

    rv = {}

    for record in metadata:
        if not isinstance(record, dict):
            continue

        resource_id = record.get("resource_id") or record.get("dataset_id")
        if not resource_id:
            continue

        rv[resource_id] = prepare_normalized_metadata_for_prompt(record)

    return rv


def prepare_normalized_metadata_for_prompt(record: dict) -> dict:
    """Adapt a normalized metadata record to the shape consumed by prompts."""
    return {
        "title": record.get("title", "N/A"),
        "description": record.get("description", "N/A"),
        "publisher": record.get("publisher", "N/A"),
        "tags": record.get("tags", []),
        "source": record.get("source", "N/A"),
        "dataset_id": record.get("dataset_id", "N/A"),
        "resource_id": record.get("resource_id", "N/A"),
        "created_at": record.get("created_at", "N/A"),
        "modified_at": record.get("modified_at", "N/A"),
        "dataset_url": record.get("dataset_url", "N/A"),
        "download_url": record.get("download_url", "N/A"),
        "format": record.get("format", "N/A"),
    }


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
    df = pl_read_dataset(dataset_path, polars_opts)

    # Build detailed column information string
    column_typings = {}
    coldetails = ""
    for col in df.columns:
        coldetails += f"\n- {col} {df[col].dtype}: {df[col].null_count()} nulls, {df[col].n_unique()} unique values."
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


def pd_read_dataset(dataset_path: Path, opts: dict = {}) -> pd.DataFrame:
    match dataset_path.suffix:
        case ".csv":
            return pd.read_csv(dataset_path, **opts.get("csv", {}))
        case ".parquet":
            return pd.read_parquet(dataset_path, **opts.get("parquet", {}))
        case _:
            raise ValueError(
                f"Unknown dataset format for file {dataset_path.absolute()}"
            )

def prepare_dataset(
    dataset_path: Path,
    involved_cols: list[str],
    limit_to_n_columns: int = 20,
    sample_size: int = 5,
    bad_tokens: list = []
) -> tuple[dict, dict]:

    df = pd_read_dataset(dataset_path,opts={"csv": {"na_values": bad_tokens, "low_memory":False},"parquet": {"na_values": bad_tokens, "low_memory":False}})
    df = df.dropna()
    # Normalize column names to handle case/whitespace mismatches
    col_map = {c.strip().lower(): c for c in df.columns}
    resolved_cols = [
        col_map[c.strip().lower()]
        for c in involved_cols
        if c.strip().lower() in col_map
    ]

    # Remove rows containing any bad token across all columns
    #if bad_tokens:
    #    mask = ~df.apply(
    #        lambda col: col.astype(str).isin([str(t) for t in bad_tokens])
    #    ).any(axis=1)
    #    df = df[mask]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        if len(df.columns) > limit_to_n_columns:
            other_cols = [c for c in df.columns if c not in resolved_cols]
            df = df[resolved_cols + other_cols[: limit_to_n_columns - len(resolved_cols)]]
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        dataset_info, _ = load_dataset_info(Path(tmp_path), {}, limit_to_n_columns, sample_size, 0)
    finally:
        os.unlink(tmp_path)
    dataset_info["dataset_name"] = dataset_path.stem
    return df, dataset_info

#def remove_bad_tokens(df,bad_tokens):
#    if bad_tokens:
#        mask = ~df.apply(
#            lambda col: col.astype(str).isin([str(t) for t in bad_tokens])
#        ).any(axis=1)
#        df = df[mask]
#    return df 
#def remove_bad_tokens(df,bad_tokens):
#    df = pd_read_dataset(dataset_path,na_values=bad_tokens)
#    df.dropna()
#    return df

def save_json(data, path: Path, indent: int = 2) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

def load_json(path: Path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def count_keywords(query: str, kind: str = "SQL") -> dict[str, int]:
        keywords = SQL_KEYWORDS if kind == "SQL" else PANDAS_KEYWORDS
        query_upper = query.upper() if kind == "SQL" else query
        counts = {}
        for kw in sorted(keywords):
            kw_pattern = kw.upper() if kind == "SQL" else kw
            pattern = rf'\b{re.escape(kw_pattern)}\b'
            matches = re.findall(pattern, query_upper, flags=re.IGNORECASE)
            if matches:
                counts[kw] = len(matches)
        return counts

