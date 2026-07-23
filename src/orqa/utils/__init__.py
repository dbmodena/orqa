from typing import Optional, Literal
import copy
import json
import logging
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
    "stack", "unstack", "explode", "resample", "rolling", "expanding", "fit","predict"
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


def select_columns(
    all_columns: list,
    limit_to_n_columns: int,
    involved_cols: list | None = None,
) -> list:
    """
    Pick at most ``limit_to_n_columns`` columns out of ``all_columns``:
    any ``involved_cols`` (known join/union columns, matched case-
    insensitively) are force-included first, then the remaining budget is
    filled with the rest of the columns in their original file order.

    With no ``involved_cols`` this is simply "the first N columns" —
    deterministic and reproducible across runs/phases for the same table.
    """
    col_map = {str(c).strip().lower(): c for c in all_columns}
    resolved_cols = [
        col_map[str(c).strip().lower()]
        for c in (involved_cols or [])
        if str(c).strip().lower() in col_map
    ]

    remaining_budget = max(0, limit_to_n_columns - len(resolved_cols))
    other_cols = [c for c in all_columns if c not in resolved_cols]
    return resolved_cols + other_cols[:remaining_budget]


def polars_column_details(df: pl.DataFrame) -> tuple[str, dict]:
    """
    Build the per-column dtype/null/unique-count text block and the
    numeric-typing map for an ALREADY column-limited polars DataFrame —
    callers are responsible for narrowing ``df`` to the columns that
    matter before calling this (no truncation happens here).
    """
    column_typings = {}
    coldetails = ""
    for col in df.columns:
        coldetails += f"\n- {col} {df[col].dtype}: {df[col].null_count()} nulls, {df[col].n_unique()} unique values."
        column_typings[col] = df[col].dtype.is_numeric()
    return coldetails, column_typings


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
    df = df.select(select_columns(df.columns, limit_to_n_columns))

    coldetails, column_typings = polars_column_details(df)

    sample = df.sample(min(sample_size, df.height), seed=seed)

    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_dataframe_shape=True,
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


def prepare_dataframe(df: pd.DataFrame, alias: str, logger=None) -> pd.DataFrame:
    """Normalize a DataFrame for pipeline processing.

    - Converts numeric column labels to strings
    - Logs warning if conversion was needed (with alias and count)

    Returns the normalized DataFrame.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Detect columns that are not already strings
    non_str_cols = [c for c in df.columns if not isinstance(c, str)]
    if non_str_cols:
        logger.warning(
            "Dataset '%s' has %d numeric column label(s) — converting to str.",
            alias,
            len(non_str_cols),
        )

    df.columns = df.columns.astype(str)
    return df


import re

def _is_valid_column_name(col) -> bool:
    """
    Returns False if the column name:
    - is a numeric type (int, float, complex)
    - is purely numeric when represented as a string
    - contains characters illegal in SQL identifiers or pandas query strings
    """
    # Reject actual numeric types (can occur with parquet/Excel column labels)
    if isinstance(col, (int, float, complex)):
        return False

    stripped = str(col).strip()

    # Discard purely numeric string representations (e.g. "0", "123", "3.14")
    try:
        float(stripped)
        return False
    except ValueError:
        pass

    # Must start with a letter or underscore (SQL/pandas identifier rule)
    if not re.match(r'^[A-Za-z_]', stripped):
        return False

    # Discard names containing characters illegal in SQL/pandas identifiers
    # Allowed: letters, digits, underscores
    if re.search(r'[^\w]', stripped, re.ASCII):
        return False

    return True


def _coerce_numeric_like_columns(
    df: pd.DataFrame, threshold: float = 0.9
) -> pd.DataFrame:
    """Convert object columns that are numeric-in-disguise to real numerics.

    Open-data columns frequently mix numeric values with formatting
    ("1,314", "34.10%", "$500") or stray non-numeric tokens ("R", "s"),
    which leaves the column as dtype object and makes row values type-
    inconsistent — joins and comparisons then crash on type mismatches.
    A column whose non-null values are >= ``threshold`` parseable as
    numbers (after stripping thousands separators, '%' and '$') is
    converted with ``errors='coerce'``: the column gets one consistent
    dtype and the stray tokens become NaN in place, instead of their rows
    being dropped. Percent values keep their face value ("34.1%" -> 34.1).

    Below-threshold columns are genuinely categorical/text and are left
    untouched.
    """
    for col in df.columns:
        if df[col].dtype != object:
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        cleaned = (
            non_null.astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.rstrip("%")
            .str.lstrip("$")
        )
        parsed = pd.to_numeric(cleaned, errors="coerce")
        if parsed.notna().mean() >= threshold:
            df[col] = parsed.reindex(df.index)
    return df


def prepare_dataset(
    dataset_path: Path,
    involved_cols: list[str],
    limit_to_n_columns: int = 20,
    sample_size: int = 5,
    bad_tokens: list = [],
    limit_to_n_rows: Optional[int] = None,
    seed: int = 0,
) -> tuple[dict, dict]:
    # ``na_values=bad_tokens`` already converts aliased-NaN tokens from open
    # data ('R', 'None', ...) into real NaN at read time, so they can never
    # appear as join/filter values downstream. No global ``df.dropna()`` here:
    # dropping every row with a null in ANY column emptied wide open-data
    # tables outright (nulls in never-used columns killed rows the selected
    # columns needed), which starved generation and validation of data.
    # Nulls are instead handled per-column below, and rows are only dropped
    # where a null actually breaks something: the mandatory link columns.
    df = pd_read_dataset(dataset_path, opts={"csv": {"na_values": bad_tokens, "low_memory": False}, "parquet": {"na_values": bad_tokens, "low_memory": False}})

    if limit_to_n_rows is not None:
        df = df.head(limit_to_n_rows)

    # ── drop columns with illegal / purely-numeric names ──────────────────────
    valid_columns = [c for c in df.columns if _is_valid_column_name(c)]
    df = df[valid_columns]
    # ──────────────────────────────────────────────────────────────────────────

    selected_cols = select_columns(df.columns.tolist(), limit_to_n_columns, involved_cols)
    df = df[selected_cols].copy()

    # Clean only the columns that survived selection (cleaning before
    # selection would let nulls/typing in discarded columns affect kept rows).
    df = _coerce_numeric_like_columns(df)

    # Rows with a null join key can never match: drop them here so merges on
    # the mandatory link columns don't crash or silently mismatch. This is the
    # only row-level null handling — single-table runs (no involved_cols) keep
    # every row.
    col_map = {str(c).strip().lower(): c for c in df.columns}
    resolved_cols = [
        col_map[str(c).strip().lower()]
        for c in involved_cols
        if str(c).strip().lower() in col_map
    ]
    if resolved_cols:
        df = df.dropna(subset=resolved_cols)

    dataset_info, column_typings = extract_dataset_info(df, sample_size=sample_size, seed=seed)
    dataset_info["dataset_name"] = dataset_path.stem
    dataset_info["id"] = dataset_path.stem

    return df, dataset_info

def extract_dataset_info(
    df: pd.DataFrame,
    sample_size: int = 5,
    seed: int = 0,
) -> tuple[dict, dict]:
    """
    Extract dataset info from an already-filtered pandas DataFrame.
    Returns a dict ready to be unpacked as kwargs for load_prompt, and column typings.
    """
    # Build detailed column information string
    column_typings = {}
    coldetails = ""
    for col in df.columns:
        null_count = df[col].isna().sum()
        n_unique = df[col].nunique()
        dtype = df[col].dtype
        coldetails += f"\n- {col} {dtype}: {null_count} nulls, {n_unique} unique values."
        column_typings[col] = pd.api.types.is_numeric_dtype(df[col])

    sample = df.sample(min(sample_size, len(df)), random_state=seed)
    sample_md = sample.to_markdown(index=False)

    info = {
        "id": "dataframe",
        "dataset_name": "dataframe",
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns_details": coldetails,
        "sample_data": sample_md,
        "columns": df.columns.tolist(),
    }

    return info, column_typings
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

