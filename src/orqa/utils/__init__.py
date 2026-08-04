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


def _pl_csv_opts_with_sniffed_separator(dataset_path: Path, opts: dict) -> dict:
    """Merge in a sniffed delimiter for Polars' CSV reader, unless the caller
    already pinned one — Polars (unlike pandas' sep=None) has no built-in
    auto-detection, so without this every non-comma-delimited file (common
    on EU open-data portals) silently mis-parses into a single column."""
    csv_defaults = {"infer_schema_length": None}
    csv_opts = {**csv_defaults, **opts.get("csv", {})}
    if "separator" not in csv_opts:
        sniffed = _sniff_csv_separator(dataset_path)
        if sniffed is not None:
            csv_opts["separator"] = sniffed
    return csv_opts


def pl_read_dataset(dataset_path: Path, opts: dict = {}) -> pl.DataFrame:
    match dataset_path.suffix:
        case ".csv":
            csv_opts = _pl_csv_opts_with_sniffed_separator(dataset_path, opts)
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
            csv_opts = _pl_csv_opts_with_sniffed_separator(dataset_path, opts)
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


def _sniff_csv_separator(dataset_path: Path, sample_bytes: int = 65_536) -> Optional[str]:
    """Cheaply detect a CSV's delimiter from a small leading sample via
    ``csv.Sniffer``, so the full read below can stay on pandas' fast C
    engine with an explicit ``sep`` instead of paying for ``sep=None`` +
    ``engine="python"`` auto-detection over the WHOLE file — some datasets
    in this corpus run to 100+MB, where the python engine is markedly
    slower. Returns ``None`` (caller falls back to pandas' comma default)
    when the sample is too ambiguous to call (e.g. a single-column file).
    """
    import csv as _csv
    try:
        with open(dataset_path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(sample_bytes)
        return _csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except (_csv.Error, OSError, UnicodeDecodeError):
        return None


def pd_read_dataset(dataset_path: Path, opts: dict = {}) -> pd.DataFrame:
    match dataset_path.suffix:
        case ".csv":
            csv_opts = dict(opts.get("csv", {}))
            if "sep" not in csv_opts and "delimiter" not in csv_opts:
                sniffed = _sniff_csv_separator(dataset_path)
                if sniffed is not None:
                    csv_opts["sep"] = sniffed
            return pd.read_csv(dataset_path, **csv_opts)
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

def _is_usable_column_label(col) -> bool:
    """
    Returns False only if the column name is unusable outright: blank or
    whitespace-only after stripping — there's no meaningful way to label,
    show, or reference such a column, so it's dropped.

    Everything else is kept, INCLUDING names that aren't safe to reference
    as a bare SQL/pandas identifier: purely numeric strings ("2023"), names
    with spaces/punctuation/non-ASCII characters, or names matching a SQL
    reserved word (e.g. "from", "group", "limit"). DuckDB and pandas accept
    any string as a column label — such names just require a quoted/bracketed
    reference (`"col name"` in SQL, `df["col name"]` in pandas) instead of a
    bare identifier, which is enforced via the statement-generation prompts
    and SQLValidator's unquoted-special-column check, not by dropping the
    column here.
    """
    return str(col).strip() != ""


def _strip_numeric_formatting(series: "pd.Series") -> "pd.Series":
    """Strip thousands separators, '%' and '$' from a string series, as a
    prelude to a numeric-parseability check — the exact cleaning
    ``clean_columns`` used to auto-coerce object columns with, factored out
    so every read-only "would this parse as a number" check in this module
    (``_numeric_parse_ratio``, ``ColumnStatistics``'s ``minority_value_groups``)
    agrees on what counts as numeric-in-disguise. Never applied to real data.
    """
    return (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.rstrip("%")
        .str.lstrip("$")
    )


def _numeric_parse_ratio(series: "pd.Series") -> Optional[float]:
    """Fraction of ``series``'s non-null values that would parse as a number
    after stripping thousands separators, '%' and '$' — kept here as a
    READ-ONLY statistic (see ``ColumnStatistics.compute``) so the planner can
    see which columns are numeric-in-disguise ("1,314", "34.10%", "$500")
    and decide for itself whether to add a ``clean`` step casting them —
    never applied automatically.

    Returns ``None`` for an all-null series (no values to judge).
    """
    non_null = series.dropna()
    if non_null.empty:
        return None
    parsed = pd.to_numeric(_strip_numeric_formatting(non_null), errors="coerce")
    return float(parsed.notna().mean())


def _generalize_value_shape(value) -> str:
    """Collapse a value to a cheap structural signature by replacing every
    run of digits with ``#`` — e.g. ``"<18"`` -> ``"<#"``, ``"90+"`` ->
    ``"#+"``, ``"18-24"`` -> ``"#-#"``, ``"minor"`` -> ``"minor"`` (unchanged,
    no digits). Used by ``ColumnStatistics.compute`` to GROUP an object
    column's rare/tail values into a handful of buckets instead of listing
    every distinct one — keeps ``minority_value_groups`` small regardless of
    table size, without making any judgment about whether a bucket is noise
    or signal (that call is left entirely to the query planner).
    """
    return re.sub(r"\d+", "#", str(value))


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level cleaning shared by ``prepare_dataset`` (the view shown to
    the LLM during analysis/planning) and ``QueryExecutor.load_tables`` (the
    view the generated code actually runs against) — kept as ONE function
    specifically so those two call sites can never silently diverge again
    (see ``QueryExecutor``'s docstring for the history of that happening).

    Stringifies non-string column labels (int/float/complex — can occur with
    parquet/Excel sources) rather than dropping them, then drops only blank/
    whitespace-only labels (see ``_is_usable_column_label``) — columns with
    spaces, punctuation, non-ASCII characters, purely-numeric names, or SQL
    reserved words are kept as-is; generated code references them via a
    quoted/bracketed identifier instead of a bare one (see the
    statement-generation prompts and ``SQLValidator``'s unquoted-special-
    column check). It deliberately does NOT touch cell values: no bad-token
    conversion, no numeric coercion, no null handling — those are now
    decisions the query planner makes explicitly (see the ``clean`` plan
    op), informed by the raw per-column statistics ``ColumnStatistics``
    computes. Also deliberately NOT row- or column-COUNT limiting: that's a
    prompt-size concern specific to the analysis view, never safe to apply
    at execution time (see ``QueryExecutor``).
    """
    df = df.rename(columns={c: str(c) for c in df.columns if not isinstance(c, str)})
    valid_columns = [c for c in df.columns if _is_usable_column_label(c)]
    return df[valid_columns]


def prepare_dataset(
    dataset_path: Path,
    involved_cols: list[str],
    limit_to_n_columns: int = 20,
    sample_size: int = 5,
    limit_to_n_rows: Optional[int] = None,
    seed: int = 0,
) -> tuple[dict, dict]:
    # Loaded raw: no bad-token->NaN conversion, no numeric coercion, no
    # null-row dropping (not even on the mandatory link columns) — those
    # were previously forced here unconditionally, but are now judgment
    # calls the query planner makes explicitly via a ``clean`` plan step,
    # informed by the raw per-column statistics ``ColumnStatistics``
    # computes from exactly this view. Pandas' own default NA-token
    # detection (blank cells, "NaN", "NULL", ...) still applies via
    # ``pd_read_dataset``'s underlying ``pd.read_csv``/``read_parquet`` —
    # that's an unambiguous "this cell is empty" signal, not a portal-
    # specific convention needing a judgment call.
    df = pd_read_dataset(dataset_path, opts={"csv": {"low_memory": False}})

    if limit_to_n_rows is not None:
        df = df.head(limit_to_n_rows)

    df = clean_columns(df)

    selected_cols = select_columns(df.columns.tolist(), limit_to_n_columns, involved_cols)
    df = df[selected_cols].copy()

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

