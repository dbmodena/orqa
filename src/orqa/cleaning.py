"""Dataset cleaning pipelines for CKAN, Socrata, and ODS sources.

Each cleaning entry point normalizes crawled datasets before later stages:
it can skip files by filename regex, probe multiple CSV separators, drop
columns matching configured regexes, remove fully null rows and columns, and
write per-dataset statistics.

CKAN datasets require one extra heuristic on top of the shared cleaning flow:
the reader tries multiple encodings and detects a simple two-row preamble
before loading the actual table.
"""

import logging
import re
from pathlib import Path
from typing import Callable

import polars as pl
from tqdm import tqdm

from conf import OrQAConfig
from orqa.utils import (
    pl_read_dataset,
    pl_write_dataset,
    remove_null_columns,
    remove_null_rows,
)

CLEANING_EXCEPTIONS = (
    pl.exceptions.ComputeError,
    pl.exceptions.NoDataError,
    UnicodeDecodeError,
    ValueError,
)


def _compile_patterns(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    """Compile a tuple of regex strings once per cleaning run."""
    return tuple(re.compile(pattern) for pattern in patterns)


def _clone_read_opts(read_opts: dict) -> dict:
    """Create a shallow copy of nested read options so per-file tweaks stay local."""
    return {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in read_opts.items()
    }


def _with_csv_opts(read_opts: dict, **updates) -> dict:
    """Return read options with updated CSV-specific parameters."""
    cloned = _clone_read_opts(read_opts)
    cloned["csv"] = dict(cloned.get("csv", {})) | updates
    return cloned


def _exclude_pattern_columns(
    df: pl.DataFrame, patterns: tuple[re.Pattern[str], ...]
) -> pl.DataFrame:
    """Drop columns whose names match any configured regex pattern."""
    if not patterns:
        return df

    excluded_columns = [
        column
        for column in df.columns
        if any(pattern.search(column) for pattern in patterns)
    ]
    if not excluded_columns:
        return df

    return df.select(pl.all().exclude(excluded_columns))


def _select_best_separator(
    path: Path,
    read_opts: dict,
    try_separators: tuple[str, ...],
    sample_rows: int = 100,
) -> str | None:
    """Pick the CSV separator that yields the widest sampled dataframe."""
    if path.suffix != ".csv" or not try_separators:
        return read_opts.get("csv", {}).get("separator")

    base_csv_opts = dict(read_opts.get("csv", {}))
    base_csv_opts["n_rows"] = sample_rows

    best_separator = base_csv_opts.get("separator")
    best_width = -1

    for separator in try_separators:
        try:
            sample_df = pl_read_dataset(
                path, {"csv": base_csv_opts | {"separator": separator}}
            )
        except CLEANING_EXCEPTIONS:
            continue

        if sample_df.width > best_width:
            best_width = sample_df.width
            best_separator = separator

    return best_separator


def _has_two_row_preamble(df: pl.DataFrame) -> bool:
    """Detect the simple CKAN preamble format already handled by the pipeline."""
    rows = df.rows()
    if len(rows) < 2:
        return False

    r0, r1 = rows[:2]
    is_first_preamble = r0[0] is not None and all(
        value is None
        or (isinstance(value, str) and value.lower().startswith("unnamed"))
        for value in r0[1:]
    )
    is_second_empty = all(value is None for value in r1)
    return is_first_preamble and is_second_empty


def _prepare_cleaning_environment(
    cfg: OrQAConfig, *, move_downloaded_datasets: bool = False
) -> logging.Logger:
    """Prepare folders and logging for a cleaning run."""
    if move_downloaded_datasets:
        try:
            cfg.crawled_datasets_path.mkdir(parents=True, exist_ok=True)
            cfg.downloaded_datasets_path.rename(cfg.crawled_datasets_path)
        except OSError:
            pass

    cleaning_logfile = cfg.logging_path / "cleaning" / "cleaning.log"
    cleaning_logfile.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=cleaning_logfile)

    cfg.datasets_path.mkdir(parents=True, exist_ok=True)
    return logging.getLogger("cleaning")


def _iter_dataset_paths(
    cfg: OrQAConfig, filename_patterns: tuple[re.Pattern[str], ...]
):
    """Yield crawled dataset paths, skipping filenames that match configured regexes."""
    for dirpath, _, filenames in tqdm(cfg.crawled_datasets_path.walk(), desc="Directories"):
        for filename in tqdm(filenames, desc="Datasets"):
            if any(pattern.search(filename) for pattern in filename_patterns):
                continue
            yield dirpath / filename


def _clean_dataframe(
    df: pl.DataFrame, column_patterns: tuple[re.Pattern[str], ...]
) -> pl.DataFrame:
    """Apply the common dataframe cleanup used by all sources."""
    df = remove_null_rows(df)
    df = remove_null_columns(df)
    return _exclude_pattern_columns(df, column_patterns)


def _build_stats_record(
    path: Path,
    raw_rows: int,
    raw_cols: int,
    df: pl.DataFrame,
    extra_stats: dict | None = None,
) -> dict:
    """Build the statistics row stored for a cleaned dataset."""
    clean_rows, clean_cols = df.shape

    type_counts = {}
    for dtype in df.dtypes:
        dtype_str = str(dtype)
        type_counts[dtype_str] = type_counts.get(dtype_str, 0) + 1

    memory_usage = df.estimated_size("mb")
    if df.width > 0:
        total_nulls = df.null_count().sum_horizontal().sum()
        total_cells = raw_rows * raw_cols
        sparsity = (total_nulls / total_cells) if total_cells > 0 else 0
    else:
        total_nulls = 0.0
        sparsity = 0

    record = {
        "filename": path.name,
        "raw_rows": raw_rows,
        "raw_cols": raw_cols,
        "clean_rows": clean_rows,
        "clean_cols": clean_cols,
        "type_counts": str(type_counts),
        "memory_mb": round(memory_usage, 2),
        "sparsity": round(sparsity, 4),
        "total_nulls": int(total_nulls),
    }
    if extra_stats:
        record.update(extra_stats)

    return record


def _write_cleaned_dataset(cfg: OrQAConfig, path: Path, df: pl.DataFrame):
    """Persist one cleaned dataset in the cleaned datasets folder."""
    cleaned_filename = cfg.datasets_path / path.name
    pl_write_dataset(df, cleaned_filename, cfg.polars_opts.write)


def _write_stats(cfg: OrQAConfig, records: list[dict]):
    """Write the aggregated statistics table for the cleaning run."""
    stats = pl.DataFrame(records)
    cfg.statistics_path.mkdir(exist_ok=True)
    stats.write_csv(cfg.statistics_path / "datasets_stats.csv")


def _read_standard_dataset(
    path: Path, read_opts: dict, logger: logging.Logger, cfg: OrQAConfig
) -> tuple[pl.DataFrame, dict]:
    """Read a dataset using the shared separator-probing logic."""
    del logger

    separator = _select_best_separator(path, read_opts, cfg.try_separators)
    if separator is not None:
        read_opts = _with_csv_opts(read_opts, separator=separator)

    df = pl_read_dataset(path, read_opts)
    return df, {"separator": read_opts.get("csv", {}).get("separator")}


def _read_ckan_dataset(
    path: Path, read_opts: dict, logger: logging.Logger, cfg: OrQAConfig
) -> tuple[pl.DataFrame, dict] | None:
    """Read a CKAN dataset, trying encodings and stripping a simple preamble."""
    for encoding in ("utf-8", "latin-1", "utf-8-sig"):
        try:
            with open(path, "r", encoding=encoding) as file:
                if "<!DOCTYPE html>" in file.read(100):
                    continue

            separator = _select_best_separator(
                path,
                _with_csv_opts(
                    read_opts, encoding=encoding, has_header=False, skip_rows=0
                ),
                cfg.try_separators,
            )
            if separator is not None:
                read_opts = _with_csv_opts(read_opts, separator=separator)

            preamble_opts = _with_csv_opts(
                read_opts, n_rows=2, has_header=False, encoding=encoding
            )
            preamble = pl_read_dataset(path, preamble_opts)
            has_preamble = _has_two_row_preamble(preamble)
            if has_preamble:
                logger.info("Preamble detected: %s", path.name)

            dataset_opts = _with_csv_opts(
                read_opts,
                n_rows=None,
                skip_rows=2 if has_preamble else 0,
                has_header=True,
                encoding=encoding,
            )
            df = pl_read_dataset(path, dataset_opts)
            return df, {
                "encoding": encoding,
                "separator": dataset_opts.get("csv", {}).get("separator"),
                "has_preamble": has_preamble,
            }
        except CLEANING_EXCEPTIONS:
            continue

    return None


def _run_cleaning_pipeline(
    cfg: OrQAConfig,
    reader: Callable[[Path, dict, logging.Logger, OrQAConfig], tuple[pl.DataFrame, dict] | None],
    *,
    move_downloaded_datasets: bool = False,
):
    """Run the shared cleaning loop for a source-specific dataset reader."""
    filter_column_patterns = _compile_patterns(cfg.filter_column_patterns)
    filter_filenames_patterns = _compile_patterns(cfg.filter_filenames_patterns)
    logger = _prepare_cleaning_environment(
        cfg, move_downloaded_datasets=move_downloaded_datasets
    )

    records = []

    for path in _iter_dataset_paths(cfg, filter_filenames_patterns):
        read_opts = _clone_read_opts(cfg.polars_opts.read)

        try:
            loaded = reader(path, read_opts, logger, cfg)
            if loaded is None:
                continue

            df, extra_stats = loaded
        except CLEANING_EXCEPTIONS:
            continue
        except BaseException as exc:  # NOTE: kept to guard against Rust panics
            print("Strange exception: ", exc)
            continue

        raw_rows, raw_cols = df.shape
        df = _clean_dataframe(df, filter_column_patterns)
        records.append(_build_stats_record(path, raw_rows, raw_cols, df, extra_stats))
        _write_cleaned_dataset(cfg, path, df)

    _write_stats(cfg, records)


def ckan_cleaning(cfg: OrQAConfig):
    """Clean CKAN datasets with encoding/preamble detection plus shared cleanup."""
    _run_cleaning_pipeline(cfg, _read_ckan_dataset, move_downloaded_datasets=True)


def socrata_cleaning(cfg: OrQAConfig):
    """Clean Socrata datasets with shared separator probing and column filtering."""
    _run_cleaning_pipeline(cfg, _read_standard_dataset, move_downloaded_datasets=True)


def ods_cleaning(cfg: OrQAConfig):
    """Clean ODS datasets after moving crawled files into the source folder layout."""
    _run_cleaning_pipeline(
        cfg, _read_standard_dataset, move_downloaded_datasets=True
    )
