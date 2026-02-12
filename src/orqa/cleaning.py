"""
Clean the downloaded datasets
That step involves:
    1. Same encoding for all datasets (utf-8 vs latin-1 generally, without BOM);
    2. Removal of empty columns or rows;
    3. Removal of misformatted headers;
        That means, in many cases we do have a first header line separated with
        an empty row to the actual dataset (a preamble). We do not want to consider
        this as valuable information, thus we try to remove these cases.
        This is only the simplest case of badly reported datasets on open data portals,
        and is probably one of the few that can be targeted with a very naive heuristic;
    4. Converting into parquet format for faster access in next steps;
    5. Save files in a flattened folder, that is, removing all nested directories;
"""

import logging

import polars as pl
from tqdm import tqdm

from conf import OrQAConfig
from orqa.utils import pl_read_dataset, remove_null_columns, remove_null_rows


def cleaning(cfg: OrQAConfig):
    records = []

    encodings = ["utf-8", "latin-1", "utf-8-sig"]

    cleaning_logfile = cfg.logging_path / "cleaning" / "cleaning.log"
    cleaning_logfile.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=cleaning_logfile)
    logger = logging.getLogger("cleaning")

    read_opts = cfg.polars_opts.read

    cfg.datasets_path.mkdir(parents=True, exist_ok=True)

    for dirpath, dirnames, filenames in tqdm(
        cfg.crawled_datasets_path.walk(), desc="Directories"
    ):
        for filename in tqdm(filenames, desc="Datasets"):
            path = dirpath / filename
            loaded = False
            preamble = None

            # check for HTML-like data

            for encoding in encodings:
                try:
                    with open(path, "r", encoding=encoding) as file:
                        data = file.read(100)
                        if "<!DOCTYPE html>" in data:
                            continue

                    # identify potential preamble and correct encoding
                    read_opts["csv"] |= {
                        "n_rows": 2,
                        "has_header": False,
                        "encoding": encoding,
                    }

                    preamble = pl_read_dataset(path, read_opts)
                    r0, r1 = preamble.rows()
                    is_first_preamble = r0[0] is not None and all(
                        v is None
                        or (isinstance(v, str) and v.lower().startswith("unnamed"))
                        for v in r0[1:]
                    )
                    is_second_empty = all(v is None for v in r1)

                    has_preamble = is_first_preamble and is_second_empty
                    if is_first_preamble and is_second_empty:
                        # we have a preamble
                        logger.info(f"Preamble detected: {path.name}")

                    # update the reading options with the preamble information
                    read_opts["csv"] |= {
                        "n_rows": None,
                        "skip_rows": 2 if has_preamble else 0,
                        "has_header": True,
                        "encoding": encoding,
                    }

                    # identify the correct encoding
                    # once loaded into polars dataframe, the string encoding should
                    # be always Utf-8
                    df = pl_read_dataset(path, read_opts)

                    raw_rows, raw_cols = df.shape
                    loaded = True
                    break
                except (
                    pl.exceptions.ComputeError,
                    pl.exceptions.NoDataError,
                    UnicodeDecodeError,
                    ValueError,
                ) as e:
                    # print(filename)
                    # print(preamble)
                    # print(e)
                    # print("-" * 100)
                    continue
                except BaseException as e:  # NOTE: quite bad but for Rust panics
                    print("Strange exception: ", e)

            if not loaded:
                continue

            # remove empty rows and columns
            df = remove_null_rows(df)
            df = remove_null_columns(df)

            clean_rows, clean_cols = df.shape

            # Type distribution
            type_counts = {}
            for dtype in df.dtypes:
                dtype_str = str(dtype)
                type_counts[dtype_str] = type_counts.get(dtype_str, 0) + 1

            # Additional info
            memory_usage = df.estimated_size("mb")
            total_nulls = df.null_count().sum_horizontal().sum()
            total_cells = raw_rows * raw_cols
            sparsity = (total_nulls / total_cells) if total_cells > 0 else 0

            records.append(
                {
                    "filename": path.name,
                    "encoding": encoding,
                    "has_preamble": has_preamble,
                    "raw_rows": raw_rows,
                    "raw_cols": raw_cols,
                    "clean_rows": clean_rows,
                    "clean_cols": clean_cols,
                    "type_counts": str(type_counts),
                    "memory_mb": round(memory_usage, 2),
                    "sparsity": round(sparsity, 4),
                    "total_nulls": int(total_nulls),
                }
            )

            # save into the cleaned datasets folder
            cleaned_filename = cfg.datasets_path / path.name
            df.write_csv(cleaned_filename)

    stats = pl.DataFrame(records)
    cfg.statistics_path.mkdir(exist_ok=True)
    stats.write_csv(cfg.statistics_path / "datasets_stats.csv")
