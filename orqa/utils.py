import logging
import os
import re
import unicodedata
from typing import Literal
from functools import lru_cache

import inflection
import polars as pl



# polars df-to-str configuration
pl_str_config = {
    'tbl_hide_dataframe_shape': True,
    'tbl_width_chars': 1000,
    'tbl_formatting': 'MARKDOWN',
    'tbl_cols': 20
}


@lru_cache(maxsize=int(1e6))
def is_num(x) -> bool:
    "Very very simple solution, but for many cases works"
    if isinstance(x, str):
        x = x.replace(',', '.').replace('%', ' ')
    try: 
        float(x)
    except: 
        return False
    return True


replace_chars = "\n \\\"()[]"
characters_translator = str.maketrans(replace_chars, "_" * len(replace_chars))

@lru_cache(maxsize=int(1e6))
def sanitize_string(s, mode: Literal["base", "complex"] | None = None) -> str:
    if not isinstance(s, str):
        return str(s)
    
    match mode:
        case None:
            return str(s)
        case "base":
            return str(s).lower()
        case "complex":
            # inflection
            s = inflection.underscore(s).lower()
            # normalize accents (e.g., é -> e)
            s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')
            # replace problematic characters with underscores
            return s.translate(characters_translator).strip()
        case _:
            raise ValueError(f"Unsupported mode: {mode}")



def get_resource_metadata(rsc_id, metadata):
    # the pure resource ID should be the one without the underscore _#value 
    rsc_id = re.sub(r'(_\d+)?.parquet$', '', rsc_id)
    rsc = next(
        filter(
            lambda r: r['id'] == rsc_id, metadata[rsc_id]['resources']))
    
    pkg_meta = metadata[rsc_id]
    
    # get metadata and tags if present
    pkg_keywords = []
    if 'keywords' in pkg_meta:
        if 'en' in pkg_meta['keywords']:
            pkg_keywords = pkg_meta['keywords']['en']
        else:
            pkg_keywords = pkg_meta['keywords']

    pkg_tags = []
    if 'tags' in pkg_meta:
        pkg_tags = pkg_meta['tags']

    pkg_id = pkg_meta['id']
    pkg_name = pkg_meta['name']     if 'name' in pkg_meta else None
    pkg_notes = pkg_meta['notes']   if 'notes' in pkg_meta else None
    
    rsc_name = rsc['name']          if 'name' in rsc else None
    
    org_name = org_title = org_desc = None
    if 'organization' in pkg_meta and 'name' in pkg_meta['organization']:
        org_name = pkg_meta['organization']['name']

    if 'organization' in pkg_meta and 'title' in pkg_meta['organization']:
        org_title = pkg_meta['organization']['title']

    if 'organization' in pkg_meta and 'description' in pkg_meta['organization']:
        org_desc = pkg_meta['organization']['description']

    jurisdiction = pkg_meta['jurisdiction'] if 'jurisdiction' in metadata[rsc_id] else None 
    
    return rsc_id, rsc_name, pkg_id, pkg_name, pkg_notes, pkg_keywords, pkg_tags, org_name, org_title, org_desc, jurisdiction




def map_dtype_to_sql(dtype: pl.DataType):    
    if isinstance(dtype, pl.Int64) or dtype.is_integer():
        return 'BIGINT'
    elif isinstance(dtype, pl.Float64) or dtype.is_numeric():
        return 'FLOAT'
    elif isinstance(dtype, pl.String):
        return 'VARCHAR(255)'
    elif isinstance(dtype, pl.Date):
        return 'DATE'
    elif isinstance(dtype, pl.Datetime):
        return 'DATETIME'
    else:
        return 'VARCHAR(255)'  # Default case for unknown types

def create_table_sql(df: pl.DataFrame, table_name):    
    columns_dtypes = []
    for column, dtype in df.schema.items():
        sql_type = map_dtype_to_sql(dtype)
        columns_dtypes.append((column, sql_type))
    
    columns_sql = ",\n    ".join(map(lambda cd: f"{cd[0], {cd[1]}}", columns_dtypes))
    create_table_stmt = f"CREATE TABLE {table_name} (\n    {columns_sql}\n);"
    return create_table_stmt, columns_dtypes



def get_all_data(rsc_id, tables_path, metadata, col_name: str, 
                 max_length_notes: int = 500, max_num_columns: int = 20, num_rows_sample: int = 5,
                 sql_table_name: str = 'R', 
                 clean_headers: Literal["base", "complex"] | None = None,
                 clean_elements: Literal["base", "complex"] | None = None,
                 pl_str_config: dict = pl_str_config):
    rsc_id, rsc_name, _, pkg_name, pkg_notes, pkg_keywords, pkg_tags, org_name, org_title, org_desc, jur = get_resource_metadata(rsc_id, metadata)
    pkg_notes = re.sub(r"((\<[^\>]+\>)|\n|\r|\t)", " ", pkg_notes)[:max_length_notes]    
    col_name= sanitize_string(col_name, clean_headers)

    df = (
        pl.scan_parquet(f'{tables_path}/{rsc_id}.parquet')
        .rename(lambda s: sanitize_string(s, clean_headers))
        .select(pl.all().map_elements(
           lambda s: sanitize_string(s, clean_elements), pl.String)
        )
        .collect()
    )

    # dtype conversion to bool/int/float
    for column in df.columns:
        try:
            if df.select(column).drop_nulls().unique().shape[0] == 2:
                df = df.with_columns(pl.col(column).cast(pl.Boolean))
                continue                
        except: pass

        try:
            dtype = pl.Float32 if any(',' in str(x) or '.' in str(x) for x in set(df.select(column).sample(1000, with_replacement=True).to_series())) else pl.Int32
            df = df.with_columns(pl.col(column).cast(dtype))
        except: continue
    
    # drop null columns
    df = df[[c for c in df.columns if df[c].is_null().sum() < df.shape[0]]]


    df = df.drop(col_name).insert_column(0, df.get_column(col_name)).select(df.columns[:max_num_columns])
    sql_schema, columns_dtypes = create_table_sql(df, sql_table_name)

    with pl.Config(**pl_str_config):
        df_str = str(df.select(df.columns[:max_num_columns]).sample(num_rows_sample, with_replacement=True))

    return (
        rsc_id, rsc_name, pkg_name, pkg_notes, pkg_keywords, pkg_tags, 
        org_name, org_title, org_desc, jur,
        col_name, df, df_str, sql_schema, columns_dtypes
    )


def setup_logger(log_path: str, logger_name: str, on_file: bool = True, on_stdout: bool = True, level: int | str = logging.INFO, keep_last: int | None = 3):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    log_formatter = logging.Formatter("%(asctime)s,[%(process)d],[%(levelname)s],%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    if on_file:
        handler = logging.FileHandler(log_path)
        handler.setFormatter(log_formatter)
        logger.addHandler(handler)

    if on_stdout:
        stdout_hanlder = logging.StreamHandler()
        logger.addHandler(stdout_hanlder)

    if keep_last:
        # keep only last n log files
        old_logs =  sorted([f for f in os.listdir(os.path.dirname(log_path)) 
                            if f.startswith(os.path.basename(log_path)[:5])], reverse=True)
        log_to_delete = old_logs[3:] if len(old_logs) > keep_last else []
        for log_to_del in log_to_delete:        
            os.remove(os.path.join(os.path.dirname(log_path), log_to_del))

    return logger

