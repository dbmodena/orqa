# import zipfile
# from typing import Literal

import inflection
import unicodedata
# import polars as pl



def sanitize_string(s):
    """
    Replaces problematic characters in column names with underscores,
    normalizes accents, and strips spaces.
    """
    if not isinstance(s, str):
        return s

    # inflection
    s = inflection.underscore(s).lower()
    # normalize accents (e.g., é -> e)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')
    # replace problematic characters with underscores
    return s.replace('\n', '_').replace(' ', '_').replace('.', '_').replace('"', '_').strip()



# define a function to read the tables directly from the .zip file
# def read_table_from_zip(table_id,  zip_folder_path:str, zip_prefix:str='datasets_CAN/', return_dataframe_format:Literal['polars', 'pandas']='polars', **pl_read_kwargs):
#     with zipfile.ZipFile(zip_folder_path, 'r') as zip_tables:
#         with zip_tables.open(f'{zip_prefix}{table_id}', 'r') as table_file:
#             try:
#                 df = pl.scan_csv(table_file, ignore_errors=True, **pl_read_kwargs)
#                 colnames = df.collect_schema().names()
#                 new_colnames = list(map(sanitize_string, colnames))
#                 mapping = {
#                     c: nc if new_colnames[:i].count(nc) == 0 else f'{nc}_{new_colnames[:i].count(nc)}'
#                     for i, (c, nc) in enumerate(zip(colnames, new_colnames))
#                 }
#                 df = df.rename(mapping)
#                 
#                 schema = df.collect_schema()
#                 for attribute in schema.names():
#                     df.with_columns(pl.col(attribute).map_elements(sanitize_string, return_dtype=schema[attribute]))
#                 
#                 match return_dataframe_format:
#                     case 'polars':
#                         return df.collect()
#                     case 'pandas':
#                         return df.collect().to_pandas(use_pyarrow_extension_array=True)
#             except Exception as e:
#                 return f'Error on reading table {table_id}: {e}'
            



