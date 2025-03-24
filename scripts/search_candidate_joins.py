import logging
from logging.handlers import RotatingFileHandler
import os
import csv
import pickle
import re
import time

import duckdb
import jsonlines
import polars as pl

from orqa.utils import sanitize_string


data_path       = f'{os.path.dirname(__file__)}/../data'
# tables_path     = f'{data_path}/datasets/CAN/tables/tables_from10000_to15000'
# metadata_path   = f'{data_path}/datasets/CAN/metadata/metadata_from10000_to15000'
tables_path     = f'{data_path}/datasets/CAN/tables/tables_from0_to10000'
metadata_path   = f'{data_path}/datasets/CAN/metadata/metadata_from0_to10000'

db_path         = f'{data_path}/datasets/CAN/database/CAN.db'
valdict_path    = f'{data_path}/datasets/CAN/database/values_dict.pickle'
log_path        = f'{os.path.dirname(__file__)}/../data/log/CAN_joinsearch.log'

candidates_path = f'{data_path}/outputs/candidate_joins.csv'

os.makedirs(os.path.dirname(log_path), exist_ok=True)

logger = logging.getLogger(f'indexerLogger')
logger.setLevel(logging.DEBUG)

handler = RotatingFileHandler(log_path, mode="a", maxBytes=1024 ** 3)
log_formatter = logging.Formatter("%(asctime)s,[%(process)d],[%(levelname)s],%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(log_formatter)
logger.addHandler(handler)

print("Started Job")
logger.info("Started Job")

# max number of results we want during search
K               = 50

# the minimum number of distinct values a column must have
MIN_NUM_VALUES  = 7

# maximum number of null values (ratio) allowed in
# one column to be accepted as candidate, i.e. drop
# columns with a lot of nulls
MAX_NULL_RATIO  = 0.5

# minimum threshold for these metrics
MIN_JACCARD     = 0.2
MIN_OVERLAP     = 0.2

N_BATCH_APPEND  = 10

# tokens that we don't want to see in headers
# because they may lead to fuzzy joins
bad_columns_tokens = {'id', 'date', 'unnamed'}

file_pattern    = re.compile(r'(_\d+)?.parquet$')

# the output list where candidates are stored
# during search
candidates      = []

# load the table IDs 
table_ids = list(sorted(os.listdir(tables_path), reverse=True))

# load the resources metadata
with jsonlines.open(metadata_path) as fr:
    metadata = {rsc['id']: md for md in fr.iter() for rsc in md['resources'] if rsc['format'] == 'CSV'}

# load the values bidictionary
with open(valdict_path, 'rb') as fr:
    values = pickle.load(fr)

# create the output directory if necessary
os.makedirs(os.path.dirname(candidates_path), exist_ok=True)

# connect to the duckdb where the AllTables index is stored
con = duckdb.connect(db_path, read_only=True)

# to check that the same tables-column pair
# is not duplicated into the final output
already_used = set()


with open(candidates_path, 'w') as file:
    wr = csv.writer(file)
    wr.writerow([
        'r_tab_id'    , 's_tab_id',
        'r_col_id'    , 's_col_id',
        'r_col_name'  , 's_col_name',
        'size_r_col'  , 'size_s_col',
        'r_pkg_id'    , 's_pkg_id',
        
        'size_intersection', 
        'size_union', 
        'jaccard', 
        'overlap'
    ])


start_batch_t = time.time()

for r_tab_id in range(len(table_ids)):
# for r_tab_id in range(5):
    # read the relative table from disk
    r_df = pl.read_parquet(f"{tables_path}/{table_ids[r_tab_id]}")

    # for each col, if it is not supposed to be an ID
    # or a date column, query the index to find potentially 
    # joinable tables
    for r_col_id in range(len(r_df.columns)):
        if any(tok in r_df.columns[r_col_id].lower() for tok in bad_columns_tokens):
            continue

        col_size = r_df.shape[0]
        if not col_size or r_df.to_series(r_col_id).is_null().sum() / col_size >= MAX_NULL_RATIO:
            continue
        
        r_col_name = r_df.columns[r_col_id]
        r_pkg_id = metadata[re.sub(file_pattern, '', table_ids[r_tab_id])]['id']
        
        r_col = set(
            filter(lambda v: v in values, 
                   map(lambda v: sanitize_string(str(v)), r_df.to_series(r_col_id))
            )
        )

        if len(r_col) < MIN_NUM_VALUES:
            continue

        r_col_int = list(map(lambda v: str(values[v]), filter(lambda v: v in values, r_col)))
        
        res = con.sql(f"""
                SELECT TableId, ColumnId, COUNT(DISTINCT CellValue) AS intersec FROM AllTables
                WHERE CellValue IN ({','.join(r_col_int)})
                AND TableId <> {r_tab_id}
                GROUP BY TableId, ColumnId
                ORDER BY COUNT(DISTINCT CellValue) DESC
                LIMIT {K};
        """).fetchall()
        
        # for each tuple, check if this is a potentially valid join candidate
        # if the overlap is over some kind of threshold, then the pair is
        # meaningful (this has to be refined, maybe with an agent?)
        for s_tab_id, s_col_id, intersection in res:
            s_pkg_id = metadata[re.sub(file_pattern, '', table_ids[s_tab_id])]['id']
            
            # if they belong to the same package, drop the pair
            if r_pkg_id == s_pkg_id:
                continue

            if candidate_id := (
                r_tab_id if r_tab_id <= s_tab_id else s_tab_id, 
                s_col_id if r_tab_id <= s_tab_id else r_tab_id,
                r_col_id if r_tab_id <= s_tab_id else s_col_id,
                s_col_id if r_tab_id <= s_tab_id else r_col_id,
                ) in already_used:
                continue
            already_used.add(candidate_id)
            
            s_df = pl.scan_parquet(f"{tables_path}/{table_ids[s_tab_id]}")
        
            s_col_series = s_df.select(pl.nth(s_col_id)).collect().to_series(0)
            size = s_col_series.shape[0]
            if not size or s_col_series.is_null().sum() / size >= MAX_NULL_RATIO:
                continue
        
            s_col = set(
                filter(lambda v: v in values, 
                       map(lambda v: sanitize_string(str(v)), 
                           s_col_series
                        )
                    )
                )
            
            s_col_name = s_df.collect_schema().names()[s_col_id]
            
            intersection    = r_col & s_col
            union           = r_col | s_col
            jaccard         = round(len(intersection) / len(union), 3)
            overlap         = round(len(intersection) / min(len(r_col), len(s_col)), 3)

            # do not consider this pair if:
            #   - one of the two columns has only few value;
            #   - both jaccard and min-overlap are lower than a threshold;
            #   - columns have the same name and have jaccard/overlap==1;
            if len(s_col) < MIN_NUM_VALUES or jaccard < MIN_JACCARD or overlap < MIN_OVERLAP \
                or r_col_name == s_col_name and jaccard == 1 and overlap == 1:
                continue

            candidates.append(
                [
                    r_tab_id    , s_tab_id,
                    r_col_id    , s_col_id,
                    r_col_name  , s_col_name,
                    len(r_col)  , len(s_col),
                    r_pkg_id    , s_pkg_id,
                    
                    len(intersection), 
                    len(union), 
                    jaccard, 
                    overlap
                ]
            )

    if r_tab_id % N_BATCH_APPEND == 0 and r_tab_id > 0:
        print(f'Up to table {r_tab_id}({round(r_tab_id * 100 / len(table_ids), 3)}%);Current candidates:{len(candidates)};time:{round(time.time() - start_batch_t, 3)}s')
        logger.info(f'Up to table {r_tab_id}({round(r_tab_id * 100 / len(table_ids), 3)}%);Current candidates:{len(candidates)};time:{round(time.time() - start_batch_t, 3)}s')
        
        with open(candidates_path, 'a') as file:
            wr = csv.writer(file)
            wr.writerows(candidates)
        candidates = []
        start_batch_t = time.time()


print(f'Up to table {r_tab_id} ({round(r_tab_id * 100 / len(table_ids), 3)}%);Current candidates:{len(candidates)};time:{round(time.time() - start_batch_t, 3)}s')
logger.info(f'Up to table {r_tab_id} ({round(r_tab_id * 100 / len(table_ids), 3)}%);Current candidates:{len(candidates)};time:{round(time.time() - start_batch_t, 3)}s')
with open(candidates_path, 'a') as file:
    wr = csv.writer(file)
    wr.writerows(candidates)

print("Done")
logger.info("Done")

