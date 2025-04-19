import re
import os
import csv
import sys
import time
import pickle
import logging
from logging.handlers import RotatingFileHandler

import duckdb
import jsonlines
import polars as pl
from tqdm import tqdm

from orqa.utils import sanitize_string


def main(tag: str = "CAN", 
         from_: int = 0, 
         to_: int = "END"):    
    
    data_path       = f'{os.path.dirname(__file__)}/../data'
    tables_path     = f'{data_path}/datasets/{tag}/tables/tables_from{from_}_to{to_}'
    metadata_path   = f'{data_path}/datasets/{tag}/metadata/metadata_from{from_}_to{to_}.jsonl'
    db_path         = f'{data_path}/datasets/{tag}/database/blend.db'
    valdict_path    = f'{data_path}/datasets/{tag}/database/values_dict.pickle'
    log_path        = f'{data_path}/log/{tag}/2_join_search.log'

    candidates_path = f'{data_path}/outputs/{tag}/candidate_joins_test.csv'


    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger('JoinSearchLogger')
    logger.setLevel(logging.DEBUG)

    handler = RotatingFileHandler(log_path, mode="a", maxBytes=1024 ** 3)
    log_formatter = logging.Formatter("%(asctime)s,[%(process)d],[%(levelname)s],%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)

    logger.info("Started Job")

    # number of tables for which do the search
    BUDGET          = 100

    # max number of results we want during search
    K               = 50

    # the minimum number of distinct values a column must have
    MIN_NUM_VALUES  = 10

    # minimum height for a table
    MIN_HEIGHT      = 30

    # maximum number of null values (ratio) allowed in
    # one column to be accepted as candidate, i.e. drop
    # columns with a lot of nulls
    MAX_NULL_RATIO  = 0.2

    # minimum threshold for these metrics
    MIN_JACCARD     = 0.2
    MIN_OVERLAP     = 0.3

    # more fine grained cleaning to get interesting  
    CLEAN_MODE = "complex"

    N_BATCH_APPEND  = 20

    # if we accept or not tables with the same schema
    ACCEPT_SAME_SCHEMA = True

    # tokens that we don't want to see in headers
    # because they may lead to fuzzy joins
    bad_columns_tokens = {'id', 'date', 'unnamed'}

    # pattern for resource name extraction
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

    # write the header row for the result CSV file
    with open(candidates_path, 'w') as file:
        wr = csv.writer(file)
        wr.writerow([
            'r_rsc_id', 
            's_rsc_id',
            'r_col_id', 
            's_col_id',
            'r_col_name', 
            's_col_name',
            'size_r_col', 
            'size_s_col',
            'r_pkg_id', 
            's_pkg_id',
            'size_intersection', 
            'size_union', 
            'jaccard', 
            'overlap'
        ])

    start_batch_t = time.time()

    for r_tab_id in tqdm(range(min(BUDGET, len(table_ids))), disable=False):
        # read the relative table from disk        
        r_df = pl.read_parquet(f"{tables_path}/{table_ids[r_tab_id]}")
        r_num_rows = r_df.shape[0]
        if r_num_rows < MIN_HEIGHT:
            continue

        # perhaps due to error in crawling and saving metadata
        if not re.sub(file_pattern, '', table_ids[r_tab_id]) in metadata:
            continue
        r_pkg_id = metadata[re.sub(file_pattern, '', table_ids[r_tab_id])]['id']
        r_column_names = set(map(lambda v: sanitize_string(v, CLEAN_MODE), r_df.columns))
        
        # for each col, if it is not supposed to be an ID
        # or a date column, query the index to find potentially 
        # joinable tables
        for r_col_id, r_col_name in enumerate(r_df.columns):
            # discard numerical columns
            if r_df.get_column(r_col_name).dtype.is_numeric():
                continue
            
            # check if any token like "id" or "date" is inside the column name
            if any(tok in r_df.columns[r_col_id].lower() for tok in bad_columns_tokens):
                continue

            # check the number of null values
            if not r_num_rows or r_df.to_series(r_col_id).is_null().sum() / r_num_rows >= MAX_NULL_RATIO:
                continue

            # extract the unique and cleaned values from the column 
            r_col = set(
                filter(lambda v: v in values, 
                    map(lambda v: sanitize_string(v, CLEAN_MODE), r_df.to_series(r_col_id))
                )
            )

            if len(r_col) < MIN_NUM_VALUES:
                continue

            r_col_int = list(map(lambda v: str(values[v]), filter(lambda v: v in values, r_col)))
            
            # apply the SC BLEND search technique, limiting to first K results
            results = con.sql(f"""
                    SELECT TableId, ColumnId, COUNT(DISTINCT CellValue) AS intersec FROM AllTables
                    WHERE CellValue IN ({','.join(r_col_int)})
                    AND TableId <> {r_tab_id}
                    GROUP BY TableId, ColumnId
                    ORDER BY COUNT(DISTINCT CellValue) DESC
                    LIMIT {K};
            """).fetchall()
            
            # for each tuple, check if this is a potentially valid join candidate
            for s_tab_id, s_col_id, intersection in results:         
                # if this candidate is valid, this record will be overriden
                candidates.append(
                    [
                        table_ids[r_tab_id].removesuffix('.parquet'), 
                        table_ids[s_tab_id].removesuffix('.parquet'),
                        r_col_id, 
                        s_col_id,
                        None, None, None, None, None, None, None, None, None, None, None
                    ]
                )

                # perhaps due to error in crawling and saving metadata?
                if not re.sub(file_pattern, '', table_ids[s_tab_id]) in metadata:
                    continue
                
                s_pkg_id = metadata[re.sub(file_pattern, '', table_ids[s_tab_id])]['id']
                
                # if they belong to the same package, drop the pair
                if r_pkg_id == s_pkg_id:
                    continue
                
                # if we have already used this tuple <TableId1, TableId2, Col1, Col2> 
                # do not save it again
                if candidate_id := (
                    r_tab_id if r_tab_id <= s_tab_id else s_tab_id, 
                    s_tab_id if r_tab_id <= s_tab_id else r_tab_id,
                    r_col_id if r_tab_id <= s_tab_id else s_col_id,
                    s_col_id if r_tab_id <= s_tab_id else r_col_id,
                    ) in already_used:
                    continue
                already_used.add(candidate_id)

                s_columns_names = set(map(lambda v: sanitize_string(v, CLEAN_MODE), 
                                          pl.scan_parquet(f"{tables_path}/{table_ids[s_tab_id]}").collect_schema().names()))
                if not ACCEPT_SAME_SCHEMA and r_column_names == s_columns_names:
                    continue 

                s_col_series = pl.scan_parquet(f"{tables_path}/{table_ids[s_tab_id]}").select(pl.nth(s_col_id)).collect().to_series(0)
                s_col_name = s_col_series.name
                                
                s_num_rows = s_col_series.shape[0]

                # the table must have a minimum height and a max null ratio
                if s_num_rows < MIN_HEIGHT or s_col_series.is_null().sum() / s_num_rows >= MAX_NULL_RATIO:
                    continue
            
                s_col = set(
                    filter(lambda v: v in values, 
                        map(lambda v: sanitize_string(v, CLEAN_MODE), 
                            s_col_series
                            )
                        )
                    )                
                
                intersection    = r_col & s_col
                union           = r_col | s_col
                jaccard         = round(len(intersection) / len(union), 3)
                overlap         = round(len(intersection) / min(len(r_col), len(s_col)), 3)

                # do not consider this pair if:
                #   - one of the two columns has only few value;
                #   - both jaccard and min-overlap are lower than a threshold (not used, to check if really useful);
                #   - columns have the same name and have jaccard/overlap==1 (not used, to check if really useful);
                if len(s_col) < MIN_NUM_VALUES: #  \
                    # or jaccard < MIN_JACCARD or overlap < MIN_OVERLAP \
                    # or r_col_name == s_col_name and jaccard == 1 and overlap == 1:
                    continue
                
                candidates.pop()
                candidates.append(
                    [
                        table_ids[r_tab_id].removesuffix('.parquet'), 
                        table_ids[s_tab_id].removesuffix('.parquet'),
                        r_col_id, 
                        s_col_id,
                        r_col_name, 
                        s_col_name,
                        len(r_col), 
                        len(s_col),
                        r_pkg_id, 
                        s_pkg_id,
                        
                        len(intersection), 
                        len(union), 
                        jaccard, 
                        overlap
                    ]
                )

        if r_tab_id % N_BATCH_APPEND == 0 and r_tab_id > 0:
            logger.info(f'Up to table {r_tab_id}({round(r_tab_id * 100 / len(table_ids), 3)}%);Current candidates:{len(candidates)};time:{round(time.time() - start_batch_t, 3)}s')
            
            with open(candidates_path, 'a') as file:
                wr = csv.writer(file)
                wr.writerows(candidates)
            candidates = []
            start_batch_t = time.time()


    logger.info(f'Up to table {r_tab_id} ({round(r_tab_id * 100 / len(table_ids), 3)}%);Current candidates:{len(candidates)};time:{round(time.time() - start_batch_t, 3)}s')
    with open(candidates_path, 'a') as file:
        wr = csv.writer(file)
        wr.writerows(candidates)
    
    logger.info("Done")


if __name__ == '__main__':
    main(*sys.argv[1:])
