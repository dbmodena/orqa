import os
import sys
import time
import yaml
import pickle
import argparse

from itertools import chain
from os.path import join as pjoin
from concurrent.futures import ProcessPoolExecutor

import bidict
import duckdb
import polars as pl
import polars.selectors as cs

from orqa.utils import clean_string, is_num, setup_logger

# TODO actually, a more simpler hash approach would be more efficient,
# since we create a 1-1 mapping int-string, and is more parallelizable...
def get_table_values(table_id):
    global clean_values, tables_path, values_bd
    
    try:
        df = pl.read_parquet(pjoin(tables_path, table_id))
        df = df[[s.name for s in df if not (s.null_count() == df.height)]]
            
        # dtype conversion to bool/int/float
        for column in df.columns:
            try:
                if df.select(column).drop_nulls().unique().shape[0] == 2:
                    df = df.with_columns(pl.col(column).cast(pl.Boolean))
                    continue
            except: pass
            try:
                dtype = pl.Float64 if any(',' in str(x) or '.' in str(x) 
                                          for x in set(df.select(column).sample(300, with_replacement=True).to_series())
                                          ) else pl.Int64
                df = df.with_columns(pl.col(column).cast(dtype))
            except: continue

        return set(
            filter(
                lambda v: not is_num(v), 
                map(
                    lambda s: clean_string(s, clean_values), 
                    chain(*(
                        df.select(col).drop_nans().drop_nulls().unique().get_column(col).to_list() 
                        for col in df.select(cs.exclude(cs.numeric(), cs.boolean())).columns
                        )
                    )                
                )
            )
        )
    except:
        return set()


def collect_table_records(data):
    global clean_values, tables_path, values_bd
    table_idx, table_id = data
    df = pl.read_parquet(pjoin(tables_path, table_id))

    # dtype conversion to bool/int/float
    for column in df.columns:
        try:
            if df.select(column).drop_nulls().unique().shape[0] == 2:
                df = df.with_columns(pl.col(column).cast(pl.Boolean))
                continue
        except: pass
        try:
            dtype = pl.Float64 if any(',' in str(x) or '.' in str(x) 
                                        for x in set(df.select(column).sample(300, with_replacement=True).to_series())
                                        ) else pl.Int64
            df = df.with_columns(pl.col(column).cast(dtype))
        except: continue

    return [
        [table_idx, col_idx, row_idx, values_bd[clean_string(cell, clean_values)]]
        for col_idx, col in enumerate(df.columns)
        if not df.get_column(col).null_count() == df.height \
            and not df.get_column(col).dtype.is_numeric()
        # all columns are considered here to keep the right index
        # (TODO optimize and more consistent with other df handling parts)
        for row_idx, cell in enumerate(df.select(col).get_column(col).to_list())
        if clean_string(cell, clean_values) in values_bd
    ]


def initializer(_clean_values, _tables_path, _values):
    global clean_values, tables_path, values_bd
    clean_values = _clean_values
    tables_path = _tables_path
    values_bd = _values


def main(tag: str = 'CAN',
         from_: int = 0,
         to_: int|str = 'END'):
    
    conf_path       = pjoin(os.path.dirname(__file__), '..', 'conf', 'configuration.yml')
    data_path       = pjoin(os.path.dirname(__file__), '..', 'data')
    tables_path     = pjoin(data_path, 'datasets', tag, 'tables', f'from{from_}_to{to_}' )
    db_path         = pjoin(data_path, 'datasets', tag, 'database', f'from{from_}_to{to_}', 'blend.db')
    values_path     = pjoin(data_path, 'datasets', tag, 'database', f'from{from_}_to{to_}', 'values_bidict.pickle')
    log_path        = pjoin(data_path, 'log', tag, f'1_indexing_{time.strftime("%y%m%d_%H_%M_%S")}.log')

    
    with open(conf_path, 'r') as file:
        raw = file.read()
        cfg = argparse.Namespace(**{**yaml.safe_load(raw)['general'], **yaml.safe_load(raw)['indexing']})

    CLEAN_HEADERS       = cfg.string_cleaning['headers']
    CLEAN_ELEMENTS      = cfg.string_cleaning['elements']
    BATCH_SIZE          = cfg.batch_size
    num_workers         = cfg.num_workers
    LOG_LEVEL           = cfg.log_level

    # set up logging
    logger = setup_logger(log_path, "index_creation_logger", on_file=True, on_stdout=False, level=LOG_LEVEL)
    
    # create the directory for the blend database
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # take the table IDs in alphabetical order
    table_ids = list(sorted(os.listdir(tables_path), reverse=True))

    logger.info(f"Headers cleaning mode: {CLEAN_HEADERS}")
    logger.info(f"Elements cleaning mode: {CLEAN_ELEMENTS}")

    logger.info("Started BLEND indexing")

    # init the duckdb database
    con = duckdb.connect(db_path)
    con.execute(f"""
        CREATE OR REPLACE TABLE AllTables (
            TableId      INT , 
            ColumnId     INT , 
            RowId        INT , 
            CellValue    INT    
        );
        """
    )

    # close connection (on Windows, fork option is not allowed)
    # TODO handle this differently on *NIX/Windows?
    con.close()

    logger.info('Index table created')

    values_bidict = {}
    records = []
    
    start_t = time.time()

    i = -1
    n_values = 0
    if not os.path.exists(values_path):
        logger.info('Create values dictionary')

        with ProcessPoolExecutor(num_workers, initializer=initializer, initargs=(CLEAN_ELEMENTS, tables_path, values_bidict)) as executor:
            for ntab in range(0, len(table_ids) + BATCH_SIZE, BATCH_SIZE):
                start_proc_t = time.time()
                results = set(chain(*executor.map(get_table_values, table_ids[ntab:ntab + BATCH_SIZE])))
                end_proc_t = time.time()

                start_filter_t = time.time()
                uniques = filter(lambda v: v not in values_bidict, results)
                end_filter_t = time.time()
                
                start_update_bd_t = time.time()
                for i, v in enumerate(uniques, start=i + 1):
                    values_bidict[v] = i
                end_update_bd_t = time.time()

                added_values, n_values = len(values_bidict) - n_values, len(values_bidict)
                
                if ntab > 0:
                    logger.debug(f"[TABLES:{ntab}/{len(table_ids)}],[TABLES_PERC:{round(ntab * 100 / len(table_ids), 1)}%)],[NEW_VALUES:{added_values}]"
                                 f"[EXTR_T:{round(end_proc_t - start_proc_t, 1)}s],"
                                 f"[FILT_T:{round(end_filter_t - start_filter_t, 1)}s],"
                                 f"[UPD_T:{round(end_update_bd_t - start_update_bd_t, 1)}s]")

        logger.info('Create bidict from values')
        values_bidict = bidict.bidict(values_bidict)

        logger.info('Save values bidict')
        with open(values_path, 'wb') as fw:
            pickle.dump(values_bidict, fw, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        logger.info('Load values dictionary')
        with open(values_path, 'rb') as fr:
            values_bidict = pickle.load(fr)

    logger.info(f'{len(values_bidict)=}')

    logger.info('Start insert values into duckdb')
    with ProcessPoolExecutor(num_workers, initializer=initializer, initargs=(CLEAN_ELEMENTS, tables_path, values_bidict)) as executor:
        for ntab in range(0, len(table_ids) + BATCH_SIZE, BATCH_SIZE):
            start_collection_t = time.time()
            results = executor.map(collect_table_records, enumerate(table_ids[ntab:ntab + BATCH_SIZE], start=ntab))
            records = []
            for table in results:
                records += table
            end_collection_t = time.time()

            start_insert_t = time.time()
            rec_df = pl.DataFrame(records, schema=['TableId', 'ColumnId', 'RowId', 'CellValue'], orient='row')
            con = duckdb.connect(db_path)
            con.execute("INSERT INTO AllTables SELECT * FROM rec_df")
            con.commit()
            con.close()
            end_insert_t = time.time()
                
            logger.debug(f"[TABLES:{ntab + BATCH_SIZE}/{len(table_ids)}],[TABLES_PERC:{round(ntab * 100 / len(table_ids), 3)}%],"
                         f"[ENTRIES:{rec_df.shape[0]}],"
                         f"[COLL_T:{round(end_collection_t - start_collection_t, 1)}s],"
                         f"[INSERT_T:{round(end_insert_t - start_insert_t, 1)}s]")

    con = duckdb.connect(db_path)
    
    logger.info('Creating indexes')
    con.execute("CREATE INDEX IF NOT EXISTS TableId_idx   ON AllTables (TableId);")
    con.execute("CREATE INDEX IF NOT EXISTS CellValue_idx ON AllTables (CellValue);")
    logger.info('Done')

    con.close()

    end_t = time.time()
    total_t = round(end_t - start_t, 3)
    logger.info(f"Total time: {total_t}s")


if __name__ == '__main__':
    main(*sys.argv[1:])