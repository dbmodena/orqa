import os
import sys
import time
import pickle
import logging

from itertools import chain
from os.path import join as pjoin
from concurrent.futures import ProcessPoolExecutor

import bidict
import duckdb
import polars as pl
import polars.selectors as cs

from orqa.utils import sanitize_string, is_num


def get_table_values(table_id):
    global clean_mode, tables_path, values_bd
    
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
                    lambda s: sanitize_string(s, clean_mode), 
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
    global clean_mode, tables_path, values_bd
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
        [table_idx, col_idx, row_idx, values_bd[sanitize_string(cell, clean_mode)]]
        for col_idx, col in enumerate(df.columns)
        if not df.get_column(col).null_count() == df.height \
            and not df.get_column(col).dtype.is_numeric()
        # all columns are considered here to keep the right index
        # (TODO optimize and more consistent with other df handling parts)
        for row_idx, cell in enumerate(df.select(col).get_column(col).to_list())
        if sanitize_string(cell, clean_mode) in values_bd
    ]


def initializer(_clean_mode, _tables_path, _values):
    global clean_mode, tables_path, values_bd
    clean_mode = _clean_mode
    tables_path = _tables_path
    values_bd = _values


def main(tag: str = 'CAN',
         from_: int = 0,
         to_: int|str = 'END'):
    data_path       = pjoin(os.path.dirname(__file__), '..', 'data')
    tables_path     = pjoin(data_path, 'datasets', tag, 'tables', f'from{from_}_to{to_}' )
    db_path         = pjoin(data_path, 'datasets', tag, 'database', f'from{from_}_to{to_}', 'blend.db')
    values_path     = pjoin(data_path, 'datasets', tag, 'database', f'from{from_}_to{to_}', 'values_bidict.pickle')
    log_path        = pjoin(data_path, 'log', tag, f'1_blend_indexing_{time.strftime("%y%m%d_%H_%M_%S")}.log')

    # num of processes spawned
    num_workers     = 10

    # string cleaning mode
    CLEAN_MODE      = "base"
    
    # update size for both values bidict
    # and BLEND index creation
    step            = 10

    # set up logging
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(f'indexerLogger')
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_path)
    log_formatter = logging.Formatter("%(asctime)s,[%(process)d],[%(levelname)s],%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)

    # keep only last three log files
    old_logs =  sorted([f for f in os.listdir(os.path.dirname(log_path)) if f.startswith('1_blend')], reverse=True)
    log_to_delete = old_logs[3:] if len(old_logs) > 3 else []
    for log_to_del in log_to_delete:        
        os.remove(pjoin(os.path.dirname(log_path), log_to_del))

    # create the directory for the blend database
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # take the table IDs in alphabetical order
    table_ids = list(sorted(os.listdir(tables_path), reverse=True))

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

    # close connection (on Windows, it must fork option is not allowed)
    # TODO handle this differently on *NIX/Windows?
    con.close()

    logger.info('Index table created')

    values_bidict = {}
    records = []
    
    start_t = time.time()

    i = 0
    if not os.path.exists(values_path):
        logger.info('Create values dictionary')

        with ProcessPoolExecutor(num_workers, initializer=initializer, initargs=(CLEAN_MODE, tables_path, values_bidict)) as executor:
            for ntab in range(0, len(table_ids) + step, step):
                results = executor.map(get_table_values, table_ids[ntab:ntab + step])
                if ntab > 0:
                    logger.debug(f"Obtained values up to table {ntab}/{len(table_ids)} ({round(ntab * 100 / len(table_ids), 3)}%)")
                
                uniques = filter(lambda v: v not in values_bidict, set(chain(*results)))
                # for uniques in results:
                for v in uniques:
                    values_bidict[v] = i
                    i += 1

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
    with ProcessPoolExecutor(num_workers, initializer=initializer, initargs=(CLEAN_MODE, tables_path, values_bidict)) as executor:
        for ntab in range(0, len(table_ids) + step, step):        
            results = executor.map(collect_table_records, enumerate(table_ids[ntab:ntab + step], start=ntab))
            records = []
            for table in results:
                records += table

            rec_df = pl.DataFrame(records, schema=['TableId', 'ColumnId', 'RowId', 'CellValue'], orient='row')
            con = duckdb.connect(db_path)
            con.execute("INSERT INTO AllTables SELECT * FROM rec_df")
            con.commit()
            con.close()
                
            logger.debug(f"Inserting batch tables {ntab + step}/{len(table_ids)} ({round(ntab * 100 / len(table_ids), 3)}%), inserted records: {rec_df.shape[0]}")

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