import os
import sys
import pickle
import logging
from itertools import chain
from logging.handlers import RotatingFileHandler
from concurrent.futures import ProcessPoolExecutor

import bidict
import duckdb
import polars as pl
import polars.selectors as cs

from orqa.utils import sanitize_string, is_num


def get_table_values(data):
    global CLEAN_MODE
    table_id, tables_path = data
    try:
        df = pl.read_parquet(f'{tables_path}/{table_id}')
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
                    lambda s: sanitize_string(s, CLEAN_MODE), 
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
    global CLEAN_MODE
    table_idx, table_id = data
    df = pl.read_parquet(f'{tables_path}/{table_id}')

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
        [table_idx, col_idx, row_idx, values[sanitize_string(cell, CLEAN_MODE)]]
        for col_idx, col in enumerate(df.columns)
        if not df.get_column(col).null_count() == df.height \
            and not df.get_column(col).dtype.is_numeric()
        # all columns are considered here to keep the right index
        # (TODO optimize and more consistent with other df handling parts)
        for row_idx, cell in enumerate(df.select(col).get_column(col).to_list())
        if sanitize_string(cell, CLEAN_MODE) in values
    ]


tag, from_, to_ = sys.argv[1:]

data_path       = f'{os.path.dirname(__file__)}/../data'
tables_path     = f'{data_path}/datasets/{tag}/tables/tables_from{from_}_to{to_}'
db_path         = f'{data_path}/datasets/{tag}/database/blend.db'
values_path     = f'{data_path}/datasets/{tag}/database/values_dict.pickle'

log_path        = f'{data_path}/log/{tag}/1_blend_indexing.log'

num_workers     = 10
CLEAN_MODE      = "base"

os.makedirs(os.path.dirname(log_path), exist_ok=True)

logger = logging.getLogger(f'indexerLogger')
logger.setLevel(logging.DEBUG)

handler = RotatingFileHandler(log_path, mode="a", maxBytes=1024 ** 3)
log_formatter = logging.Formatter("%(asctime)s,[%(process)d],[%(levelname)s],%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(log_formatter)
logger.addHandler(handler)

os.makedirs(os.path.dirname(db_path), exist_ok=True)

# take the table IDs in alphabetical order
table_ids = list(sorted(os.listdir(tables_path), reverse=True))

logger.info(f"\n{'-' * 100}\n{tag} {from_=}-{to_=}")

# init the duckdb database
con = duckdb.connect(db_path)
logger.info('Connected to DuckDB')

con.execute(f"""
    CREATE OR REPLACE TABLE AllTables (
    TableId      INT , 
    ColumnId     INT , 
    RowId        INT , 
    CellValue    INT    
    );
    """
)
logger.info('Index table created')

step = 100
values = {}
records = []

i = 0
if not os.path.exists(values_path):
    logger.info('Create values dictionary')
    
    with ProcessPoolExecutor(num_workers) as executor:
        for ntab in range(0, len(table_ids) + step, step):
            results = executor.map(get_table_values, ((table_id, tables_path) for table_id in table_ids[ntab:ntab + step]))
            logger.debug(f"Obtained values up to table {ntab}/{len(table_ids)} ({round(ntab * 100 / len(table_ids), 3)}%)")
            
            uniques = filter(lambda v: v not in values, set(chain(*results)))
            # for uniques in results:
            for v in uniques:
                if v not in values:
                    values[v] = i
                    i += 1

    logger.info('Create bidict from values')
    values = bidict.bidict(values)

    logger.info('Save values bidict')
    with open(values_path, 'wb') as fw:
        pickle.dump(values, fw, protocol=pickle.HIGHEST_PROTOCOL)
else:
    logger.info('Load values dictionary')
    with open(values_path, 'rb') as fr:
        values = pickle.load(fr)

logger.info(f'{len(values)=}')

logger.info('Start insert values into duckdb')
with ProcessPoolExecutor(num_workers) as executor:
    for ntab in range(0, len(table_ids) + step, step):        
        results = executor.map(collect_table_records, enumerate(table_ids[ntab:ntab + step], start=ntab))
        records = []
        for table in results:
            records += table

        rec_df = pl.DataFrame(records, schema=['TableId', 'ColumnId', 'RowId', 'CellValue'], orient='row')
        con.execute("INSERT INTO AllTables SELECT * FROM rec_df")
        con.commit()
            
        logger.debug(f"Inserting batch tables {ntab + step}/{len(table_ids)} ({round(ntab * 100 / len(table_ids), 3)}%), inserted records: {rec_df.shape[0]}")

logger.info('Creating indexes')
con.execute("CREATE INDEX IF NOT EXISTS TableId_idx   ON AllTables (TableId);")
con.execute("CREATE INDEX IF NOT EXISTS CellValue_idx ON AllTables (CellValue);")
logger.info('Done')

