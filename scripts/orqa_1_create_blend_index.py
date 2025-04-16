import os
import sys
import pickle
import logging
from itertools import chain
from logging.handlers import RotatingFileHandler
from concurrent.futures import ProcessPoolExecutor

os.environ["POLARS_MAX_THREADS"] = "10"

import bidict
import duckdb
import polars as pl
import polars.selectors as cs
from wrapt_timeout_decorator import timeout

from orqa.utils import sanitize_string, is_num




def get_table_values(table_id: str, tables_path: str):
    try:
        df = pl.read_parquet(f'{tables_path}/{table_id}')
        df = df[[s.name for s in df if not (s.null_count() == df.height)]]
            
        # dtype conversion to bool/int/float
        for column in df.columns:
            try:
                if df.select(column).drop_nulls().unique().shape[0] == 2:
                    df = df.with_columns(pl.col(column).cast(dtype))
                    continue                
            except: pass

            try:
                dtype = pl.Float64 if any(',' in str(x) or '.' in str(x) for x in set(df.select(column).sample(1000, with_replacement=True).to_series())) else pl.Int64
                df = df.with_columns(pl.col(column).cast(dtype))
            except: continue


        return set(
            filter(
                lambda v: not is_num(v), 
                map(
                    sanitize_string, 
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
    table_idx, table_id = data
    df = pl.read_parquet(f'{tables_path}/{table_id}')

    return [
        [table_idx, col_idx, row_idx, values[sanitize_string(cell)]]
        for col_idx, col in enumerate(df.columns)
        if not df.get_column(col).null_count() == df.height
        for row_idx, cell in enumerate(df.select(col).get_column(col).to_list())        
        if sanitize_string(cell) in values
    ]



tag, from_, to_ = sys.argv[1:]

data_path       = f'{os.path.dirname(__file__)}/../data'
tables_path     = f'{data_path}/datasets/{tag}/tables/tables_from{from_}_to{to_}'
db_path         = f'{data_path}/datasets/{tag}/database/blend.db'
values_path     = f'{data_path}/datasets/{tag}/database/values_dict.pickle'

log_path        = f'{data_path}/log/{tag}/1_blend_indexing.log'

num_cpu         = 10


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

values = {}
CHECKPOINT = 100
records = []


i = 0
if not os.path.exists(values_path):
    logger.info('Create values dictionary')
    
    with ProcessPoolExecutor(num_cpu) as executor:
        for ntab in range(0, len(table_ids) + 100, 100):
            results = executor.map(get_table_values, table_ids[ntab:ntab+100])
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
step = 100
with ProcessPoolExecutor(num_cpu) as executor:
    for ntab in range(0, len(table_ids) + step, step):        
        results = executor.map(collect_table_records, enumerate(table_ids[ntab:ntab+step], start=ntab))
        records = []
        for table in results:
            records += table

        rec_df = pl.DataFrame(records, schema=['TableId', 'ColumnId', 'RowId', 'CellValue'], orient='row')
        con.execute("INSERT INTO AllTables SELECT * FROM rec_df")
        con.commit()
            
        logger.debug(f"Inserting batch tables {ntab}/{len(table_ids)} ({round(ntab * 100 / len(table_ids), 3)}%), inserted records: {rec_df.shape[0]}")


logger.info('Creating indexes')
con.execute("CREATE INDEX IF NOT EXISTS TableId_idx   ON AllTables (TableId);")
# these should have a very low selectivity thus is prob not necessary
# if we can get the records relative to our tables with the TableId_idx the main
# work is done (?)
# con.execute("CREATE INDEX IF NOT EXISTS ColumnId_idx  ON AllTables (ColumnId);")
# con.execute("CREATE INDEX IF NOT EXISTS RowId_idx     ON AllTables (RowId);")
con.execute("CREATE INDEX IF NOT EXISTS CellValue_idx ON AllTables (CellValue);")
logger.info('Done')

