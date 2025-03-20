import os
import pickle
import logging
from itertools import chain
from logging.handlers import RotatingFileHandler

import tqdm
import bidict
import duckdb
import polars as pl
from wrapt_timeout_decorator import timeout

from orqa.utils import sanitize_string, is_num


# tables_path     = f'{os.path.dirname(__file__)}/../data/datasets/CAN/tables/tables_from0_to10000'
# metadata_path   = f'{os.path.dirname(__file__)}/../data/datasets/CAN/metadata/metadata_from0_to10000.jsonl'
tables_path     = f'{os.path.dirname(__file__)}/../data/datasets/CAN/tables/tables_from10000_to15000'
metadata_path   = f'{os.path.dirname(__file__)}/../data/datasets/CAN/metadata/metadata_from10000_to15000.jsonl'
db_path         = f'{os.path.dirname(__file__)}/../data/datasets/CAN/database/CAN.db'
values_path     = f'{os.path.dirname(__file__)}/../data/datasets/CAN/database/values_dict.pickle'
failures_path   = f'{os.path.dirname(__file__)}/../data/datasets/CAN/database/failures.txt'
log_path        = f'{os.path.dirname(__file__)}/../data/log/CAN_indexing.log'


os.makedirs(os.path.dirname(log_path), exist_ok=True)

logger = logging.getLogger(f'indexerLogger')
logger.setLevel(logging.DEBUG)

handler = RotatingFileHandler(log_path, mode="a", maxBytes=1024 ** 3)
log_formatter = logging.Formatter("%(asctime)s,[%(process)d],[%(levelname)s],%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(log_formatter)
logger.addHandler(handler)

if os.path.exists(db_path): os.remove(db_path)
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# take the table IDs in alphabetical order
table_ids = list(sorted(os.listdir(tables_path), reverse=True))

logger.info(' Start new run '.center(50, '#'))

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



# setting this timeout we will prob have some keyerror next
# @timeout(20)
def get_table_values(table_id):
    df = pl.read_parquet(f'{tables_path}/{table_id}')
    df = df[[s.name for s in df if not (s.null_count() == df.height)]]

    return set(
        filter(
            lambda v: v not in values and not is_num(v), 
            map(
                lambda s: sanitize_string(str(s)), 
                chain(*(
                    df.select(col).drop_nans().drop_nulls().unique().get_column(col).to_list() 
                    for col in df.columns
                    )
                )                
            )
        )
    )


i = 0
if not os.path.exists(values_path):
    logger.info('Create values dictionary')
    
    for table_id in tqdm.tqdm(table_ids, desc="Creating values dictionary:"):
        try:
            unique_vals = get_table_values(table_id)
            for v in unique_vals:
                values[v] = i
                i += 1
        except TimeoutError:
            continue

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


@timeout(20)
def collect_table_records(table_idx, table_id):
    df = pl.read_parquet(f'{tables_path}/{table_id}')

    return [
        [table_idx, col_idx, row_idx, values[sanitize_string(str(cell))]]
        for col_idx, col in enumerate(df.columns)
        if not df.get_column(col).null_count() == df.height
        for row_idx, cell in enumerate(df.select(col).get_column(col).to_list())        
        if sanitize_string(str(cell)) in values
    ]

n = 0
logger.info('Start insert values into duckdb')
for table_idx, table_id in tqdm.tqdm(enumerate(table_ids), total=len(table_ids), desc="Inserting values:"):
    try:
        records += collect_table_records(table_idx, table_id)
        n += 1
    except TimeoutError:
        continue
    except KeyError:
        continue
    
    if n % CHECKPOINT == 0 and n > 0:
        rec_df = pl.DataFrame(records, schema=['TableId', 'ColumnId', 'RowId', 'CellValue'], orient='row')
        con.execute("INSERT INTO AllTables SELECT * FROM rec_df")
        con.commit()
        records = []
        logger.debug(f'Inserted tables [{n - CHECKPOINT}, {n}]')

rec_df = pl.DataFrame(records, schema=['TableId', 'ColumnId', 'RowId', 'CellValue'], orient='row')
con.execute("INSERT INTO AllTables SELECT * FROM rec_df")
con.commit()
logger.debug(f'Inserted tables [{n - CHECKPOINT}, {n}]')
logger.info('Done')        

logger.info('Creating indexes')
con.execute("CREATE INDEX IF NOT EXISTS TableId_idx   ON AllTables (TableId);")
con.execute("CREATE INDEX IF NOT EXISTS ColumnId_idx  ON AllTables (ColumnId);")
con.execute("CREATE INDEX IF NOT EXISTS RowId_idx     ON AllTables (RowId);")
con.execute("CREATE INDEX IF NOT EXISTS CellValue_idx ON AllTables (CellValue);")
logger.info('Done')        

