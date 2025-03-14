from itertools import chain
import itertools
import os
import threading
import duckdb
import pandas as pd
import polars as pl
from wrapt_timeout_decorator import timeout
import bidict
import pickle
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from tqdm import tqdm


tables_path     = f'{os.path.dirname(__file__)}/../data/datasets/CAN/tables/tables_from10000_to15000'
metadata_path   = f'{os.path.dirname(__file__)}/../data/datasets/CAN/metadata/metadata_from10000_to15000'
db_path         = f'{os.path.dirname(__file__)}/../data/datasets/CAN/database/CAN.db'
values_path     = f'{os.path.dirname(__file__)}/../data/datasets/CAN/database/values_dict.pickle'
failures_path   = f'{os.path.dirname(__file__)}/../data/datasets/CAN/database/failures.txt'

if os.path.exists(db_path): os.remove(db_path)
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# take the table IDs in alphabetical order
table_ids = list(sorted(os.listdir(tables_path), reverse=True))

# init the duckdb database
con = duckdb.connect(db_path)

con.execute(f"""
    CREATE OR REPLACE TABLE OrqaIndex (
       TableID      INT , 
       ColumnID     INT , 
       RowID        INT , 
       Value        INT    
       );
    """
)

values = {}

NUM_WORKERS = 3
n = CHECKPOINT = 10
records = []


@timeout(20)
def get_unique_values(table_id):
    lf = pl.scan_parquet(f'{tables_path}/{table_id}')
    return set(chain(*(lf.select(col).unique().collect().get_column(col).drop_nans().drop_nulls().to_list() for col in lf.collect_schema().names())))

print('Create values dictionary')
with ThreadPoolExecutor(NUM_WORKERS) as executor:
    jobs = [executor.submit(get_unique_values, table_id) for table_id in table_ids]
    
    for job in tqdm(as_completed(jobs), total=len(jobs)):
        try:
            unique = job.result()
            for v in unique:
                if v not in values:
                    values[v] = len(unique)
        except TimeoutError:
            continue    
    
print(f'{len(values)=}')

values = bidict.bidict(values)
    
with open(values_path, 'wb') as fw:
    pickle.dump(values, fw, protocol=pickle.HIGHEST_PROTOCOL)



@timeout(20)
def collect_table_records(table_idx, table_id):
    df = pl.scan_parquet(f'{tables_path}/{table_id}').filter(~pl.all_horizontal(pl.all().is_null()) & ~pl.all_horizontal(pl.all().is_nan()))
    
    columns = df.collect_schema().names()

    return [
        [table_idx, col_idx, row_idx, values[cell]]
        # for col_idx, col in tqdm(enumerate(columns), total=len(columns), disable=True, leave=False, position=(threading.get_native_id() % NUM_WORKERS) + 1)
        # for row_idx, cell in tqdm(enumerate(df.select(col).collect().rows()), disable=True, total=df.select(pl.len()).collect().item(), leave=False, position=2)
        for col_idx, col in enumerate(columns)
        for row_idx, cell in enumerate(df.select(col).collect().rows())
        
        if not pd.isna(cell)
    ]

print('Create index')

with ThreadPoolExecutor(NUM_WORKERS) as executor:
    jobs = [executor.submit(collect_table_records, table_idx, table_id) for table_idx, table_id in enumerate(table_ids)]

    for job in tqdm(as_completed(jobs), total=len(jobs)):
        try:
            records += job.result()
            n -= 1
        except TimeoutError:
            continue

        if n == 0:
            rec_df = pl.DataFrame(records, schema=['TableID', 'ColumnID', 'RowID', 'Value'], orient='row')
            con.execute("INSERT INTO OrqaIndex SELECT * FROM rec_df")
            con.commit()
            n = CHECKPOINT
            records = []

con.execute("CREATE INDEX IF NOT EXISTS TableID_idx   ON OrqaIndex (TableID);")
con.execute("CREATE INDEX IF NOT EXISTS ColumnID_idx  ON OrqaIndex (ColumnID);")
con.execute("CREATE INDEX IF NOT EXISTS RowID_idx     ON OrqaIndex (RowID);")
con.execute("CREATE INDEX IF NOT EXISTS Value_idx     ON OrqaIndex (Value);")

# with open(failures_path, 'w') as fw:
#     fw.writelines(list(map(lambda)))
