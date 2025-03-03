import os
import time
import zipfile
import unicodedata
from functools import lru_cache
from itertools import chain, product

import warnings
warnings.filterwarnings('ignore')

import inflection
import numpy as np
import polars as pl
from tqdm import tqdm
import polars.selectors as cs

from sqlalchemy import create_engine
from sqlalchemy import Integer, String, Float, Boolean, DateTime

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def map_dtype_pl_to_sql(dtype: pl.DataType):
    match str(dtype.to_python()):
        case 'int':
            return Integer
        case 'float':
            return Float
        case 'bool':
            return Boolean
        case 'datetime.datetime':
            return DateTime
        case 'str':
            return String
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def extract_representative_rows(df: pl.DataFrame, n_repr_rows: int, max_unique_value: int):
    df = df.drop_nans()
    df_copy = df.clone()

    numerical_column    = df.select(cs.numeric()).columns
    categorical_columns = df.select(cs.categorical()).columns
    
    columns_to_mantain  = [df.select(col) for col in categorical_columns if df.select(col).n_unique() < max_unique_value]
    df                  = df.select(numerical_column + columns_to_mantain)
    
    transformers = []

    if len(numerical_column) > 0:
        transformers.append(('num', StandardScaler(), numerical_column))
    if len(categorical_columns) > 0:
        ('cat', OneHotEncoder(), columns_to_mantain)

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )
    
    # Pipeline to preprocess and cluster
    pipeline = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('clustering', KMeans(n_clusters=n_repr_rows, random_state=42))
        ]
    )

    # Fit the pipeline to the data
    try:
        # little mix of pandas and polars...
        pipeline.fit(df)
        df = df.to_pandas()
        df['Cluster'] = pipeline.named_steps['clustering'].labels_

        # Get the cluster centers in the original feature space
        cluster_centers = pipeline.named_steps['clustering'].cluster_centers_
        transformed_features = pipeline.named_steps['preprocessor'].transform(df)
    
        # Find the most representative row for each cluster
        repr_rows_indices = []
        for cluster_num in range(cluster_centers.shape[0]):
            cluster_data = transformed_features[df['Cluster'] == cluster_num]
            centroid = cluster_centers[cluster_num]
            distances = np.linalg.norm(np.float64(cluster_data - centroid), axis=1)
            representative_index = distances.argmin()
            repr_rows_indices.append(df[df['Cluster'] == cluster_num].index[representative_index])

    except Exception as e:
        return df_copy.sample(n_repr_rows)

    # Extract the representative rows
    # representative_rows = df_copy.loc[representative_rows_indices]
    representative_rows = df_copy[repr_rows_indices] # [df_copy.row(idx) for idx in repr_rows_indices]
    return representative_rows


@lru_cache
def sanitize_string(s):
    """
    Replaces problematic characters in column names with underscores,
    normalizes accents, and strips spaces.
    """
    if not isinstance(s, str):
        return s
    
    s = inflection.underscore(s).lower()
    # Normalize accents (e.g., é -> e)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')
    # Replace problematic characters with underscores
    return s.replace('\n', '_').replace(' ', '_').replace('.', '_').replace('"', '_').strip()
    

# define a function to read the tables directly from the .zip file
def read_table_pl(table_id, zip_dataset_path, open_dataset_name):
    with zipfile.ZipFile(zip_dataset_path, 'r') as zip_tables:
        with zip_tables.open(f'datasets_{open_dataset_name}/{table_id}', 'r') as table_file:
            try:
                df = pl.scan_csv(table_file, ignore_errors=True)
                colnames = df.collect_schema().names()
                new_colnames = list(map(sanitize_string, colnames))
                mapping = {
                    c: nc if new_colnames[:i].count(nc) == 0 else f'{nc}_{new_colnames[:i].count(nc)}'
                    for i, (c, nc) in enumerate(zip(colnames, new_colnames))
                }
                df = df.rename(mapping)
                
                schema = df.collect_schema()
                for attribute in schema.names():
                    df.with_columns(pl.col(attribute).map_elements(sanitize_string, return_dtype=schema[attribute]))
                return df.collect()
            except Exception as e:
                return f'Error on reading table {table_id}: {e}'
            

def preprocess_dataset_pl(open_dataset_name='CAN', task='join', data_dir='data/'):
    # the dataset (stored as an archive, it's large)
    DATASET_PATH            = f"{data_dir}/datasets/datasets_{open_dataset_name}.zip"

    # a simple SQLite database where are stored the tables
    # extracted from the mega zip file
    DATABASE_PATH           = f'sqlite:///{data_dir}/databases/{open_dataset_name}_{task}.db'

    STATS_PATH              = f'{data_dir}/statistics/{open_dataset_name}_{task}_preprocessing.csv'

    # the Ground Truth file with table paris (from LakeBench)
    GT_PATH                 = f"{data_dir}/{task}/{open_dataset_name}_ground_truth.csv"

    # the directory where are stored for each table a small snapshot (<5) of its rows
    REPR_ROWS_DIR_PATH      = f'{data_dir}/representative_rows/{open_dataset_name}_{task}'

    # Used to select the representative rows (by now, they are just randomly sampled)
    MAX_UNIQUE_VALUES       = 10

    # the number of rows to consider as an example snapshot 
    # passed to the generator agent
    N_REPRESENTATIVE_ROWS   = 3

    for dir in [REPR_ROWS_DIR_PATH, os.path.dirname(STATS_PATH)]:
        if not os.path.isdir(dir):
            os.makedirs(dir)

    # setup the SQLAlchemy engine
    engine = create_engine(DATABASE_PATH)
    conn = engine.connect()

    gt = pl.read_csv(GT_PATH)

    tab = gt.unique(subset=['query_table', 'candidate_table'])
    table_ids = set(chain(*tab[['query_table', 'candidate_table']].rows()))

    stats = []

    for table_id in tqdm(table_ids):
        if not os.path.exists(f'{REPR_ROWS_DIR_PATH}/{table_id}'):
            # read the table cleaning its values and column names
            start = time.time()
            df = read_table_pl(table_id, DATASET_PATH, open_dataset_name)
            read_time = time.time() - start

            # store the table into the database
            start = time.time()
            try:
                df.to_pandas(use_pyarrow_extension_array=True).to_sql(table_id, conn, if_exists='replace', index=False, )
            except Exception as e:
                print(table_id)
                raise e
            sql_time = time.time() - start
            
            # save the tables representative rows
            start = time.time()
            repr_rows = extract_representative_rows(df, N_REPRESENTATIVE_ROWS, MAX_UNIQUE_VALUES)
            repr_rows_time = time.time() - start
            repr_rows.write_csv(f'{REPR_ROWS_DIR_PATH}/{table_id}')

            stats.append([round(read_time, 5), round(sql_time, 5), round(repr_rows_time, 5)])
    
    pl.DataFrame(stats, schema=['read_time(s)', 'to_db_time(s)', 'repr_rows_extr_time(s)'], orient='row').write_csv(STATS_PATH)



def main():
    open_datasets = ['CAN', 'SG', 'USA', 'UK']
    tasks = ['join', 'union']
    done = [('CAN', 'join')] # [('SG', 'join'), ('SG', 'union'), ('CAN', 'join')]

    for open_dataset, task in product(open_datasets, tasks):
        if (open_dataset, task) in done:
            continue
        print(f'WORKING ON {open_dataset} - {task}')
        start_prep = time.time()
        preprocess_dataset_pl(open_dataset, task, data_dir=f'{os.path.dirname(__file__)}/../data')
        end_prep = time.time()
        print(f'Total preprocessing time: {round(end_prep - start_prep, 3)}s')

    
if __name__ == '__main__':
    main()

    # df = read_table_pl('CAN_CSV0000000000015696.csv', '/home/giovanni.malaguti/projects/tabgen/data/datasets/datasets_CAN.zip', 'CAN')
    # repr_rows = extract_representative_rows(df, 3, 10)
    # print(repr_rows)