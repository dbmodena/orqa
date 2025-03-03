import os
import json
import os
import shutil
import zipfile
import urllib3
import threading
from io import StringIO
from itertools import chain
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from tqdm import tqdm
from jsonlines import jsonlines


data_path                   = f'{os.path.dirname(__file__)}/../data'
tmp_path                    = f'{data_path}/tmp'

# the tables metadata obtained from the Open Data Canada 
# filtering by date, around July 2023
snapshot_path               = f'{data_path}/snapshot.jsonl'

# the above metadata enriched with the relative table schema
snapshot_with_schema_path   = f'{data_path}/snapshot_with_schema_also_for_zip.jsonl'

with jsonlines.open(snapshot_path) as reader:
    metadata = list(reader.iter())


def fetch_schema(csv_md, http: urllib3.PoolManager):
    url = csv_md['url']
    try:
        # the format field isn't always totally accurate: sometimes
        # the URL links to a ZIP folder which contains one or more CSVs
        format = url.split('.')[-1].upper()
        format = format if format in {'ZIP', 'CSV'} else csv_md['format']
        
        response = http.request('GET', url, preload_content=format == 'CSV')
        
        if response.status != 200:
            csv_md['schema'] = None
            return csv_md
        
        match format:
            case 'ZIP':
                # download the zip to the tmp location
                tmp_file = f'{tmp_path}/{threading.get_ident()}.zip'
                with open(tmp_file, 'wb') as fw:
                    fw.write(response.data)                

                # open the donwloaded ZIP 
                with zipfile.ZipFile(tmp_file, 'r') as z:
                    files = []
                    for fname in z.namelist():
                        if 'metadata' not in fname.lower():
                            files.append(fname)
                    
                    # if we have only one file read its schema, 
                    # otherwise concat the schema for the files (not the best solution)
                    if len(files) == 1:
                        csv_md['schema'] = pd.read_csv(z.open(files[0]), nrows=0, sep=None, encoding_errors='ignore', engine='python').columns.to_list()
                    else:
                        csv_md['schema'] = list(chain(*[pd.read_csv(z.open(fname), nrows=0, sep=None, encoding_errors='ignore', engine='python').columns.to_list() for fname in files]))
            case 'CSV':
                first_line = StringIO(response.data.decode('latin-1')).readline()
                csv_md['schema'] = first_line.split(',')
            case _:
                csv_md['schema'] = None
                return csv_md
    except Exception as e:
        print(f'{url=}, {url[-3:]=}, {format}, {csv_md["format"]=}, {e=}')
        csv_md['schema'] = None
    return csv_md


# create the tmp directory if it doesn't exist
os.makedirs(tmp_path, exist_ok=True)

n_threads = 100

# instantiate a single connection manager
# it should keep a single pool for each thread
# they access different hosts
conn_manager = urllib3.PoolManager(num_pools=n_threads)

# Use ThreadPoolExecutor to process CSVs concurrently with tqdm for progress tracking
with ThreadPoolExecutor(max_workers=n_threads) as executor:
    futures = {executor.submit(fetch_schema, csv_md, conn_manager): csv_md for csv_md in metadata}
    
    # Initialize tqdm for tracking progress
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing URLs"):
        csv_md = future.result()


# remove the tmp dir
shutil.rmtree(f'{tmp_path}')


for csv_md in metadata:
    if 'schema' not in csv_md:
        csv_md['schema'] = None


with open(snapshot_with_schema_path, 'w') as fw:
    json.dump(metadata, fw)


