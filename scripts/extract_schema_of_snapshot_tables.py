import os
import re
import shutil
import time
import zipfile
import urllib3
import threading
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By

from tqdm import tqdm
from jsonlines import jsonlines


data_path                   = f'{os.path.dirname(__file__)}/../data'
tmp_path                    = f'{data_path}/tmp'

# the tables metadata obtained from the Open Data Canada 
# filtered by date (around July 2023)
snapshot_path               = f'{data_path}/snapshot.jsonl'

# the above metadata enriched with the relative table schema
snapshot_with_schema_path   = f'{data_path}/snapshot_with_schema_also_for_zip.jsonl'

with jsonlines.open(snapshot_path) as reader:
    metadata = list(reader.iter())


def fetch_schema(csv_md, http: urllib3.PoolManager):
    url = csv_md['url']
    schema = []
    start_get = end_get = start_read = end_read = 0
    
    try:
        # the format field isn't always totally accurate: sometimes
        # the URL links to a ZIP folder which contains one or more CSVs
    
        format = 'CSV' if url.endswith('.csv') else 'ZIP'
        
        start_get = time.time()
        response = http.request('GET', url, preload_content=format == 'CSV')
        end_get = time.time()
        
        # if no valid response, return
        if response.status != 200:
            csv_md['schema'] = []
            return csv_md
        
        def zip_schema():
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
                    schema = pd.read_csv(z.open(files[0])  , nrows=5, sep=None, encoding='latin-1', encoding_errors='ignore', engine='python').columns.to_list()
                else:
                    schema = []
                    for fname in files:
                        schema += pd.read_csv(z.open(files[0])  , nrows=5, sep=None, encoding='latin-1', encoding_errors='ignore', engine='python').columns.to_list()                
            return schema
        
        def nested_link_schema():
            # initialize the WebDriver
            options = webdriver.FirefoxOptions()
            options.add_argument('--headless')
            driver = webdriver.Firefox(options=options)

            # open the webpage
            driver.get(url)

            # wait for the page to fully load
            time.sleep(5)            
            
            response = http.request('GET', re.search(r'href="(https?://[^\"]+\.csv)"', driver.page_source).group(1))        
            return pd.read_csv(StringIO(response.data.decode('latin-1'))  , nrows=5, sep=None, encoding='latin-1', encoding_errors='ignore', engine='python').columns.to_list()
        
        def csv_schema():
            return pd.read_csv(StringIO(response.data.decode('latin-1'))  , nrows=5, sep=None, encoding='latin-1', encoding_errors='ignore', engine='python').columns.to_list()

        start_read = time.time()
        for f in [csv_schema, zip_schema, nested_link_schema]:
            try:
                schema = f()
                break
            except:
                continue

        # create a set from the schema
        # here I expect a list of tuples as schema: [(h1, h2, hA), (h1, h2, hB), (h1, h3, hC), ...]
        schema = [attr for header in schema for attr in (header if isinstance(header, tuple) else [header])]
        
    except Exception as e:
        print(f'{url=}, {e=}')
    end_read = time.time()

    csv_md['schema'] = schema
    csv_md['get_time'] = round(end_get - start_get, 5)
    csv_md['read_schema'] = round(end_read - start_read, 5)
    return csv_md


# create the tmp directory if it doesn't exist
os.makedirs(tmp_path, exist_ok=True)

n_threads = 20

# instantiate a single connection manager, it should keep 
# a single pool for each thread, since they access different hosts
conn_manager = urllib3.PoolManager(num_pools=n_threads, timeout=urllib3.Timeout(total=240))

metadata = metadata[1680:1690]
with ThreadPoolExecutor(max_workers=n_threads) as executor:
    futures = {executor.submit(fetch_schema, csv_md, conn_manager): csv_md for csv_md in metadata}    
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing URLs"):
        csv_md = future.result()        

# remove the tmp dir
shutil.rmtree(f'{tmp_path}')

# set a default empty value for all those cases
# where the schema fetching phase has failed
for csv_md in metadata:
    if 'schema' not in csv_md:
        csv_md['schema'] = []

with jsonlines.open(snapshot_with_schema_path, 'w') as fw:
    fw.write_all(metadata)

