from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import time
import warnings
import requests
import multiprocessing as mp

import polars as pl
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')



import zipfile
import threading


def fetch_schema(i, ckan_url, formats):
    csv_files = []
    package_list_url = f'{ckan_url}package_search?rows=1000&start={i}'
    try:
        # start_request = time.time()
        response = requests.get(package_list_url)
        # end_request = time.time()
        
        data = response.json()
        packages = data['result']['results']
        
        for dataset in packages:
            dataset_id = dataset['id']
            resources = dataset['resources']    

            for resource in resources:
                if resource['format'].upper() in formats:
                    csv_files.append({
                        'dataset_id'    : dataset_id,
                        'resource_id'   : resource['id'],
                        'title'         : resource['name'],
                        'url'           : resource['url'],
                        'date_modified' : resource['last_modified'],
                        'tags'          : resource['tag']
                    })
    except Exception as e:
        print(i, e)


def foo(ckan_url):
    package_list_url = ckan_url + 'package_search?rows=1000&start=0'
    csv_files = []
    
    start_request = time.time()
    response = requests.get(package_list_url)
    end_request = time.time()
    print(f'Total time for initial request: {round(end_request - start_request, 3)}')

    if response.status_code == 200:
        n_datasets = response.json()['result']['count']
        print(f"Found {n_datasets} datasets.")
    else:
        print(f"Error fetching dataset list: {response.status_code}")
        return

    # Use ThreadPoolExecutor to process CSVs concurrently with tqdm for progress tracking
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = {
            executor.submit(fetch_schema, i): i for i in range(n_datasets)
        }
        
        # Initialize tqdm for tracking progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing CSVs"):
            csv_md = future.result()


# function to get datasets that have CSV resources
def get_metadata_information(ckan_url, formats=['CSV']):
    # get the list of dataset IDs
    package_list_url = ckan_url + 'package_search?rows=1000&start=0'
    csv_files = []
    
    start_request = time.time()
    response = requests.get(package_list_url)
    end_request = time.time()
    print(f'Total time for initial request: {round(end_request - start_request, 3)}')

    if response.status_code == 200:
        data = response.json()
        n_datasets = data['result']['count']
        packages = data['result']['results']
        print(f"Found {n_datasets} datasets.")
    else:
        print(f"Error fetching dataset list: {response.status_code}")
        return

    for i in tqdm(range(0, n_datasets, 1000)):
        package_list_url = f'{ckan_url}package_search?rows=1000&start={i}'
        try:
            start_request = time.time()
            response = requests.get(package_list_url)
            end_request = time.time()
            
            data = response.json()
            packages = data['result']['results']
            
            for dataset in packages:
                dataset_id = dataset['id']
                resources = dataset['resources']    

                for resource in resources:
                    if resource['format'].upper() in formats:
                        csv_files.append({
                            'dataset_id'    : dataset_id,
                            'resource_id'   : resource['id'],
                            'title'         : resource['name'],
                            'url'           : resource['url'],
                            'date_modified' : resource['last_modified']
                        })
        except Exception as e:
            print(i, e)
            continue
    
    print(f'Obtained {len(csv_files)} resource metadata')
    return csv_files


def downloader(csv_md):
    global csv_folder
    try:
        url = csv_md['url']
        csv_id = csv_md['resource_id']
        if url.endswith('.csv'):
            pd.read_csv(url, encoding_errors='ignore', low_memory=True).to_csv(f'{csv_folder}/{csv_id}.csv', index=False)
        elif url.endswith('.xlsx'):
            raise Exception('XLSX endoding')
        #     pd.read_excel(url, encoding_errors='ignore' ).write_csv(f'{csv_folder}/{csv_id}.csv')
        return ()
    except Exception as exception:
        return (str(exception).replace('\n', ' '), url)
        

def initializer(_csv_folder):
    global csv_folder
    csv_folder = _csv_folder



def download_csv_files_from_metadata(metadata: list, csv_folder: str):
    with mp.Pool(initializer=initializer, initargs=(csv_folder, )) as pool:
        errors = list(filter(bool, pool.map(downloader, metadata)))
    return errors



def main():
    data_path = f'{os.path.dirname(__file__)}/../data'
    metadata_folder = f'{data_path}/datasets/metadata'
    
    download_metadata = False
    download_csv = True

    if not os.path.isdir(metadata_folder):
        os.makedirs(metadata_folder)

    open_data_portals = {
        'UK'    : 'https://data.gov.uk/api/action/',
        'US'    : 'https://catalog.data.gov/api/3/action/',
        'CAN'   : 'https://open.canada.ca/data/api/action/',
        'AFR'   : 'https://open.africa/api/action/',
    }

    if download_metadata:
        print('Download metadata...')
        for country_tag, country_ckan_base_url in open_data_portals.items():
            print(f' {country_tag} '.center(50, '#'))
            # get the metadata for the CSV files
            csv_metadata = get_metadata_information(country_ckan_base_url)

            # save the metadata
            metadata_path = f'{metadata_folder}/metadata_{country_tag}.json'
            with open(metadata_path, 'w') as fw:
                json.dump(csv_metadata, fw)

    if download_csv:
        for country_tag, country_ckan_base_url in open_data_portals.items():
            print(f' {country_tag} '.center(50, '#'))
    
            print('Load metadata...')
            metadata_path = f'{metadata_folder}/metadata_{country_tag}.json'
            with open(metadata_path, 'r') as fr:
                csv_metadata = json.load(fr)

            csv_metadata = csv_metadata[:1_000]
            
            print('Download files...')
            dataset_path = f'{data_path}/datasets/{country_tag}'
            if not os.path.isdir(dataset_path):
                os.makedirs(dataset_path)

            errors = download_csv_files_from_metadata(csv_metadata, dataset_path)
            perr = len(errors) * 100 / len(csv_metadata)
            print(f'{perr=}%')
            with open(f'{metadata_folder}/errors_{country_tag}.csv', 'w') as fw:
                for e, u in errors:
                    fw.write(f'{e},{u}\n')

            
        

if __name__ == '__main__':
    main()