import os
import sys
import time
import queue
import shutil
import logging
import urllib3
import zipfile
import warnings
from io import BytesIO
from typing import Tuple
from statistics import mean
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from logging.handlers import QueueHandler, RotatingFileHandler, QueueListener

import tqdm
import tqdm.auto
import jsonlines
import pandas as pd

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')


MAX_THREADS_NUM = 10
MAX_PROC_NUM = 10


def init_logger(log_directory):
    root = logging.getLogger(f'crawlerLogger_{os.getpid()}')
    root.setLevel(logging.INFO)
    que = queue.Queue(-1)
    queue_handler = QueueHandler(que)
    if root.hasHandlers():
        root.handlers.clear()

    # TODO keep only the 2 most recent log directories
    old_dirs =  sorted([d for d in os.listdir(os.path.dirname(log_directory))], reverse=True)
    dirs_to_delete = old_dirs[3:] if len(old_dirs) > 3 else []

    for dir_to_delete in dirs_to_delete:        
        dir_path = os.path.join(os.path.dirname(log_directory), dir_to_delete)     
        shutil.rmtree(dir_path)
    
    if not root.hasHandlers():
        handler = RotatingFileHandler(f"{log_directory}/{os.getpid()}.log", mode="a", maxBytes=1024 ** 3)
        log_formatter = logging.Formatter("%(asctime)s,[%(process)d],[%(threadName)s],[%(levelname)s],%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(log_formatter)
        root.addHandler(queue_handler)
    
    listener = QueueListener(que, handler)
    return root, listener


def download_resource_csv(data: Tuple[urllib3.PoolManager | urllib3.ProxyManager, str, str, str, int, logging.Logger|None]):
    http, rsc_url, rsc_name, download_directory, max_content_length, logger = data

    def csv_base():
        # sometimes the data are encoded, sometimes not
        # and we do not want to start reading zip files here
        assert not rsc_url.endswith('.zip')
        assert 'DOCTYPE' not in response.data[:100].decode('latin-1')
        try:
            pd.read_csv(response.data, **pd_read_csv_kwargs).to_parquet(f'{download_directory}/{rsc_name}.parquet', **pd_to_parquet_kwargs)
        except:
            # in some cases data are encoded, thus we try again with a byte io stream
            pd.read_csv(BytesIO(response.data), **pd_read_csv_kwargs).to_parquet(f'{download_directory}/{rsc_name}.parquet', **pd_to_parquet_kwargs)
            
    def zip():
        # open the donwloaded ZIP
        with zipfile.ZipFile(BytesIO(response.data), 'r') as z:
            file_names = list(filter(lambda fname: 'metadata' not in fname.lower() and 'fr' not in fname.lower() and fname.endswith('.csv'), z.namelist()))
            for i, fname in enumerate(file_names):
                (
                    pd
                    .read_csv(z.open(fname), **pd_read_csv_kwargs)
                    .to_parquet(f'{download_directory}/{rsc_name}{"" if len(file_names) == 1 else f"_{i}"}.parquet', **pd_to_parquet_kwargs)
                )

    try:
        # try to get the size of the file
        response = http.request('GET', rsc_url, preload_content=False, redirect=True)
        content_bytes = response.headers.get("Content-Length")

        # Accept files with limited size
        if content_bytes and int(content_bytes) > max_content_length:
            if logger: logger.warning(f'Large content-length for {rsc_url=}')
            return False

        pd_read_csv_kwargs = {'sep': None, 'encoding': 'latin-1', 'encoding_errors': 'ignore', 'on_bad_lines': 'skip', 'engine': 'python'}
        pd_to_parquet_kwargs = {'index': False, 'compression': 'gzip'}            

        # download all the resource data at once
        response = http.request('GET', rsc_url, redirect=True)
        if logger: logger.debug(f"Resource {rsc_name} size: {len(response.data)}")
    except urllib3.exceptions.TimeoutError:
        if logger: logger.warning(f"Timeout with resource: {rsc_url}")
        return False

    if logger: logger.debug(f"Extracting data from resource {rsc_name}...")

    # try each method to get the data
    success = False
    for method in [csv_base, zip]:
        try:             
            method()
            success = True
            break
        except Exception as e:
            if logger: logger.debug(f"Method {method} failed with resource {rsc_name}: {e}")
            continue
    
    if logger: logger.debug(f"{'SUCCESS' if success else 'FAILURE'} download resource {rsc_url}")
    return success


def process_task(
        package_search_url: str,
        download_directory: str,
        temporary_directory: str,
        log_directory: str,
        start: int,
        n_workers: int = 10,
        packages_per_worker:int = 1_000,
        accepted_formats: list = ['CSV'],
        proxy_kwargs: dict | None = None, 
        keep_logging: bool = True):
    
    if keep_logging:
        logger, listener = init_logger(log_directory)
        listener.start()
    else:
        logger = None
    
    start_ptask = time.time()
    
    # create the HTTP manager for this process (shared among its threads)
    if proxy_kwargs:
        http = urllib3.ProxyManager(
            maxsize=n_workers,
            retries=False,
            timeout=urllib3.Timeout(connect=7.0, read=5.0),        
            headers={
                'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0'
            },
            **proxy_kwargs
        )
    else:
        http = urllib3.PoolManager(
            maxsize=n_workers,
            retries=False,
            timeout=urllib3.Timeout(connect=7.0, read=5.0),        
            headers={
                'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0'
            }
        )

    # the success ratio of tables correctly downloaded
    success = 0
    success_rate = 0
    resources = []
    packages_metadata = []

    try:
        # get the URL
        pkg_url = f'{package_search_url}?start={start}&rows={packages_per_worker - 1}'

        # create a process-specific tmp directory 
        # (which will be removed at the end of current job)
        temporary_directory = f'{temporary_directory}/{os.getpid()}'
        os.makedirs(temporary_directory, exist_ok=True)

        # get the packages metadata
        response = http.request('GET', pkg_url, redirect=True)
        if response.status != 200:
            if logger: logger.warning(f'Invalid response for {pkg_url=}: {response.status=}.')
            return start, os.getpid(), round(time.time() - start_ptask), []
        
        packages_metadata = response.json()['result']['results']
        
        # get all the resources metadata
        resources = [
            rsc 
            for pkg in packages_metadata 
            for rsc in pkg['resources'] 
            if rsc['format'] in accepted_formats and ('language' not in rsc or 'language' in rsc and 'en' in rsc['language'])
        ]
        res_urls = [rsc['url'] for rsc in resources]

        # Does this duplicates any package downloads?
        if logger: logger.info(f'Downloaded packages metadata, from {start=} to {start + len(packages_metadata)}, total resources: {len(res_urls)} (urls={res_urls})')
        
        with ThreadPoolExecutor(max_workers=min(n_workers, MAX_THREADS_NUM)) as thread_executor:
            success = sum(thread_executor.map(download_resource_csv, [[http, rsc['url'], rsc['id'], download_directory, 2**29, logger] for rsc in resources]))
        success_rate = round((success * 100 / len(resources)) if resources else 0, 3)

    except Exception as e:
        if logger: logger.error(f'Proccess failure: {e}')
    finally:
        if logger: logger.info(f'Process completed current task. {success}/{len(resources)} ({success_rate}%) success resource downloads.')
        if keep_logging: listener.stop()
        http.clear()
        if logger: logger.debug(f'Process released resources')
        
    shutil.rmtree(temporary_directory, ignore_errors=True)
    return start, os.getpid(), round(time.time() - start_ptask, 3), packages_metadata, success, success_rate



def download_tables(url_basepoint: str, 
                    download_directory: str,
                    temporary_directory: str,
                    log_directory: str,
                    accepted_formats: list = ["CSV"],
                    logger: logging.Logger|None = None,                    
                    n_workers: int = 10, 
                    packages_per_worker: int = 1_000,
                    from_n_package:int | None = None,
                    to_n_package: int | str | None = "END",
                    proxy_kwargs: None | dict = None, 
                    keep_logging: bool = True):
    """
    Download all the available resources from the given Open Data URL basepoint.

    It spawns a pool of n_workers processes, and each worker spawns a pool of n_worker threads.

    All the downloaded data are saved to the given download directory.

    Some corner-cases may not be covered by the download logic.
    """
    
    try: from_n_package = int(from_n_package)
    except: pass
    try: to_n_package = int(to_n_package)
    except: pass
    assert isinstance(to_n_package, str) or (isinstance(to_n_package, int) and to_n_package > from_n_package)
    
    if keep_logging:
        logger, listener = init_logger(log_directory)        
        listener.start()
    else:
        logger = None
    
    # the basepoint available for all the relevant Open Data 
    # portals is "package_search"
    package_search_url = f'{url_basepoint}/package_search'

    try:
        # get initial response, to get some basic stats
        init_url = f'{package_search_url}?start=0&rows=1000'
        if not proxy_kwargs:
            http = urllib3.PoolManager()
        else:
            http = urllib3.ProxyManager(
                **proxy_kwargs
                # cert_reqs='CERT_NONE',  # Disables SSL certificate verification (equivalent to -k in curl)
                # retries=3,  # Retry on failure (optional)
                # proxy_url='http://localhost:9999',  # Proxy configuration (equivalent to -x in curl)
            )

        response = http.request(
            'GET',
            init_url,
            redirect=True
        )

        if response.status != 200:
            if logger: logger.warning(f'Impossible to get initial response for {init_url=}')
            return
        
        response = response.json()                
        n_total_packages = response['result']['count']
        if logger:
            n_valid_resources = sum(1 for pkg in response['result']['results'] for rsc in pkg['resources'] if rsc['format'] in accepted_formats)    
            logger.info(f'{n_total_packages=} for current Open Data portal')
            logger.info(f'{n_valid_resources=} in first 1000 packages ({round(n_valid_resources / 1000, 1)} on average)')
        http.clear()
        
        to_n_package_usr    = to_n_package
        from_n_package      = max(from_n_package, 0) if from_n_package else 0
        to_n_package        = n_total_packages if isinstance(to_n_package, str) else min(n_total_packages, to_n_package) if to_n_package else n_total_packages
        n_total_packages    = to_n_package - from_n_package
        packages_per_worker = min(packages_per_worker, to_n_package)

        metadata_jsonl      = f'{download_directory}/metadata/metadata_from{from_n_package}_to{to_n_package if isinstance(to_n_package_usr, int) else to_n_package_usr}.jsonl'
        download_directory  = f'{download_directory}/tables/tables_from{from_n_package}_to{to_n_package if isinstance(to_n_package_usr, int) else to_n_package_usr}'
        
        # remove old existent data
        shutil.rmtree(download_directory, ignore_errors=True)
        if os.path.exists(metadata_jsonl):
            os.remove(metadata_jsonl)

        # create the directories
        os.makedirs(os.path.dirname(metadata_jsonl), exist_ok=True)
        os.makedirs(download_directory, exist_ok=True)
        
        if logger: logger.info(f'Downloading tables into {download_directory}')
        if logger: logger.info(f'Submitting download tasks for {n_total_packages=}, from {from_n_package} to {to_n_package}...')

        # start the process pool
        with ProcessPoolExecutor(max_workers=min(n_workers, MAX_PROC_NUM), mp_context=mp.get_context('spawn')) as executor:
            futures = [
                executor.submit(
                    process_task, package_search_url, download_directory, temporary_directory, log_directory, i, n_workers, packages_per_worker, accepted_formats, proxy_kwargs, keep_logging)
                    for i in range(from_n_package, to_n_package, packages_per_worker)
                ]
            if logger: logger.debug(f'Total work steps: {len(range(from_n_package, to_n_package, packages_per_worker))}')

            total_success = []
            packages_metadata = []
            
            # if logger: logger.info('Saving packages metadata to JSON Lines...')
            with jsonlines.open(metadata_jsonl, 'a') as w:
                for future in tqdm.auto.tqdm(as_completed(futures), total=len(futures), desc="Global Processing Status:"):
                    start, pid, ptime, metadata, success, success_rate = future.result()
                    total_success.append((success, success_rate))
                    if logger: logger.info(f'[PID:{pid}],[PROC_TIME:{ptime}s],[SUCCESS:{success}],[SUCCESS_RATE:{success_rate}],Step completed: [{start},{start + packages_per_worker-1}]')
                    # once a step is completed, write the collected metadata and release these resources
                    for md in metadata:
                        packages_metadata.append(md)
                    w.write_all(packages_metadata)
                    packages_metadata = []
                    del future
            
        if logger: 
            logger.info('Done')
            logger.info(f'Total success downloads: {sum(s[0] for s in total_success)}. Average success rate (per process): {mean(s[1] for s in total_success)}')
        
    
    except Exception as e:
        if logger: logger.error(e)
        raise e
    finally:
        if keep_logging: listener.stop()


def main(country_tag: str = 'CAN',
         from_: int = 0,
         to_: int|str = 'END',
         country_ckan_base_url: str = 'https://open.canada.ca/data/api/action'):
    
    data_path       = f'{os.path.dirname(__file__)}/../data'
    tmp_dir         = f'{data_path}/tmp'
    log_dir         = f'{data_path}/log'

    # clean and create directories
    # currently is not used, TODO check if is ok
    # to store large files on disk and do then IO 
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # 'CAN'   : 'https://open.canada.ca/data/api/action',
    # 'US'    : 'https://catalog.data.gov/api/3/action',
    # 'UK'    : 'https://data.gov.uk/api/action',
    # 'AFR'   : 'https://open.africa/api/action',
    # 'SG'    : '???'
    
    download_dir = f'{data_path}/datasets/{country_tag}'
    log_dir = f"{log_dir}/{country_tag}/crawling/{time.strftime('%y%m%d_%H_%M_%S')}"

    shutil.rmtree(download_dir, ignore_errors=True)
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    download_tables(
        country_ckan_base_url, 
        download_dir, 
        tmp_dir, 
        log_dir, 
        ['CSV'], 
        n_workers=10,
        packages_per_worker=10,
        from_n_package=from_,
        to_n_package=to_,
        keep_logging=True
    )

    # remove the temporary directory
    shutil.rmtree(tmp_dir, ignore_errors=True)
    # shutil.move(log_dir, f"{log_dir}_{time.strftime('%y%m%d_%H_%M_%S')}")

if __name__ == '__main__':
    main(*sys.argv[1:])
