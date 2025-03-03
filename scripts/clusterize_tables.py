import os
import json
import zipfile
from itertools import chain, product
from collections import defaultdict

import polars as pl
from tqdm import tqdm
from datasketch import MinHashLSHForest, MinHash

from orqa.utils import sanitize_string



def create_minhash_for_(words: list, num_perm: int, n: int=3):
    mh = MinHash(num_perm=num_perm)
    # from nltk import ngrams
    # for w in words:
    #     for ngram in ngrams(w, n):
    #         mh.update(''.join(ngram).encode('utf8'))
    mh.update_batch(list(map(str.encode, words)))
    return mh


def create_clusters(open_dataset_name="CAN", task="join", data_path="data", 
                    num_perm : int = 16, 
                    l : int = 4,
                    k : int = 200,
                    max_num_of_occurrences : int = 1
                    ):
    OUTPUT_CLEANED_GT   = f'{data_path}/{task}/{open_dataset_name}_ground_truth.csv'
    DATASET_PATH        = f'{data_path}/datasets/datasets_{open_dataset_name}.zip'
    CLUSTERS_PATH       = f'{data_path}/{task}/clusters/clusters_{open_dataset_name}.json'
    TABLES_SCHEMA       = f'{data_path}/{task}/schema/tables_schema_{open_dataset_name}.json'

    cat_gt_ids = set(chain(*pl.read_csv(OUTPUT_CLEANED_GT).rows()))

    print(f'Total tables in {open_dataset_name} Ground Truth: {len(cat_gt_ids)}')

    errors = []
    schema = {}

    # Read the schema of each table
    with zipfile.ZipFile(DATASET_PATH, 'r') as zip_tables:
        table_names = [f.removeprefix(f'datasets_{open_dataset_name}/') for f in zip_tables.namelist() if not f.startswith('__MACOSX') and not f.endswith('/') and not f.split('/')[-1].startswith('._')]

        for table_id in tqdm(cat_gt_ids.intersection(table_names), leave=False):
            with zip_tables.open(f'datasets_{open_dataset_name}/{table_id}', 'r') as table_file:
                try:
                    schema[table_id] = list(map(sanitize_string, (pl.read_csv(table_file, n_rows=0).columns)))
                except:
                    errors.append(table_id)


    all_attributes = set(chain(*schema.values()))
    print(f'Total number of distinct attributes: {len(all_attributes)}')

    # Save the tables schema
    os.makedirs(os.path.dirname(TABLES_SCHEMA), exist_ok=True)
    with open(TABLES_SCHEMA, 'w') as fw:
        json.dump(schema, fw)

    # Create the LSHForest index
    lsh_forest = MinHashLSHForest(num_perm=num_perm, l=l)
    minhashes = []

    for table_id, attributes in tqdm(schema.items(), leave=False):
        mh = create_minhash_for_(attributes, num_perm)
        minhashes.append((table_id, mh))

    for table_id, mh in tqdm(minhashes, leave=False):
        lsh_forest.add(table_id, mh)
    lsh_forest.index()


    all_table_ids = set(schema.keys())
    clusters = defaultdict(set)

    while all_table_ids:
        table_id = all_table_ids.pop()

        # check if the current table is already in any cluster
        if sum(table_id in cluster for cluster in clusters.values()) >= max_num_of_occurrences:
            continue
        
        # init the new cluster
        cluster_id = len(clusters)
        clusters[cluster_id].add(table_id)

        # query the index for a large number of results
        mtab = MinHash(num_perm=num_perm)
        mtab.update_batch(list(map(str.encode, schema[table_id])))
        neighbors = lsh_forest.query(mtab, k=k)

        # add each item to the cluster (if it is not in another cluster yet)
        for n in neighbors:
            if sum(n in cluster for cluster in clusters.values()) >= max_num_of_occurrences:
                continue
            clusters[cluster_id].add(n)

    print(f'Number of clusters: {len(clusters)}')


    # For each cluster, take all the attributes in its tables' schemas
    cluster_attributes = [
        [cluster_key, {attribute for table_id in cluster for attribute in schema[table_id]}]
        for cluster_key, cluster in clusters.items()
    ]

    cluster_attributes_dict = {ckey: list(cschema) for ckey, cschema in cluster_attributes}


    # Save the clusters
    os.makedirs(os.path.dirname(CLUSTERS_PATH), exist_ok=True)
    with open(CLUSTERS_PATH, 'w') as fw:
        json.dump({
            'max_num_of_cooccurrences': max_num_of_occurrences,
            'num_perm': num_perm,
            'l': l,
            'k': k,
            'clusters': [
                {
                    'key': ckey, 
                    'schema': cluster_attributes_dict[ckey],
                    'ids': list(set(cluster))

                } for ckey, cluster in clusters.items()
            ]
            
        }, fw, indent=4)



def main():
    open_datasets = ['CAN', 'USA', 'UK', 'SG']
    tasks = ['join', 'union']

    for open_dataset, task in product(open_datasets, tasks):
        print(f'WORKING ON {open_dataset} - {task}')
        create_clusters(open_dataset, task, data_path=f'{os.path.dirname(__file__)}/../data')

    
if __name__ == '__main__':
    main()


