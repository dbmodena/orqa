import random
import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Any, Tuple, Dict
import sys
import os
import polars as pl 
import pandas as pd
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed  # ADD THIS LINE
from valentine import valentine_match
from valentine.algorithms import Coma






DATASET_DIR = Path("D:/uk_small/uk_small_copy/datasets/csv")
DOCUMENT_TYPE = "csv"
MATCHES_JSON = "matches.json"
THRESHOLD = 0.5
MAX_WORKERS = 12

### calcs the Valentine averages
def calcAvgValentine(scores,q_columns:list=None,r_columns:list=None):
    averageAll = sum(scores.values()) / len(scores)
    specificAvg = 0
    filteredValues = []
    if q_columns is not None and r_columns is not None:
        filteredValues = [ value for ((_, col1), (_, col2)), value in scores.items()if col1 in q_columns and col2 in r_columns]
        specificAvg = sum(filteredValues) / len(filteredValues)
    elif q_columns is not None:
        filteredValues = [ value for ((_, col1), (_, col2)), value in scores.items()if col1 in q_columns and col2 in q_columns]
        specificAvg = sum(filteredValues) / len(filteredValues)
    return averageAll,specificAvg

def prepare_datasets(dataset_dir,Q,R):
    try:
        l_path = dataset_dir.joinpath(f"{Q}.{DOCUMENT_TYPE}")
        r_path = dataset_dir.joinpath(f"{R}.{DOCUMENT_TYPE}")
        l_table = pd.read_csv(l_path,thousands=",").dropna(axis=1, how="all")
        r_table = pd.read_csv(r_path,thousands=",").dropna(axis=1, how="all")
        return l_table,r_table
    except Exception as e:
        print(e)
        return None

def valentineHandler(dataset_dir,Q, R, q_columns:list =None,r_indices:list=None) -> Dict:
    try:
        l_table, r_table = prepare_datasets(dataset_dir,Q,R)
        matcher = Coma(use_instances=True)
        if r_indices is not None:
            matches= valentine_match(l_table, r_table, matcher)
            r_columns = r_table.columns[r_indices].tolist()
            avg, avg_q= calcAvgValentine(matches,q_columns,r_columns)
            print(f"average overall {avg}")
            print(f"average score columns involved {avg_q}")
            return matches,avg,avg_q
        else:
            matches= valentine_match(l_table, r_table, matcher)
            avg, avg_q= calcAvgValentine(matches,q_columns)
            print(f"average overall {avg}")
            print(f"average score columns involved {avg_q}")
            return matches,avg,avg_q
    except Exception as e:
        print(e)








def find_matches(matches_path:Path, dataset_dir:Path):
    try:
        with open(matches_path) as f:
            data = json.load(f)
            # Add all nodes first
            for entry in data:
                print(f"Analyzing the overlap between {entry["Q"]} and {entry["R"]}")
                if entry["task"] == "U":
                    print("smell ya later")
                    #valentineHandler(dataset_dir,entry["Q"], entry["R"], entry["q_columns"])
                elif entry["task"] == "MJ":
                    valentineHandler(dataset_dir,entry["Q"], entry["R"], entry["q_join_keys"], entry["r_join_keys_pos"])
                elif entry["task"] == "JC":
                    print("still not supported")
    except Exception as e:
        print(e)






if __name__ == "__main__":
    #table_id = "2019-March-return__3f436d14-4e17-476c-a3e4-66d18e7f6c90"
    #operation = "JC"
    path_matches = Path(MATCHES_JSON)
    path_dataset = Path(DATASET_DIR)
    find_matches(path_matches, path_dataset)