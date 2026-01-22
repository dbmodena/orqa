import random
import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Any, Tuple, Dict
import sys
import os
from sloth import sloth
import polars as pl 
import pandas as pd
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed  # ADD THIS LINE


DATASET_DIR = Path("D:/uk_small/uk_small_copy/datasets/csv")
DOCUMENT_TYPE = "csv"
MATCHES_JSON = "matches.json"
THRESHOLD = 0.5
MAX_WORKERS = 8

def calculateOverlapRatio(rTab: List[List[Any]], result: List[tuple], metrics: list) -> Dict:
    """
    Calculate the overlap ratio using metrics and rTab
    
    Args:
        rTab: The R table as list of columns
        result: The SLOTH result [(mapping, overlap_rows), ...]
        metrics: The metrics returned by SLOTH
        
    Returns:
        Dictionary with overlap statistics
    """
    if not result or not result[0] or not metrics:
        return {
            'overlapArea': 0,
            'rInvolvedArea': 0,
            'overlapRatio': 0.0,
            'numColumnsInvolved': 0,
            'numRowsInR': 0,
            'numRowsOverlapping': 0
        }
    
    # Extract from metrics (already calculated by SLOTH)
    overlapArea = metrics[12]  # Area from metrics
    numRowsOverlapping = metrics[11]  # Height from metrics
    numColumnsInvolved = metrics[10]  # Width from metrics
    
    # Extract mapping from result
    mapping, overlapRows = result[0]
    
    # Get R column indices involved
    rColumnIndices = [colPair[0] for colPair in mapping]
    
    # Calculate R involved area
    numRowsInR = len(rTab[rColumnIndices[0]]) if rColumnIndices else 0
    rInvolvedArea = numColumnsInvolved * numRowsInR
    
    # Calculate ratio
    overlapRatio = (overlapArea / rInvolvedArea * 100) if rInvolvedArea > 0 else 0.0
    
    return {
        'overlapArea': overlapArea,
        'rInvolvedArea': rInvolvedArea,
        'overlapRatio': overlapRatio,
        'numColumnsInvolved': numColumnsInvolved,
        'numRowsInR': numRowsInR,
        'numRowsOverlapping': numRowsOverlapping,
        'rColumnIndices': rColumnIndices
    }

def analyzeTablePair(rTab: List[List[Any]], sTab: List[List[Any]], 
                     verbose: bool = True) -> Dict:
    """
    Analyze a single pair of tables that are already in list of lists format
    
    Args:
        rTab: The R table as list of columns
        sTab: The S table as list of columns
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary containing result, metrics, and statistics
    """
    metrics = []
    
    # Run SLOTH
    result, metrics = sloth(rTab, sTab, metrics=metrics, verbose=verbose)
    # Calculate overlap statistics
    stats = calculateOverlapRatio(rTab, result, metrics)
    print(stats['overlapRatio'])
    return stats['overlapRatio']



def analyze_table_pair(left_table:Path=None,right_table:Path=None)-> float:
    try:
        l_table, r_table = prepare_tables(left_table,right_table)
        return analyzeTablePair(l_table, r_table,False) 
    except Exception as e:
        print(e)
        return 0

def prepare_tables(left_table:Path,right_table:Path):
    l_table = loadCsvAsLists(pd.read_csv(left_table,thousands=","))
    r_table = loadCsvAsLists(pd.read_csv(right_table,thousands=","))
    return l_table, r_table  


def loadCsvAsLists(df) -> List[List[Any]]:
    # Convert to column-oriented list of lists
    columns = []
    for colName in df.columns:
        columns.append(df[colName].to_list())   
    return columns

def process_edge(entry, dataset_dir):
    """Helper function to process a single edge in parallel"""
    Q_node = entry["Q"]
    R_node = entry["R"]
    task_label = entry["task"]
    
    if Q_node == R_node:
        return None
    
    try:
        r_path = dataset_dir.joinpath(f"{Q_node}.{DOCUMENT_TYPE}")
        l_path = dataset_dir.joinpath(f"{R_node}.{DOCUMENT_TYPE}")
        score = analyze_table_pair(r_path, l_path)
        return (Q_node, R_node, task_label, score)
    except Exception as e:
        print(e)
        return None

def build_matches_graph(matches_path:Path, dataset_dir:Path):
    G = nx.Graph()
    try:
        with open(matches_path) as f:
            data = json.load(f)
            
            # Add all nodes first
            for entry in data:
                print(f"Analyzing {entry}")
                G.add_node(entry["Q"])
                G.add_node(entry["R"])
            
            # Process edges in parallel
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(process_edge, entry, dataset_dir): entry 
                    for entry in data
                }
                
                # Collect results as they complete
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        Q_node, R_node, task_label, score = result
                        G.add_edge(Q_node, R_node, label=task_label, score=score)
                        
    except Exception as e:
        print(e)
        return nx.Graph()
    return G

def expand_one_hop(G, source_nodes, edge_label, score_threshold, selected_edges, visited):
    next_hop = set()
    for src in source_nodes:
        for u, v, data in G.edges(src, data=True):
            if (
                data.get("label") == edge_label and
                data.get("score", 0) >= score_threshold
            ):
                next_node = v if u == src else u

                # prevent going backwards or revisiting
                if next_node in visited:
                    continue

                selected_edges.add((u, v))
                next_hop.add(next_node)

    return next_hop

def fetch_matches(G, node, edge_label, score_treshold=THRESHOLD, show_edge_labels=False, edge_label_format="label_score", second_hop_check=True):
    if node not in G:
        raise ValueError(f"Node '{node}' not found in graph")

    selected_edges = set()
    visited = {node}

    # -------- FIRST HOP --------
    first_hop = expand_one_hop(
        G,
        source_nodes={node},
        edge_label=edge_label,
        score_threshold=score_treshold,
        selected_edges=selected_edges,
        visited=visited,
    )
    visited |= first_hop

    # -------- SECOND HOP --------
    if second_hop_check:
        second_hop = expand_one_hop(
            G,
            source_nodes=first_hop,
            edge_label=edge_label,
            score_threshold=score_treshold,
            selected_edges=selected_edges,
            visited=visited,
        )
        visited |= second_hop

    # -------- BUILD SUBGRAPH (EDGE-DRIVEN) --------
    subG = G.edge_subgraph(selected_edges).copy()
    subG.add_node(node)  # keep isolated root if needed

    return subG

if __name__ == "__main__":
    table_id = "2019-March-return__3f436d14-4e17-476c-a3e4-66d18e7f6c90"
    operation = "JC"
    path_matches = Path(MATCHES_JSON)
    path_dataset = Path(DATASET_DIR)
    G = build_matches_graph(path_matches, path_dataset)
    matching_subgraph = fetch_matches(G, node=table_id, edge_label=operation)
    print(matching_subgraph.edges())


