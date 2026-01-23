from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import networkx as nx
from .sloth import sloth

from ..utils import pl_read_dataset

DATASET_DIR = Path("D:/uk_small/uk_small_copy/datasets/csv")
DOCUMENT_TYPE = "csv"
MATCHES_JSON = "matches.json"
THRESHOLD = 0.5
MAX_WORKERS = 8
OVERLAP_RATIO_THRESHOLD = 0.5


def load_dataset_as_list_of_columns(path: Path, opts: dict = {}) -> list[list[Any]]:
    df = pl_read_dataset(path, opts)

    # Convert to column-oriented list of lists
    columns = []
    for col_name in df.columns:
        columns.append(df[col_name].to_list())
    return columns


def calculate_overlap_ratio(
    table: list[list[Any]], result: list[tuple], metrics: list
) -> dict:
    """
    Calculate the overlap ratio using metrics and r_tab

    Args:
        r_tab: The R table as list of columns
        result: The SLOTH result [(mapping, overlap_rows), ...]
        metrics: The metrics returned by SLOTH

    Returns:
        dictionary with overlap statistics
    """
    if not result or not result[0] or not metrics:
        return {
            "overlap_area": 0,
            "r_involved_area": 0,
            "overlap_ratio": 0.0,
            "num_columns_involved": 0,
            "num_rows_in_r": 0,
            "num_rows_overlapping": 0,
        }

    # Extract from metrics (already calculated by SLOTH)
    overlap_area = metrics[12]  # Area from metrics
    num_rows_overlapping = metrics[11]  # Height from metrics
    num_columns_involved = metrics[10]  # Width from metrics

    # Extract mapping from result
    mapping, overlap_rows = result[0]

    # Get R column indices involved
    r_column_indices = [col_pair[0] for col_pair in mapping]

    # Calculate R involved area
    num_rows_in_r = len(table[r_column_indices[0]]) if r_column_indices else 0
    r_involved_area = num_columns_involved * num_rows_in_r

    # Calculate ratio
    overlap_ratio = (
        (overlap_area / r_involved_area * 100) if r_involved_area > 0 else 0.0
    )

    return {
        "overlap_area": overlap_area,
        "r_involved_area": r_involved_area,
        "overlap_ratio": overlap_ratio,
        "num_columns_involved": num_columns_involved,
        "num_rows_in_r": num_rows_in_r,
        "num_rows_overlapping": num_rows_overlapping,
        "r_column_indices": r_column_indices,
    }


def compute_overlap_metrics(
    left_table: list[list[Any]], right_table: list[list[Any]], verbose: bool = True
) -> dict:
    """
    Analyze a single pair of tables that are already in list of lists format.
    Returns statistics about the overlap computed with respect to the left table only (overlap ratio).

    :param left_table: The left table as list of columns
    :param right_table: The right table as list of columns
    :param verbose: Whether to print detailed information
    :return dictionary containing result, metrics, and statistics
    """
    metrics = []

    # Run SLOTH
    result, metrics = sloth(left_table, right_table, metrics=metrics, verbose=verbose)
    # Calculate overlap statistics
    stats = calculate_overlap_ratio(left_table, result, metrics)
    return {"overlap_ratio": stats["overlap_ratio"]}


def process_edge(
    entry: dict, datasets_folder: Path, polars_opts: dict = {}
) -> tuple[str, str, str, dict] | None:
    """Helper function to process a single edge in parallel"""
    q_node = entry["Q"]
    r_node = entry["R"]
    task_label = entry["task"]

    if q_node == r_node:
        return None

    try:
        left_path = datasets_folder.joinpath(f"{q_node}.{DOCUMENT_TYPE}")
        right_path = datasets_folder.joinpath(f"{r_node}.{DOCUMENT_TYPE}")

        left_table = load_dataset_as_list_of_columns(left_path, polars_opts)
        right_table = load_dataset_as_list_of_columns(right_path, polars_opts)

        overlap_metrics = compute_overlap_metrics(left_table, right_table)
        return (q_node, r_node, task_label, overlap_metrics)
    except Exception as e:
        print(e)
        return None


def build_matches_graph(
    matches: list[dict], datasets_folder: Path, polars_opts: dict = {}
) -> nx.MultiDiGraph:
    """
    From the discovered matches, build a graph where nodes are
    tables and edges are their connections, labelled with overlap
    ratio, schema matching metrics and other useful metrics.

    :param matches: A list of discovered matches.
    :param datasets_folder: Where the datasets are placed.
    :param polars_opts: A dictionary of polars.read_* with entry for any expected filetype.
    :return The Graph representing the matches enriched with metadata.
    """
    G = nx.MultiDiGraph()
    try:
        # Add all nodes first
        for entry in matches:
            # print(f"Analyzing {entry}")
            G.add_node(entry["Q"])
            G.add_node(entry["R"])

        # Process edges in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    process_edge, entry, datasets_folder, polars_opts
                ): entry
                for entry in matches
            }

            # Collect results as they complete
            for future in as_completed(futures):
                result = future.result()
                if result:
                    q_node, r_node, task_label, metrics = result
                    G.add_edge(q_node, r_node, label=task_label, **metrics)

    except Exception as e:
        print(e)
        return nx.MultiDiGraph()
    return G


def overlap_ratio_predicate(node: dict) -> bool:
    return node["overlap_ratio"] >= OVERLAP_RATIO_THRESHOLD


# if __name__ == "__main__":
#     table_id = "2019-March-return__3f436d14-4e17-476c-a3e4-66d18e7f6c90"
#     operations = ["JC"]
#     path_matches = Path(MATCHES_JSON)
#     path_dataset = Path(DATASET_DIR)
#
#     with open(path_matches) as file:
#         matches = json.load(file)
#
#     G = build_matches_graph(matches, path_dataset)  # FIXED: path -> path_matches
#
#     matching_subgraph = fetch_matches(
#         G,
#         node_id=table_id,
#         predicate=overlap_ratio_predicate,
#         edge_labels=operations,  # ty: ignore
#     )
#
#     print(matching_subgraph.edges())
