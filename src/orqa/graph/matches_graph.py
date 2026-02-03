from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import networkx as nx
import polars as pl
from tqdm import tqdm

from orqa.sloth import sloth
from orqa.utils import pl_read_dataset, remove_null_columns, remove_null_rows

DOCUMENT_TYPE = "csv"
THRESHOLD = 0.5
MAX_WORKERS = 8
OVERLAP_RATIO_THRESHOLD = 0.5


def overlap_ratio_only_predicate(edge_data: dict, overlap_threshold: float) -> bool:
    return edge_data["overlap_ratio"] >= overlap_threshold


def load_dataset_as_list_of_columns(
    path: Path, columns: list, opts: dict = {}
) -> list[list[Any]]:
    df = pl_read_dataset(path, opts)
    if columns != []:
        if isinstance(columns[0], int):
            columns = [pl.nth(i) for i in columns]
        df = df.select(*columns)
    df = remove_null_columns(df)
    df = remove_null_rows(df)

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
    overlap_area = metrics[-2]  # Area from metrics
    num_rows_overlapping = metrics[-3]  # Height from metrics
    num_columns_involved = metrics[-4]  # Width from metrics

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
    left_table: list[list[Any]],
    right_table: list[list[Any]],
    min_width: Optional[int] = None,
    verbose: bool = False,
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
    entry: dict, datasets_folder: Path, polars_opts: dict = {}, verbose: bool = False
) -> tuple[str, str, str, dict] | None:
    """Helper function to process a single edge in parallel"""
    q_node = entry["Q"]
    r_node = entry["R"]
    task_label = entry["task"]

    if q_node == r_node:
        return None

    match task_label:
        case "U":
            left_columns = entry["q_columns"]
            right_columns = []
        case "J":
            left_columns = entry["q_join_keys"]
            right_columns = entry["r_join_keys"]
        case "JC":
            left_columns = [entry["q_key"], entry["q_target"]]
            right_columns = [entry["r_key"], entry["r_target"]]

    try:
        left_path = datasets_folder.joinpath(f"{q_node}.{DOCUMENT_TYPE}")
        right_path = datasets_folder.joinpath(f"{r_node}.{DOCUMENT_TYPE}")

        left_table = load_dataset_as_list_of_columns(
            left_path, left_columns, polars_opts
        )
        right_table = load_dataset_as_list_of_columns(
            right_path, right_columns, polars_opts
        )

        # we force an overlap with at least #(left_table_involved_columns) width
        overlap_metrics = compute_overlap_metrics(
            left_table, right_table, min_width=len(left_columns), verbose=verbose
        )
        return (q_node, r_node, task_label, overlap_metrics)
    except Exception as e:
        print(e)
        raise e
        return None


def overlap_ratio_predicate(node: dict) -> bool:
    return node["overlap_ratio"] >= OVERLAP_RATIO_THRESHOLD


class DatasetMatchesGraph:
    def __init__(self):
        self._G = nx.MultiDiGraph()

    def add(
        self,
        matches: list[dict],
        datasets_folder: Path,
        opts: dict,
        max_workers: Optional[int] = None,
        verbose: bool = False,
    ):
        try:
            # Add all nodes first
            for entry in matches:
                # print(f"Analyzing {entry}")
                self._G.add_node(entry["Q"])
                self._G.add_node(entry["R"])

            # Process edges in parallel
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(
                        process_edge,
                        entry,
                        datasets_folder,
                        opts,
                        verbose,
                    ): entry
                    for entry in matches
                }

                # Collect results as they complete
                for future in tqdm(
                    as_completed(futures),
                    desc="Adding edges to the graph",
                    total=len(matches),
                ):
                    result = future.result()
                    if result:
                        q_node, r_node, task_label, metrics = result
                        self._G.add_edge(q_node, r_node, label=task_label, **metrics)

        except Exception as e:
            print(e)

    def expand_one_hop(
        self,
        source_nodes: set[str],
        edge_labels: list[Literal["J", "U", "JC"]],
        predicate: Callable,
        selected_edges,
        visited: set[str],
    ) -> set:
        next_hop = set()
        for src in source_nodes:
            for u, v, data in self._G.edges(src, data=True):
                # with data=True, g.edges return a tuple with
                # three values, and the third is the data-dict of the node
                if data.get("label") in edge_labels and predicate(data):
                    next_node = v if u == src else u

                    # prevent going backwards or revisiting
                    if next_node in visited:
                        continue

                    selected_edges.add((u, v))
                    next_hop.add(next_node)

        return next_hop

    def fetch_matches(
        self,
        node: str,
        predicate: Optional[Callable],
        edge_labels: list[Literal["J", "U", "JC"]],
        num_hops: int = 1,
    ) -> nx.Graph:
        """
        Explore the neighbors of a given node to fetch potentially
        interesting chains of operations.

        :param g: A Graph representing relationships among datasets.
        :param node_id: The node from where to start the search.
        :param edge_label: The kind of relationships we're looking for:
            "Join-", "Union-", "Join-Correlation-" operations are possible.
        :param predicate: A boolean predicate that will to be applied on a node
            within the search to check if this has to be considered or not as
            a valid and relevant match.
        :param num_hops: Number of hops to perform before stopping the search.
        :raises ValueError: If the input node is not found in the graph.
        """
        if not predicate:
            return self._G

        if node not in self._G:
            raise ValueError(f"Node '{node}' not found in graph")

        selected_edges = set()
        visited = {node}
        source_nodes = {node}

        while num_hops > 0:
            # -------- FIRST HOP --------
            first_hop = self.expand_one_hop(
                source_nodes=source_nodes,
                edge_labels=edge_labels,
                predicate=predicate,
                selected_edges=selected_edges,
                visited=visited,
            )
            visited |= first_hop
            source_nodes = first_hop
            num_hops -= 1

        # -------- BUILD SUBGRAPH (EDGE-DRIVEN) --------
        sub_g = self._G.edge_subgraph(selected_edges).copy()
        sub_g.add_node(node)  # keep isolated root if needed

        return sub_g

    def generate_random_walks(
        self,
        dataset_id: str,
        n_paths_to_generate: int,
        max_path_length: int,
        overlap_ratio_threshold: Optional[float],
        seed: int,
    ) -> list:
        _overlap_ratio_only_predicate = partial(
            overlap_ratio_only_predicate,
            overlap_threshold=overlap_ratio_threshold,
        )

        random_walks = []

        weight = "overlap_ratio" if overlap_ratio_threshold else None
        predicate = _overlap_ratio_only_predicate if overlap_ratio_threshold else None

        # FIX: Here we do not check whether a Join column is considered
        # also in any other Join/Join-Correlation candidate match during
        # search.
        for edge_labels in [["U"], ["J", "JC"]]:
            sub_graph = self.fetch_matches(
                dataset_id,
                predicate,
                edge_labels,  # ty: ignore
                max_path_length,
            )

            for random_walk in nx.generate_random_paths(
                sub_graph,
                n_paths_to_generate,
                max_path_length,
                weight=weight,
                seed=seed,
                source=dataset_id,
            ):
                random_walks.append(
                    {
                        "Q": dataset_id,
                        "operation_type": edge_labels,
                        "datasets": random_walk,  # this should be a list
                    }
                )

        return random_walks

    def load(self, path: Path):
        self._G = nx.read_gml(path)

    def save(self, path: Path):
        nx.write_gml(self._G, path)
