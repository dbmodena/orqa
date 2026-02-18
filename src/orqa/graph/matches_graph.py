import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import networkx as nx
from tqdm import tqdm

from orqa.schema_matching.valentine_matcher import instantiate_matcher, schema_matching
from orqa.sloth import sloth
from orqa.utils import pl_read_dataset

DOCUMENT_TYPE = "csv"
THRESHOLD = 0.5
MAX_WORKERS = 10
OVERLAP_RATIO_THRESHOLD = 0.5
ROUND = 5


def overlap_ratio_only_predicate(edge_data: dict, overlap_threshold: float) -> bool:
    return edge_data["overlap_ratio"] >= overlap_threshold


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

    left_area = len(left_table) * len(left_table[0])
    right_area = len(right_table) * len(right_table[0])

    min_area = min(left_area, right_area)
    overlap_ratio = overlap_area / min_area * 100

    return {
        "overlap_area": overlap_area,
        "overlap_ratio": round(overlap_ratio, ROUND),
        "num_columns_involved": num_columns_involved,
        "num_rows_overlapping": num_rows_overlapping,
    }


def process_edge(
    entry: dict,
    entry_idx: int,
    datasets_folder: Path,
    read_opts: dict = {},
    matcher_name: str = "coma",
    matcher_kwargs: Optional[dict] = None,
    verbose: bool = False,
) -> tuple[int, str, str, str, list, dict, list] | None:
    """Helper function to process a single edge in parallel"""
    q_node = entry["Q"]
    r_node = entry["R"]
    task = entry["task"]

    metrics = {}

    if q_node == r_node:
        return None

    Q = pl_read_dataset(datasets_folder / f"{q_node}.csv", read_opts)
    R = pl_read_dataset(datasets_folder / f"{r_node}.csv", read_opts)

    q_columns = r_columns = None
    q_key = r_key = None
    q_target = r_target = None

    match task:
        case "U":
            q_columns = entry["q_columns"]
            r_columns = []
        case "J" | "MJ":
            q_columns = entry["q_columns"]
            r_columns = entry["r_columns"]

            r_columns = [R.columns[idx] for idx in r_columns]
        case "JC":
            q_key = entry["q_key"]
            q_target = entry["q_target"]

            r_key = R.columns[entry["r_key"]]
            r_target = R.columns[entry["r_target"]]

            q_columns = [q_key]
            r_columns = [r_key]
        case _:
            raise ValueError(f"Unknown task: {task}")

    try:
        # Prepare datasets for SLOTH
        q_columns_values = [Q.get_column(col).to_list() for col in q_columns]
        r_columns_values = [
            R.get_column(col).to_list()
            for col in (r_columns if r_columns else R.columns)
        ]

        # we force an overlap with at least #(left_table_involved_columns) width
        overlap_t = time.time()
        metrics = compute_overlap_metrics(
            q_columns_values,
            r_columns_values,
            min_width=len(q_columns),
            verbose=verbose,
        )
        overlap_t = time.time() - overlap_t
    except Exception as e:
        print(f"Error while processing edge with SLOTH: {e}")
        metrics["overlap_ratio"] = None
        metrics["overlap_time"] = None
    else:
        metrics["overlap_time"] = round(overlap_t, ROUND)

    try:
        # Prepare datasets for Schema Matching
        Q = Q.to_pandas()
        R = R.to_pandas()

        matcher_kwargs = {} if matcher_kwargs is None else matcher_kwargs
        matcher = instantiate_matcher(matcher_name, **matcher_kwargs)

        match_t = time.time()
        matches, macro_avg, micro_avg = schema_matching(
            matcher,
            task,
            Q,
            R,
            q_columns,
            r_columns,
            q_key,
            r_key,
        )
        match_t = time.time() - match_t
    except Exception as e:
        print(f"Error wile processing edge with Schema Matcher {matcher_name}: {e}")
        metrics["sm_macro_avg"] = None
        metrics["sm_micro_avg"] = None
        metrics["sm_n_matches"] = None
        metrics["sm_time"] = None
    else:
        metrics["sm_macro_avg"] = round(macro_avg, ROUND)
        metrics["sm_micro_avg"] = round(micro_avg, ROUND)
        metrics["sm_n_matches"] = len(matches)
        metrics["sm_time"] = round(match_t, ROUND)

    if task == "JC":
        r_columns = [r_key, r_target]

    return (
        entry_idx,
        q_node,
        r_node,
        task,
        r_columns,
        metrics,
        [(c1, c2, round(s, ROUND)) for (c1, c2), s in matches.items()],
    )


def overlap_ratio_predicate(node: dict) -> bool:
    return node["overlap_ratio"] >= OVERLAP_RATIO_THRESHOLD


class DatasetMatchesGraph:
    def __init__(self):
        self._G = nx.MultiDiGraph()

    def add(
        self,
        blend_matches: list[dict],
        datasets_folder: Path,
        read_opts: dict,
        matcher_name: str,
        matcher_kwargs: Optional[dict] = None,
        max_workers: Optional[int] = None,
        verbose: bool = False,
    ):
        for entry in blend_matches:
            self._G.add_node(entry["Q"])
            self._G.add_node(entry["R"])

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    process_edge,
                    entry,
                    idx,
                    datasets_folder,
                    read_opts,
                    matcher_name,
                    matcher_kwargs,
                    verbose,
                )
                for idx, entry in enumerate(blend_matches)
            }

            for future in tqdm(
                as_completed(futures),
                desc="Adding edges to the graph",
                total=len(blend_matches),
            ):
                try:
                    result = future.result(60)
                    if result:
                        entry_idx, q_node, r_node, task, r_columns, metrics, matches = (
                            result
                        )

                        if task != "JC":
                            blend_matches[entry_idx]["r_columns"] = r_columns
                        else:
                            blend_matches[entry_idx]["r_key"] = r_columns[0]
                            blend_matches[entry_idx]["r_target"] = r_columns[1]

                        blend_matches[entry_idx]["matches"] = matches
                        blend_matches[entry_idx]["metrics"] = metrics
                        self._G.add_edge(q_node, r_node, task=task, **metrics)
                except Exception as e:
                    print(f"Error within main process: {e}")

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
                nx.to_undirected(sub_graph),
                n_paths_to_generate,
                max_path_length,
                weight=weight,
                seed=seed,
                source=dataset_id,
            ):
                random_walks.append(
                    {
                        # "Q": dataset_id,
                        "operation_type": edge_labels,
                        "datasets": random_walk,  # this should be a list
                    }
                )

        return random_walks

    def load(self, path: Path):
        self._G = nx.read_gml(path)

    def save(self, path: Path):
        nx.write_gml(self._G, path)
