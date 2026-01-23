from typing import Callable, Literal, Optional

import networkx as nx


def expand_one_hop(
    g: nx.MultiDiGraph,
    source_nodes: set[str],
    edge_labels: list[Literal["J", "U", "JC"]],
    predicate: Callable,
    selected_edges,
    visited: set[str],
) -> set:
    next_hop = set()
    for src in source_nodes:
        for u, v, data in g.edges(src, data=True):
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
    G: nx.MultiDiGraph,
    node_id: str,
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
        return G

    if node_id not in G:
        raise ValueError(f"Node '{node_id}' not found in graph")

    selected_edges = set()
    visited = {node_id}
    source_nodes = {node_id}

    while num_hops > 0:
        # -------- FIRST HOP --------
        first_hop = expand_one_hop(
            G,
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
    sub_g = G.edge_subgraph(selected_edges).copy()
    sub_g.add_node(node_id)  # keep isolated root if needed

    return sub_g
