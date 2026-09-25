import os
import sys
from pathlib import Path

sys.path.append("..")

import matplotlib.pyplot as plt
import networkx as nx

from orqa.embedding_discovery.pipeline import get_seed_datasets
from conf import load_config

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_large_multidigraph(
    G: nx.MultiDiGraph, seeds: list, sinks: list, graph_png_path: Path
):
    # 1. Increase figure size for clarity
    plt.figure(figsize=(16, 10))

    # 2. Connectivity Specs
    # Weakly: Undirected version is connected
    # Strongly: Every node is reachable from every other node
    is_weak = nx.is_weakly_connected(G)
    is_strong = nx.is_strongly_connected(G)
    conn_text = f"Weakly Conn: {is_weak}\nStrongly Conn: {is_strong}"

    # 3. Define Node Colors
    node_colors = []
    for node in G.nodes():
        if node in seeds:
            node_colors.append("orange")
        elif node in sinks:
            node_colors.append("green")
        else:
            node_colors.append("skyblue")

    # 4. Layout
    # Use 'k' parameter in spring_layout to increase distance between nodes
    pos = nx.spring_layout(G, k=0.15, seed=42)

    # 5. Draw (Labels omitted)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=120, alpha=0.9)

    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="->",
        arrowsize=10,
        connectionstyle="arc3,rad=0.1",  # Crucial for MultiDiGraph overlaps
        edge_color="gray",
        alpha=0.3,
        width=0.5,
    )

    # 6. Legend and Info Box
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Seeds (Orange)",
            markerfacecolor="orange",
            markersize=12,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Sinks (Green)",
            markerfacecolor="green",
            markersize=12,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Other (Blue)",
            markerfacecolor="skyblue",
            markersize=12,
        ),
        Line2D([0], [0], marker="", color="w", label="---"),
        Line2D([0], [0], marker="", color="w", label=conn_text),
    ]

    plt.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Graph Specifications",
        fontsize=10,
    )

    plt.title(f"MultiDiGraph Visualization ({G.number_of_nodes()} nodes)", fontsize=15)
    plt.tight_layout()
    print(f"Saving plots to {graph_png_path}...")
    plt.savefig(graph_png_path)
    print("Done.")


def modena():
    modena_yaml_path = Path(
        os.path.dirname(__file__), "..", "..", "conf", "workflow", "modena.yaml"
    )
    data_path = Path(os.environ["DATADIR"], "open_data", "ckan", "modena")

    cfg = load_config(modena_yaml_path, data_path)

    seeds = get_seed_datasets(cfg)
    G = nx.read_gml(cfg.candidates_discovery.matches_graph_path)

    sinks = [node for node, degree in G.out_degree() if degree == 0]

    graph_png_path = Path("..", "..", "plots", "modena_matches.png")
    plot_large_multidigraph(G, seeds, sinks, graph_png_path)


if __name__ == "__main__":
    modena()
