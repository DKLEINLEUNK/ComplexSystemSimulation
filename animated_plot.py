import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
import numpy as np
from network import Network
from network_modifier import (
    create_shock,
    get_weak_nodes,
    threshold_test,
    propagate_shock,
    degrade,
    fail,
    save_state,
    load_state,
)


ANIMATION = True  # True = animation ove time, False = just plot initial state
EDGE_LABELS = False  # True fi want to see the edge labels
NODE_SIZE = 100  # Node size


def update_node_status(iteration):
    """
    For each iterations, it propagates a shock and plots teh results
    """

    global graph
    global nodes
    global color_mapping
    nodes.remove()

    propagate_shock(network, 0.6, 0.8)

    node_statuses = nx.get_node_attributes(graph, "status")

    node_colors = [color_mapping[node_statuses[node]] for node in graph.nodes]

    nodes = nx.draw_networkx_nodes(
        graph, pos=pos, node_color=node_colors, node_size=NODE_SIZE
    )

    ax.set_title(f"Cycle: {iteration}")


if __name__ == "__main__":
    ## Network creation
    network = Network(n=50, p=0.3)
    graph = network.graph
    network.set_all_statuses(2)
    network.set_all_edges()
    create_shock(network, 10)

    ## Network plot
    pos = nx.spring_layout(graph)
    fig, ax = plt.subplots()

    ## Color setter
    color_mapping = {0: "red", 1: "yellow", 2: "green"}

    ## PLots
    node_statuses = nx.get_node_attributes(graph, "status")
    node_colors = [color_mapping[node_statuses[node]] for node in graph.nodes]
    nodes = nx.draw_networkx_nodes(
        graph, pos=pos, node_color=node_colors, node_size=NODE_SIZE
    )

    edges = nx.draw_networkx_edges(
        graph, pos=pos, width=0.1, edge_color="black", arrowsize=3
    )

    if EDGE_LABELS == True:
        edge_labels = {
            (u, v): f'{d["ownership"]:.2f}' for u, v, d in graph.edges(data=True)
        }
        print(edge_labels)
        edge_label = nx.draw_networkx_edge_labels(
            graph, pos=pos, edge_labels=edge_labels, font_size=8
        )

    if ANIMATION == True:
        animation = FuncAnimation(
            fig, update_node_status, frames=10, interval=1000, blit=False
        )

    plt.show()
