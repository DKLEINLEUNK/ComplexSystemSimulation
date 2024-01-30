import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
import numpy as np
from network import Network
from network_modifier import create_shock, get_weak_nodes, threshold_test, propagate_shock, degrade, fail, save_state, load_state




def update_node_status(iteration):
    # Update node status based on your logic
    ### EXAMPLE USAGE ###

    # Creating a network
    global graph
    ax.clear()
    
    propagate_shock(network, 0.6, 0.8)

    # Draw the updated graph with node colors based on status
    color_mapping = {0: 'red', 1: 'yellow', 2: 'green'}

    node_statuses = nx.get_node_attributes(graph, 'status')

    node_colors = [color_mapping[node_statuses[node]] for node in graph.nodes]
    
    nodes = nx.draw_networkx_nodes(graph, pos=pos, node_color=node_colors, node_size = 8)
    edges = nx.draw_networkx_edges(graph, pos=pos, width = 0.1, edge_color='black', arrowsize = 3)

    # Return the elements that are being animated
    ax.set_title(f'Iteration: {iteration}')

    
   

if __name__ == "__main__":

    ## Network creation
    network = Network(n=50, p= 0.3)
    graph = network.graph
    network.set_all_statuses(2)
    network.set_all_edges()
    create_shock(network, 3)

    ## Network plot
    pos = nx.spring_layout(graph)
    fig, ax = plt.subplots()
    animation = FuncAnimation(fig, update_node_status, frames=100, interval=1000, blit = False)
    plt.show()