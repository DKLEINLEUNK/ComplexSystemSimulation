"""
This file contains functions which do not use network as input
"""

import numpy as  np
import matplotlib.pyplot as plt
import networkx as nx


def custom_sort(edge,weights):
    """
    Sort the edges by connection, edges of a high node to a small node are put first, while edges of a small node and high node are last
    """

    return weights[edge[1]] - weights[edge[0]]


def ownership_matrix(graph,exponent):
    """
    Create a matrix of ownerships and sets the edges values following a power law distribution
    """
    edges = np.array(graph.edges())
    weights = np.bincount(edges[:,0])
    power_law_ownerships = np.random.power(exponent, graph.number_of_edges())
    
    sort_indices = np.array([custom_sort(edge, weights) for edge in edges])
    edges = edges[np.argsort(sort_indices)]
    power_law_ownerships = np.sort(power_law_ownerships)[::-1]

    A = np.zeros((graph.number_of_nodes(),graph.number_of_nodes()))
    
    A[edges[:, 0], edges[:, 1]] = power_law_ownerships
    
    sum_owns = np.sum(A, axis=0)
    mask = sum_owns > 1
    A[:, mask] /= sum_owns[mask]
    return A



if __name__ == "__main__":
    ### Graph generation of Matrix A ownerships
    n = 1000
    graph = nx.gnm_random_graph(n, n*0.3, seed = 100)
    graph = graph.to_directed()

    A = ownership_matrix(graph, 0.55)

    edges = np.array(graph.edges())
    weights = np.bincount(edges[:,0])
    sort_indices = np.array([custom_sort(edge, weights) for edge in edges])
    edges = edges[np.argsort(sort_indices)]
    values = np.array([A[i, j] for i, j in edges])
    fraction = np.round(len(values[values>0.5])/len(values[values<0.5]),2)
    
    plt.scatter(np.arange(len(values)), values, c='blue', marker='o', s = 1)

    plt.ylabel('Value')
    plt.title(f'Sparse Matrix Ownerships distribution. {fraction*100}% of ownerships above 50% for the more connected companies')
    plt.show()



