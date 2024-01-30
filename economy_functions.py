"""
This file contains functions which are not called in the terminal, support functions
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
    '''
    Create a matrix of ownerships and sets the edges values following a power law distribution
    '''
    edges = np.array(graph.edges())
    weights = np.bincount(edges[:,0])
    power_law_ownerships = np.random.power(exponent, graph.number_of_edges())
    
    sort_indices = np.array([custom_sort(edge, weights) for edge in edges])
    edges = edges[np.argsort(sort_indices)]
    power_law_ownerships = np.sort(power_law_ownerships)[::-1]

    A = np.zeros((graph.number_of_nodes(),graph.number_of_nodes()))
    
    for i,edge in enumerate(edges):
        A[edge[0],edge[1]] = power_law_ownerships[i]
    
    sum_owns = np.sum(A, axis=0)
    mask = sum_owns > 1
    A[:, mask] /= sum_owns[mask]
    return A




if __name__ == "__main__":
    ### Graph generation
    n = 10_000
    graph = nx.gnm_random_graph(n, n*0.3, seed = 100)
    graph = graph.to_directed()
    ## Mode

    A = ownership_matrix(graph, 0.2)

    edges = np.array(graph.edges())
    weights = np.bincount(edges[:,0])
    sort_indices = np.array([custom_sort(edge, weights) for edge in edges])
    edges = edges[np.argsort(sort_indices)]
    values = np.array([A[i, j] for i, j in edges])
    
    plt.scatter(np.arange(len(values)), values, c='blue', marker='o', s = 1)


    # Add labels and a title
    plt.xlabel('edge [i, j]')
    plt.ylabel('Value')
    plt.title('Sparse Matrix Values vs Positions')
    plt.show()

        #print(edges)

