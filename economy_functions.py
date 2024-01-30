"""
This file contains functions which are not called in the terminal, support functions
"""

import numpy as  np
import networkx as nx


def custom_sort(edge,weights):
    """
    Sort the edges by connection, edges of a high node to a small node are put first, while edges of a small node and high node are last
    """
    diff = weights[edge[1]] - weights[edge[0]]
    if diff>0:
        return 1/diff
    elif diff == 0:
        return  1
    else:
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
    
    
    sum_ =np.sum(A,axis = 0)
    
    for j in range(A.shape[0]):
        if sum_[j]>1:
            A[:,j] = A[:,j]/sum_[j]
    return A




if __name__ == "__main__":
    ### Graph generation
    n = 10
    graph = nx.gnm_random_graph(n, 4, seed = 100)
    graph = graph.to_directed()
    ## Mode
    A = ownership_matrix(graph, 0.2)


    #print(edges)

