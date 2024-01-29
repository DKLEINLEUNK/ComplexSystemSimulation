"""
This file contains functions which are not called in the terminal, support functions
"""

import numpy as  np
from scipy.stats import beta
import networkx as nx

def ownership_matrix(n,edges):
    '''
    Create a matrix of ownerships
    '''
    A = np.zeros((n,n))
    edges = np.array(edges)
    weights_a = np.bincount(edges[:,0])/(np.sum(np.bincount(edges[:,0]))) ##How close to 1, wight of itself
    weights_b = np.bincount(edges[:,1])/np.sum(np.bincount(edges[:,1])) ## How close to 0, weight of other, must be average left tail, so, by x multiply

    for i,v in edges:
        A[i,v] = (np.random.beta(weights_a[i], 2*weights_b[v]))
    return A


if __name__ == "__main__":
    n = 5
    graph = nx.gnm_random_graph(n, 2)
    graph = graph.to_directed()
    graph.add_edges_from(graph.reverse().edges())
    edges = graph.edges()
    print(edges)
    A = ownership_matrix(n,edges)
    print(A)
    #print(counts)

    #A = ownership_matrix(n,graph.edges())

