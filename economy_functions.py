import numpy as np
import networkx as nx
from scipy.sparse import csr_array


def ownership_matrix(n,edges):
    '''
    Create a matrix of ownerships
    '''
    edges = np.array(edges)
    ownerships = np.random.rand(edges.shape[0])
    A = csr_array((ownerships,(edges[:,0], edges[:,1])), shape = (n,n))
    
    return A.toarray()

if __name__== "__main__":
    G = nx.gnm_random_graph(5, 3, directed = True)
    A = ownership_matrix(G.number_of_nodes(), G.edges())
    print(A)
    for u, v in G.edges():
        G[u][v]['ownership'] = A[u,v]

    print(G.edges(data=True))


