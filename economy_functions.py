"""
This file contains functions which are not called in the terminal, support functions
"""

import numpy as  np
import networkx as nx
from scipy.sparse import csr_array

def ownership_matrix(n,edges):
    '''
    Create a matrix of ownerships
    '''
    edges = np.array(edges)
    ownerships = np.random.rand(edges.shape[0])
    A = csr_array((ownerships,(edges[:,0], edges[:,1])), shape = (n,n))
    A = A.toarray()
    #A = A/np.sum(A)
    return A


