"""
This module contains the Network class, which represents a network of nodes.

Running this module as a script will generate an example network.
"""


import networkx as nx
import numpy as np
from scipy.sparse import csr_array
from network_modifier import *


LIMIT_FAIL = 0.3  # Company fails if 30% of its EPS drops
LOSS_IF_INFECTED = 0.85



class Network:

    """
    A class to represent a network of nodes.

    Attributes
    ----------
    graph : networkx.Graph
            The network graph


    Methods
    -------
    set_status(node, status)
            Set the status of a node
    get_status(node)
            Get the status of a node
    get_all_statuses()
            Get the statuses of all nodes
    """

    def __init__(self, n, m=None, p=None, _lambda=None):
        """
        Description
        -----------
        Initializes a network with `n` nodes.
        Either `m` edges or `p` probability of an edge should be provided.

        Parameters
        ----------
        `n` : int
        Number of nodes
        `m` : int
        Number of edges
        `p` : float
        Probability of an edge
        """
        if m is not None:
            self.graph = nx.gnm_random_graph(n, m, seed= 100, directed=True)
        elif p is not None:
            self.graph = nx.erdos_renyi_graph(n, p, directed=True)

        else:
            raise ValueError("Either m or p must be provided.")

        _lambda = _lambda or 1.5
        self.eps = np.random.exponential(_lambda, n)
        self.eps_ini = self.eps
        self.A = ownership_matrix(self.graph.number_of_nodes(), self.graph.edges())

    def set_edge(self,edge):
        return None

    def set_edges(self,edges):
        return None

    def set_all_edges(self):
        """
        Sets all edges weights
        """

        for u, v in G.edges():
            graph[u][v]["ownership"] = self.A[u, v]
        return None

    def set_status(self, node, status):
        ### SET AT 0 and 1 status, and add counter EPS. MEYBE WE CLEANING CODE SET IT AS CHILD CLASS
        if status in [0, 1, 2]:
            nx.set_node_attributes(self.graph, {node: status}, "status")
        else:
            raise ValueError("Status must be 0, 1, or 2.")

    def create_shock(self, size):
        """
        Creates the default shock into the system, by failing n businesses

        Parameters
        ----------
            size: Number of companies that fail
        """
        for i in range(size):
            node = np.random.randint(0, len(self.graph.nodes))
            self.set_status(node, 1)

    def propogate_shock(self):
        failed_nodes = list(self.get_all_statuses().keys())
        neighbours = set(
            sum(
                self.get_multiple_neighbors(failed_nodes, as_list=True), []
            )  # Gets all neighbours, combines into single set
        )

        # Calculate change in EPS here, dependent on A
        delta_eps = self.eps * LOSS_IF_INFECTED * select_values(failed_nodes)
        init_eps = self.eps.copy()
        self.eps -= delta_eps

        # For 90 days
        for i in range(90):
            delta_eps = delta_eps @ 
            self.eps = self.eps - delta_eps
            print((init_eps - self.eps) / init_eps < 1 - LIMIT_FAIL)
            print(f"i: {i}, EPS: {self.eps}")

    def set_statuses(self, nodes, statuses):
        """
        Sets the statuses of an array of nodes.
        """
        if len(nodes) != len(statuses):
            raise ValueError("Nodes and statuses must be the same length.")
        for node, status in zip(nodes, statuses):
            self.set_status(node, status)

    def set_all_statuses(self, status):
        """
        Sets the statuses of all nodes.
        """
        nx.set_node_attributes(self.graph, status, "status")

    def get_status(self, node):
        """
        Returns the status of a node.
        """
        return nx.get_node_attributes(self.graph, "status")[node]

    def get_statuses(self, nodes):
        """
        Returns the statuses of an array of nodes.
        """
        return np.array(
            [self.get_status(node) for node in nodes]
        )  # TODO check if this is a bottleneck

    def get_all_statuses(self):
        """
        Returns the statuses of all nodes.
        """
        return nx.get_node_attributes(self.graph, "status")

    def get_neighbors(self, node, as_list=False):
        """
        Returns the neighbors of a node.
        """
        return (
            self.graph.neighbors(node)
            if not as_list
            else list(self.graph.neighbors(node))
        )

    def get_multiple_neighbors(self, nodes, as_list=False):
        """
        Returns the neighbors of an array of nodes.
        """
        # TODO see if this is a bottleneck
        neighbors = []
        for node in nodes:
            neighbors.append(
                self.get_neighbors(node)
            ) if not as_list else neighbors.append(list(self.graph.neighbors(node)))
        return neighbors


##### We could use two status, EPS and Fail or not. Send array with Statuses, and recive array. I will send you matrix A.

if __name__ == "__main__":
    ### EXAMPLE USAGE ###

    # Creating a network
    network = Network(n=1_0000, m=4000)
    create_shock(network,300)
    propagate_shock(network,0.85)

