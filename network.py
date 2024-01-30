"""
This module contains the Network class, which represents a network of nodes.

Running this module as a script will generate an example network.
"""


import networkx as nx
import numpy as np
from network_modifier import create_shock, get_weak_nodes, threshold_test, propagate_shock, degrade, fail, save_state, load_state
from economy_functions import ownership_matrix



LIMIT_FAIL = 0.8  # Company fails if 30% of its EPS drops
LOSS_IF_INFECTED = 0.6
SECTOR_MPE = np.array(
    [7, 9, 10, 12, 14, 15, 16, 17, 18, 19, 22, 31]
)  # Median MPE per sector

POWER_LAW_OWNS = 0.2 ## Need to improve with real data, or maybe we can research it's effect


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
            self.graph = nx.gnm_random_graph(n, m, seed=100)
        elif p is not None:
            self.graph = nx.erdos_renyi_graph(
                n,
                p,
                seed=100,
            )

        else:
            raise ValueError("Either m or p must be provided.")

        ## Turning graph to directed
        self.graph = self.graph.to_directed()
        edges = self.graph.edges()

        _lambda = _lambda or 1.5
        self.eps = np.random.exponential(_lambda, n)
        self.eps_ini = self.eps

        self.sector = np.random.randint(0, 12, n)
        self.mpe = SECTOR_MPE[self.sector]  # Now uses the PE of a sector
        self.mpe_ini = self.mpe

        self.pi = self.mpe * self.eps
        self.pi_ini = self.pi

        self.A = ownership_matrix(self.graph,POWER_LAW_OWNS)

    def set_edge(self, edge):
        return None

    def set_edges(self, edges):
        return None

    def set_all_edges(self):
        """
        Sets all edges weights:

        This function is only for cool graphs, the code can run without it since the info is stored in matrix A
        """
        for u, v in self.graph.edges():
            self.graph[u][v]["ownership"] = self.A[u, v]
        return None

    def set_status(self, node, status):
        ### SET AT 0 and 1 status, and add counter EPS. MEYBE WE CLEANING CODE SET IT AS CHILD CLASS
        if status in [0, 1, 2]:
            nx.set_node_attributes(self.graph, {node: status}, "status")
        else:
            raise ValueError("Status must be 0, 1, or 2.")

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
    network = Network(n=1000, p=0.2)
    network.set_all_statuses(2)

    network.set_all_edges()
    print("i")
    create_shock(network, 10)
    for i in range(10):
        propagate_shock(network, LOSS_IF_INFECTED, LIMIT_FAIL)
        total_failures = len(
            list(
                filter(
                    lambda item: item[1] in {0, 1}, network.get_all_statuses().items()
                )
            )
        )
        print(
            f"Total fraction of failures: {total_failures/network.graph.number_of_nodes()}"
        )
