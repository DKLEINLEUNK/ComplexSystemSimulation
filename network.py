"""
This module contains the Network class, which represents a network of nodes.

Running this module as a script will generate an example network and make simulations (see __name__ == "__main__"")
"""

import multiprocessing
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


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
from economy_functions import ownership_matrix
from tqdm import tqdm, trange


LIMIT_FAIL = 0.3  
LOSS_IF_INFECTED = 0.85 
USE_REAL_DATA = False
POWER_LAW_OWNS = (
    0.55
)

if USE_REAL_DATA == True:
    path = "data\companies_data.csv"
    data = pd.read_csv(path, delimiter=";", decimal=",")

    EPS = np.array(data["EPS(rial)"]).astype(float)
    NETWORK_SIZE = len(EPS)
    SECTOR_MPE = data["Group P/E"].unique().astype(float)

else:
    NETWORK_SIZE = 100
    SECTOR_MPE = np.array(
        [7, 9, 10, 12, 14, 15, 16, 17, 18, 19, 22, 31]
    )  # Median MPE per sector


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
            self.graph = nx.barabasi_albert_graph(n,m)
        elif p is not None:
            self.graph = nx.erdos_renyi_graph(
                n,
                p,
            )
        else:
            raise ValueError("Either m or p must be provided.")

        ## Turning graph to directed
        self.graph = self.graph.to_directed()
        edges = self.graph.edges()

        _lambda = _lambda or 1.5
        if USE_REAL_DATA == True:
            self.eps = EPS
        else:
            self.eps = _lambda = _lambda or 1.5
            self.eps = np.random.exponential(_lambda, n)
        
        self.eps_ini = self.eps
        self.eps_ini_o = self.eps
        self.sector = np.random.randint(0, 12, n)
        self.mpe = SECTOR_MPE[self.sector]  
        self.mpe_ini = self.mpe

        self.pi = self.mpe * self.eps
        self.pi_ini = self.pi

        self.A = ownership_matrix(self.graph, POWER_LAW_OWNS)


    def set_all_edges(self):
        """
        Sets all edges weights/ownerships.
        Note:
            - The code can rune without this function, edges and ownerships are stored and edited in Matrix A. This is just for the plots

        """
        for u, v in self.graph.edges():
            self.graph[u][v]["ownership"] = self.A[u, v]
        return None

    def set_status(self, node, status):
        """
        Sets status of a node to 0,1,2
        """
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
        neighbors = []
        for node in nodes:
            neighbors.append(
                self.get_neighbors(node)
            ) if not as_list else neighbors.append(list(self.graph.neighbors(node)))
        return neighbors


def single_simulation(args):
    """Generates a single iteration of the network simulation"""
    loss_if_infected = args[0]
    limit_fail = args[1]
    shock_size = args[2]
    p = args[3]
    recovery_rate = args[4]
    network = Network(n=NETWORK_SIZE, p=p)
    network.set_all_statuses(2)
    network.set_all_edges()
    create_shock(network, shock_size)
    for i in range(10):
        propagate_shock(network, loss_if_infected, limit_fail, recovery_rate)
        total_failures = len(
            list(
                filter(
                    lambda item: item[1] in {0, 1},
                    network.get_all_statuses().items(),
                )
            )
        )
        fraction_failure = total_failures / network.graph.number_of_nodes()
        print(f"Total fraction of failures: {fraction_failure}")
        return fraction_failure


def simulate_failures(
    simulation_size,
    loss_if_infected,
    limit_fail,
    store_hist=False,
    shock_size=10,
    p=0.1,
    recovery_rate=0.1,
    change = "not_specified", 
    file_number = 101
):
    """
    Generate a simulation of the entire netwrok:
        - Simulation size = Number of simulations with specified conditions
        - store_hist =  True, if you want to store the data
    """


    pool = multiprocessing.Pool()
    simulation_results = pool.map(
        single_simulation,
        [
            (loss_if_infected, limit_fail, shock_size, p, recovery_rate)
            for i in range(simulation_size)
        ],
    )
    pool.close()

    fraction_failure_results = np.array(list(simulation_results))
    print(f"Average failure rate: {fraction_failure_results.mean()}")

    if store_hist:
        np.save(
            f"data/{change}{file_number}/p{p}-fail{limit_fail}-recov{recovery_rate}-loss{loss_if_infected}",
            fraction_failure_results,
        )


if __name__ == "__main__":


    ## Specifications simulations
    shock = 0.1 
    simulation_size = 1000

    prob_sim = [0.1,0.2,0.4]
    limit_fail_sim = [0.1,0.4,0.6,0.8]
    recovery_rate_sim = [0.1,0.4,0.6,1.0]
    eps_sim = [0.3,0.6,0.7,0.85,0.95]


    ## Storing specs
    store_data = True
    file_number = 6



    ## Check if the folder exists, if not create the folder
    for ch in ["recovery", "failure", "EPS"]:
        if not os.path.exists(f"data/{ch}{file_number}"):
            os.makedirs(f"data/{ch}{file_number}")


    for prob in prob_sim:

        ## Change limit of failure
        for limits_fail in limit_fail_sim:
            simulate_failures(simulation_size, LOSS_IF_INFECTED, limit_fail = limits_fail, store_hist=store_data, shock_size = int(shock*NETWORK_SIZE), p = prob, change = "failure", file_number = file_number)
        ## Change recovery raters
        for recovery_rates in recovery_rate_sim:
            simulate_failures(simulation_size, LOSS_IF_INFECTED, LIMIT_FAIL, store_hist=store_data, shock_size = int(shock*NETWORK_SIZE), p = prob, recovery_rate= recovery_rates, change = "recovery", file_number = file_number)
        ## Change EPS drop
        for eps in eps_sim:
            simulate_failures(simulation_size, eps, LIMIT_FAIL, store_hist=store_data, shock_size = int(shock*NETWORK_SIZE),p = prob, change = "EPS", file_number = file_number)
