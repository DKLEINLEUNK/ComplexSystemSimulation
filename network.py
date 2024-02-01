"""
This module contains the Network class, which represents a network of nodes.

Running this module as a script will generate an example network.
"""

import multiprocessing
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import powerlaw as powerlaw


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


LIMIT_FAIL = 0.2  # Company fails if 20% of its EPS drops
LOSS_IF_INFECTED = 0.85
USE_REAL_DATA = False
POWER_LAW_OWNS = (
    0.55
)
NETWORK_SIZE = 100

if USE_REAL_DATA == True:
    file_path = "companies_data.csv"
    data = pd.read_csv(file_path, delimiter=";", decimal=",")

    EPS = np.array(data["EPS(rial)"]).astype(float)
    SECTOR_MPE = data["Group P/E"].unique().astype(float)

else:
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
        self.mpe = SECTOR_MPE[self.sector]  # Now uses the PE of a sector
        self.mpe_ini = self.mpe

        self.pi = self.mpe * self.eps
        self.pi_ini = self.pi

        self.A = ownership_matrix(self.graph, POWER_LAW_OWNS)

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
    change = "not_specified"
):
    """Generate the entire simulation of the network simulation"""
    #  = np.zeros(simulation_size)

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
        plt.title(
            f"ER $p={p}$, $f$ = {limit_fail}, $r$ = {recovery_rate}, $l$ = {loss_if_infected}"
        )

        sns.kdeplot(fraction_failure_results, bw_adjust=0.5, cut=0)
        plt.yscale("log")
        plt.xscale('log')
        plt.xlabel("$s$")
        plt.ylabel("$P(s)$")
        #plt.savefig(
            #f"figures/{change}/p{p}-fail{limit_fail}-recov{recovery_rate}-loss{loss_if_infected}.png"
        #)
        plt.clf()
    
        np.save(
            f"data/{change}2/p{p}-fail{limit_fail}-recov{recovery_rate}-loss{loss_if_infected}",
            fraction_failure_results,
        )


##### We could use two status, EPS and Fail or not. Send array with Statuses, and recive array. I will send you matrix A.

if __name__ == "__main__":
    # Change Network Structure
    """
    network = Network(100, m = 90)
    network.set_all_statuses(2)
    network.set_all_edges()
    create_shock(network, int(100*0.3))
    
    for i in range(10):
        propagate_shock(network, LOSS_IF_INFECTED, LIMIT_FAIL)
    """
    shock = 0.1
    """
    ## Change Network structure
    
    simulate_failures(1000, LOSS_IF_INFECTED, LIMIT_FAIL,  store_hist= True, shock_size = int(shock*NETWORK_SIZE), p = 0.1, change = "network")
    simulate_failures(1000, LOSS_IF_INFECTED, LIMIT_FAIL, store_hist=True, shock_size = int(shock*NETWORK_SIZE),  p = 0.2,change = "network")
    simulate_failures(1000, LOSS_IF_INFECTED, LIMIT_FAIL, store_hist= True, shock_size = int(shock*NETWORK_SIZE), p = 0.4,change = "network")
    
    # Change limit of failure
    
    simulate_failures(1000, LOSS_IF_INFECTED, limit_fail = 0.1, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.1, change = "failure")
    simulate_failures(1000, LOSS_IF_INFECTED, limit_fail = 0.4, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.1, change = "failure")
    simulate_failures(1000, LOSS_IF_INFECTED, limit_fail = 0.6, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.1, change = "failure")
    simulate_failures(1000, LOSS_IF_INFECTED, limit_fail = 0.8, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p =0.1, change = "failure")
    
    simulate_failures(1000, LOSS_IF_INFECTED, limit_fail = 0.1, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.2, change = "failure")
    simulate_failures(1000, LOSS_IF_INFECTED, limit_fail = 0.4, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.2, change = "failure")
    simulate_failures(1000, LOSS_IF_INFECTED, limit_fail = 0.6, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.2, change = "failure")
    simulate_failures(1000, LOSS_IF_INFECTED, limit_fail = 0.8, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p =0.2, change = "failure")

    simulate_failures(1000, LOSS_IF_INFECTED, limit_fail = 0.1, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.4, change = "failure")
    simulate_failures(1000, LOSS_IF_INFECTED, limit_fail = 0.4, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.4, change = "failure")
    simulate_failures(1000, LOSS_IF_INFECTED, limit_fail = 0.6, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.4, change = "failure")
    simulate_failures(1000, LOSS_IF_INFECTED, limit_fail = 0.8, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.4, change = "failure")

    # Change recovery rates
    simulate_failures(1000, LOSS_IF_INFECTED, LIMIT_FAIL, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.1, recovery_rate= 0.1, change = "recovery")
    simulate_failures(1000, LOSS_IF_INFECTED, LIMIT_FAIL, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.1, recovery_rate= 0.4, change = "recovery")
    simulate_failures(1000, LOSS_IF_INFECTED, LIMIT_FAIL, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.1, recovery_rate= 0.6, change = "recovery")
    simulate_failures(1000, LOSS_IF_INFECTED, LIMIT_FAIL, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.1, recovery_rate= 1.0, change = "recovery")

    simulate_failures(1000, LOSS_IF_INFECTED, LIMIT_FAIL, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.2, recovery_rate= 0.1, change = "recovery")
    simulate_failures(1000, LOSS_IF_INFECTED, LIMIT_FAIL, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.2, recovery_rate= 0.4, change = "recovery")
    simulate_failures(1000, LOSS_IF_INFECTED, LIMIT_FAIL, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.2, recovery_rate= 0.6, change = "recovery")
    simulate_failures(1000, LOSS_IF_INFECTED, LIMIT_FAIL, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.2, recovery_rate= 1.0, change = "recovery")

    simulate_failures(1000, LOSS_IF_INFECTED, LIMIT_FAIL, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.4, recovery_rate= 0.1, change = "recovery")
    simulate_failures(1000, LOSS_IF_INFECTED, LIMIT_FAIL, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.4, recovery_rate= 0.4, change = "recovery")
    simulate_failures(1000, LOSS_IF_INFECTED, LIMIT_FAIL, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.4, recovery_rate= 0.6, change = "recovery")
    simulate_failures(1000, LOSS_IF_INFECTED, LIMIT_FAIL, store_hist=True, shock_size = int(shock*NETWORK_SIZE), p = 0.4, recovery_rate= 1.0, change = "recovery")
    """
    # Change drop in EPS
    simulate_failures(1000, 0.3, LIMIT_FAIL, True, shock_size = int(shock*NETWORK_SIZE),p =  0.1, change = "EPS")
    simulate_failures(1000, 0.6, LIMIT_FAIL, True, shock_size = int(shock*NETWORK_SIZE), p = 0.1, change = "EPS")
    simulate_failures(1000, 0.7, LIMIT_FAIL, True, shock_size = int(shock*NETWORK_SIZE), p = 0.1, change = "EPS")
    simulate_failures(1000, 0.85, LIMIT_FAIL, True, shock_size = int(shock*NETWORK_SIZE), p = 0.1, change = "EPS")
    simulate_failures(1000, 0.95, LIMIT_FAIL, True, shock_size = int(shock*NETWORK_SIZE), p = 0.1,change = "EPS")
    """
    simulate_failures(1000, 0.3, LIMIT_FAIL, True, shock_size = int(shock*NETWORK_SIZE),p =  0.2, change = "EPS")
    simulate_failures(1000, 0.6, LIMIT_FAIL, True, shock_size = int(shock*NETWORK_SIZE), p = 0.2, change = "EPS")
    simulate_failures(1000, 0.7, LIMIT_FAIL, True, shock_size = int(shock*NETWORK_SIZE), p = 0.2, change = "EPS")
    simulate_failures(1000, 0.85, LIMIT_FAIL, True, shock_size = int(shock*NETWORK_SIZE), p = 0.2, change = "EPS")
    simulate_failures(1000, 0.95, LIMIT_FAIL, True, shock_size = int(shock*NETWORK_SIZE), p = 0.2, change = "EPS")

    simulate_failures(1000, 0.3, LIMIT_FAIL, True, shock_size = int(shock*NETWORK_SIZE),p =  0.4, change = "EPS")
    simulate_failures(1000, 0.6, LIMIT_FAIL, True, shock_size = int(shock*NETWORK_SIZE), p = 0.4, change = "EPS")
    simulate_failures(1000, 0.7, LIMIT_FAIL, True, shock_size = int(shock*NETWORK_SIZE), p = 0.4, change = "EPS")
    simulate_failures(1000, 0.85, LIMIT_FAIL, True, shock_size = int(shock*NETWORK_SIZE), p = 0.4, change = "EPS")
    simulate_failures(1000, 0.95, LIMIT_FAIL, True, shock_size = int(shock*NETWORK_SIZE), p = 0.4, change = "EPS" )

    """ 