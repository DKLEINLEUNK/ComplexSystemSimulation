"""
This module contains functions for modifying a network.

It is meant to be used in conjunction with the `Network` class.

Running this module as a script shows some example use cases.


Guide:
	- Healthy nodes: Nodes with status = 2, which means they have not become weak/infected,
	- Weak nodes: Nodes with status = 1, that in the next quarter will have an 85% decrease in it's eps.
	- Failed nodes: Nodes with status = 0, have been already become weak, thus they will not have another 85% in the next quarter and their eps is 0.
"""


import numpy as np
from economy_functions import ownership_matrix, custom_sort


def create_shock(network, size):
    """
    Creates the default shock into the system, by failing n businesses

    Parameters
    ----------
            size: Number of companies that fail
    """
    nodes = np.random.choice(len(network.graph.nodes), size=size, replace=False)
    network.set_statuses(nodes, np.ones(len(nodes)))


def get_weak_nodes(network, loss_if_infected):
    """
    Given a network, it check if there are any new infected nodes (node = 1)
    Return the weak_nodes array and loss_infected for multiplication in algorithm and loss reduction
    """

    weak_nodes = np.array(
        list(filter(lambda item: item[1] == 1, network.get_all_statuses().items()))
    )
    loss_infected = np.zeros(network.graph.number_of_nodes())

    if weak_nodes.shape[0] == 0:
        return weak_nodes, loss_infected
    
    else:
        weak_nodes = weak_nodes[:, 0].astype(int)
        loss_infected[weak_nodes] = 1 - loss_if_infected
    
    return weak_nodes, loss_infected

def get_failed_nodes(network):
    """
    Given a network, it check if there are any new infected nodes (node = 1)
    Return the weak_nodes array and loss_infected for multiplication in algorithm
    """

    failed_nodes = np.array(
        list(filter(lambda item: item[1] == 0, network.get_all_statuses().items()))
    )
    failed_nodes = failed_nodes[:, 0].astype(int)
    
    return failed_nodes


def threshold_test(network, threshold):
    """
    Returns an array weak_nodes which are all healthy nodes that had a threshold reduction.
    Only set as new_weak_nodes the nodes that come from healthy.
    Inputs:
            - Network
            - Array of healthy nodes in the network
    """
    healthy_nodes = np.array(
        list(filter(lambda item: item[1] == 2, network.get_all_statuses().items()))
    )
    possible_weak_nodes = np.where(
        ((network.pi_ini - network.pi) / (network.pi_ini)) > threshold
    )[0]

    new_weak_nodes = np.intersect1d(healthy_nodes, possible_weak_nodes)
    return new_weak_nodes


def propagate_shock(network, loss_if_infected, threshold, recovery_rate = 0.1):
    """
    Given a network, and a loss_if_infected, propagates a shock during 90 days.
    After 90 days:
            - new infected nodes turn to 1.
            - Already weak nodes turn to 0 or 2 (2 means they recover)
            - eps_initial is set at the end eps.
    Inputs:
            - Loss_if infected: % loss of eps if a node fails
            - Threshold: % threshold to see a failure
    """

    weak_nodes, loss_infected = get_weak_nodes(
        network, loss_if_infected
    )  

    delta_eps = network.eps * loss_infected
    network.eps -= delta_eps

    network.pi = network.mpe * network.eps


    # For 90 days
    for i in range(90):
        delta_eps = delta_eps @ network.A
        network.eps = network.eps - delta_eps
        network.pi = network.mpe * network.eps


    ## Setting olds weak nodes to weak, status = 0
    fail(network,weak_nodes)
    failed_nodes = get_failed_nodes(network)

    ## Setting all new weak nodes to weak, status = 1
    new_weak_nodes = threshold_test(network, threshold)
    network.set_statuses(new_weak_nodes, np.ones(len(new_weak_nodes)))
    
    #Recovery of already failed nodes
    recovery_nodes = np.random.choice(failed_nodes, size=int(recovery_rate * len(failed_nodes)), replace=False)
    improve(network, recovery_nodes)
    

    print(len(failed_nodes)/network.graph.number_of_nodes())

    ## Setting new conditions as initial for next period
    network.eps_ini = network.eps
    network.mpe_ini = network.mpe
    network.pi_ini = network.pi


def improve(network, nodes):
    """
    Given a set fo nodes, it returns them to state two
    """
    network.set_statuses(nodes, np.full(len(nodes),2))
    if len(nodes) > 0:
        network.eps[nodes] = network.eps_ini_o[nodes]
    

def fail(network, nodes):
    """
    Description
    -----------
    Fails an array of nodes.

    Parameters
    ----------
    `network` : Network
            The network to fail.
    `nodes` : NDArray
            The array of nodes to fail.
    """
    network.set_statuses(nodes, np.zeros(len(nodes), dtype=int))
    if len(nodes) > 0:
        network.eps[nodes] = (
            network.eps[nodes]*0
        )

def degrade(network, node=None, random=False):
    """
    Description
    -----------
    Degrades a node's status by 1.
    Either a specific `node` or a random node can be degraded.

    Parameters
    ----------
    `network` : Network
            The network to degrade.
    `node` : int
            The node to degrade.
    `random` : bool
            Whether to choose a random node to degrade.
    """
    if node is None and not random:
        raise ValueError("Either a node or random must be provided.")
    if random:
        node = np.random.choice(network.graph.nodes)
    network.set_status(node, int(network.get_status(node) - 1))


def save_state(network, verbose=False):
    """
    Description
    -----------
    Saves the current state of the network.

    Parameters
    ----------
    `network` : Network
            The network to save the state of.
    """
    print("Saving state...") if verbose else None
    return network.get_all_statuses()


def load_state(network, previous_state):
    """
    Description
    -----------
    Loads a previous network state.

    May be used in conjunction with `save_state` to reset a network after a cascade.

    Parameters
    ----------
    `network` : Network
            The network to reset.
    `previous_state` : NDArray
            The previous state of the network.
    """
    network.set_all_statuses(previous_state)


def reinforce(network, nodes, epsilon):
    """
    Description
    -----------
    Reinforce weak an array of nodes with probability epsilon.

    TODO add numpy enhancement to remove this additional loop (use an array of probabilities)
    """
    for node in nodes:
        if np.random.rand() < epsilon:
            network.set_status(node, 2)
        else:
            network.set_status(node, 1)
