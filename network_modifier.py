'''
This module contains functions for modifying a network.

It is meant to be used in conjunction with the `Network` class.

Running this module as a script shows some example use cases.
'''


import numpy as np
from network import Network
from economy_functions import *


def create_shock(network:Network, size):
	"""
	Creates the default shock into the system, by failing n businesses

	Parameters
	----------
		size: Number of companies that fail
	"""
	nodes = np.random.choice(len(network.graph.nodes),size = size, replace = False)
	network.set_statuses(nodes, np.ones(len(nodes)))

def get_failed_nodes(network:Network, loss_if_infected):
	"""
	Given a network, it check if there are any new infected nodes (node = 1)
	Return the failed_nodes array and loss_infected for multiplication in algorithm
	"""

	failed_nodes = np.array(list(filter(lambda item: item[1] == 1, network.get_all_statuses().items())))
	if failed_nodes.shape[0] == 0:
		### THIS NEEDS TO BE IMPROVED; I SET AS IF THERE ARE NO NEW INFECTED NODES THEN WE ARE IN STABILITY AND THE PROGRAMS STOP
		raise ValueError("There are not any new failed nodes, stability has been reached")
	else:
		failed_nodes = failed_nodes[:,0].astype(int)
		loss_infected = np.zeros(network.graph.number_of_nodes())
		loss_infected[failed_nodes] = loss_if_infected
	return failed_nodes,loss_infected

def propagate_shock(network:Network, loss_if_infected, threshold):
	"""
	Given a network, and a loss_if_infected, propagates a shocl during 90 days.
	After 90 days:
		- new infected nodes turn to 1.
		- Already failed nodes turn to 0 or 2 (2 means they recover)
		- eps_initial is set at the end eps.
	Inputs:
		- Loss_if infected: % loss of eps if a node fails
		- Threshold: % threshold to see a failure
	"""

	failed_nodes,loss_infected = get_failed_nodes(network,loss_if_infected)
	# Calculate change in EPS here, dependent on A
	delta_eps = network.eps * loss_infected
	network.eps -= delta_eps

	network.pi = network.mpe * network.eps

	# For 90 days
	for i in range(90):
		delta_eps = delta_eps @ network.A
		network.eps = network.eps - delta_eps
		network.pi = network.mpe * network.eps
		network.mpe = np.average(network.eps/network.pi) ## Compute new network mpe
		#print(f"i: {i}, EPS: {network.eps}")

	## Setting all new failed nodes to failed and setting already failed nodes to dead(0)
	new_failed_nodes = np.where(((network.pi_ini-network.pi)/(network.pi_ini)) > threshold)[0]
	network.set_statuses(new_failed_nodes, np.ones(len(new_failed_nodes)))
	fail(network,failed_nodes)
	
	## Setting new conditions as initial for next period
	network.eps_ini = network.eps 
	network.mpe_ini = network.mpe
	network.pi_ini = network.pi


def degrade(network:Network, node=None, random=False):
	'''
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
	'''
	if node is None and not random:
		raise ValueError("Either a node or random must be provided.")
	if random:
		node = np.random.choice(network.graph.nodes)
	network.set_status(node, int(network.get_status(node) - 1))

def fail(network:Network, nodes):
	'''
	Description
	-----------
	Fails an array of nodes.

	Parameters
	----------
	`network` : Network
		The network to fail.
	`nodes` : NDArray
		The array of nodes to fail.
	'''
	network.set_statuses(nodes, np.zeros(len(nodes), dtype=int))


def save_state(network:Network, verbose=False):
	'''
	Description
	-----------
	Saves the current state of the network.

	Parameters
	----------
	`network` : Network
		The network to save the state of.
	'''
	print("Saving state...") if verbose else None
	return network.get_all_statuses()


def load_state(network:Network, previous_state):
	'''
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
    '''
	network.set_all_statuses(previous_state)


def reinforce(network:Network, nodes, epsilon):
	'''
	Description
	-----------
	Reinforce failed an array of nodes with probability epsilon.

	TODO add numpy enhancement to remove this additional loop (use an array of probabilities)
	'''
	for node in nodes:
		if np.random.rand() < epsilon:
			network.set_status(node, 2)
		else:
			network.set_status(node, 1)


if __name__ == "__main__":
	
	### EXAMPLE USAGE ###
	
	# Creating a network
	network = Network(n=100_001, m=300_000)
	nodes = np.array([1, 10, 100, 1_000, 10_000, 100_000])  # nodes to check status of
	print(f"Status of nodes 1, 10, 100, 1_000, 10_000, 100_000: {network.get_statuses(nodes)}")
	
	# Degrading a specific node (if random use node=None, random=True)
	degrade(network, node=10_000, random=False)
	print(f"Status of nodes after degradation: {network.get_statuses(nodes)}")

	# Failing an array of nodes (useful for cascading failures)
	fail(network, np.array([1, 10, 100]))
	print(f"Status of nodes after failure: {network.get_statuses(nodes)}")

	# Saving the current state
	previous_state = save_state(network, verbose=True)

	# Setting the statuses of all nodes to 1
	network.set_all_statuses(1)
	print(f"Status of nodes after setting all to 1: {network.get_statuses(nodes)}")

	# Loading the previous state
	load_state(network, previous_state)
	print(f"Status of nodes after loading previous state: {network.get_statuses(nodes)}")