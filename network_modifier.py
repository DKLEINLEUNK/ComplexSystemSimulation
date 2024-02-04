'''
This module contains functions for modifying a network.

It is meant to be used in conjunction with the `Network` class.

Running this module as a script shows some example use cases.
'''


import numpy as np

from network import Network


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
		# if network.get_status(node) == 1:  # to ensure strong nodes remain strong
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