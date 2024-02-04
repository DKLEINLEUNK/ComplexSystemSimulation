'''
This module contains the Network class, which represents a network of nodes.

Running this module as a script will generate an example network.
'''


import networkx as nx
import numpy as np


class Network:

	'''
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
	'''

	def __init__(self, n, m=None, p=None, d=None, BA=False):
		'''
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
		`d` : int
			Number of dimensions for random regular graph
		'''
		
		self.n = int(np.ceil(n / 2) * 2)
		if m is not None:
			self.graph = nx.gnm_random_graph(self.n, m)
		elif p is not None:
			self.graph = nx.erdos_renyi_graph(self.n, p)
		elif d is not None:
			self.graph = nx.random_regular_graph(d, self.n)
		elif BA:
			self.graph = nx.barabasi_albert_graph(self.n, 3)
		else:
			raise ValueError("Either m, p, d must be provided.")
		

	def set_status(self, node, status):
		if status in [0, 1, 2]:
			nx.set_node_attributes(self.graph, {node: status}, 'status')
		else:
			raise ValueError("Status must be 0, 1, or 2.")


	def set_statuses(self, nodes, statuses):
		'''
		Sets the statuses of an array of nodes.
		'''
		if len(nodes) != len(statuses):
			raise ValueError("Nodes and statuses must be the same length.")
		for node, status in zip(nodes, statuses):
			self.set_status(node, status)

	
	def set_all_statuses(self, status):
		'''
		Sets the statuses of all nodes.
		'''
		nx.set_node_attributes(self.graph, status, 'status')
	

	def get_status(self, node):
		'''
		Returns the status of a node.
		'''
		return nx.get_node_attributes(self.graph, 'status')[node]


	def get_statuses(self, nodes):
		'''
		Returns the statuses of an array of nodes.
		'''
		return np.array([self.get_status(node) for node in nodes])  # TODO check if this is a bottleneck


	def get_all_statuses(self):
		'''
		Returns the statuses of all nodes.
		'''
		return nx.get_node_attributes(self.graph, 'status')
	

	def get_neighbors(self, node, as_list=False):
		'''
		Returns the neighbors of a node.
		'''
		return self.graph.neighbors(node) if not as_list else list(self.graph.neighbors(node))
	
	
	def get_multiple_neighbors(self, nodes, as_list=False):
		'''
		Returns the neighbors of an array of nodes.
		'''
		# TODO see if this is a bottleneck
		neighbors = []
		for node in nodes:
			neighbors.append(self.get_neighbors(node)) if not as_list else neighbors.append(list(self.graph.neighbors(node)))
		return neighbors


if __name__ == "__main__":
	
	### EXAMPLE USAGE ###
	
	# Creating a network
	network = Network(n=100_001, m=300_000)
	nodes = np.array([1, 10, 100, 1_000, 10_000, 100_000])  # nodes to check status of
	print(f"Generated a network with {network.graph.number_of_nodes()} nodes and {network.graph.number_of_edges()} edges.")
	print(f"Status of nodes 1, 10, 100, 1_000, 10_000, 100_000: {network.get_statuses(nodes)}")

	# Getting the neighbors of a specific node (WARNING: only use as_list=True for testing and illustration purposes, as it is a bottleneck)
	neighbors = network.get_neighbors(10, as_list=True)
	print(f"Neighbors of node 10: {neighbors}")

	# Getting the neighbors of an array of nodes (WARNING: only use as_list=True for testing and illustration purposes, as it is a bottleneck)
	neighbors = network.get_multiple_neighbors(nodes, as_list=True)
	print(f"Neighbors of nodes 1, 10, 100, 1_000, 10_000, 100_000: {neighbors}")

