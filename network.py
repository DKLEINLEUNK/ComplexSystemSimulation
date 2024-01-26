'''
This module contains the Network class, which represents a network of nodes.

Running this module as a script will generate an example network.
'''


import networkx as nx
import numpy as np
from economy_functions import ownership_matrix

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

	def __init__(self, n, m=None, p=None):
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
		'''
		if m is not None:
			self.graph = nx.gnm_random_graph(n, m, directed = True)
		elif p is not None:
			self.graph = nx.erdos_renyi_graph(n, p, directed = True)
	
		else:
			raise ValueError("Either m or p must be provided.")
	
	def set_ownerships(self):
		"""
		Sets all edges weights
		"""
		A = ownership_matrix(graph.number_of_nodes(), graph.edges())
		for u, v in G.edges():
			graph[u][v]['ownership'] = A[u,v]


	def set_status(self, node, status):
		### SET AT 0 and 1 status, and add counter EPS. MEYBE WE CLEANING CODE SET IT AS CHILD CLASS
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
##### We could use two status, EPS and Fail or not. Send array with Statuses, and recive array. I will send you matrix A.

if __name__ == "__main__":
	
	### EXAMPLE USAGE ###
	
	# Creating a network
	network = Network(n=10, m=5)
	nodes = np.array([1, 10, 100, 1_000, 10_000, 100_000])  # nodes to check status of
	print(f"Generated a network with {network.graph.number_of_nodes()} nodes and {network.graph.number_of_edges()} edges.")
	print(nx.is_weighted(network.graph))
	print(np.array(network.graph.edges()))
	print(f"Status of nodes 1, 10, 100, 1_000, 10_000, 100_000: {network.get_statuses(nodes)}")

	# Getting the neighbors of a specific node (WARNING: only use as_list=True for testing and illustration purposes, as it is a bottleneck)
	#neighbors = network.get_neighbors(10, as_list=True)
	#print(f"Neighbors of node 10: {neighbors}")

	# Getting the neighbors of an array of nodes (WARNING: only use as_list=True for testing and illustration purposes, as it is a bottleneck)
	#neighbors = network.get_multiple_neighbors(nodes, as_list=True)
	#print(f"Neighbors of nodes 1, 10, 100, 1_000, 10_000, 100_000: {neighbors}")

