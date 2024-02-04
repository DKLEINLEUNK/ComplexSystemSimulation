import networkx as nx
import matplotlib.pyplot as plt


G = nx.random_regular_graph(3, 26)

# Set status of nodes
for node in G.nodes():
    if node < 6:
        G.nodes[node]['status'] = 0
    else:
        G.nodes[node]['status'] = 1

# Color nodes
node_color = ['green' if G.nodes[node]['status'] == 0 else 'yellow' for node in G.nodes()]


# Draw the graph
plt.figure(figsize=(6, 6))
nx.spring_layout(G, iterations=100)
nx.draw(G, node_color=node_color)
plt.show()