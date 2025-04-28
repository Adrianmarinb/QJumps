# -*- coding: utf-8 -*-
"""
Graph Generator
@author: adria
"""

import networkx as nx
import numpy as np
import random

# ---------------------------------------------------------------------------
# Functions to construct graphs
# ---------------------------------------------------------------------------

def construct_scale_free_graph(n_nodes, m_edges, seed=None):
    if seed is not None:
        np.random.seed(seed)

    G = nx.barabasi_albert_graph(n=n_nodes, m=m_edges, seed=seed)

    node_types = {node: np.random.randint(0, 1) for node in G.nodes()}
    node_degrees = dict(G.degree())

    return G, node_types, node_degrees

def construct_clustered_random_graph(num_clusters, cluster_size, p_intra, p_inter, num_node_types, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    G = nx.Graph()
    node_offset = 0
    cluster_nodes = []

    for c in range(num_clusters):
        n = 55 if (c == num_clusters - 1) else cluster_size
        cluster = nx.erdos_renyi_graph(n=n, p=p_intra, seed=seed)
        mapping = {node: node + node_offset for node in cluster.nodes}
        cluster = nx.relabel_nodes(cluster, mapping)
        G = nx.compose(G, cluster)

        cluster_nodes.append(list(mapping.values()))
        node_offset += cluster_size

    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            for u in cluster_nodes[i]:
                for v in cluster_nodes[j]:
                    if random.random() < p_inter:
                        G.add_edge(u, v)

    node_types = {node: np.random.randint(0, num_node_types) for node in G.nodes()}
    node_degrees = dict(G.degree())

    return G, node_types, node_degrees

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

graph_type = "clustered"  # Choose: "scale-free" or "clustered"
seed = 13

# Parameters for scale-free
n_nodes = 100
m_edges = 2

# Parameters for clustered
num_clusters = 4
cluster_size = 15
p_intra = 0.25
p_inter = 0.0015
num_node_types = 1

# ---------------------------------------------------------------------------
# Graph Generation
# ---------------------------------------------------------------------------

if graph_type == "scale-free":
    G, node_types, node_degrees = construct_scale_free_graph(n_nodes, m_edges, seed)
else:
    G, node_types, node_degrees = construct_clustered_random_graph(
        num_clusters, cluster_size, p_intra, p_inter, num_node_types, seed)

# ---------------------------------------------------------------------------
# Saving files
# ---------------------------------------------------------------------------

# Save adjacency matrix
adj_matrix = nx.to_numpy_array(G, dtype=int)
np.savetxt("adjacency_matrix.txt", adj_matrix, fmt="%d")

# Save edges
with open("edges.txt", "w") as f:
    for u, v in G.edges():
        f.write(f"{u} {v}\n")

# Save nodes (node id and type)
with open("nodes.txt", "w") as f:
    for node in G.nodes():
        f.write(f"{node} {node_types[node]}\n")
