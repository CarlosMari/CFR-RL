import numpy as np
import networkx as nx

def generate_topology(num_nodes: int =12,w=10,seed=1):
    """
    Code to generate random topologies

    num_nodes: int
        Number of nodes of the topology
    num_links: int
        Number of links of the topology. Must be greater or equal to the number of nodes.
    """

    M = nx.random_internet_as_graph(num_nodes, seed=seed)
    print(M)
    print(f'Generating topology {seed}')
    Z = nx.DiGraph()
    for u,v,data in M.edges(data=True):
        Z.add_edge(u, v, weight=w)
        Z.add_edge(v, u, weight=w)

    return Z, Z.number_of_edges()