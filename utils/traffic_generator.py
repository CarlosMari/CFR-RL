import numpy as np

def generate_tm(num_nodes=12, scale=1, total_traffic=2e8, N=1) -> np.array:
    """
    Generate synthetic traffic matrices based on the gravity model.
    num_nodes: int
        Number of nodes in the topology
    scale: float
        Parameter of the exponential rv
    Total traffic: float
        Total traffic of the network
    N: int
        Number of traffic matrices
    """
    # Generate the traffic in
    traffic_in = np.random.exponential(scale=scale, size=(N,num_nodes))
    # Generate the traffic out
    traffic_out = np.random.exponential(scale=scale, size=(N,num_nodes))
    # Normalize traffic in and traffic out
    prob_traffic_in = (traffic_in / np.sum(traffic_in, axis=1)[:, np.newaxis]).reshape(N,num_nodes,1)
    prob_traffic_out = (traffic_out / np.sum(traffic_out, axis=1)[:, np.newaxis]).reshape(N,1,num_nodes)
    # Multiply them, result at [i][j] will equal traffinc_in[i] * traffic_out[j] * total_traffic
    tm = total_traffic * np.einsum('ijk,ikl->ijl', prob_traffic_in, prob_traffic_out)
    return tm