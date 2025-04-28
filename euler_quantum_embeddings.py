# -*- coding: utf-8 -*-
"""
Euler‑Trotter Algorithm for simulating Quantum Jump Walks in Graphs
"""
import networkx as nx
import numpy as np
import time
from gensim.models import Word2Vec

'''
Inputs:
  - Adjacency matrix of the graph (adjacency_matrix.txt)
  - config.txt: Configuration file with parameters (num_traj, desired_jumps)
    * num_traj: number of trajectories to simulate
    * desired_jumps: number of jumps per trajectory
Outputs:
    - Embeddings for different gamma values (Qwalks_gamma={gamma}_traj_{num_traj}.txt)
'''

# ---------------------------------------------------------------------------
# Read Config Parameters
# ---------------------------------------------------------------------------

def read_config(path="config.txt"):
	params = {}
	with open(path, 'r') as f:
		for line in f:
			if "=" in line:
				key, val = line.strip().split("=")
				params[key.strip()] = float(val.strip())
	return int(params["num_traj"]), int(params["desired_jumps"]), int(params["window"])

# ---------------------------------------------------------------------------
# Prepare Hamiltonian (H), Possible Jumps and Rates
# ---------------------------------------------------------------------------

def prepare_quantum_operators(adj_matrix):
    """
    From adjacency matrix (adj), produce:
      - H: Hamiltonian 
      - collapse_indices: list of (i,j) with adj[i,j]>0
      - rates[(i,j)] = sqrt(P[i,j]) with P = row‑normalized adj
      // In Quantum Physics, the probability of jumping from i to j is given by the square of the amplitude of the wave function at node i, multiplied by the rate of transition from i to j. Here we precaulate the rates for each possible jump.
    """
    num_nodes = adj_matrix.shape[0]
    H = adj_matrix.astype(np.complex128)

    out_deg = adj_matrix.sum(axis=1)
    P = adj_matrix / out_deg[:, None]

    collapse_indices = [
        (i, j)
        for i in range(num_nodes)
        for j in range(num_nodes)
        if adj_matrix[i, j] > 0
    ]
    rates = { (i, j): np.sqrt(P[i, j]) for (i, j) in collapse_indices }

    return H, collapse_indices, rates

# ---------------------------------------------------------------------------
# Euler-Trotter Quantum Jump Walks
# ---------------------------------------------------------------------------

def quantum_trajectories_euler(
        H, collapse_indices, rates,
        num_nodes, gamma,
        num_traj, jumps_per_traj, start_node):
    '''
    This function simulates quantum trajectories using the Euler-Trotter method for a given start node.
    It performs 'gamma' coherent steps followed by a jump, until the desired number of jumps is reached.
    '''

    # Gamma will be the number of times the coherent step is repeated
    Coherent_steps_before_jump = max(1, gamma)

    # We define the time step size in terms of the Hamiltonian so that dt << 1 / (gamma * H_norm)
    H_norm = np.max(np.abs(H).sum(axis=1)) # Norm of the Hamiltonian
    dt = 0.1 / (gamma * H_norm) # time step size

    # Precompute Euler step operator V for coherent evolution
    # V = exp(-i * H * dt) ≈ I - i * H * dt (for small dt)
    V = np.eye(num_nodes, dtype=np.complex128) - 1j * H * dt

    all_trajs = []

    # Number of trajectories to simulate for the given start node
    for _ in range(num_traj):

        # Initialize the wave function at the start node
        psi = np.zeros(num_nodes, dtype=np.complex128)
        psi[start_node] = 1
        jumps = []

        for _ in range(jumps_per_traj):

            # Coherent Euler block
            for _ in range(Coherent_steps_before_jump):
                psi = V.dot(psi)
                # Normalize the wave function after each step
                psi /= np.linalg.norm(psi)

            # Compute jump probabilities p_k ∝ |ψ_i|^2 * rates^2
            # In Quantum Physics, the probability of jumping from i to j is given by the square of the amplitude of the wave function at node i, multiplied by the rate of transition from i to j. Here we precaulate the rates for each possible jump.
            probs = np.array([
                (abs(psi[i])**2) * (rates[(i, j)]**2)
                for (i, j) in collapse_indices
            ])
            total_p = probs.sum()

            # If total probability is zero, break the loop
            if total_p <= 0:
                break
            probs /= total_p

            # Sample a jump based on the probabilities
            _, j = collapse_indices[np.random.choice(len(collapse_indices), p=probs)]

            # Collapse the wave function to the jump target
            psi = np.zeros_like(psi)
            psi[j] = 1

            # Store the jump target
            jumps.append(j)

        all_trajs.append(jumps)

    return all_trajs

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    # Load graph from adjacency matrix
    adj_matrix = np.loadtxt("adjacency_matrix.txt", dtype=int)
    G = nx.from_numpy_array(adj_matrix)

    # Print graph information
    print(f"Graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # Build operators
    H, collapse_indices, rates = prepare_quantum_operators(adj_matrix)

    # Read configuration parameters
    # num_traj: number of trajectories to simulate for each start node
    # jumps_per_traj: number of jumps per trajectory
    # window: window size for Word2Vec

    num_traj, jumps_per_traj, window = read_config()

    # Loop over desired gamma values
    for gamma in range(3, 10):

        # Gamma will be the number of times the coherent step is repeated
        # before a jump is made. This is the number of Euler steps per block.

        embedding_path = f"Qwalks_embedding_gamma={gamma}_ntraj={num_traj}.txt"

        start_time = time.time()

        print(f"\n--- gamma = {gamma:.1f}, generating embedding to {embedding_path}")

        with open(embedding_path, 'w') as outfile:
            # Loop over all nodes in the graph
            for node in G.nodes():

                trajs = quantum_trajectories_euler(
                    H, collapse_indices, rates,
                    num_nodes=adj_matrix.shape[0],
                    gamma=gamma,
                    num_traj=num_traj,
                    jumps_per_traj=jumps_per_traj,
                    start_node=node
                )

            model = Word2Vec(
                    sentences=trajs,
                    vector_size=128,
                    window=window,
                    sg=1,
                    negative=10,
                    min_count=1,
                    workers=4
                            )
            
            # Write the embedding for the current 'node' and 'gamma' to the file
            with open(embedding_path, "w") as f:
                for node in model.wv.index_to_key:
                    embedding = model.wv[node]
                    emb_str = " ".join(map(str, embedding))
                    f.write(f"{node} {emb_str}\n")    

        elapsed = time.time() - start_time
        print(f"Done in {elapsed:.2f}s")
