import numpy as np
import networkx as nx
from scipy.linalg import fractional_matrix_power


def compute_rwr(graph: nx.Graph, c=0.1):
    """
    This function computes the random walk with restart as the initial positional embedding.
    Other graph diffusion matrices such as Personalized PageRank or Heat diffusion might also be used.
    :param graph: input graph
    :param c: restart probability
    :return: RWR transition matrix
    """
    a = nx.convert_matrix.to_numpy_array(graph)
    d = np.diag(np.sum(a, 1))
    d_inv = fractional_matrix_power(d, -0.5)
    w_tilda = np.matmul(d_inv, a)
    w_tilda = np.matmul(w_tilda, d_inv)
    q = np.eye(w_tilda.shape[0]) - c * w_tilda
    q_inv = np.linalg.inv(q)
    e = np.eye(w_tilda.shape[0])
    r = (1 - c) * q_inv
    r = np.matmul(r, e)
    return r


def normalize_adjacency(adj):
    """
    :param adj: Adjancy matrix
    :return: Normalized adjancy matrix
    """
    adj = adj + np.diag(np.ones(adj.shape[0]))
    sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
    adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
    return adj

