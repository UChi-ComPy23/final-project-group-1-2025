# src/utils.py
import numpy as np

def vectorize_u(u: np.ndarray) -> np.ndarray:
    """
    Take a symmetric p x p matrix with zero diagonal and return the
    vector of upper-triangular entries (s < t), in row-major order.
    """
    u = np.asarray(u)
    p = u.shape[0]
    idx = np.triu_indices(p, k=1)
    return u[idx]

def unvectorize_u(u_vec: np.ndarray, p: int) -> np.ndarray:
    """
    Inverse of vectorize_u: construct a symmetric p x p matrix with
    zero diagonal from a vector of length p*(p-1)/2.
    """
    u_mat = np.zeros((p, p), dtype=float)
    idx = np.triu_indices(p, k=1)
    u_mat[idx] = u_vec
    u_mat[(idx[1], idx[0])] = u_vec  # copy to lower triangle
    return u_mat
