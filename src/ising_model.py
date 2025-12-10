# src/ising_model.py
import numpy as np
from itertools import product
from numba import njit, prange

def pair_energy(y: np.ndarray, u: np.ndarray, T: float = 1.0) -> float:
    """
    Return the exponent term sum_{s<t} u_{st} y_s y_t / T.
    """
    y = np.asarray(y)
    u = np.asarray(u)
    return (1.0 / T) * np.sum(np.triu(u, k=1) * np.outer(y, y))

def partition(u: np.ndarray, T: float = 1.0) -> float:
    """
    Exact partition function Z(u) = sum_y exp(sum_{s<t} u_{st} y_s y_t / T).
    Only feasible for small p (e.g. p <= 10).
    """
    u = np.asarray(u)
    p = u.shape[0]
    Z = 0.0
    for bits in product([-1, 1], repeat=p):
        y = np.array(bits, dtype=int)
        Z += np.exp(pair_energy(y, u, T=T))
    return Z

# Numba-parallel exact partition function
@njit(parallel=True)
def _partition_numba(u: np.ndarray, T: float) -> float:
    """
    Numba-accelerated exact partition function for the Ising model.
    The outer loop over configurations s is parallelized with prange.
    """
    p = u.shape[0]
    n_states = 1 << p  # 2^p
    total = 0.0

    # Each iteration corresponds to one spin configuration encoded by s
    for s in prange(n_states):
        # Compute energy E(y; u) for configuration y determined by bits of s
        energy = 0.0
        for i in range(p):
            # spin y_i ∈ {-1, +1} from bit i of s
            yi = 1.0 if (s >> i) & 1 else -1.0
            for j in range(i + 1, p):
                yj = 1.0 if (s >> j) & 1 else -1.0
                energy += u[i, j] * yi * yj / T
        total += np.exp(energy)

    return total

def partition_parallel(u: np.ndarray, T: float = 1.0) -> float:
    """
    Exact partition function Z(u) = sum_y exp(sum_{s<t} u_{st} y_s y_t / T).
    """
    u = np.asarray(u, dtype=np.float64)
    return float(_partition_numba(u, T))

def conditional_prob_spin_plus_one(y: np.ndarray, r: int, u: np.ndarray, T: float = 1.0) -> float:
    """
    P(y_r = +1 | y_{V\\r}) for the Ising model with interaction matrix u.

    Using:
        P(y_r | y_{-r}) = exp( (2 y_r / T) sum_{t!=r} u_{rt} y_t ) /
                          (exp( (2 / T) sum_{t!=r} u_{rt} y_t ) + 1 )
    We return the probability of y_r = +1.
    """
    y = np.asarray(y)
    u = np.asarray(u)
    A = np.sum(u[r, :] * y) - u[r, r] * y[r]   # sum_{t != r} u_{rt} y_t
    A = A / T
    p_plus = np.exp(2 * A) / (np.exp(2 * A) + 1.0)
    return float(p_plus)
