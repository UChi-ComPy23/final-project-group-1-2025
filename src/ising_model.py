# src/ising_model.py
import numpy as np
from itertools import product

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
