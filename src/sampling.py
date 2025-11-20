# src/sampling.py
import numpy as np
from .ising_model import conditional_prob_spin_plus_one

def ising_gibbs_sampler(u: np.ndarray,
                        y_init: np.ndarray,
                        n_samples: int,
                        burn_in: int = 0,
                        T: float = 1.0,
                        random_state = None) -> np.ndarray:
    """
    Run Gibbs sampling for the Ising model with interaction matrix u.
    """
    if random_state is not None:
        np.random.seed(random_state)

    y_current = np.asarray(y_init, dtype=int).copy()
    p = len(y_current)
    total_iters = n_samples + burn_in
    samples = np.zeros((total_iters, p), dtype=int)

    for n in range(total_iters):
        for r in range(p):
            p_plus = conditional_prob_spin_plus_one(y_current, r, u, T=T)
            if np.random.rand() < p_plus:
                y_current[r] = 1
            else:
                y_current[r] = -1
        samples[n] = y_current.copy()

    return samples[burn_in:]
