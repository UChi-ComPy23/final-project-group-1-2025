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

## Numba JIT core
from numba import njit

@njit
def _ising_gibbs_core_numba(u, y_init, n_samples, burn_in, T):
    """
    Numba-compiled core Gibbs sampler.
    """
    p = y_init.shape[0]
    total_iters = n_samples + burn_in
    samples = np.empty((total_iters, p), dtype=np.int8)

    y_current = y_init.copy()

    for n in range(total_iters):
        for r in range(p):
            # inline conditional_prob_spin_plus_one(y_current, r, u, T)
            A = 0.0
            for t in range(p):
                if t != r:
                    A += u[r, t] * y_current[t]
            A /= T
            p_plus = np.exp(2.0 * A) / (np.exp(2.0 * A) + 1.0)

            # Gibbs update
            if np.random.rand() < p_plus:
                y_current[r] = 1
            else:
                y_current[r] = -1

        # store current state
        samples[n, :] = y_current

    # discard burn-in
    return samples[burn_in:, :]

def ising_gibbs_sampler_parallel(u: np.ndarray,
                        y_init: np.ndarray,
                        n_samples: int,
                        burn_in: int = 0,
                        T: float = 1.0,
                        random_state=None) -> np.ndarray:
    """
    Run Gibbs sampling for the Ising model with interaction matrix u.
    Using a Numba-compiled core routine for speed.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Ensure dtypes are friendly to numba
    u_arr = np.asarray(u, dtype=np.float64)
    y0 = np.asarray(y_init, dtype=np.int8)

    samples = _ising_gibbs_core_numba(u_arr, y0, n_samples, burn_in, T)
    return samples