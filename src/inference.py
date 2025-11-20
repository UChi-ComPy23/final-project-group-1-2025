# src/inference.py
import numpy as np
from scipy.stats import laplace
from .ising_model import partition
from .utils import unvectorize_u

def metropolis_hastings(log_target,
                        initial_state: np.ndarray,
                        proposal_std: float = 0.1,
                        n_samples: int = 1000,
                        burn_in: int = 10_000,
                        random_state = None) -> np.ndarray:
    """
    Generic Random Walk Metropolis--Hastings sampler on R^d.
    """
    if random_state is not None:
        np.random.seed(random_state)

    initial_state = np.asarray(initial_state, dtype=float)
    d = initial_state.size
    total = n_samples + burn_in

    samples = np.zeros((total, d), dtype=float)
    samples[0] = initial_state.copy()

    current_state = initial_state.copy()
    current_log_target = log_target(current_state)

    for n in range(1, total):
        proposal = current_state + proposal_std * np.random.randn(d)
        proposal_log_target = log_target(proposal)

        log_alpha = proposal_log_target - current_log_target
        if np.log(np.random.rand()) < log_alpha:
            current_state = proposal
            current_log_target = proposal_log_target

        samples[n] = current_state

    return samples[burn_in:]


def make_log_posterior(data: np.ndarray,
                       p: int,
                       lamb: float = 1.0,
                       T: float = 1.0):
    """
    Construct a log-posterior function log p(u_vec | data) for the Ising model.
    """
    data = np.asarray(data, dtype=int)

    def log_posterior(u_vec: np.ndarray) -> float:
        u_vec = np.asarray(u_vec, dtype=float)
        u_mat = unvectorize_u(u_vec, p)

        # log-likelihood
        log_lik = 0.0
        for sample in data:
            # energy term sum_{s<t} u_{st} y_s y_t / T
            energy = 0.0
            for i in range(p):
                for j in range(i + 1, p):
                    energy += u_mat[i, j] * sample[i] * sample[j] / T
            log_lik += energy

        Z = partition(u_mat, T=T)
        log_lik -= len(data) * np.log(Z)

        # Laplace prior
        log_prior = np.sum(laplace.logpdf(u_vec, loc=0.0, scale=1.0 / lamb))

        return log_lik + log_prior

    return log_posterior


# PGD Pseudo-likelihood MAP Pending to do

def pgd_pseudolikelihood_map(
    data: np.ndarray,
    p: int,
    lamb: float = 1.0,
    step_size: float = 1e-2,
    max_iter: int = 100,
    tol: float = 1e-5,
):
    """
    Placeholder for the pseudo-likelihood MAP estimator using Proximal Gradient Descent.

    This will be fully implemented in the final project. For the checkpoint,
    we only define the interface.
    """
    raise NotImplementedError("PGD pseudo-likelihood MAP will be implemented in the final project.")
