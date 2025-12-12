# src/inference.py
import numpy as np
from scipy.stats import laplace
from .ising_model import partition
from .utils import unvectorize_u
from numba import njit, prange

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

# Numba-accelerated log-likelihood
@njit(parallel=True)
def _log_likelihood_numba(u_mat: np.ndarray,
                          data: np.ndarray,
                          p: int,
                          T: float) -> float:
    """
    Numba-accelerated computation of the Ising log-likelihood term
    The outer loop (over samples) uses prange for data-parallel speedup.
    """
    n_samples = data.shape[0]
    log_lik = 0.0

    for k in prange(n_samples):           # parallel over samples
        sample = data[k]
        energy = 0.0
        for i in range(p):
            si = sample[i]
            for j in range(i + 1, p):
                energy += u_mat[i, j] * si * sample[j] / T
        log_lik += energy

    return log_lik

def make_log_posterior_parallel(data: np.ndarray,
                       p: int,
                       lamb: float = 1.0,
                       T: float = 1.0):
    """
    Construct a log-posterior function log p(u_vec | data) for the Ising model.
    parallel
    """
    # Numba prefers simple integer dtype for the observed spins
    data = np.asarray(data, dtype=np.int8)
    n_samples = data.shape[0]

    def log_posterior(u_vec: np.ndarray) -> float:
        u_vec = np.asarray(u_vec, dtype=np.float64)
        # Convert vectorized parameters into full symmetric matrix
        u_mat = unvectorize_u(u_vec, p)
        # Accelerated log-likelihood term
        log_lik = _log_likelihood_numba(u_mat, data, p, T)

        # Exact partition function (same as original)
        Z = partition(u_mat, T=T)
        log_lik -= n_samples * np.log(Z)

        # Laplace prior over u_vec (same as original)
        log_prior = np.sum(
            laplace.logpdf(u_vec, loc=0.0, scale=1.0 / lamb)
        )
        return log_lik + log_prior
    return log_posterior

# PGD Pseudo-likelihood MAP
def compute_conditional_probability(y, u_mat, r):
    p = len(y)
    A_r = 0
    for t in range(p):
        if t != r:
            A_r += u_mat[r, t] * y[t]

    exp_2A = np.exp(2 * A_r)
    if y[r] == 1:
        p_r = exp_2A / (exp_2A + 1)
    else:  # y_r = -1
        p_r = 1 / (exp_2A + 1)

    return p_r, A_r

def compute_negative_log_likelihood(Y, u_mat):
    N, p = Y.shape
    neg_ll = 0.0

    for i in range(N):
        y = Y[i]
        for r in range(p):
            p_r, _ = compute_conditional_probability(y, u_mat, r)
            neg_ll -= np.log(p_r + 1e-15)  # Add small epsilon for stability

    return neg_ll

def compute_gradient(Y, u_mat):
    """
    Compute gradient of negative pseudo log-likelihood analytically
    Derived from the conditional probability formula
    """
    N, p = Y.shape
    grad = np.zeros((p, p))

    for i in range(N):
        y = Y[i]

        for r in range(p):

            A_r = 0
            for t in range(p):
                if t != r:
                    A_r += u_mat[r, t] * y[t]

            z = 2 * y[r] * A_r
            if z > 0:
                sigma = 1 / (1 + np.exp(-z))
            else:
                exp_z = np.exp(z)
                sigma = exp_z / (1 + exp_z)

            # Gradient contribution from this conditional
            for t in range(p):
                if t != r:
                    grad_rt = -2 * y[r] * y[t] * (1 - sigma)  # Negative gradient for negative log-likelihood
                    grad[r, t] += grad_rt

    # Average over samples and enforce symmetry
    grad = grad / N
    grad_sym = (grad + grad.T) / 2
    np.fill_diagonal(grad_sym, 0)

    return grad_sym

def soft_threshold(x, threshold):
    """Soft thresholding operator for L1 regularization"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def pgd_pseudolikelihood_map(Y, lambda_reg=10.0, step_size=0.001,
                              max_iter=1000, tol=1e-6):
    """
    Proximal Gradient Descent with corrected gradient computation
    lambda_reg: regularization parameter (should be larger for weaker regularization)
                The objective is: negative_log_likelihood + (1/lambda_reg) * ‖u‖₁
    """
    N, p = Y.shape

    # Initialize with small values
    np.random.seed(1)
    u_mat = np.random.randn(p, p) * 0.01
    u_mat = (u_mat + u_mat.T) / 2  # Make symmetric
    np.fill_diagonal(u_mat, 0)  # Zero diagonal

    losses = []

    for iteration in range(max_iter):
        u_prev = u_mat.copy()

        # Compute gradient
        grad = compute_gradient(Y, u_mat)

        # Gradient step
        u_gd = u_mat - step_size * grad

        # Proximal step (soft thresholding)
        # Threshold = step_size / lambda_reg
        threshold = step_size / lambda_reg

        u_new = np.zeros((p, p))
        for i in range(p):
            for j in range(i + 1, p):
                u_ij = soft_threshold(u_gd[i, j], threshold)
                u_new[i, j] = u_ij
                u_new[j, i] = u_ij

        np.fill_diagonal(u_new, 0)
        u_mat = u_new

        # Compute loss
        neg_ll = compute_negative_log_likelihood(Y, u_mat)
        l1_penalty = np.sum(np.abs(u_mat)) / lambda_reg
        total_loss = neg_ll + l1_penalty
        losses.append(total_loss)

        # Check convergence
        u_change = np.linalg.norm(u_mat - u_prev, 'fro')

        if (iteration % 100 == 0 or iteration == max_iter - 1):
            print(f"Iter {iteration:4d}: Loss = {total_loss:.6f}, "
                  f"delta_u = {u_change:.6f}")

        if u_change < tol and iteration > 10:
            print(f"Converged at iteration {iteration}")
            break

    return u_mat, losses