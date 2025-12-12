import numpy as np
import pytest
from src.inference import metropolis_hastings, make_log_posterior, pgd_pseudolikelihood_map, soft_threshold
from src.sampling import ising_gibbs_sampler

def test_mh_standard_normal():
    def log_t(x): return -0.5*x[0]**2
    s=metropolis_hastings(log_t, np.array([0.0]), proposal_std=1.0, n_samples=300, burn_in=200, random_state=0)
    assert abs(s.mean())<0.3

def test_log_posterior_runs():
    # small fake data
    p = 3
    data = np.random.choice([-1, 1], size=(10, p))
    log_post = make_log_posterior(data, p=p, lamb=1.0, T=1.0)

    u_vec = np.zeros(int(p * (p - 1) / 2))  # all zeros
    val = log_post(u_vec)

    assert np.isfinite(val)  # shouldnot be nan or inf

def test_soft_threshold_prox_properties():
    """
    Validates correctness of the proximal update (soft-thresholding) used in
    pseudo-likelihood MAP estimation.
    """

    # vector input, compare against the closed-form definition
    x = np.array([-0.2, -0.05, 0.0, 0.05, 0.2], dtype=float)
    tau = 0.1

    out = soft_threshold(x, tau)
    expected = np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)

    assert np.allclose(out, expected, atol=1e-12), (
        f"soft_threshold mismatch.\n"
        f"x={x}\n"
        f"tau={tau}\n"
        f"out={out}\n"
        f"expected={expected}\n"
    )

    # threshold=0 should return the input
    out0 = soft_threshold(x, 0.0)
    assert np.allclose(out0, x, atol=1e-12), (
        f"threshold=0 should return x.\n"
        f"x={x}\n"
        f"out0={out0}\n"
    )

    # scalar input should also match the same formula
    x_scalar = 0.37
    tau_scalar = 0.15
    out_s = soft_threshold(x_scalar, tau_scalar)
    expected_s = np.sign(x_scalar) * max(abs(x_scalar) - tau_scalar, 0.0)

    assert np.isclose(out_s, expected_s, atol=1e-12), (
        f"scalar soft_threshold mismatch.\n"
        f"x={x_scalar}, tau={tau_scalar}\n"
        f"out={out_s}, expected={expected_s}\n"
    )
def test_pseudo_likelihood_map_monotone_and_symmetric():
    """
    Tests confirm that the objective function decreases monotonically
    along PGD iterations and that the output matrix is symmetric
    with zero diagonal.
    """
    np.random.seed(0)

    # Small Ising model for fast and stable testing
    p = 4
    u_star = np.array([
        [0.0, 0.5, 0.0, 0.2],
        [0.5, 0.0, 0.3, 0.0],
        [0.0, 0.3, 0.0, 0.4],
        [0.2, 0.0, 0.4, 0.0],
    ])

    y_init = np.ones(p, dtype=int)

    # Generate observations via Gibbs sampling
    Y = ising_gibbs_sampler(
        u=u_star,
        y_init=y_init,
        n_samples=300,
        burn_in=2000,
        T=1.0,
        random_state=123,
    )

    # Run PGD for pseudo-likelihood MAP estimation
    u_hat, losses = pgd_pseudolikelihood_map(
        Y,
        lambda_reg=50.0,
        step_size=0.01,
        max_iter=500,
        tol=1e-6,
    )

    losses = np.asarray(losses)

    # Objective function should decrease monotonically
    # Allow tiny numerical fluctuations
    assert losses[-1] <= losses[0]
    assert np.max(np.diff(losses)) <= 1e-2

    # Estimated interaction matrix must be symmetric with zero diagonal
    assert np.allclose(u_hat, u_hat.T, atol=1e-6)
    assert np.allclose(np.diag(u_hat), 0.0, atol=1e-6)
