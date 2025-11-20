import numpy as np
import pytest
from src.inference import metropolis_hastings, make_log_posterior, pgd_pseudolikelihood_map

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
    
def test_pgd_map_not_implemented_yet():
    data = np.random.choice([-1, 1], size=(10, 3))

    with pytest.raises(NotImplementedError):
        pgd_pseudolikelihood_map(data, p=3)