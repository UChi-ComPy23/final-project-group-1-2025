import numpy as np
from src.sampling import ising_gibbs_sampler
from src.ising_model import conditional_prob_spin_plus_one

'''
def test_gibbs_sampler_basic():
    u=np.array([[0.,1.],[1.,0.]])
    y_init=np.array([1,-1])
    samples=ising_gibbs_sampler(u,y_init, n_samples=200, burn_in=100, random_state=0)
    assert samples.shape==(200,2)
'''
def test_gibbs_single_spin_bias():
    u = np.array([[0., 1.0],
                  [1.0, 0.]])
    y_init = np.array([-1, 1], dtype=int)

    samples = ising_gibbs_sampler(
        u=u,
        y_init=y_init,
        n_samples=2000,
        burn_in=1000,
        T=1.0,
        random_state=0
    )
    # fisrt spin
    m1 = samples[:, 0].mean()
    # mean > 0
    assert m1 > 0.0