import numpy as np
from src.ising_model import pair_energy, partition, conditional_prob_spin_plus_one

def test_partition_p2():
    # p=2 only one interaction u_12 = J
    J=0.7
    u=np.array([[0.,J],[J,0.]])
    Z_manual=2*np.exp(J)+2*np.exp(-J)
    Z_code = partition(u, T=1.0)
    assert np.allclose(Z_code, Z_manual)

def test_conditional_prob_range():
    u=np.array([[0.,1.],[1.,0.]])
    y = np.array([1, 1], dtype=int)
    p_plus = conditional_prob_spin_plus_one(y, r=0, u=u, T=1.0)
    assert 0.0 < p_plus < 1.0 and p_plus > 0.5
