import numpy as np
from src.utils import vectorize_u, unvectorize_u

def test_vectorize_unvectorize_roundtrip():
    p = 5
    u = np.zeros((p, p))
    u[0, 1] = 0.5
    u[1, 3] = -0.2
    u[2, 4] = 1.0

    u[1, 0] = 0.5
    u[3, 1] = -0.2
    u[4, 2] = 1.0

    u_vec = vectorize_u(u)
    u_rec = unvectorize_u(u_vec, p=p)

    assert u_rec.shape == (p, p)
    # symmetry
    assert np.allclose(u_rec, u_rec.T)
    # diagonal 0
    assert np.allclose(np.diag(u_rec), 0.0)
    # same as the original matrix
    assert np.allclose(u_rec, u)
