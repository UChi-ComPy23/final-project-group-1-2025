# src/tools.py
import matplotlib.pyplot as plt
import numpy as np
def plot_data(dats, legs, xlab, ylab, title=None):
    """
    Plot multiple 1D data series on the same figure.
    """
    for d in dats:
        plt.plot(d)

    if legs is not None:
        plt.legend(legs)

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.show()

def load_matrix_from_txt(u_txt_path: str) -> np.ndarray:
    """
    Load a square interaction matrix u from a txt file.
    """
    u_star = np.loadtxt(u_txt_path)
    if u_star.ndim != 2 or u_star.shape[0] != u_star.shape[1]:
        raise ValueError(
            f"Matrix loaded from {u_txt_path!r} must be square (p x p), "
            f"got shape {u_star.shape}."
        )
    return u_star