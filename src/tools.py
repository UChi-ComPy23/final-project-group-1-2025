# src/tools.py
import matplotlib.pyplot as plt

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
