# src/experiments.py
import numpy as np
import matplotlib.pyplot as plt

from .sampling import ising_gibbs_sampler
from .utils import vectorize_u, unvectorize_u
from .inference import make_log_posterior, metropolis_hastings
from .tools import plot_data


def run_gibbs_demo(random_state: int = 1):
    """
    Reproduce the small p=5 Gibbs sampling example from the notebook.
    """
    np.random.seed(random_state)

    u_star = np.array([
        [0,   0.5, 0,   0.5, 0],
        [0.5, 0,   0.5, 0,   0.5],
        [0,   0.5, 0,   0.5, 0],
        [0.5, 0,   0.5, 0,   0],
        [0,   0.5, 0,   0,   0]
    ])

    y_init = np.array([1, -1, -1, 1, 1])
    p = len(y_init)

    samples = ising_gibbs_sampler(u_star, y_init,
                                  n_samples=1000,
                                  burn_in=5000,
                                  T=1.0,
                                  random_state=random_state)

    sample_mean = np.mean(samples, axis=0)
    sample_var = np.var(samples, axis=0)

    print("\nSample Statistics:")
    print(f"{'Variable':<10} {'Mean':<10} {'Variance':<10}")
    for i in range(p):
        print(f"{i+1:<10} {sample_mean[i]:<10.3f} {sample_var[i]:<10.3f}")

    # Correlation matrix
    corr = np.corrcoef(samples.T)
    print("\nCorrelation Matrix:")
    print(corr)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, label='Correlation')
    plt.title('Correlation Matrix of Ising Model Variables')
    plt.xlabel('Variable Index')
    plt.ylabel('Variable Index')

    for i in range(p):
        for j in range(p):
            plt.text(j, i, f'{corr[i, j]:.2f}',
                     ha='center', va='center', fontsize=10)
    plt.tight_layout()
    plt.show()

    return u_star, samples


def run_mh_parameter_inference(random_state: int = 100):
    """
    Reproduce the MH-on-u example from the notebook (p=5).
    """
    # First generate data
    u_star, samples = run_gibbs_demo(random_state=random_state)
    p = u_star.shape[0]

    # Build log posterior
    lamb = 1.0  # Laplace prior parameter (you can adjust)
    log_post = make_log_posterior(samples, p=p, lamb=lamb, T=1.0)

    # Initial state: small positive values
    initial_u_vec = np.zeros(int(p * (p - 1) / 2)) + 0.1

    # Run MH
    samples_u = metropolis_hastings(
        log_target=log_post,
        initial_state=initial_u_vec,
        proposal_std=0.1,
        n_samples=1000,
        burn_in=10_000,
        random_state=random_state
    )

    # Analysis
    true_u_vec = vectorize_u(u_star)
    posterior_means = np.mean(samples_u, axis=0)
    posterior_vars = np.var(samples_u, axis=0)

    print("\nCOMPARISON: TRUE vs ESTIMATED")
    param_labels = []
    for i in range(p):
        for j in range(i + 1, p):
            param_labels.append(f"u_{i+1}{j+1}")

    print(f"{'Parameter':<10} {'True':<10} {'Sample_Mean':<12} "
          f"{'Error':<10} {'Sample_Variance':<10}")
    for idx, label in enumerate(param_labels):
        true_val = true_u_vec[idx]
        est_mean = posterior_means[idx]
        est_var = posterior_vars[idx]
        error = abs(true_val - est_mean)
        print(f"{label:<10} {true_val:<10.3f} {est_mean:<11.3f}  "
              f"{error:<10.3f} {est_var:<10.6f}")

    estimated_u_mat = unvectorize_u(posterior_means, p=p)
    print("\nTRUE u*:")
    print(np.round(u_star, 3))
    print("\nESTIMATED u (posterior mean):")
    print(np.round(estimated_u_mat, 3))

    return u_star, estimated_u_mat, samples_u
