def frob(A, B):
    return np.linalg.norm(A - B, ord='fro')

def compute_errors(u_star, name):
    samples_true = ising_gibbs_sampler(
        u_star, y_init,
        n_samples=200000,
        burn_in=50000
    )
    C_true = np.corrcoef(samples_true.T)

    samples_hat = ising_gibbs_sampler(
        u_star, y_init,
        n_samples=2000,
        burn_in=5000
    )
    C_hat = np.corrcoef(samples_hat.T)

    gibbs_err = frob(C_hat, C_true)

    true_vec = vectorize_u(u_star)
    samples_u = metropolis_hastings(
        log_target=log_posterior(samples),
        initial_state=np.zeros_like(true_vec),
        n_samples=1000,
        burn_in=10000
    )

    u_hat_vec = np.mean(samples_u, axis=0)
    u_hat = unvectorize_u(u_hat_vec)

    print(f"{name} MODEL:")
    print(f"  Gibbs Error: {gibbs_err:.4f}")
    print(f"  u-Estimation Error:{u_err:.4f}\n")


for name, U in u_variants.items():
    compute_errors(U, name.upper())
