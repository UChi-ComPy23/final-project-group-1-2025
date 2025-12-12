def frob(A, B):
    return np.linalg.norm(A - B, ord='fro')

def compute_errors(u_star):
    C_true = estimate_true_correlations(u_star, y_init, n_samples=200000, burn_in=50000)
    C_hat = estimate_gibbs_correlations(u_star, y_init, n_samples=2000, burn_in=5000)
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
    u_err = frob(u_hat, u_star)

    return gibbs_err, u_err

print("Model Comparison:\n")
for name, U in u_variants.items():
    gibbs_err, u_err = compute_errors(U)
    print(f"{name.upper()} MODEL:")
    print(f"  Gibbs Error:            {gibbs_err:.4f}")
    print(f"  u-Estimation Error:     {u_err:.4f}\n")
