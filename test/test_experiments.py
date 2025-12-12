from src.experiments import run_gibbs, run_mh_parameter_inference

def test_run_gibbs_smoke():
    u_star, samples = run_gibbs(random_state=0)
    assert samples.ndim == 2
    assert samples.shape[1] == u_star.shape[0]

def test_run_mh_parameter_inference_smoke():
    u_star, u_est, samples_u = run_mh_parameter_inference(random_state=0)
    assert u_est.shape == u_star.shape