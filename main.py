from src.experiments import run_pgd_parameter_inference, run_mh_parameter_inference, run_mh_parameter_inference_parallel

example_path = "examples/u_example_8x8.txt"

if __name__ == "__main__":

    u_star1, u_est1, samples_u1 = run_mh_parameter_inference(random_state=123, u_txt_path=example_path)
    u_star1, u_est1 = run_pgd_parameter_inference(random_state=123, u_txt_path=example_path)