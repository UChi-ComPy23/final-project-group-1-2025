'''
from src.experiments import run_mh_parameter_inference, run_mh_parameter_inference_parallel

# demo example for checkpoint
u_star, u_est, samples_u = run_mh_parameter_inference(random_state=123)

u_star, u_est, samples_u = run_mh_parameter_inference_parallel(random_state=123)
'''
import numpy as np
import time
from src.experiments import (
    run_mh_parameter_inference,
    run_mh_parameter_inference_parallel
)

if __name__ == "__main__":

    print("\n=== Benchmark: MH Parameter Inference (Original vs Parallel) ===\n")

    t0 = time.perf_counter()
    u_star1, u_est1, samples_u1 = run_mh_parameter_inference(random_state=123)
    t1 = time.perf_counter()
    time_original = t1 - t0
    print(f"Original MH inference time : {time_original:.4f} seconds")

    t2 = time.perf_counter()
    u_star2, u_est2, samples_u2 = run_mh_parameter_inference_parallel(random_state=123)
    t3 = time.perf_counter()
    time_parallel = t3 - t2
    print(f"Parallel MH inference time : {time_parallel:.4f} seconds")

    if time_parallel > 0:
        speedup = time_original / time_parallel
        print(f"Speedup (original / parallel): {speedup:.2f}x\n")
    else:
        print("Parallel version time is too small to compute speedup safely.\n")