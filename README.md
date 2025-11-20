# Simulation and Recovery of Ising Sparse Graphical Model

This repository implements simulation and inference methods for sparse Ising graphical models, including:

- Gibbs sampling for spin configurations  
- Metropolis–Hastings (MH) sampling for Bayesian estimation of the interaction matrix  
- (To be completed) Pseudo-likelihood MAP via Proximal Gradient Descent (PGD) 

Code is modularized under `src/` and fully tested with `pytest`.

## Directory Structure
```text
src/
├── __init__.py
├── utils.py              # vectorize_u / unvectorize_u
├── ising_model.py        # energy, partition, conditional prob
├── sampling.py           # Gibbs sampler
├── inference.py          # MH sampler + log posterior
├── experiments.py
└── tools.py

test/
├── conftest.py
├── test_utils.py
├── test_ising_model.py
├── test_sampling.py
├── test_inference.py
└── test_experiments.py
```

## Run the Program

To run the full project (Gibbs sampling + MH inference), simply execute:
```bash
python main.py
```

## Testing the Project

All tests are located in the `test/` directory and can be executed using **pytest**.

To run the entire test suite, simply run the following command from the project root directory:

```bash
pytest
```

## Running Different Examples

Currently, the example Ising models used in the project (such as `u_star` and
`y_init`) are defined directly inside `src/experiments.py`.

To run a different example at the moment, you must manually modify these lines
inside `run_gibbs_demo()`:

```python
# Inside src/experiments.py
u_star = <your interaction matrix>
y_init = <your initial spin configuration>
```
After updating these values, simply run:
```bash
python main.py
```

In the final project, we will implement a clean mechanism for loading examples:

Add an examples/ directory containing predefined Ising model configurations.

Implement functions to read these examples directly.




## Final Work (Upcoming)

The following will be implemented for the final report:

- **Pseudo-likelihood MAP estimator using Proximal Gradient Descent (PGD)**
- **Symmetry & zero-diagonal constraints**
- **Accuracy, runtime, and sparsity recovery comparison vs MH**