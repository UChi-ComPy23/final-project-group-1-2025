# Simulation and Recovery of Ising Sparse Graphical Model

This repository implements simulation and inference methods for sparse Ising graphical models, including:

- Gibbs sampling for spin configurations  
- Metropolis–Hastings (MH) sampling for Bayesian estimation of the interaction matrix  
- Pseudo-likelihood MAP via Proximal Gradient Descent (PGD) 

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

To run the full project (Gibbs sampling + MH inference + PGD), simply execute:
```bash
python main.py
```
To run the Original vs Parallel simply execute:
```bash
python main_parallel.py
```

## Testing the Project

All tests are located in the `test/` directory and can be executed using **pytest**.

To run the entire test suite, simply run the following command from the project root directory:

```bash
pytest
```

## Running Different Examples

We implement a clean mechanism for loading examples: the example Ising models used in the project are defined separately saved as `*.txt` inside `examples`.

To run a different example at the moment, you just need to modify the `example_path` inside `main.py` or `main_parallel.py`.

After updating these, simply run:
```bash
python main.py
```
or
```bash
python main_parallel.py
```

