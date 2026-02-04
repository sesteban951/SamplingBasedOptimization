# Sampling-Based Optimization
Repo for testing sampling-based optimization methods.

## Installation
Use `conda` and install via:
```bash
conda env create -f environment.yml
conda activate env_sbo
```

## Directories
- `scripts/`: Contains the main scripts for running experiments and testing optimization methods.
- `models/`: Contains various dynamical system models described by an XML file. You can view a model using `visualize_model_mujoco.py`.
- `utils/`: Contains utility functions and classes used throughout the codebase. For example, parallel simulation, optimization methods, and splines.
- `results/`: Directory to store results from experiments.