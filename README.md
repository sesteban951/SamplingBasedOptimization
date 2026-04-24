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
- `srb/`: Contains the scripts for Single Rigid Body (SRB) non-linear trajectory optimization.
- `models/`: Contains various dynamical system models described by an XML file. You can view a model using `visualize_model_mujoco.py`.
- `utils/`: Contains utility functions and classes used throughout the codebase. For example, parallel simulation, optimization methods, and splines.
- `results/`: Directory to store results from experiments. To view results, run `view_results.py` which will load the data,  visualize it, and plot. Just change the `experiment` and `xml path` at the top of the file to view different results. Similarly, you can use `view_results_srb.py` to view results from the SRB optimization.
- `logs/`: Directory to store tensorboard logs from experiments. View with `tensorboard --logdir=./logs`.

## Usage
All scripts are run from the root directory. The main scripts are located in the `scripts/` directory.

If you can't run a particular script, try running it as a module. For example, for `scripts/cartpole/cartpole_cem.py`, use:
```bash
python -m scripts.cartpole.cartpole_cem
```
instead of:
```bash
python scripts/cartpole/cartpole_cem.py
```


srb stuff: 
```bash
python -m srb.srb_jump
python -m results.view_srb_results srb_jump
```

## Viewing Results
### Optimization Results
Optimization results are saved in `results/`. To view, run `view_results.py` which will load the data, visualize it, and plot. Just change the `experiment` and `xml_path` at the top of the file to view different results. Similarly, you can use `view_results_srb.py` to view results from the SRB optimization.

### Tensorboard Logs
To view tensorboard logs for optimization progress, run:
```bash
tensorboard --logdir=./logs
```
and open the provided URL in your browser (e.g., `http://localhost:6006`). To enable logging for a task, pass a `Logger_Config` to the optimizer.
