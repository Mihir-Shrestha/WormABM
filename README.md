# C. elegans Agent-Based Model (Main Branch)

## Overview
This branch contains the baseline model with bacteria diffusion and logistic
growth solved using Fourier-space Strang splitting. Worms follow a
run-and-tumble style motion on a 2D grid with a single initial bacteria patch.

## Requirements
- Python 3.x
- NumPy 2.4.1
- OpenCV 4.13.0
- Matplotlib 3.10.8
- h5py 3.15.1

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start
Run a default simulation:
```bash
python run_simulation.py
```

## Outputs
Results are saved under experiments/ with:
- the config file used for that run (.cfg)
- time-series data (.h5)

## Model Notes
- Diffusion and growth are updated with exact Fourier steps and analytic
	logistic growth (Strang splitting) in modules/Environment.py.
- Initial bacteria patch is centered at the origin.
- The parameter sweep is driven by config/config_src.py.

## Movie Generation
Create a visualization after running a simulation:
```bash
python make_movie.py -p <experiment_folder_name> -r 5 -s 1
```

Flags:
- -p / --path: experiment folder name inside experiments/
- -r / --fps: frames per second (default: 5)
- -s / --stepsize: frame stride (default: 1)

Note: stepsize controls the number of frames by skipping timesteps.