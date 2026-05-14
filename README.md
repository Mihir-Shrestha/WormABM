# C. elegans Agent-Based Model (Method2 Branch)

## Overview
This branch models feeding and deposition using multiple bacterial patches.
Patch area and density follow coupled logistic ODEs, and feeding depends on
total patch area, density, and the effective number of worms on patch.

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
- Multiple bacteria patches evolve by logistic ODEs for area and density
	(modules/Environment.py). The grid map is built from all active patches.
- Feeding uses global patch properties plus the effective number of worms on
	patch, and depletion is applied across patches.
- Worms follow an SDE motility model with density-dependent speed and noise.
- Key parameters and sweeps live in config/config_src.py.

## Diagnostics
Optional checks and plots:
```bash
python verify_worm_feeding.py
python plot_feeding_diagnostics.py
```

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