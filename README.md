# C. elegans Agent-Based Model (Method3 Branch)

## Overview
This branch implements a simplified two-patch environment for behavioral
experiments. The two sources are static, and deposited bacteria are non-growing
trails that are not consumed or used for chemotaxis.

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
- Two fixed sources are placed at +/- source_x_offset with a hard or soft
	boundary controlled by source_boundary_k.
- Deposited bacteria are written to the map but do not grow, diffuse, or feed
	worms; they are not used in chemotactic gradients.
- Worms use a density-dependent SDE motility model with tuned speed/noise.
- Key parameters live in config/config_src.py.

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