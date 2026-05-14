# C. elegans Agent-Based Model

## Overview
This repository tracks the full semester development of a C. elegans
agent-based model. The history is linear and can be checked out by branch or
commit to reproduce each iteration.

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

## Iterations and How to Access Them
Use the branch that matches the model you want, or check out a specific commit
from the linear history.

### Method 1 (Baseline Fourier + Logistic Growth)
- Fourier-space Strang splitting for diffusion with analytic logistic growth.
- Single initial bacteria patch, baseline run-and-tumble style motion.

Check out:
```bash
git checkout main
git log --oneline --decorate --graph
```

### Method 2 (Patch ODEs + Feeding/Deposition)
- Multiple bacteria patches with logistic ODEs for area and density.
- Feeding depends on patch properties and effective worms on patch.
- Includes diagnostics and feeding verification tools.

Check out:
```bash
git checkout method2
```

### Method 3 (Simplified Two-Patch Model)
- Two static sources; deposited bacteria are non-growing and non-feeding.
- Designed for behavior-focused experiments and clean comparisons.

Check out:
```bash
git checkout method3
```

## Diagnostics (Method 2 and later)
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

## Report
The final report is included as WormABM_Report.pdf in the repo root.