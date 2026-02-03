# *Caenorhabditis elegans* Agent-Based Model

## Overview:
Agent-based model for simulating *C. elegans* worm behavior and interactions.

## Requirements:
- Python 3.x
- NumPy 2.4.1
- OpenCV 4.13.0
- Matplotlib 3.10.8
- H5py 3.15.1

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Usage:
Run a simulation with default parameters:
```bash
python run_simulation.py
```

## Output:
Simulation results are saved in the `experiments/` folder with timestamped subfolders containing:
- Configuration file (.cfg)
- Time-series data (.h5)

## Video Visualization:
After running a simulation, generate a visualization video:
```bash
python make_movie.py -p <experiment_folder_name> -r 5 -s 1
```

**Command line parameters:**
- `-p` or `--path`: Path to experiment folder in `experiments/` directory
- `-r` or `--fps`: Frame rate for output movie (default: `5`)
- `-s` or `--stepsize`: Step size for plotting frames (default: `1`)

The output video will be saved in the experiment folder.