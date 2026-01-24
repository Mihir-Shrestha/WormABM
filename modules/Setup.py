import os 
import argparse
import shutil
import numpy as np
from datetime import datetime

import modules.Environment as Environment
import modules.Worms as Worms
import modules.Keeper as Keeper

def config_options():
    # Parse command-line arguments and config file
    class LoadFromFile(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            with values as f:
                parser.parse_args(f.read().split(), namespace)
                setattr(namespace, 'config_file', values.name)

    # Instantiate parser
    parser = argparse.ArgumentParser()

    # Simulation parameters
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--measurements_on", type=bool, default=True)

    # Environment parameters
    parser.add_argument("--x_min", type=float, default=-1.5)
    parser.add_argument("--x_max", type=float, default=1.5)
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--t_min", type=float, default=0)
    parser.add_argument("--t_max", type=float, default=0.125)
    parser.add_argument("--dt", type=float, default=0.005)

    # Worm parameters
    parser.add_argument("--num_worms", type=int, default=1)
    parser.add_argument("--worm_step_size", type=float, default=0.1)
    parser.add_argument("--worm_turn_noise", type=float, default=0.2)
    parser.add_argument("--worm_mean_run_duration", type=float, default=3)
    parser.add_argument("--worm_mean_tumble_duration", type=float, default=2)

    # Config file
    parser.add_argument("--file", type=open, action=LoadFromFile)
    parser.add_argument("--base_dir", type=str, default="experiments")

    # Read arguments from parser
    args = parser.parse_args()

    return args


def directory(config):
    # Create experiment directory and copy config file
    # timestamp = datetime.now().strftime("%m-%d-%y_%H-%M-%S")

    if not hasattr(config, 'config_file'):
        cfg_name = 'test'
    else:
        cfg_name = config.config_file.split(os.path.sep)[-1].replace('.cfg', '')

    # Create folder name with parameters
    N = config.num_worms
    seed = config.random_seed
    # params_name = f"N{N}_seed{seed}_{timestamp}"
    params_name = f"N{N}_seed{seed}"
    model_dir = os.path.join(config.base_dir, params_name)

    os.makedirs(model_dir, exist_ok=True)

    # Copy config file to model dir
    if hasattr(config, 'config_file'):
        shutil.copyfile(config.config_file, os.path.join(model_dir, f"{cfg_name}.cfg"))

    return model_dir


def world_parameters(cfg, model_dir):
    # Organize parameters into dictionaries
    keeper_params = {
        "worm_path": os.path.join(model_dir, "worm_hist.h5"),
        "environment_path": os.path.join(model_dir, "envir_hist.h5"),
        "sleeping": not cfg.measurements_on,
    }

    environment_params = {
        "x_min": cfg.x_min,
        "x_max": cfg.x_max,
        "dx": cfg.dx,
        "t_min": cfg.t_min,
        "t_max": cfg.t_max,
        "dt": cfg.dt,
    }

    worm_params = {
        "worm_step_size": cfg.worm_step_size,
        "worm_turn_noise": cfg.worm_turn_noise,
        "worm_mean_run_duration": cfg.worm_mean_run_duration,
        "worm_mean_tumble_duration": cfg.worm_mean_tumble_duration,
    }

    world_params = {
        "keeper": keeper_params,
        "environment": environment_params,
        "worm": worm_params,
    }

    return world_params

def convert_index_to_xy(idx, idx_min=0, idx_max=300, xy_min=-1.5, xy_max=1.5):
    # Convert grid index to real-world coordinate
    xy = np.interp(idx, [idx_min, idx_max], [xy_min, xy_max])
    return xy

def generate_points_with_min_distance(num_worms, shape, min_dist):
    # Generate initial positions for multiple worms
    # Ensures minimum distance between starting positions

    # Handle edge case: single worm
    if num_worms <= 1:
        # Return center position
        center_x = shape[1] / 2
        center_y = shape[0] / 2
        return np.array([[center_x, center_y]])

    # Compute grid shape based on number of worms
    width_ratio = shape[1] / shape[0]
    num_y = int(np.sqrt(num_worms / width_ratio)) + 1
    num_x = int(num_worms / num_y) + 1

    # Create regularly spaced points
    x = np.linspace(0, shape[1], int(num_x))[1:-1]
    y = np.linspace(0, shape[0], int(num_y))[1:-1]
    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)

    # Compute spacing safely
    if len(x) > 1 and len(y) > 1:
        init_dist = np.min((x[1] - x[0], y[1] - y[0]))
    else:
        init_dist = np.min(shape)

    # Perturb points with random noise
    max_movement = (init_dist - min_dist) / 2
    noise = np.random.uniform(low=-max_movement,
                              high=max_movement,
                              size=(len(coords), 2))
    coords += noise
    return coords

def create_worms(coords, dim, cfg, worm_params):
    # Create worm objects with initial positions from grid coordinates
    np.random.shuffle(coords)
    worms = []
    
    for worm_i in range(cfg.num_worms):
        # Copy params for this worm
        worm_params_copy = worm_params
        worm_params_copy["num"] = worm_i
        
        # Create worm object
        worm = Worms.Worm(worm_params_copy)

        # Convert grid indices to real coordinates
        if cfg.num_worms > 1:
            # Only use grid placement if multiple worms
            worm_x_idx = coords[worm_i][0]
            worm_y_idx = coords[worm_i][1]
            worm_x = convert_index_to_xy(worm_x_idx, idx_min=0, idx_max=dim, 
                                        xy_min=cfg.x_min, xy_max=cfg.x_max)
            worm_y = convert_index_to_xy(worm_y_idx, idx_min=0, idx_max=dim, 
                                        xy_min=cfg.x_min, xy_max=cfg.x_max)
            worm.x = worm_x
            worm.y = worm_y
        else:
            # Single worm: start at origin
            worm.x = 0.0
            worm.y = 0.0

        worms.append(worm)
    
    return worms

def world_objects(cfg_options, world_params):
    # Instantiate environment and worm objects
    
    # Calculate grid dimension
    dim = len(np.arange(cfg_options.x_min, cfg_options.x_max, cfg_options.dx)) + 1

    # Create environment and keeper objects
    environment = Environment.Environment(world_params["environment"])
    keeper = Keeper.Keeper(world_params["keeper"])

    # Create worm(s)
    coords = generate_points_with_min_distance(
        cfg_options.num_worms * 2,
        shape=(dim, dim),
        min_dist=10
    )
    worms = create_worms(coords, dim, cfg_options, world_params["worm"])

    world_objs = {
        "environment": environment,
        "worms": worms,
        "keeper": keeper,
    }

    return world_objs