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
    # parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--measurements_on", type=bool, default=True)
    parser.add_argument("--save_interval_minutes", type=float, default=0.0)

    # Environment parameters
    parser.add_argument("--x_min", type=float)
    parser.add_argument("--x_max", type=float)
    parser.add_argument("--dx", type=float)
    parser.add_argument("--t_min", type=float)
    parser.add_argument("--t_max", type=float)
    parser.add_argument("--dt", type=float)\
    
    # ODE parameters (defaults from Table S1)
    parser.add_argument("--g_A", type=float)           # area growth rate (min^-1)
    parser.add_argument("--g_rho", type=float)         # density growth rate (min^-1)
    # parser.add_argument("--K_A", type=float, default=np.pi*2*2)              # area carrying capacity (cm^2)
    parser.add_argument("--R", type=float)              # plate radius used for K_A and naming only
    parser.add_argument("--K_rho", type=float)         # density carrying capacity (cells/cm^2)
    parser.add_argument("--A_B_0", type=float)  # initial patch area = pi * 0.5^2 cm^2
    parser.add_argument("--rho_0", type=float)          # initial density (cells/cm^2)
    parser.add_argument("--boundary_k", type=float)      # Boundary sharpness (cm^-1). Large (e.g. 100) = sharp ring, small (e.g. 2) = wide sensing range
    parser.add_argument("--feed_c", type=float)          # Amplitude of Gaussian food source (cells/cm^2). 0 means no food source.
    parser.add_argument("--feed_sigma", type=float)      # Width of Gaussian food source (
    parser.add_argument("--patch_bnorm_threshold", type=float, default=0.01)
    parser.add_argument("--local_feed_b_half", type=float, default=0.1)
    parser.add_argument("--local_feed_hill_n", type=float, default=2.0)
    parser.add_argument("--movie_min_density_factor", type=float, default=0.1)
    parser.add_argument("--deposit_patch_radius", type=float, default=0.03)
    parser.add_argument("--deposit_patch_radius_pixels", type=int, default=1)
    parser.add_argument("--deposit_cells_multiplier", type=float, default=20.0)
    parser.add_argument("--deposit_merge_distance", type=float, default=0.03)

    # Worm parameters
    parser.add_argument("--num_worms", type=int)
    parser.add_argument("--v_max", type=float)                            # speed at low bacterial density (exploring)
    parser.add_argument("--v_min", type=float)                           # speed at high bacterial density (dwelling)
    parser.add_argument("--alpha", type=float)                             # how strongly speed decreases with bacterial density
    parser.add_argument("--beta_b", type=float)                            # how strongly bacterial density suppresses speed
    parser.add_argument("--chi_theta", type=float)                         # chemotactic turning sensitivity
    parser.add_argument("--dtheta", type=float)           # max rotational diffusion (at b=0)

    # Bacteria drop parameters
    # parser.add_argument("--bacteria_enabled", type=bool, default=False)
    parser.add_argument("--bacteria_enabled", action='store_true', help="Enable bacteria dropping")
    parser.add_argument("--bacteria_drop_interval", type=int, default=5)
    parser.add_argument("--bacteria_amount", type=float, default=1.0)
    parser.add_argument("--deposition_enabled", type=int, default=1, help="Enable deposition from eaten cells (1/0)")
    parser.add_argument("--deposition_fraction", type=float, default=1.0)
    parser.add_argument("--deposition_interval_steps", type=int, default=5)
    parser.add_argument("--deposition_max_per_event", type=float, default=0.0)
    parser.add_argument("--surface_shedding_enabled", type=int, default=1)
    parser.add_argument("--surface_pickup_rate", type=float, default=2.0)
    parser.add_argument("--surface_carry_capacity", type=float, default=40.0)
    parser.add_argument("--surface_shed_mean_steps", type=int, default=300)
    parser.add_argument("--surface_shed_fraction_min", type=float, default=0.02)
    parser.add_argument("--surface_shed_fraction_max", type=float, default=0.15)
    parser.add_argument("--surface_shed_max_cells", type=float, default=0.0)
    parser.add_argument("--surface_drop_jitter_radius", type=float, default=0.08)
    parser.add_argument("--gut_drop_enabled", type=int, default=1)
    parser.add_argument("--gut_drop_trigger_cells", type=float, default=35.0)
    parser.add_argument("--gut_drop_conversion_fraction", type=float, default=0.25)
    parser.add_argument("--gut_drop_delay_steps", type=int, default=1000)
    parser.add_argument("--gut_drop_jitter_radius", type=float, default=0.05)
    parser.add_argument("--gut_drop_max_events_per_step", type=int, default=5)

    # Config file
    parser.add_argument("--file", type=open, action=LoadFromFile)
    parser.add_argument("--base_dir", type=str, default="experiments")

    # Read arguments from parser
    args = parser.parse_args()

    return args


def directory(config):
    # Create experiment directory and copy config file
    if not hasattr(config, 'config_file'):
        cfg_name = 'test'
    else:
        cfg_name = config.config_file.split(os.path.sep)[-1].replace('.cfg', '')

    # Create folder name with parameters
    N = config.num_worms
    # seed = config.random_seed
    dx_value = f"{config.dx:.3f}".rstrip('0').rstrip('.')
    dt_value = f"{config.dt:.10f}".rstrip('0').rstrip('.')
    R_value = f"{config.R:.2f}".rstrip('0').rstrip('.')
    vmax_value = f"{config.v_max:.2f}".rstrip('0').rstrip('.')
    vmin_value = f"{config.v_min:.2f}".rstrip('0').rstrip('.')
    total_time = f"{config.t_max:.1f}".rstrip('0').rstrip('.')
    params_name = f"N{N}_dx{dx_value}_dt{dt_value}_k{config.boundary_k}_R{R_value}_vmax{vmax_value}_vmin{vmin_value}_T{total_time}"
    model_dir = os.path.join(config.base_dir, params_name)

    os.makedirs(model_dir, exist_ok=True)

    # Copy config file to model dir
    if hasattr(config, 'config_file'):
        shutil.copyfile(config.config_file, os.path.join(model_dir, f"{cfg_name}.cfg"))

    return model_dir


def world_parameters(cfg, model_dir):
    # Organize parameters into dictionaries
    if cfg.save_interval_minutes > 0.0 and cfg.dt > 0.0:
        save_interval_steps = max(int(round(cfg.save_interval_minutes / cfg.dt)), 1)
    else:
        save_interval_steps = 1

    keeper_params = {
        "worm_path": os.path.join(model_dir, "worm_hist.h5"),
        "environment_path": os.path.join(model_dir, "environment_hist.h5"),
        "sleeping": not cfg.measurements_on,
        "save_interval_steps": save_interval_steps,
        "save_interval_minutes": cfg.save_interval_minutes,
    }

    environment_params = {
        "x_min": cfg.x_min,
        "x_max": cfg.x_max,
        "dx": cfg.dx,
        "t_min": cfg.t_min,
        "t_max": cfg.t_max,
        "dt": cfg.dt,
        "g_A": cfg.g_A,
        "g_rho": cfg.g_rho,
        # "K_A": cfg.K_A,
        "R": cfg.R,
        "K_rho": cfg.K_rho,
        "A_B_0": cfg.A_B_0,
        "rho_0": cfg.rho_0,
        "boundary_k": cfg.boundary_k,
        "num_worms": cfg.num_worms,
        "feed_c": cfg.feed_c,
        "feed_sigma": cfg.feed_sigma,
        "patch_bnorm_threshold": cfg.patch_bnorm_threshold,
        "local_feed_b_half": cfg.local_feed_b_half,
        "local_feed_hill_n": cfg.local_feed_hill_n,
        "movie_min_density_factor": cfg.movie_min_density_factor,
        "deposit_patch_radius": cfg.deposit_patch_radius,
        "deposit_patch_radius_pixels": cfg.deposit_patch_radius_pixels,
        "deposit_cells_multiplier": cfg.deposit_cells_multiplier,
        "deposit_merge_distance": cfg.deposit_merge_distance,
    }

    worm_params = {
        "num_worms": cfg.num_worms,
        # SDE motility parameters
        "v_max": cfg.v_max,
        "v_min": cfg.v_min,
        "alpha": cfg.alpha,
        "beta_b": cfg.beta_b,
        "chi_theta": cfg.chi_theta,
        "dtheta": cfg.dtheta,
        # Bacteria dropping parameters
        "bacteria_enabled": cfg.bacteria_enabled,
        "bacteria_drop_interval": cfg.bacteria_drop_interval,
        "bacteria_amount": cfg.bacteria_amount,
        "deposition_enabled": bool(cfg.deposition_enabled),
        "deposition_fraction": cfg.deposition_fraction,
        "deposition_interval_steps": cfg.deposition_interval_steps,
        "deposition_max_per_event": cfg.deposition_max_per_event,
        "surface_shedding_enabled": bool(cfg.surface_shedding_enabled),
        "surface_pickup_rate": cfg.surface_pickup_rate,
        "surface_carry_capacity": cfg.surface_carry_capacity,
        "surface_shed_mean_steps": cfg.surface_shed_mean_steps,
        "surface_shed_fraction_min": cfg.surface_shed_fraction_min,
        "surface_shed_fraction_max": cfg.surface_shed_fraction_max,
        "surface_shed_max_cells": cfg.surface_shed_max_cells,
        "surface_drop_jitter_radius": cfg.surface_drop_jitter_radius,
        "gut_drop_enabled": bool(cfg.gut_drop_enabled),
        "gut_drop_trigger_cells": cfg.gut_drop_trigger_cells,
        "gut_drop_conversion_fraction": cfg.gut_drop_conversion_fraction,
        "gut_drop_delay_steps": cfg.gut_drop_delay_steps,
        "gut_drop_jitter_radius": cfg.gut_drop_jitter_radius,
        "gut_drop_max_events_per_step": cfg.gut_drop_max_events_per_step,
    }

    world_params = {
        "keeper": keeper_params,
        "environment": environment_params,
        "worm": worm_params,
    }

    return world_params

def convert_index_to_xy(idx, idx_min, idx_max, xy_min, xy_max):
    # Convert grid index to real-world coordinate
    xy = np.interp(idx, [idx_min, idx_max], [xy_min, xy_max])
    return xy

def generate_points_with_min_distance(num_worms, R, min_dist):
    """
    Generate initial worm positions uniformly inside circular arena of radius R,
    with a minimum distance between worms.
    Works in real (cm) coordinates directly — no index space needed.
    """
    if num_worms <= 1:
        return np.array([[0.0, 0.0]])

    positions = []
    max_attempts = 10000
    attempts = 0

    while len(positions) < num_worms and attempts < max_attempts:
        # uniform random point inside circle
        x = np.random.uniform(-R, R)
        y = np.random.uniform(-R, R)
        if x**2 + y**2 >= R**2:
            attempts += 1
            continue

        # check min distance from already placed worms
        too_close = False
        for px, py in positions:
            if np.sqrt((x - px)**2 + (y - py)**2) < min_dist:
                too_close = True
                break

        if not too_close:
            positions.append([x, y])
        attempts += 1

    if len(positions) < num_worms:
        raise ValueError(
            f"Could not place {num_worms} worms inside R={R} "
            f"with min_dist={min_dist}. Try reducing min_dist."
        )

    return np.array(positions)

def create_worms(positions, cfg, worm_params):
    """
    positions: real (cm) coordinates, shape (num_worms, 2)
    """
    worms = []

    for worm_i in range(cfg.num_worms):
        worm_params_copy = worm_params.copy()
        worm_params_copy["num"] = worm_i
        worm_params_copy["R"] = cfg.R   # pass R so boundary check works

        worm = Worms.Worm(worm_params_copy)

        # override position set by __init_position
        worm.x = float(positions[worm_i][0])
        worm.y = float(positions[worm_i][1])
        worm.theta = np.random.uniform(-np.pi, np.pi)

        worms.append(worm)

    return worms

def world_objects(cfg_options, world_params):
    environment = Environment.Environment(world_params["environment"])
    keeper = Keeper.Keeper(world_params["keeper"])

    # generate positions directly in cm, inside circle
    coords = generate_points_with_min_distance(
        num_worms=cfg_options.num_worms,
        R=cfg_options.R,
        min_dist=1.0       # minimum 1 cm between worms at start
    )

    worms = create_worms(coords, cfg_options, world_params["worm"])

    world_objs = {
        "environment": environment,
        "worms": worms,
        "keeper": keeper,
    }

    return world_objs