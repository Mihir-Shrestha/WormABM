## python make_movie.py -p N10_dx0.05_dt0.01_kNone_R4.25_vmax2_vmin0.2_T6000 -s 100 -r 5

import numpy as np

config_opts = {
    # Simulation parameters
    "random_seed" : 42,
    "save_interval_minutes" : 1.0,

    # Environment parameters
    "x_min" : -4.25,
    "x_max" : 4.25,
    "t_min" : 0,
    "t_max" : 6000,
    "dx" : [0.05],
    "dt": [0.01],
    "R" : [4.25],

    # Two initial static sources (equidistant from origin)
    "source_density" : 1.0e8,
    # Temporary single-patch test: overlapping left/right sources at x=0.
    "source_x_offset" : 1,
    "source_radius" : 0.2,
    "source_boundary_k" : 0.0,
    "on_patch_density_epsilon" : 1e-12,

    # Normalisation used by movie scripts and speed/noise scaling helpers
    "K_rho" : 1.0e8,
    "movie_min_density_factor" : 0.0001,

    # Worm SDE motility parameters
    "num_worms" : 10,
    "v_max" : 2,
    "v_min" : 0.2,
    "alpha" : 4.0,
    "beta_b" : 4.0,
    "chi_theta" : 6.0,
    "dtheta" : np.pi/8,

    # Initial worm placement around origin, away from both sources
    "spawn_radius" : 0.5,
    "spawn_min_distance" : 0.1,
    "spawn_source_clearance" : 0.05,

    # Fed-state deposition (non-growing, non-food trail)
    "deposition_enabled" : 1,
    "deposit_patch_radius" : 0.05,
    "deposit_patch_radius_pixels" : 1,
    "deposit_cells_multiplier" : 1.0,
    "fed_deposit_delay_steps" : 100,
    "fed_deposit_interval_steps" : 1000,
    "fed_deposit_amount" : 35,
    "fed_deposit_jitter_radius" : 0.05,

    # Other parameters
    "measurements_on" : True,
}