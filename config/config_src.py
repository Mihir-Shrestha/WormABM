config_opts = {
    # Simulation parameters
    "verbose"     : True,
    "random_seed" : 42,

    # Environment parameters
    "x_min" : -1.5,
    "x_max" : 1.5,
    "dx" : 0.01,
    "t_min" : 0,
    "t_max" : 0.0025,
    "dt" : 0.000025,

    # Worm parameters
    "num_worms" : 1,
    "worm_step_size" : 0.1,
    "worm_turn_noise" : 0.2,
    "worm_mean_run_duration" : 3,
    "worm_mean_tumble_duration" : 2,
    "bacteria_enabled" : True,
    "bacteria_drop_interval" : 5,
    "bacteria_amount" : 1,

    # Other parameters
    "measurements_on" : True,
}


# reference calculations:
# arena_size = x_max - x_min  # 1.5 - (-1.5) = 3.0
# grid_size = (x_max - x_min) / dx + 1  # 3.0 / 0.01 + 1 = 301
# total_duration = t_max - t_min  # 0.125 - 0 = 0.125
# num_timesteps = total_duration / dt  # 0.125 / 0.005 = 25