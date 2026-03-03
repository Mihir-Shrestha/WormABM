config_opts = {
    # Simulation parameters
    # "verbose"     : False,
    "random_seed" : 42,

    # Environment parameters
    "x_min" : -1.5,
    "x_max" : 1.5,
    "t_min" : 0,
    "t_max" : 6000,
    "dx" : [0.01],               # dx: spatial resolution (smaller = more accurate but slower)
    "dt": [1],                   # dt: time step (smaller = more accurate but slower)

    # Bacteria ODE parameters (Eqs. S1 & S2, Table S1)
    "g_A"   : 1.13e-4,           # area growth rate (min^-1)
    "g_rho" : 1.07e-3,           # density growth rate (min^-1)
    "K_rho" : 4.58e8,            # density carrying capacity (cells/cm^2)
    # K_A is computed automatically in Environment as (x_max - x_min)^2 = 9 cm^2
    "A_B_0" : 0.7854,            # initial patch area = pi * 0.5^2 cm^2
    "rho_0" : 1.27e8,            # initial density (cells/cm^2)

    # Worm parameters
    "num_worms" : 1,
    "worm_step_size" : 0.1,
    "worm_turn_noise" : 0.2,
    "worm_mean_run_duration" : 3,
    "worm_mean_tumble_duration" : 2,
    # "bacteria_enabled" : False,
    "bacteria_drop_interval" : 5,
    "bacteria_amount" : 0,

    # Other parameters
    "measurements_on" : True,
}


# reference calculations:
# arena_size = x_max - x_min  # 1.5 - (-1.5) = 3.0
# grid_size = (x_max - x_min) / dx + 1  # 3.0 / 0.01 + 1 = 301
# total_duration = t_max - t_min  # 0.125 - 0 = 0.125
# num_timesteps = total_duration / dt  # 0.125 / 0.005 = 25