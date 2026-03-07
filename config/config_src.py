import numpy as np

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
    "boundary_k" : [10],   # cm^-1  boundary sharpness
                            # large (e.g. 100) = sharp ring
                            # small (e.g. 2)   = wide sensing range

    # Worm SDE motility parameters
    "num_worms" : 10,
    "v_max" : 0.02,              # speed at low bacterial density (exploring) (cm/min)
    "v_min" : 0.001,             # speed at high bacterial density (dwelling) (cm/min)
    "alpha" : 4.0,             # how strongly speed decreases with bacterial density (speed suppresion sensitivity)
    "beta_b" : 4.0,            # how strongly bacterial density suppresses speed (noise suppressioin sensitivity)
    "chi_theta" : 6.0,         # chemotactic turning sensitivity (rad)
    "D_theta" : (np.pi/8)**2 / 2.0,           # max rotational diffusion (at b=0) (rad^2/min)

    # "bacteria_enabled" : False,  (Not active yet)
    "bacteria_drop_interval" : 5,
    "bacteria_amount" : 0,

    # Other parameters
    "measurements_on" : True,
}

    # reference calculations:
    # arena_size  = x_max - x_min        = 3.0 cm
    # grid_size   = arena_size / dx + 1  = 301 x 301
    # K_A         = arena_size^2         = 9.0 cm^2
    # A_B_0       = pi * 0.5^2          = 0.7854 cm^2
    # total_time  = t_max - t_min        = 6000 min (~4.2 days)
    # num_steps   = total_time / dt      = 6000
    # D_theta     = (pi/8)^2 / 2        = ~0.0308 rad^2/min