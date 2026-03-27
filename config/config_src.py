import numpy as np

config_opts = {
    # Simulation parameters
    # "verbose"     : False,
    "random_seed" : 42,

    # Environment parameters
    "x_min" : -4.25,          # cm
    "x_max" : 4.25,           # cm
    "t_min" : 0,            # min
    "t_max" : 1000,         # min
    "dx" : [0.05],          # cm
    "dt": [0.01],              # min
    "R" : [4.25],              # cm ( plate radius used for K_A)

    # Bacteria ODE parameters (Eqs. S1 & S2, Table S1)
    "g_A"   : 1.13e-4,      # area growth rate (min^-1)
    "g_rho" : 1.07e-3,      # density growth rate (min^-1)
    "K_rho" : 4.58e8,       # density carrying capacity (cells/cm^2)
    # "K_A" : [np.pi*2*2, np.pi*25*25],
    "A_B_0" : np.pi * 0.5 * 0.5,            # initial bacteria patch area = pi * 0.5^2 cm^2
    "rho_0" : 1.27e8,                       # initial bacteria density (cells/cm^2)
    "boundary_k" : [10],                    # Boundary sharpness (cm^-1). Large (e.g. 100) = sharp ring, small (e.g. 2) = wide sensing range. Negative value means no boundary.
    "feed_c" : [2.5e-9],                       # Amplitude of Gaussian food source (cells/cm^2). 0 means no food source.
    "feed_sigma" : [1000],                   # Width of Gaussian food source (

    # Worm SDE motility parameters
    "num_worms" : 10,
    "v_max" : 2,              # speed at low bacterial density (exploring) (cm/min)
    "v_min" : 0.2,             # speed at high bacterial density (dwelling) (cm/min)
    "alpha" : 4.0,             # how strongly speed decreases with bacterial density (speed suppresion sensitivity)
    "beta_b" : 4.0,            # how strongly bacterial density suppresses speed (noise suppressioin sensitivity)
    "chi_theta" : 6.0,         # chemotactic turning sensitivity (rad)
    "dtheta" : np.pi/8,           # max rotational diffusion (at b=0) (rad^2/min)

    # "bacteria_enabled" : False,  (Not active yet)
    "bacteria_drop_interval" : 5,
    "bacteria_amount" : 0,

    # Other parameters
    "measurements_on" : True,
}