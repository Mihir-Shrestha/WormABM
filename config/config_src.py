import numpy as np

config_opts = {
    # Simulation parameters
    # "verbose"     : False,
    "random_seed" : 42,  # RNG seed for reproducibility (unitless)
    "save_interval_minutes" : 60.0,  # write measurements every N simulation minutes; <=0 writes every step (min)

    # Environment parameters
    "x_min" : -4.25,  # domain minimum x coordinate (cm)
    "x_max" : 4.25,  # domain maximum x coordinate (cm)
    "t_min" : 0,  # simulation start time (min)
    "t_max" : 6000,  # simulation end time (min)
    "dx" : [0.05],  # spatial grid spacing (cm)
    "dt": [0.01],  # time step (min)
    "R" : [4.25],  # arena radius used for geometry and K_A derivation (cm)

    # Bacteria ODE parameters (Eqs. S1 & S2, Table S1)
    "g_A"   : 1.13e-4,  # logistic growth rate of patch area A_B (min^-1)
    "g_rho" : 1.07e-3,  # logistic growth rate of density rho (min^-1)
    "K_rho" : 4.58e6,  # carrying capacity for bacteria density (cells/cm^2)
    # "K_A" : [np.pi*2*2, np.pi*25*25],
    "A_B_0" : np.pi * 0.5 * 0.5,  # initial patch area (cm^2)
    "rho_0" : 1.27e6,  # initial patch density (cells/cm^2)
    "boundary_k" : [0],  # boundary sharpness of patch profile; larger => sharper edge (cm^-1)
    "feed_c" : [2.5e-9],  # feeding-response scale in F(A_B, rho, R) (approximately 1/cells)
    "feed_sigma" : [1000],  # shape/scale factor in psi term of global feeding function (unitless)
    "feeding_cells_per_worm" : 70.0,  # paper a term: nominal feeding per worm (cells/min/worm)
    "patch_bnorm_threshold" : 0.01,  # legacy threshold on local normalized density (currently unused)
    "on_patch_density_epsilon" : 1e-12,  # classify on-source when local absolute density is above this tiny floor (cells/cm^2)
    "local_feed_b_half" : 0.1,  # legacy local-feeding parameter (currently unused)
    "local_feed_hill_n" : 2.0,  # legacy local-feeding parameter (currently unused)
    "movie_min_density_factor" : 0.01,  # visualization floor as fraction of rho_0 (unitless)
    "deposit_patch_radius" : 0.05,  # physical radius of deposition kernel if pixel radius not forced (cm)
    "deposit_patch_radius_pixels" : 0,  # deposition kernel radius in grid cells; >=0 overrides physical radius (pixels)
    "deposit_cells_multiplier" : 1.0,  # multiplier converting dropped amount to deposited-map cells (unitless)
    "deposit_merge_distance" : 0.03,  # legacy merge-distance knob for deposited spots (cm)

    # Worm SDE motility parameters
    "num_worms" : 10,  # number of worms (count)
    "v_max" : 2,  # speed at low bacteria density, exploratory regime (cm/min)
    "v_min" : 0.2,  # speed at high bacteria density, dwelling regime (cm/min)
    "alpha" : 3.0,  # strength of speed suppression vs normalized bacteria (unitless)
    "beta_b" : 3.0,  # strength of angular-noise suppression vs normalized bacteria (unitless)
    "chi_theta" : 6.0,  # chemotactic turning gain for gradient-following (rad)
    "dtheta" : np.pi/8,  # angular noise scale used to build rotational diffusion (rad)

    # Worm deposition parameters
    "deposition_enabled" : 1,  # master switch for deposition pathways (0/1)
    "deposition_fraction" : 1.0,  # fraction of eaten cells routed toward delayed gut-drop intake (unitless)
    "deposition_interval_steps" : 5,  # legacy deposition interval setting (timesteps)
    "deposition_max_per_event" : 0.0,  # legacy max per event; <=0 disables cap (cells/event)
    "surface_shedding_enabled" : 0,  # enable surface-shedding pathway (0/1); typically off for one-drop-per-worm experiments
    "surface_pickup_rate" : 8.0,  # carried-load pickup rate on patch (cells/min)
    "surface_carry_capacity" : 40.0,  # max carried load available for shedding (cells)
    "surface_shed_mean_steps" : 300,  # mean waiting time between shedding opportunities (timesteps)
    "surface_shed_fraction_min" : 0.02,  # minimum random fraction shed per shedding event (unitless)
    "surface_shed_fraction_max" : 0.15,  # maximum random fraction shed per shedding event (unitless)
    "surface_shed_max_cells" : 20.0,  # optional hard cap per shedding event (cells/event)
    "surface_drop_jitter_radius" : 0.08,  # random spatial offset radius for surface-drop location (cm)
    "gut_drop_enabled" : 1,  # enable gut-drop pathway (0/1)
    "gut_drop_trigger_cells" : 35.0,  # gut reservoir threshold needed to trigger one drop event (cells)
    "gut_drop_conversion_fraction" : 1.0,  # fraction of trigger amount converted to deposited amount (unitless)
    "gut_drop_delay_steps" : 5000,  # delay from intake to gut availability for dropping (timesteps)
    "gut_drop_jitter_radius" : 0.05,  # random spatial offset radius for gut-drop location (cm)
    "gut_drop_max_events_per_step" : 5,  # maximum gut-drop events processed in one timestep (events/step)
    "single_deposit_per_worm" : 1,  # one-time deposition mode: each worm can create at most one deposited patch (0/1)

    # Legacy (kept for compatibility)
    # "bacteria_enabled" : False,
    "bacteria_drop_interval" : 5,  # legacy drop interval from earlier pathway (timesteps)
    "bacteria_amount" : 0,  # legacy amount per legacy drop event (cells/event)

    # Other parameters
    "measurements_on" : True,  # enable time-series logging to HDF5 outputs (bool)
}