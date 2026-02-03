import numpy as np
import warnings

warnings.filterwarnings("ignore")

import modules.Setup as Setup

def main(cfg_options, environment, worms, keeper):
    
    print("Begin simulation")
    
    try:
        for global_i, t_i in enumerate(environment):
            
            if cfg_options.verbose:
                print(f"\rTimestep: {global_i+1}/{environment.t_grid.shape[0]}")

            # Initialize or update bacteria map
            environment.update_bacteria_map()

            for worm in worms:
                worm.step(environment)

                # Measure and store worm info
                keeper.measure_worms(worm, global_i)

            # Store environment info after all worms have moved
            keeper.measure_environment(environment)
        
        # Save data to h5 files
        keeper.log_data_to_handy_dandy_notebook()
    
    except KeyboardInterrupt:
        print("\nEnding early.")
        keeper.log_data_to_handy_dandy_notebook()

if __name__ == '__main__':
    # Parse configuration
    cfg_options = Setup.config_options()
    np.random.seed(cfg_options.random_seed)

    # Create experiment directory
    model_dir = Setup.directory(cfg_options)
    print(f"Experiment directory: {model_dir}")

    # Organize parameters
    world_params = Setup.world_parameters(cfg_options, model_dir)

    # Instantiate world objects
    world_objects = Setup.world_objects(cfg_options, world_params)
    
    # Run simulation
    main(cfg_options, **world_objects)           




# Temporary simulation runs for visualization purposes
# import numpy as np
# import warnings

# warnings.filterwarnings("ignore")

# import modules.Setup as Setup
# import matplotlib.pyplot as plt

# def main(cfg_options, environment, worms, keeper):
#     print("Begin simulation...")
    
#     # Setup figure
#     fig, ax = plt.subplots(figsize=(8, 8))
    
#     try:
#         for global_i, t_i in enumerate(environment):
            
#             # Update worms
#             for worm in worms:
#                 worm.step(environment)
            
#             # Visualize every N timesteps
#             if global_i % 1 == 0:  # Every timestep
#                 ax.clear()
                
#                 # Draw arena
#                 ax.set_xlim(environment.x_min, environment.x_max)
#                 ax.set_ylim(environment.x_min, environment.x_max)
#                 ax.set_aspect('equal')
#                 ax.grid(True, alpha=0.3)
                
#                 # Plot worms
#                 for worm in worms:
#                     ax.plot(worm.x, worm.y, 'ro', markersize=10, label=f'Worm {worm.num}')
                
#                 ax.set_title(f"Worm ABM - Timestep {global_i}/{len(environment.t_grid)}")
#                 ax.legend()
                
#                 plt.pause(0.5)  # Small pause for animation
        
#         plt.close()
#         print("\nSimulation complete!")
    
#     except KeyboardInterrupt:
#         print("\nEnding early.")
#         plt.close()

# if __name__ == '__main__':
#     # Parse configuration
#     cfg_options = Setup.config_options()
#     np.random.seed(cfg_options.random_seed)

#     # Create experiment directory
#     model_dir = Setup.directory(cfg_options)
#     print(f"Experiment directory: {model_dir}")

#     # Organize parameters
#     world_params = Setup.world_parameters(cfg_options, model_dir)

#     # Instantiate world objects
#     world_objects = Setup.world_objects(cfg_options, world_params)
    
#     # Run simulation
#     main(cfg_options, **world_objects)