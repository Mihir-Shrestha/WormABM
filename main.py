import numpy as np
import warnings

warnings.filterwarnings("ignore")

import modules.Setup as Setup

def main(cfg_options, environment, worms, keeper):
    
    print("Begin simulation")
    
    try:
        for global_i, t_i in enumerate(environment):
            record_now = keeper.should_record_step(global_i)
            
            if cfg_options.verbose and global_i % 100 == 0:  # Print every 100 timesteps
                print(f"\rTimestep: {global_i+1}/{environment.t_grid.shape[0]}")

            for worm in worms:
                worm.step(environment)

                # Measure and store worm info
                if record_now:
                    keeper.measure_worms(worm, global_i)

            # Update environment state (static sources + deposited trail bookkeeping)
            environment.update_bacteria_map()

            # Store environment info after all worms have moved
            if record_now:
                keeper.measure_environment(environment, global_i)
        
        # Save data to h5 files
        keeper.log_data_to_handy_dandy_notebook()
    
    except KeyboardInterrupt:
        print("\nEnding early.")
        keeper.log_data_to_handy_dandy_notebook()

if __name__ == '__main__':
    # Parse configuration
    cfg_options = Setup.config_options()
    np.random.seed(cfg_options.random_seed)
    # print(f"Verbose mode is set to: {cfg_options.verbose}")  # Debugging line

    # Create experiment directory
    model_dir = Setup.directory(cfg_options)
    print(f"\n"+"="*60+"\n")
    print(f"Experiment directory: {model_dir}")

    # Organize parameters
    world_params = Setup.world_parameters(cfg_options, model_dir)

    # Instantiate world objects
    world_objects = Setup.world_objects(cfg_options, world_params)
    
    # Run simulation
    main(cfg_options, **world_objects)