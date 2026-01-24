import numpy as np
import h5py

class Keeper(object):
    def __init__(self, params):
        self.__set_params(params)
        self.__init_history()
        self.environment_history = []

    def __set_params(self, params):
        for key, val in params.items():
            self.__dict__[key] = val

    def __init_history(self):
        """Initialize history dictionaries to store worm data"""
        self.worm_history = {
            "t"     : [],
            "worm_i" : [],
            "x"     : [],
            "y"     : [],
            "state" : [], 
            "angle" : [],
            "timestep" : [],
        }

        states = ["run", "tumble"]
        self.state_encoding = { state : i for i, state in enumerate(states) }

    def __update_worm_history(self, worm_info):
        for key, val in worm_info.items():
            self.worm_history[key].append(val)

    def __write_worm_data(self):
        """Save worm history to HDF5 file"""
        with h5py.File(self.worm_path, 'w') as outfile:
            for key, val in self.worm_history.items():
                outfile.create_dataset(key, data=val)
    
    # def __write_environment_data(self):
    #     with h5py.File(self.environment_path, 'w') as outfile:
    #         outfile.create_dataset("concentration", data=self.environment_history)

    def measure_worms(self, worm, global_i):
        """Record worm state at current timestep"""
        if self.sleeping:
            return

        worm_info = {
            "t"      : global_i,
            "worm_i" : worm.num,
            "x"      : worm.x,
            "y"      : worm.y,
            "state"  : self.state_encoding[worm.state],
            "angle"  : worm.angle,
            "timestep": worm.timestep,
        }
        self.__update_worm_history(worm_info)

    def log_data_to_handy_dandy_notebook(self):
        if self.sleeping:
            return
        
        # self.__write_environment_data()
        self.__write_worm_data()