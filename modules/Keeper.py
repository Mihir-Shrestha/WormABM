import numpy as np
import h5py

class Keeper(object):
    def __init__(self, params):
        self.__set_params(params)
        self.__init_history()
        self.environment_history = []   # List of 2D grids

        self.B_before_hist = []  # Total bacteria before feeding
        self.B_after_hist = []   # Total bacteria after feeding
        self.dB_feed_requested_hist = []  # Summed worm consumption before clamping
        self.dB_feed_hist = []   # Total bacteria consumed by worms
        self.F_hist = []         # Feeding rate F(A_B, rho, R)


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
            "theta" : [],
            "on_patch" : [],
            "cells_eaten_step" : [],
            "cells_eaten_total" : [],
        }

    def __update_worm_history(self, worm_info):
        for key, val in worm_info.items():
            self.worm_history[key].append(val)

    def __write_environment_data(self):
        """Save bacteria grid time series (3D: [time, height, width])"""
        with h5py.File(self.environment_path, 'w') as outfile:
            outfile.create_dataset("bacteria", data=self.environment_history)
        
            # 1D scalar time series
            outfile.create_dataset("B_before", data=self.B_before_hist)
            outfile.create_dataset("B_after", data=self.B_after_hist)
            outfile.create_dataset("dB_feed_requested", data=self.dB_feed_requested_hist)
            outfile.create_dataset("dB_feed", data=self.dB_feed_hist)
            outfile.create_dataset("F", data=self.F_hist)
            
    def __write_worm_data(self):
        """Save worm history to HDF5 file"""
        with h5py.File(self.worm_path, 'w') as outfile:
            for key, val in self.worm_history.items():
                outfile.create_dataset(key, data=val)

    def measure_environment(self, environment):
        """Record bacteria grid snapshot at current timestep"""
        if self.sleeping:
            return
        self.environment_history.append(environment.bacteria_map.copy())
        self.B_before_hist.append(environment.B_before)
        self.B_after_hist.append(environment.B_after)
        self.dB_feed_requested_hist.append(environment.dB_feed_requested)
        self.dB_feed_hist.append(environment.dB_feed)
        self.F_hist.append(environment.F)

    
    def measure_worms(self, worm, global_i):
        """Record worm state at current timestep"""
        if self.sleeping:
            return

        worm_info = {
            "t"      : global_i,
            "worm_i" : worm.num,
            "x"      : worm.x,
            "y"      : worm.y,
            "theta"  : worm.theta,
            "on_patch" : worm.on_patch,
            "cells_eaten_step" : worm.cells_eaten_step,
            "cells_eaten_total" : worm.cells_eaten_total,
        }
        self.__update_worm_history(worm_info)

    def log_data_to_handy_dandy_notebook(self):
        if self.sleeping:
            return
        
        self.__write_environment_data()
        self.__write_worm_data()