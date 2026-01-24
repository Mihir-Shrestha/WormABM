import numpy as np

class Environment:
    def __init__(self, params):
        self.__set_params(params)
        self.__init_environment_grid()
        self.__init_timecourse()

    def __getitem__(self, idx):
        # Allow indexing: env[i] returns time at index i
        return self.t_grid[idx]
    
    def __set_params(self, params):
        # Store all parameters as instance variables
        for key, val in params.items():
            self.__dict__[key] = val
    
    def __init_environment_grid(self):
        # Create 2D spatial grid using x_min, x_max, dx
        print("Creating environment grid...")
        X1 = np.arange(self.x_min, self.x_max + self.dx, self.dx)
        X2 = np.arange(self.x_min, self.x_max + self.dx, self.dx)
        self.x_grid, self.y_grid = np.meshgrid(X1, X2)
    
    def __init_timecourse(self):
        # Create 1D temporal/time grid using t_min, t_max, dt
        print("Creating timecourse...")
        self.t_grid = np.arange(self.t_min, self.t_max, self.dt)

    def convert_xy_to_index(self, xy):
        # Convert real coordinates (x or y) to grid indices
        index = ((xy - self.x_min) / (self.x_max - self.x_min)) * self.x_grid.shape[0]
        return index
    

    

    

