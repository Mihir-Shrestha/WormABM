import numpy as np

class Environment:
    """
        Pieces of the environment
        ---------------------------
        1. Bacteria Concentration Map (Grid)
          - Diffusion equation
        2. Time-course
    """
    def __init__(self, params):
        self.__set_params(params)
        self.__init_environment_grid()
        self.__init_timecourse()

        # Bacteria concentration grid
        self.bacteria_map = []
        self.__init_bacteria_map()

    def __getitem__(self, idx):
        """Allow indexing: env[i] returns time at index i"""
        return self.t_grid[idx]
    
    def __set_params(self, params):
        """Store all parameters as instance variables"""
        for key, val in params.items():
            self.__dict__[key] = val
    
    def __init_environment_grid(self):
        """Create 2D spatial grid using x_min, x_max, dx"""
        print("Creating environment grid...")
        X1 = np.arange(self.x_min, self.x_max + self.dx, self.dx)
        X2 = np.arange(self.x_min, self.x_max + self.dx, self.dx)
        self.x_grid, self.y_grid = np.meshgrid(X1, X2)
    
    def __init_timecourse(self):
        """Create 1D temporal/time grid using t_min, t_max, dt"""
        print("Creating timecourse...")
        self.t_grid = np.arange(self.t_min, self.t_max, self.dt)

    def __init_bacteria_map(self):
        """Initialize bacteria concentration map to zeros"""
        self.bacteria_map = np.zeros_like(self.x_grid, dtype=float)
        self.init_bacteria_patch(x_center=0.0, y_center=0.0, radius=0.1, amplitude=1)

    def init_bacteria_patch(self, x_center, y_center, radius, amplitude):
        """
        Initialize a Gaussian patch of bacteria at (x_center, y_center)
        Only sets bacteria within a certain radius, keeps rest at zero
        """
        dx = self.x_grid - x_center
        dy = self.y_grid - y_center
        dist_sq = dx**2 + dy**2
        
        # Create Gaussian
        gaussian = amplitude * np.exp(-dist_sq / (2 * radius**2))
        
        # Only apply where distance < 3*radius (99.7% of distribution) and everything outside the mask stays at 0
        mask = np.sqrt(dist_sq) < (3 * radius)
        self.bacteria_map[mask] += gaussian[mask]

        # Clip to prevent exceeding carrying capacity to value of 1
        self.bacteria_map = np.clip(self.bacteria_map, 0, 1)

    def update_bacteria_map(self):
        """
        Solve ∂b/∂t = ∇²b + b(1-b) using finite differences
        """
        # Compute Laplacian (diffusion term)
        laplacian = self.__compute_laplacian(self.bacteria_map)
        # Compute logistic growth term with r = 1
        growth = self.bacteria_map * (1 - self.bacteria_map)
        # Forward Euler time step: b_new = b_old + dt * (∇²b + b(1-b))
        self.bacteria_map += self.dt * (laplacian + growth)
        # Clamp to [0, 1] to avoid numerical instability
        self.bacteria_map = np.clip(self.bacteria_map, 0, 1)

    def __compute_laplacian(self, field):
        """
        Compute ∇²b using 9-point stencil (includes diagonals)
        """
        laplacian = np.zeros_like(field)
        dx2 = self.dx ** 2
        
        # 9-point stencil: includes diagonal neighbors
        laplacian[1:-1, 1:-1] = (
            # Cardinal directions (weight = 1)
            field[2:, 1:-1] + field[:-2, 1:-1] +
            field[1:-1, 2:] + field[1:-1, :-2] +
            # Diagonal directions (weight = 0.5)
            0.5 * (field[2:, 2:] + field[:-2, :-2] + 
                   field[2:, :-2] + field[:-2, 2:]) -
            # Center (weight = 6)
            6 * field[1:-1, 1:-1]
        ) / dx2
        return laplacian

    def add_bacteria_source(self, x, y, amount):
        """Deposits bacteria as a small patch at (x,y)"""
        self.init_bacteria_patch(x_center=x, y_center=y, radius=0.03, amplitude=amount)

    def convert_xy_to_index(self, xy):
        """Convert real coordinates (x or y) to grid indices"""
        index = ((xy - self.x_min) / (self.x_max - self.x_min)) * self.x_grid.shape[0]
        return index