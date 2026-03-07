import numpy as np

class Environment:
    """
    Environment containing:
      1. A single circular bacterial patch governed by logistic ODEs (S1, S2):
            dA_B/dt  = g_A   * A_B * (1 - A_B  / K_A)
            drho/dt  = g_rho * rho * (1 - rho   / K_rho)
      2. A 2D spatial bacteria_map grid (density rho inside patch, 0 outside)
      3. A time-course grid
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
        """
        Initialise ODE state variables and build the first spatial map.
        A_B_0 = pi * 0.5^2 cm^2  (initial patch radius = 0.5 cm from paper)
        rho_0 = 1.27e8 cells/cm^2 (from Table S1)
        K_A defaults to arena area = (x_max - x_min)^2 if not supplied.
        """
        arena_side = self.x_max - self.x_min           # cm
        # default_K_A = arena_side ** 2                   # cm^2 (square arena)
        default_K_A = np.pi * 10**2      # cm^2 (circular arena inscribed in square)

        # ODE parameters (fall back to Table S1 values if not in params)
        self.g_A   = getattr(self, "g_A")           # min^-1
        self.g_rho = getattr(self, "g_rho")           # min^-1
        self.K_A   = getattr(self, "K_A", default_K_A)       # cm^2
        self.K_rho = getattr(self, "K_rho")            # cells/cm^2

        # Scalar ODE state variables
        self.A_B = getattr(self, "A_B_0", np.pi * 0.5**2)      # cm^2
        self.rho = getattr(self, "rho_0", 1.27e8)               # cells/cm^2

        self.bacteria_map = np.zeros_like(self.x_grid, dtype=float)
        self.__init_bacteria_patch()

    def __init_bacteria_patch(self):
        # """
        # Paint the circular patch onto the grid.
        # Radius is derived from current A_B: r = sqrt(A_B / pi)
        # Cells inside the circle get value rho, everything outside is 0.
        # Patch is always centred at (0, 0).
        # """
        # r    = np.sqrt(self.A_B / np.pi)               # current radius (cm)
        # dist = np.sqrt(self.x_grid**2 + self.y_grid**2)
        # self.bacteria_map = np.where(dist <= r, self.rho, 0.0)
        """
        Paint the circular patch onto the grid with a sigmoid falloff at the boundary.
        
        b(r) = rho / (1 + exp(k * (r - R)))
        
        where:
            R = current patch radius (cm)
            k = sharpness of boundary (cm^-1)
                large k -> sharp edge (current behaviour)
                small k -> smooth falloff, worms sense gradient from distance
        """
        R    = np.sqrt(self.A_B / np.pi)               # current patch radius (cm)
        dist = np.sqrt(self.x_grid**2 + self.y_grid**2)

        # Sigmoid falloff — k controls sensing range
        k = getattr(self, "boundary_k")          # cm^-1, tunable parameter
        self.bacteria_map = self.rho / (1.0 + np.exp(k * (dist - R)))

    def __logistic_step(self, x, g, K):
        """
        Exact analytical solution of logistic ODE over one timestep dt:
            x(t+dt) = K*x / (x + (K - x)*exp(-g*dt))
        """
        if g == 0.0 or x <= 0.0:
            return x
        exp_term = np.exp(-g * self.dt)
        return K * x / (x + (K - x) * exp_term)
    
    def update_bacteria_map(self):
        """
        Advance A_B and rho by dt minutes using exact logistic solutions (S1, S2),
        then rebuild the spatial bacteria map.
        """
        self.A_B = self.__logistic_step(self.A_B, self.g_A,   self.K_A)
        self.rho = self.__logistic_step(self.rho,  self.g_rho, self.K_rho)
        self.__init_bacteria_patch()

        # Debug print
        # print(
        #     f"A_B={self.A_B:.4f} cm^2  "
        #     f"r={np.sqrt(self.A_B/np.pi):.4f} cm  "
        #     f"rho={self.rho:.3e} cells/cm^2  "
        #     f"B={self.A_B * self.rho:.3e} cells"
        # )

    def add_bacteria_source(self, x, y, amount):
        """Deposits bacteria as a small patch at (x,y)"""
        self.__init_bacteria_patch(x_center=x, y_center=y, radius=0.03, amplitude=amount)

    def convert_xy_to_index(self, xy):
        """Convert real coordinates (x or y) to grid indices"""
        index = ((xy - self.x_min) / (self.x_max - self.x_min)) * self.x_grid.shape[0]
        return index