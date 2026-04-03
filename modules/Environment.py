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
        """

        # ODE parameters (fall back to Table S1 values if not in params)
        self.g_A   = getattr(self, "g_A")           # min^-1
        self.g_rho = getattr(self, "g_rho")           # min^-1
        self.R = getattr(self, "R")              # cm, used for K_A
        self.K_A   = np.pi * self.R**2           # cm^2, area carrying capacity (derived from plate radius)
        self.K_rho = getattr(self, "K_rho")            # cells/cm^2
        self.patch_bnorm_threshold = getattr(self, "patch_bnorm_threshold", 0.01)

        # Scalar ODE state variables
        self.A_B = getattr(self, "A_B_0")      # cm^2
        self.rho = getattr(self, "rho_0")               # cells/cm^2

        # self.bacteria_map = np.zeros_like(self.x_grid, dtype=float)
        # self.__init_bacteria_patch()

        self.bacteria_map = np.zeros_like(self.x_grid, dtype=float)
        self.__init_bacteria_patch()
        self.__update_bacteria_gradient()
        self.pending_worm_consumption = 0.0
        self.dB_feed_requested = 0.0

    def is_on_patch(self, b_norm):
        """Return True when local normalised bacteria is high enough to count as patch."""
        return b_norm >= self.patch_bnorm_threshold

    def register_worm_consumption(self, dB_worm):
        """Accumulate bacteria consumed by worms during the current timestep."""
        if dB_worm > 0.0:
            self.pending_worm_consumption += dB_worm

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

    def __update_bacteria_gradient(self):
        """
        Compute graident of normalised bacteria map b_norm = bacteria_map / K_rho
        and store as grad_bn_x (d b_norm / dx) and grad_bn_y (d b_norm / dy).
        """
        b_norm = self.bacteria_map/self.K_rho
        grad_row, grad_col = np.gradient(b_norm, self.dx)
        self.grad_bn_x = grad_col
        self.grad_bn_y = grad_row

    def __logistic_step(self, x, g, K):
        """
        Exact analytical solution of logistic ODE over one timestep dt:
            x(t+dt) = K*x / (x + (K - x)*exp(-g*dt))
        """
        if g == 0.0 or x <= 0.0:
            return x
        exp_term = np.exp(-g * self.dt)
        return K * x / (x + (K - x) * exp_term)

    def feeding_rate(self):
        A_B = self.A_B
        rho = self.rho
        R = self.R
        
        if A_B <= 0.0 or rho <= 0.0:
            return 0.0
        
        c = getattr(self, "feed_c")
        sigma = getattr(self, "feed_sigma") 
        psi = np.exp(-np.pi * R**2 / (sigma * A_B))
        F = (c * psi * rho * A_B) / (1.0 + c * psi * rho * A_B)
        return F

    def feeding_rate_from_bnorm(self, b_norm):
        """
        Local feeding modulation from normalised concentration at worm position.
        Returns a factor in [0, 1] used to scale the 70 cells/min per-worm baseline.
        """
        b = max(float(b_norm), 0.0)
        b_half = getattr(self, "local_feed_b_half", 0.1)
        hill_n = getattr(self, "local_feed_hill_n", 2.0)

        if b_half <= 0.0:
            return 1.0 if b > 0.0 else 0.0

        num = b ** hill_n
        den = (b_half ** hill_n) + num
        if den <= 0.0:
            return 0.0
        return num / den

    def update_bacteria_map(self):
        """
        Advance A_B and rho by dt minutes using exact logistic solutions (S1, S2),
        then rebuild the spatial bacteria map.
        """
        # 1) logistic growth
        self.A_B = self.__logistic_step(self.A_B, self.g_A,   self.K_A)
        self.rho = self.__logistic_step(self.rho,  self.g_rho, self.K_rho)
        
        B_before = self.A_B * self.rho
        # 2) total bacteria before feeding

        # 3) feeding term from actual summed per-worm intake this step
        F = self.feeding_rate()
        dB_feed_requested = self.pending_worm_consumption
        dB_feed = min(dB_feed_requested, B_before)
        self.pending_worm_consumption = 0.0
        B_after = max(B_before - dB_feed, 0.0) # Ensure non-negative bacteria count after feeding

        self.B_before = B_before
        self.B_after = B_after
        self.dB_feed_requested = dB_feed_requested
        self.dB_feed = dB_feed
        self.F = F
        B = B_after

        # 4) update density from new total bacteria
        if self.A_B > 0.0:
            self.rho = B / self.A_B
        else:
            self.rho = 0.0

        # 5) rebuild spatial map
        self.__init_bacteria_patch()
        
        # 6) update gradient of normalised bacteria map
        self.__update_bacteria_gradient()

    def add_bacteria_source(self, x, y, amount):
        """Deposits bacteria as a small patch at (x,y)"""
        self.__init_bacteria_patch(x_center=x, y_center=y, radius=0.03, amplitude=amount)

    def convert_xy_to_index(self, xy):
        """Convert real coordinates (x or y) to grid indices"""
        index = ((xy - self.x_min) / (self.x_max - self.x_min)) * self.x_grid.shape[0]
        return index