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
        self.deposited_map = np.zeros_like(self.x_grid, dtype=float)
        self._deposit_event_count = 0
        self.num_deposited_patches = 0
        self.B_deposited = 0.0

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

    def __patch_profile(self, x_center, y_center, A_B, rho):
        """Return spatial contribution of one patch using the same boundary model as the main patch."""
        if A_B <= 0.0 or rho <= 0.0:
            return np.zeros_like(self.x_grid, dtype=float)

        radius = np.sqrt(A_B / np.pi)
        dist = np.sqrt((self.x_grid - x_center) ** 2 + (self.y_grid - y_center) ** 2)
        k = getattr(self, "boundary_k")

        if k <= 0.0:
            return np.where(dist <= radius, rho, 0.0)

        return rho / (1.0 + np.exp(k * (dist - radius)))

    def __total_bacteria_count(self):
        total_main = self.A_B * self.rho
        total_deposited = np.sum(self.deposited_map) * (self.dx ** 2)
        return total_main + total_deposited

    def __advance_patch_growth(self):
        """Advance growth for the main patch only (deposited food does not grow)."""
        self.A_B = self.__logistic_step(self.A_B, self.g_A, self.K_A)
        self.rho = self.__logistic_step(self.rho, self.g_rho, self.K_rho)

    def __apply_global_depletion(self, dB_feed):
        """Apply total worm depletion proportionally across all patches by density scaling."""
        B_before = self.__total_bacteria_count()
        if B_before <= 0.0:
            self.rho = 0.0
            self.deposited_map.fill(0.0)
            return

        scale = max((B_before - dB_feed) / B_before, 0.0)
        self.rho *= scale
        self.deposited_map *= scale

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
        main_patch = self.__patch_profile(0.0, 0.0, self.A_B, self.rho)
        bacteria_map = main_patch + self.deposited_map

        # Keep normalised density bounded for motility/feed factors.
        self.bacteria_map = np.minimum(bacteria_map, self.K_rho)

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
        deposited_area = np.count_nonzero(self.deposited_map > 0.0) * (self.dx ** 2)
        A_B = self.A_B + deposited_area
        if A_B <= 0.0:
            return 0.0

        rho = self.__total_bacteria_count() / A_B
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
        # 1) logistic growth for main + deposited patches
        self.__advance_patch_growth()
        
        B_before = self.__total_bacteria_count()
        # 2) total bacteria before feeding

        # 3) paper-style global feeding term:
        #    d_rho_feed = (a / A_B_eff) * F(A_B_eff, rho_eff, R) * W
        #    dB_feed    = d_rho_feed * A_B_eff * dt
        F = self.feeding_rate()
        a_cells_per_min = max(float(getattr(self, "feeding_cells_per_worm", 70.0)), 0.0)
        W = max(int(getattr(self, "num_worms", 0)), 0)
        deposited_area = np.count_nonzero(self.deposited_map > 0.0) * (self.dx ** 2)
        A_B_eff = self.A_B + deposited_area

        if A_B_eff > 0.0 and a_cells_per_min > 0.0 and W > 0:
            d_rho_feed = (a_cells_per_min / A_B_eff) * F * W
            dB_feed_requested = d_rho_feed * A_B_eff * self.dt
        else:
            dB_feed_requested = 0.0

        dB_feed = min(dB_feed_requested, B_before)
        # Reset legacy accumulator (no longer used for depletion).
        self.pending_worm_consumption = 0.0
        self.__apply_global_depletion(dB_feed)
        B_after = self.__total_bacteria_count()

        self.B_before = B_before
        self.B_after = B_after
        self.dB_feed_requested = dB_feed_requested
        self.dB_feed = dB_feed
        self.F = F
        self.num_deposited_patches = self._deposit_event_count
        self.B_deposited = np.sum(self.deposited_map) * (self.dx ** 2)

        # 4) rebuild spatial map from all active patches
        self.__init_bacteria_patch()
        
        # 5) update gradient of normalised bacteria map
        self.__update_bacteria_gradient()

    def add_bacteria_source(self, x, y, amount):
        """Add a non-growing local deposited food patch at (x, y) as edible density."""
        dep_mult = max(float(getattr(self, "deposit_cells_multiplier", 1.0)), 0.0)
        cells = float(amount) * dep_mult
        if cells <= 0.0:
            return

        radius_cm = max(float(getattr(self, "deposit_patch_radius", 0.03)), 1e-9)
        radius_px_cfg = int(getattr(self, "deposit_patch_radius_pixels", -1))
        if radius_px_cfg >= 0:
            radius_px = radius_px_cfg
        else:
            radius_px = max(int(np.ceil(radius_cm / self.dx)), 1)

        x = float(np.clip(x, self.x_min, self.x_max))
        y = float(np.clip(y, self.x_min, self.x_max))

        col_center = int(np.clip(round((x - self.x_min) / self.dx), 0, self.x_grid.shape[1] - 1))
        row_center = int(np.clip(round((y - self.x_min) / self.dx), 0, self.x_grid.shape[0] - 1))
        half_width = max(radius_px, 1)

        row0 = max(0, row_center - half_width)
        row1 = min(self.x_grid.shape[0], row_center + half_width + 1)
        col0 = max(0, col_center - half_width)
        col1 = min(self.x_grid.shape[1], col_center + half_width + 1)

        rr, cc = np.ogrid[row0:row1, col0:col1]
        dist_px = np.sqrt((rr - row_center) ** 2 + (cc - col_center) ** 2)
        mask = dist_px <= radius_px
        if not np.any(mask):
            mask = np.zeros_like(dist_px, dtype=bool)
            mask[row_center - row0, col_center - col0] = True

        # Radial kernel gives each drop a tiny patch-like profile in a few cells.
        if radius_px > 0:
            weights = np.clip(1.0 - (dist_px / (radius_px + 1e-12)), 0.0, None)
        else:
            weights = np.zeros_like(dist_px)
            weights[row_center - row0, col_center - col0] = 1.0

        weights *= mask
        sum_w = float(np.sum(weights))
        if sum_w <= 0.0:
            weights = np.zeros_like(dist_px)
            weights[row_center - row0, col_center - col0] = 1.0
            sum_w = 1.0

        delta_rho_per_weight = cells / ((self.dx ** 2) * sum_w)

        patch_view = self.deposited_map[row0:row1, col0:col1]
        patch_view[mask] = np.minimum(
            patch_view[mask] + (delta_rho_per_weight * weights[mask]),
            self.K_rho,
        )

        self._deposit_event_count += 1

    def convert_xy_to_index(self, xy):
        """Convert real coordinates (x or y) to grid indices"""
        index = ((xy - self.x_min) / (self.x_max - self.x_min)) * self.x_grid.shape[0]
        return index