import numpy as np

class Environment:
    """
    Environment containing:
    1. Multiple circular bacterial patches, each governed by logistic ODEs (S1, S2):
            dA_B/dt  = g_A   * A_B * (1 - A_B  / K_A)
            drho/dt  = g_rho * rho * (1 - rho   / K_rho)
    2. A 2D spatial bacteria_map grid from the sum of all active patches
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
        self.on_patch_density_epsilon = max(float(getattr(self, "on_patch_density_epsilon", 1e-12)), 0.0)

        # Scalar ODE state variables
        self.A_B = getattr(self, "A_B_0")      # cm^2
        self.rho = getattr(self, "rho_0")               # cells/cm^2
        self.patches = [self.__make_patch(0.0, 0.0, self.A_B, self.rho, is_initial=True)]
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
        self.pending_feeding_worms = 0
        self.dB_feed_requested = 0.0

    def __make_patch(self, x, y, A_B, rho, is_initial=False):
        """Create one independent patch state that follows the same ODEs as the main patch."""
        return {
            "x": float(np.clip(x, self.x_min, self.x_max)),
            "y": float(np.clip(y, self.x_min, self.x_max)),
            "A_B": max(float(A_B), 0.0),
            "rho": max(float(rho), 0.0),
            "is_initial": bool(is_initial),
        }

    def __sync_legacy_scalars(self):
        """Maintain legacy scalar fields from the initial patch for compatibility with existing outputs."""
        if not self.patches:
            self.A_B = 0.0
            self.rho = 0.0
            return
        self.A_B = self.patches[0]["A_B"]
        self.rho = self.patches[0]["rho"]

    def is_on_patch(self, local_density):
        """Return True when local absolute bacteria density is above a tiny epsilon."""
        return float(local_density) > self.on_patch_density_epsilon

    def register_worm_consumption(self, dB_worm):
        """Accumulate bacteria consumed by worms during the current timestep."""
        if dB_worm > 0.0:
            self.pending_worm_consumption += dB_worm

    def register_feeding_worm(self, is_feeding):
        """Register whether a worm is currently on a bacteria source for W_eff."""
        if bool(is_feeding):
            self.pending_feeding_worms += 1

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

    def __paint_hard_patch_local(self, target_map, x_center, y_center, A_B, rho):
        """Paint a hard-edged circular patch into a local window to reduce per-step cost."""
        if A_B <= 0.0 or rho <= 0.0:
            return

        radius = np.sqrt(A_B / np.pi)
        if radius <= 0.0:
            return

        col_center = int(np.clip(round((x_center - self.x_min) / self.dx), 0, self.x_grid.shape[1] - 1))
        row_center = int(np.clip(round((y_center - self.x_min) / self.dx), 0, self.x_grid.shape[0] - 1))
        half_width = max(int(np.ceil(radius / self.dx)), 1)

        row0 = max(0, row_center - half_width)
        row1 = min(self.x_grid.shape[0], row_center + half_width + 1)
        col0 = max(0, col_center - half_width)
        col1 = min(self.x_grid.shape[1], col_center + half_width + 1)

        xs = self.x_grid[row0:row1, col0:col1] - x_center
        ys = self.y_grid[row0:row1, col0:col1] - y_center
        mask = (xs * xs + ys * ys) <= (radius * radius)
        if not np.any(mask):
            return

        local_view = target_map[row0:row1, col0:col1]
        local_view[mask] += rho

    def __total_bacteria_count(self):
        return sum(p["A_B"] * p["rho"] for p in self.patches)

    def __total_patch_area(self):
        return sum(p["A_B"] for p in self.patches)

    def __advance_patch_growth(self):
        """Advance patch area growth; density is updated by the coupled rho ODE in update_bacteria_map."""
        for patch in self.patches:
            patch["A_B"] = self.__logistic_step(patch["A_B"], self.g_A, self.K_A)
        self.__sync_legacy_scalars()

    def __apply_global_depletion(self, dB_feed):
        """Apply total worm depletion proportionally across all patches by density scaling."""
        B_before = self.__total_bacteria_count()
        if B_before <= 0.0:
            for patch in self.patches:
                patch["rho"] = 0.0
            self.__sync_legacy_scalars()
            return

        scale = max((B_before - dB_feed) / B_before, 0.0)
        for patch in self.patches:
            patch["rho"] *= scale
        self.__sync_legacy_scalars()

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
        bacteria_map = np.zeros_like(self.x_grid, dtype=float)
        deposited_map = np.zeros_like(self.x_grid, dtype=float)
        hard_boundary = float(getattr(self, "boundary_k", 0.0)) <= 0.0
        for i, patch in enumerate(self.patches):
            if hard_boundary:
                self.__paint_hard_patch_local(
                    bacteria_map,
                    patch["x"],
                    patch["y"],
                    patch["A_B"],
                    patch["rho"],
                )
                if i > 0:
                    self.__paint_hard_patch_local(
                        deposited_map,
                        patch["x"],
                        patch["y"],
                        patch["A_B"],
                        patch["rho"],
                    )
            else:
                profile = self.__patch_profile(patch["x"], patch["y"], patch["A_B"], patch["rho"])
                bacteria_map += profile
                if i > 0:
                    deposited_map += profile

        # Keep normalised density bounded for motility/feed factors.
        self.deposited_map = np.minimum(deposited_map, self.K_rho)
        self.bacteria_map = np.minimum(bacteria_map, self.K_rho)
        self.__sync_legacy_scalars()

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
        A_B = self.__total_patch_area()
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
        Legacy local Hill feeding helper (currently not used in active simulation).
        The active model now uses paper-style global depletion with constant a
        per active feeding worm via W_eff.
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
        Advance area and density, then rebuild the spatial bacteria map.

        Area is still advanced with exact logistic steps.
        Density follows a single coupled paper-style ODE term each timestep:
            d(rho)/dt = g_rho * rho * (1 - rho/K_rho) - (a/A_B) * F(A_B, rho, R) * W
        """
        # 1) area growth for main + deposited patches
        self.__advance_patch_growth()

        B_before = self.__total_bacteria_count()
        A_B_eff = self.__total_patch_area()
        rho_eff = (B_before / A_B_eff) if A_B_eff > 0.0 else 0.0

        # 2) coupled paper-style rho ODE over one Euler step
        # F = self.feeding_rate()
        F = 1
        a_cells_per_min = max(float(getattr(self, "feeding_cells_per_worm", 70.0)), 0.0)
        W_eff = int(np.clip(self.pending_feeding_worms, 0, int(getattr(self, "num_worms", 0))))
        if A_B_eff > 0.0 and rho_eff > 0.0 and a_cells_per_min > 0.0 and W_eff > 0:
            d_rho_feed = (a_cells_per_min / A_B_eff) * F * W_eff
        else:
            d_rho_feed = 0.0

        if self.K_rho > 0.0 and rho_eff > 0.0:
            d_rho_growth = self.g_rho * rho_eff * (1.0 - (rho_eff / self.K_rho))
        else:
            d_rho_growth = 0.0

        rho_eff_new = rho_eff + (d_rho_growth - d_rho_feed) * self.dt
        rho_eff_new = float(np.clip(rho_eff_new, 0.0, self.K_rho if self.K_rho > 0.0 else np.inf))

        if rho_eff <= 0.0:
            for patch in self.patches:
                patch["rho"] = 0.0
        else:
            scale = rho_eff_new / rho_eff
            for patch in self.patches:
                patch["rho"] = float(np.clip(patch["rho"] * scale, 0.0, self.K_rho))

        dB_feed_requested = d_rho_feed * A_B_eff * self.dt
        dB_feed_available = max(B_before + (d_rho_growth * A_B_eff * self.dt), 0.0)
        dB_feed = min(dB_feed_requested, dB_feed_available)

        # Reset legacy accumulator (no longer used for depletion).
        self.pending_worm_consumption = 0.0
        self.pending_feeding_worms = 0
        self.__sync_legacy_scalars()
        B_after = self.__total_bacteria_count()

        self.B_before = B_before
        self.B_after = B_after
        self.dB_feed_requested = dB_feed_requested
        self.dB_feed = dB_feed
        self.F = F
        self.num_deposited_patches = max(len(self.patches) - 1, 0)
        self.B_deposited = sum(p["A_B"] * p["rho"] for p in self.patches[1:])

        # 4) rebuild spatial map from all active patches
        self.__init_bacteria_patch()
        
        # 5) update gradient of normalised bacteria map
        self.__update_bacteria_gradient()

    def add_bacteria_source(self, x, y, amount):
        """Add a new deposited source patch with its own independent ODE state."""
        dep_mult = max(float(getattr(self, "deposit_cells_multiplier", 1.0)), 0.0)
        cells = float(amount) * dep_mult
        if cells <= 0.0:
            return

        radius_cm = max(float(getattr(self, "deposit_patch_radius", 0.03)), 1e-9)
        radius_px_cfg = int(getattr(self, "deposit_patch_radius_pixels", -1))
        if radius_px_cfg >= 0:
            radius_px = max(radius_px_cfg, 1)
            radius_cm = max(radius_px * self.dx, 1e-9)

        x = float(np.clip(x, self.x_min, self.x_max))
        y = float(np.clip(y, self.x_min, self.x_max))

        A_init = np.pi * (radius_cm ** 2)
        if A_init <= 0.0:
            return

        rho_init = cells / A_init
        if rho_init > self.K_rho:
            rho_init = self.K_rho
            A_init = cells / max(self.K_rho, 1e-12)

        self.patches.append(self.__make_patch(x, y, A_init, rho_init, is_initial=False))

        self._deposit_event_count += 1

    def convert_xy_to_index(self, xy):
        """Convert real coordinates (x or y) to grid indices"""
        index = ((xy - self.x_min) / (self.x_max - self.x_min)) * self.x_grid.shape[0]
        return index