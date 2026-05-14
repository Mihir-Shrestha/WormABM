import numpy as np

class Environment:
    """
        Environment for two fixed bacteria sources and a deposited, non-food trail.

        Sources are static (effectively infinite for this simulation):
            - left source at (-source_x_offset, 0)
            - right source at (+source_x_offset, 0)

        Deposited bacteria:
            - is written directly into bacteria_map (no separate deposition layer)
            - does not grow
            - is not consumed
            - is not used as feeding source or for chemotactic gradient sensing
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
        self.R = getattr(self, "R")
        self.K_rho = float(getattr(self, "K_rho", 1.0e8))
        self.source_density = float(getattr(self, "source_density", 1.0e8))
        self.source_x_offset = float(getattr(self, "source_x_offset", 1.5))
        self.source_radius = float(getattr(self, "source_radius", 0.2))
        self.source_boundary_k = float(getattr(self, "source_boundary_k", 0.0))
        self.on_patch_density_epsilon = max(float(getattr(self, "on_patch_density_epsilon", 1e-12)), 0.0)

        self.deposit_patch_radius = max(float(getattr(self, "deposit_patch_radius", 0.05)), 1e-9)
        self.deposit_patch_radius_pixels = int(getattr(self, "deposit_patch_radius_pixels", 0))
        self.deposit_cells_multiplier = max(float(getattr(self, "deposit_cells_multiplier", 1.0)), 0.0)

        # Legacy scalar diagnostics kept for compatibility with existing plotting/logging scripts.
        self.A_B = 0.0
        self.rho = self.source_density

        self._init_source_map()
        self.bacteria_map = self.source_map.copy()
        self._update_bacteria_gradient()

        self._deposit_event_count = 0
        self.num_deposited_patches = 0
        self.B_deposited = 0.0
        self.B_before = self._total_cells_in_map(self.bacteria_map)
        self.B_after = self.B_before
        self.dB_feed_requested = 0.0
        self.dB_feed = 0.0
        self.F = 1.0

    def _circle_profile(self, x_center, y_center, radius, density, boundary_k):
        if radius <= 0.0 or density <= 0.0:
            return np.zeros_like(self.x_grid, dtype=float)

        dist = np.sqrt((self.x_grid - x_center) ** 2 + (self.y_grid - y_center) ** 2)
        if boundary_k <= 0.0:
            return np.where(dist <= radius, density, 0.0)
        return density / (1.0 + np.exp(boundary_k * (dist - radius)))

    def _init_source_map(self):
        left_x = -self.source_x_offset
        right_x = self.source_x_offset

        left_source = self._circle_profile(
            left_x,
            0.0,
            self.source_radius,
            self.source_density,
            self.source_boundary_k,
        )
        right_source = self._circle_profile(
            right_x,
            0.0,
            self.source_radius,
            self.source_density,
            self.source_boundary_k,
        )

        self.source_map = np.minimum(left_source + right_source, self.source_density)
        self.source_mask = self.source_map > self.on_patch_density_epsilon
        self._update_source_gradient()

    def _update_source_gradient(self):
        """Compute gradients of normalised source map for chemotactic turning."""
        norm_scale = max(self.source_density, 1e-12)
        b_norm = self.source_map / norm_scale
        grad_row, grad_col = np.gradient(b_norm, self.dx)
        self.grad_bn_x = grad_col
        self.grad_bn_y = grad_row

    def _update_bacteria_gradient(self):
        """Compute gradients of normalised bacteria map for chemotactic turning."""
        if self.K_rho <= 0.0:
            self.grad_bn_x = np.zeros_like(self.bacteria_map)
            self.grad_bn_y = np.zeros_like(self.bacteria_map)
            return
        b_norm = self.bacteria_map / self.K_rho
        grad_row, grad_col = np.gradient(b_norm, self.dx)
        self.grad_bn_x = grad_col
        self.grad_bn_y = grad_row

    def _total_cells_in_map(self, density_map):
        return float(np.sum(density_map) * (self.dx ** 2))

    def _enforce_source_floor(self):
        """Keep fixed source regions at least at source_map density (infinite-source assumption)."""
        self.bacteria_map = np.maximum(self.bacteria_map, self.source_map)

    def is_on_source(self, local_source_density):
        """Only the two initial sources can set worms into fed state."""
        return float(local_source_density) > self.on_patch_density_epsilon

    def is_on_patch(self, local_source_density):
        """True when local bacteria density is above the epsilon threshold."""
        return float(local_source_density) > self.on_patch_density_epsilon

    def __update_bacteria_gradient(self):
        """Legacy wrapper retained for compatibility."""
        self._update_bacteria_gradient()

    def update_bacteria_map(self):
        """
        Static-source model update.

        - Sources remain unchanged (infinite reservoir assumption).
        - Deposited trail remains in place and does not grow.
        - No depletion term is applied.
        """
        self.B_before = self._total_cells_in_map(self.bacteria_map)
        self._enforce_source_floor()
        self.B_after = self._total_cells_in_map(self.bacteria_map)
        self.dB_feed_requested = 0.0
        self.dB_feed = 0.0
        self.F = 1.0
        self.num_deposited_patches = self._deposit_event_count
        deposited_component = np.maximum(self.bacteria_map - self.source_map, 0.0)
        self.B_deposited = self._total_cells_in_map(deposited_component)
        self._update_bacteria_gradient()

    def add_bacteria_source(self, x, y, amount):
        """Add non-growing deposited bacteria near (x, y). Deposits are not food sources."""
        dep_mult = self.deposit_cells_multiplier
        cells = float(amount) * dep_mult
        if cells <= 0.0:
            return

        radius_cm = self.deposit_patch_radius
        radius_px_cfg = self.deposit_patch_radius_pixels
        if radius_px_cfg > 0:
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

        weights = np.clip(1.0 - (dist_px / (radius_px + 1e-12)), 0.0, None)
        weights *= mask
        sum_w = float(np.sum(weights))
        if sum_w <= 0.0:
            weights = np.zeros_like(dist_px)
            weights[row_center - row0, col_center - col0] = 1.0
            sum_w = 1.0

        delta_rho_per_weight = cells / ((self.dx ** 2) * sum_w)
        patch_view = self.bacteria_map[row0:row1, col0:col1]
        patch_view[mask] += delta_rho_per_weight * weights[mask]

        self._enforce_source_floor()

        self._deposit_event_count += 1

    def convert_xy_to_index(self, xy):
        """Convert real coordinates (x or y) to grid indices"""
        index = ((xy - self.x_min) / (self.x_max - self.x_min)) * self.x_grid.shape[0]
        return index