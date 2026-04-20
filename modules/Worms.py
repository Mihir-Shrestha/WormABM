import numpy as np
from collections import deque

class Worm(object):
    """
    Single worm using continuous SDE motility (Euler-Maruyama):

        dtheta = chi_theta * (u_perp . grad_b_norm) * dt
                 + sqrt(2 * D(b_norm)) * dW

        dx     = v(b_norm) * u * dt

    where b_norm = b / K_rho is the bacterial density normalised to [0, 1],
    so that alpha, beta_b, chi_theta are dimensionless and scale-independent.

    Speed and noise are suppressed at high b (dwelling behaviour):
        v(b_norm)  = v_min + (v_max - v_min) * exp(-alpha  * b_norm)
        D(b_norm)  = D_theta * exp(-beta_b * b_norm)
    """

    def __init__(self, params):
        self.__set_params(params)
        self.__init_position()
        self.on_patch = False       
        self.cells_eaten_step = 0.0
        self.cells_eaten_total = 0.0
        self.cells_available_to_deposit = 0.0

        # Deposition state
        self.surface_cells_carried = 0.0
        self.gut_cells_for_drop = 0.0
        self.surface_drop_step = 0.0
        self.gut_drop_step = 0.0
        self.surface_drop_total = 0.0
        self.gut_drop_total = 0.0
        self.timestep = 0

        self.deposition_enabled = getattr(
            self,
            "deposition_enabled",
            getattr(self, "bacteria_enabled", True),
        )
        self.deposition_fraction = max(float(getattr(self, "deposition_fraction", 1.0)), 0.0)

        # Surface shedding pathway (random intervals)
        self.surface_shedding_enabled = bool(getattr(self, "surface_shedding_enabled", True))
        self.surface_pickup_rate = max(float(getattr(self, "surface_pickup_rate", 2.0)), 0.0)  # cells/min
        self.surface_carry_capacity = max(float(getattr(self, "surface_carry_capacity", 40.0)), 0.0)
        default_interval = int(getattr(self, "deposition_interval_steps", getattr(self, "bacteria_drop_interval", 5)))
        self.surface_shed_mean_steps = max(int(getattr(self, "surface_shed_mean_steps", default_interval)), 1)
        self.surface_shed_fraction_min = float(getattr(self, "surface_shed_fraction_min", 0.02))
        self.surface_shed_fraction_max = float(getattr(self, "surface_shed_fraction_max", 0.15))
        if self.surface_shed_fraction_max < self.surface_shed_fraction_min:
            self.surface_shed_fraction_min, self.surface_shed_fraction_max = self.surface_shed_fraction_max, self.surface_shed_fraction_min
        default_surface_cap = float(getattr(self, "deposition_max_per_event", getattr(self, "bacteria_amount", 0.0)))
        self.surface_shed_max_cells = float(getattr(self, "surface_shed_max_cells", default_surface_cap))
        self.surface_drop_jitter_radius = max(float(getattr(self, "surface_drop_jitter_radius", 0.08)), 0.0)
        self.next_surface_shed_timestep = self.__sample_surface_shed_interval()

        # Drop pathway (triggered by eaten amount)
        self.gut_drop_enabled = bool(getattr(self, "gut_drop_enabled", True))
        self.gut_drop_trigger_cells = max(float(getattr(self, "gut_drop_trigger_cells", 35.0)), 1e-9)
        self.gut_drop_conversion_fraction = float(np.clip(getattr(self, "gut_drop_conversion_fraction", 0.25), 0.0, 1.0))
        self.gut_drop_delay_steps = max(int(getattr(self, "gut_drop_delay_steps", 1000)), 0)
        self.gut_drop_jitter_radius = max(float(getattr(self, "gut_drop_jitter_radius", 0.05)), 0.0)
        self.gut_drop_max_events_per_step = max(int(getattr(self, "gut_drop_max_events_per_step", 5)), 1)
        self._gut_drop_release_queue = deque()
        self.single_deposit_per_worm = bool(getattr(self, "single_deposit_per_worm", True))
        if self.single_deposit_per_worm:
            self.max_deposits_per_worm = 1
        else:
            self.max_deposits_per_worm = max(int(getattr(self, "max_deposits_per_worm", 0)), 0)
        self.num_deposits_done = 0
    
    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __set_params(self, params):
        for key, val in params.items():
            self.__dict__[key] = val

    def __init_position(self):
        """Worm starts at (0,0) facing a random direction."""
        self.x     = getattr(self, "x",     0.0)
        self.y     = getattr(self, "y",     0.0)
        self.theta = np.random.uniform(-np.pi, np.pi)

    # ------------------------------------------------------------------
    # Motility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def __unit_vectors(theta):
        """
        Return direction vector u and perpendicular u_perp for scalar theta.
            u      = ( cos θ,  sin θ)
            u_perp = (-sin θ,  cos θ)
        """
        c, s = np.cos(theta), np.sin(theta)
        u      = np.array([ c,  s])
        u_perp = np.array([-s,  c])
        return u, u_perp

    @staticmethod
    def __speed_from_b(b_norm, v_min, v_max, alpha):
        """
        Speed as function of normalised bacterial density b_norm in [0, 1].
            v(b_norm) = v_min + (v_max - v_min) * exp(-alpha * b_norm)
        - b_norm ~ 0  ->  v ~ v_max  (fast, exploring)
        - b_norm ~ 1  ->  v ~ v_min  (slow, dwelling)
        """
        return v_min + (v_max - v_min) * np.exp(-alpha * b_norm)
    
    def __get_bacteria_and_gradient(self, environment):
        """
        Sample bacteria density and gradient at current worm position.
        
        1. Convert (x, y) to grid indices
        2. Sample bacteria_map at that index
        3. Look up precomputed gradient at that index
        
        Returns
        -------
        b_norm  : float  normalised density in [0, 1]  (b / K_rho)
        grad_bn : (2,)   normalised gradient [d/dx, d/dy]
        """
        bmap = environment.bacteria_map          # 2D array (row=y, col=x)
        K    = environment.K_rho                 # carrying capacity for normalisation

        # Grid spacing in cm
        dx = environment.dx

        # Convert worm (x,y) to grid indices — clamp to valid range
        nx = bmap.shape[1]
        ny = bmap.shape[0]

        col = int(np.clip(
            round((self.x - environment.x_min) / dx), 0, nx - 1))
        row = int(np.clip(
            round((self.y - environment.x_min) / dx), 0, ny - 1))

        # Local and normalised density at worm position
        b_local = bmap[row, col]
        if K > 0.0:
            b_norm = b_local / K
        else:
            b_norm = 0.0

        # Gradient of normalised bacteria map, precomputed once per timestep
        grad_bn_x = environment.grad_bn_x[row, col]  # d b_norm / dx
        grad_bn_y = environment.grad_bn_y[row, col]  # d b_norm / dy

        grad_bn = np.array([grad_bn_x, grad_bn_y])

        # On-patch uses an absolute concentration check so low-density regimes still count as source.
        self.on_patch = environment.is_on_patch(b_local)

        return b_norm, grad_bn
    
    def __check_arena_boundary(self, environment, new_x, new_y):        
        """
        Reflective circular boundary of radius R.
        If the proposed position exits the circle, reflect the velocity
        vector off the normal at the boundary, and flip heading accordingly.
        """
        R = environment.R
        dist = np.sqrt(new_x**2 + new_y**2)

        if dist > R:
            # --- reflect position ---
            # normal at boundary point (pointing inward)
            nx = new_x / dist
            ny = new_y / dist

            # reflect position back inside
            new_x = new_x - 2 * (dist - R) * nx
            new_y = new_y - 2 * (dist - R) * ny

            # clamp in case of floating point overshoot
            dist2 = np.sqrt(new_x**2 + new_y**2)
            if dist2 > R:
                new_x = new_x * (R / dist2)
                new_y = new_y * (R / dist2)

            # --- reflect heading angle ---
            # reflect theta off the inward normal
            # inward normal angle
            normal_angle = np.arctan2(-ny, -nx)
            # angle of current heading relative to normal
            incident = self.theta - normal_angle
            # reflected heading
            self.theta = normal_angle - incident

        return new_x, new_y

    def __sample_surface_shed_interval(self):
        """Sample random shedding interval in timesteps (exponential waiting time)."""
        return max(1, int(np.random.exponential(self.surface_shed_mean_steps)))

    def __deposit_with_jitter(self, environment, amount, jitter_radius):
        """Deposit amount near current location with small random spatial jitter."""
        if amount <= 0.0:
            return 0.0

        if jitter_radius > 0.0:
            ang = np.random.uniform(0.0, 2.0 * np.pi)
            r = np.sqrt(np.random.uniform(0.0, 1.0)) * jitter_radius
            x_drop = self.x + r * np.cos(ang)
            y_drop = self.y + r * np.sin(ang)
        else:
            x_drop = self.x
            y_drop = self.y

        environment.add_bacteria_source(x_drop, y_drop, amount)
        return amount

    def __release_delayed_gut_cells(self):
        """Move matured eaten cells into gut reservoir once their delay has elapsed."""
        while self._gut_drop_release_queue and self._gut_drop_release_queue[0][0] <= self.timestep:
            _, amount = self._gut_drop_release_queue.popleft()
            self.gut_cells_for_drop += amount

    
    # ------------------------------------------------------------------
    # Public step
    # ------------------------------------------------------------------
    def step(self, environment):
        """
        Advance worm by one timestep dt using Euler-Maruyama SDE:

            dtheta = chi_theta * (u_perp . grad_b_norm) * dt
                     + sqrt(2 * D_eff * dt) * N(0,1)

            dx     = v(b_norm) * u(theta_new) * dt
        """
        dt = environment.dt

        # --- sample local field ---
        b_norm, grad_bn = self.__get_bacteria_and_gradient(environment)
        self.surface_drop_step = 0.0
        self.gut_drop_step = 0.0

        # --- local intake used for deposition bookkeeping
        # Global environment depletion is handled by the paper-style term in Environment.
        # Intensity is constant per feeding worm (paper a term), gated only by on_patch.
        rate_per_worm = max(float(getattr(self, "feeding_cells_per_worm", 70.0)), 0.0)
        if self.on_patch:
            dB_worm = rate_per_worm * dt
        else:
            dB_worm = 0.0
        self.cells_eaten_step = dB_worm
        self.cells_eaten_total += dB_worm
        environment.register_feeding_worm(self.on_patch)

        if self.deposition_enabled:
            if (not self.single_deposit_per_worm) and self.surface_shedding_enabled and self.on_patch:
                pickup = self.surface_pickup_rate * dt
                self.surface_cells_carried = min(self.surface_cells_carried + pickup, self.surface_carry_capacity)

            deposits_remaining = (self.max_deposits_per_worm == 0) or (self.num_deposits_done < self.max_deposits_per_worm)
            if self.gut_drop_enabled and self.deposition_fraction > 0.0 and deposits_remaining:
                delayed_cells = self.deposition_fraction * dB_worm
                release_step = self.timestep + self.gut_drop_delay_steps
                self._gut_drop_release_queue.append((release_step, delayed_cells))

        # --- diagnostic suppression inside patch ---
        suppress = getattr(self, "suppress_patch_chemotaxis", False)
        on_patch = self.on_patch

        if suppress and on_patch:
            # Force pure random walk: no gradient bias, no speed/noise suppression
            b_norm  = 0.0               # v -> v_max, D_eff -> D_theta
            grad_bn = np.zeros(2)       # no chemotactic turning

        # --- unit vectors from current angle ---
        u, u_perp = self.__unit_vectors(self.theta)

        # --- chemotactic turning (deterministic) ---
        proj       = np.dot(u_perp, grad_bn)            # scalar
        dtheta_det = self.chi_theta * proj * dt
        D_theta = (self.dtheta ** 2) / (2.0 * environment.dt)

        # --- angular noise suppressed by bacteria (stochastic) ---
        D_eff        = D_theta * np.exp(-self.beta_b * b_norm)
        dtheta_noise = np.sqrt(2.0 * D_eff * dt) * np.random.normal()

        # --- update angle ---
        self.theta = self.theta + dtheta_det + dtheta_noise

        # --- speed suppressed by bacteria ---
        v = self.__speed_from_b(b_norm, self.v_min, self.v_max, self.alpha)

        # --- update position using new angle ---
        u_new, _ = self.__unit_vectors(self.theta)
        new_x    = self.x + v * u_new[0] * dt
        new_y    = self.y + v * u_new[1] * dt

        # --- boundary check ---
        self.x, self.y = self.__check_arena_boundary(environment, new_x, new_y)

        # --- deposit after movement ---
        self.__release_delayed_gut_cells()
        self.__drop_bacteria(environment)
        self.cells_available_to_deposit = self.surface_cells_carried + (self.gut_cells_for_drop * self.gut_drop_conversion_fraction)
        self.timestep += 1

    def __drop_bacteria(self, environment):
        """Deposit via two pathways: random surface shedding and threshold-triggered drop."""
        if not self.deposition_enabled:
            return

        if self.max_deposits_per_worm > 0 and self.num_deposits_done >= self.max_deposits_per_worm:
            return

        # 1) Surface shedding at random intervals
        if self.surface_shedding_enabled and self.timestep >= self.next_surface_shed_timestep:
            # Shedding occurs only outside patch; carrying can still be acquired on patch.
            if (not self.on_patch) and self.surface_cells_carried > 0.0:
                frac = np.random.uniform(self.surface_shed_fraction_min, self.surface_shed_fraction_max)
                amount = self.surface_cells_carried * max(frac, 0.0)
                if self.surface_shed_max_cells > 0.0:
                    amount = min(amount, self.surface_shed_max_cells)

                dropped = self.__deposit_with_jitter(environment, amount, self.surface_drop_jitter_radius)
                self.surface_cells_carried = max(self.surface_cells_carried - dropped, 0.0)
                self.surface_drop_step += dropped
                self.surface_drop_total += dropped

                self.next_surface_shed_timestep += self.__sample_surface_shed_interval()

        # 2) Gut-drop events after consuming threshold amount
        if not self.gut_drop_enabled:
            return

        # Gut-drop deposition is allowed only outside the patch.
        if self.on_patch:
            return

        events = 0
        while self.gut_cells_for_drop >= self.gut_drop_trigger_cells and events < self.gut_drop_max_events_per_step:
            if self.max_deposits_per_worm > 0 and self.num_deposits_done >= self.max_deposits_per_worm:
                break
            self.gut_cells_for_drop -= self.gut_drop_trigger_cells
            amount = self.gut_drop_trigger_cells * self.gut_drop_conversion_fraction
            dropped = self.__deposit_with_jitter(environment, amount, self.gut_drop_jitter_radius)
            self.gut_drop_step += dropped
            self.gut_drop_total += dropped
            if dropped > 0.0:
                self.num_deposits_done += 1
            events += 1