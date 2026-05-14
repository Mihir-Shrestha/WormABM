import numpy as np

class Worm(object):
    """
    Single worm with random walk + source-dependent dwelling.

    Motion uses chemotactic turning with a stochastic term:
        dtheta = chi_theta * (u_perp . grad_b_norm) * dt
                 + sqrt(2 * D_eff * dt) * N(0,1)
        dx     = v(b_norm) * u(theta_new) * dt

    A worm becomes "fed" once it visits one of the two initial bacteria sources.
    After a fixed delay, fed worms deposit forever at a fixed interval.
    """

    def __init__(self, params):
        self.__set_params(params)
        self.__init_position()
        self.on_patch = False       
        self.on_source = False
        self.fed = False
        self.first_fed_timestep = -1
        self.next_deposit_timestep = None

        self.cells_eaten_step = 0.0
        self.cells_eaten_total = 0.0
        self.cells_available_to_deposit = 0.0

        # Legacy deposition diagnostics (kept so existing logging continues to work).
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

        # Fed-state deposition parameters.
        self.fed_deposit_delay_steps = max(
            int(getattr(self, "fed_deposit_delay_steps", getattr(self, "gut_drop_delay_steps", 1000))),
            0,
        )
        self.fed_deposit_interval_steps = max(
            int(getattr(self, "fed_deposit_interval_steps", getattr(self, "deposition_interval_steps", 100))),
            1,
        )
        self.fed_deposit_amount = max(
            float(getattr(self, "fed_deposit_amount", getattr(self, "gut_drop_trigger_cells", 35.0))),
            0.0,
        )
        self.fed_deposit_jitter_radius = max(
            float(getattr(self, "fed_deposit_jitter_radius", getattr(self, "gut_drop_jitter_radius", 0.05))),
            0.0,
        )
    
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
        b_norm : float
            Bacteria density normalised to [0, 1] using K_rho.
        grad_bn : np.ndarray
            Local bacteria gradient [d b_norm / dx, d b_norm / dy].
        """
        bmap = environment.bacteria_map
        norm_scale = max(environment.K_rho, 1e-12)
        source_map = environment.source_map

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
        b_norm = float(np.clip(b_local / norm_scale, 0.0, 1.0))
        grad_bn = np.array([
            environment.grad_bn_x[row, col],
            environment.grad_bn_y[row, col],
        ])

        # Only the initial sources can feed worms, but on_patch tracks any bacteria.
        source_local = source_map[row, col]
        self.on_source = environment.is_on_source(source_local)
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
            # outward normal at boundary point
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

            # --- reflect heading by reflecting direction vector ---
            # v_ref = v - 2 * (v · n) * n
            vx = np.cos(self.theta)
            vy = np.sin(self.theta)
            dot_vn = vx * nx + vy * ny
            vx_ref = vx - 2.0 * dot_vn * nx
            vy_ref = vy - 2.0 * dot_vn * ny
            self.theta = np.arctan2(vy_ref, vx_ref)

        return new_x, new_y

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

    def __deposit_if_ready(self, environment):
        """After fed-state delay, deposit forever at fixed intervals."""
        if not self.deposition_enabled or (not self.fed) or self.fed_deposit_amount <= 0.0:
            return

        if self.next_deposit_timestep is None or self.timestep < self.next_deposit_timestep:
            return

        # Catch up if a long delay created multiple due events.
        due_events = 1 + (self.timestep - self.next_deposit_timestep) // self.fed_deposit_interval_steps
        for _ in range(int(due_events)):
            dropped = self.__deposit_with_jitter(environment, self.fed_deposit_amount, self.fed_deposit_jitter_radius)
            self.gut_drop_step += dropped
            self.gut_drop_total += dropped

        self.next_deposit_timestep += int(due_events) * self.fed_deposit_interval_steps

    
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

        # --- sample local bacteria field ---
        b_norm, grad_bn = self.__get_bacteria_and_gradient(environment)
        self.surface_drop_step = 0.0
        self.gut_drop_step = 0.0

        # "Feeding" is a state flag: once worm reaches either source it remains fed forever.
        if self.on_source and (not self.fed):
            self.fed = True
            self.first_fed_timestep = self.timestep
            self.next_deposit_timestep = self.timestep + self.fed_deposit_delay_steps

        self.cells_eaten_step = 1.0 if self.on_source else 0.0
        self.cells_eaten_total += self.cells_eaten_step

        # --- diagnostic suppression inside patch ---
        suppress = getattr(self, "suppress_patch_chemotaxis", False)
        on_patch = self.on_patch

        if suppress and on_patch:
            # Suppress source-dependent modulation inside source patch if requested.
            b_norm  = 0.0
            grad_bn = np.zeros(2)

        # --- chemotactic turning with stochastic term ---
        u, u_perp = self.__unit_vectors(self.theta)
        proj = np.dot(u_perp, grad_bn)
        dtheta_det = self.chi_theta * proj * dt
        D_theta = (self.dtheta ** 2) / (2.0 * environment.dt)
        D_eff = D_theta * np.exp(-self.beta_b * b_norm)
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

        # --- fed worms deposit forever after delay ---
        self.__deposit_if_ready(environment)
        self.cells_available_to_deposit = 1.0 if self.fed else 0.0
        self.timestep += 1