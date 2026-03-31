import numpy as np

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

        # Normalised density at worm position
        b_norm = bmap[row, col] / K

        # # Gradient of normalised bacteria map (returns [grad_row, grad_col])
        # # np.gradient returns derivatives w.r.t. array indices; divide by dx for cm units
        # grad_row, grad_col = np.gradient(bmap / K, dx)

        # # grad_col -> d/dx,  grad_row -> d/dy
        # grad_bn = np.array([grad_col[row, col], grad_row[row, col]])

        # Gradient of normalised bacteria map, precomputed once per timestep
        grad_bn_x = environment.grad_bn_x[row, col]  # d b_norm / dx
        grad_bn_y = environment.grad_bn_y[row, col]  # d b_norm / dy

        grad_bn = np.array([grad_bn_x, grad_bn_y])

        return b_norm, grad_bn
    
    def __check_arena_boundary(self, environment, new_x, new_y):
        # """
        # Keep worm inside arena. If proposed move exits boundary, 
        # stay at current position (wall absorption).
        # Returns valid (x, y).
        # """
        # x = np.clip(new_x, environment.x_min, environment.x_max)
        # y = np.clip(new_y, environment.x_min, environment.x_max)
        # return x, y
        
        # """
        # Keep worm inside circular arena of radius R centered at origin.
        # If proposed move exits boundary, clamp to the boundary circle.
        # """
        # R = environment.R
        # dist = np.sqrt(new_x**2 + new_y**2)
        # if dist > R:
        #     # project back onto circle boundary
        #     scale = R / dist
        #     new_x = new_x * scale
        #     new_y = new_y * scale
        # return new_x, new_y
        
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

        # --- diagnostic suppression inside patch ---
        suppress = getattr(self, "suppress_patch_chemotaxis", False)
        on_patch = b_norm > 0.0

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

    def __drop_bacteria(self, environment):
        """Drop bacteria source at current location at fixed intervals"""
        # if not self.bacteria_enabled:
        #     return
        if self.timestep >= self.next_drop_timestep:
            environment.add_bacteria_source(self.x, self.y, self.bacteria_amount)
            self.next_drop_timestep += int(self.bacteria_drop_interval)