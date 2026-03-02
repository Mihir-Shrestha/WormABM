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
        self.__precompute_diffusion_factor()

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
        self.init_bacteria_patch(x_center=0.0, y_center=0.0, radius=0.1, amplitude=0.5)

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

    def __precompute_diffusion_factor(self):
        """
        Precompute the Fourier space diffusion factor for half timestep.
        This only needs to be computed once since D, dt, dx are fixed.
        exp(-D * k^2 * dt/2) in Fourier space
        """
        nx, ny = self.bacteria_map.shape

        # Compute wavenumbers for each axis
        kx = np.fft.fftfreq(nx, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, d=self.dx) * 2 * np.pi

        # Create 2D wavenumber grid
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K2 = KX**2 + KY**2

        # Diffusion factor for half timestep (Strang splitting)
        self.diff_factor = np.exp(-self.diffusion_coefficient * K2 * self.dt / 2)

    def __diffusion_step(self, b):
        """
        Apply exact diffusion for half timestep in Fourier space.
        b_new = IFFT(FFT(b) * exp(-D*k^2*dt/2))
        """
        return np.real(np.fft.ifft2(np.fft.fft2(b) * self.diff_factor))

    def __growth_step(self, b):
        """
        Apply exact logistic growth for full timestep using analytical solution.
        db/dt = r * b * (1 - b/K)
        b(t+dt) = K*b / (b + (K-b)*exp(-r*dt))
        """
        r = getattr(self, 'bacteria_growth_rate', 1.0)
        K = getattr(self, 'bacteria_carrying_capacity', 1.0)

        # Clip negatives before growth
        b = np.clip(b, 0.0, None)

        if r == 0.0:
            return b

        exp_term = np.exp(-r * self.dt)
        numerator = K * b
        denominator = b + (K - b) * exp_term

        with np.errstate(divide='ignore', invalid='ignore'):
            b_new = numerator / denominator

        # Handle edge cases (nan, inf)
        b_new = np.nan_to_num(b_new, nan=0.0, posinf=K, neginf=0.0)
        return b_new
    
    def update_bacteria_map(self):
        """
        Solve db/dt = D*∇²b + r*b*(1-b) using Strang splitting:
        1. Half diffusion step (exact, Fourier space)
        2. Full logistic growth step (exact, analytical)
        3. Half diffusion step (exact, Fourier space)
        """
        # Step 1: half diffusion
        b = self.__diffusion_step(self.bacteria_map)

        # Step 2: full logistic growth
        b_before_growth = b.copy()  # b before growth for diagnostics
        b = self.__growth_step(b)

        # Diagnostic: print max growth change per step
        growth_change = np.max(np.abs(b - b_before_growth))
        print(f"Max growth change this step: {growth_change:.6f}")

        # Step 3: half diffusion
        b = self.__diffusion_step(b)

        # Clip only negative values (unphysical)
        self.bacteria_map = np.clip(b, 0.0, None)

    def add_bacteria_source(self, x, y, amount):
        """Deposits bacteria as a small patch at (x,y)"""
        self.init_bacteria_patch(x_center=x, y_center=y, radius=0.03, amplitude=amount)

    def convert_xy_to_index(self, xy):
        """Convert real coordinates (x or y) to grid indices"""
        index = ((xy - self.x_min) / (self.x_max - self.x_min)) * self.x_grid.shape[0]
        return index