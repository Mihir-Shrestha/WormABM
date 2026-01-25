import numpy as np

class Worm(object):
    def __init__(self, params):
        self.__set_params(params)
        self.__init_position()
        self.__init_conditions()
    
    def __set_params(self, params):
        # Store all parameters as instance variables
        for key, val in params.items():
            self.__dict__[key] = val
    
    def __init_position(self):
        # Initialize worm position at origin
        self.x = 0
        self.y = 0
    
    def __init_conditions(self):
        # Initialize worm state variables
        self.angle = np.random.uniform(0, 2 * np.pi) # Random initial angle
        self.timestep = 0

        # Run and tumble state
        self.state = 'run'
        self.state_timer = 0
        self.run_duration = self.__sample_run_duration()
        self.tumble_duration = self.__sample_tumble_duration()

        # Bacteria drop (fixed interval)
        self.bacteria_enabled = self.bacteria_enabled
        self.bacteria_drop_interval = self.bacteria_drop_interval
        self.bacteria_amount = self.bacteria_amount
        self.next_drop_timestep = 0

    def __sample_run_duration(self):
        # Sample run duration from exponential distribution
        # Adjust rate parameter (1/mean) as needed
        return np.random.exponential(self.worm_mean_run_duration)
    
    def __sample_tumble_duration(self):
        return np.random.exponential(self.worm_mean_tumble_duration)
        
    def __check_arena_boundary(self, environment, coord):
        # Check if coordinate is within bounds
        return environment.x_min < coord < environment.x_max
    
    def __update_movement(self, environment):
        # Update worm position based on current angle and speed
        if self.state == "run":
            # Continue in current direction
        # Sample tumble duration from exponential distribution
            next_x = self.x + self.worm_step_size * np.cos(self.angle)
            next_y = self.y + self.worm_step_size * np.sin(self.angle)
        else:  # tumble
            # Stay in place during tumble
            next_x = self.x
            next_y = self.y

        # Check arena boundaries
        move_in_x = self.__check_arena_boundary(environment, next_x)
        move_in_y = self.__check_arena_boundary(environment, next_y)

        if not move_in_x:
            next_x = self.x
        if not move_in_y:
            next_y = self.y
        
        self.x = next_x
        self.y = next_y

    def __update_angle(self):
        # Update heading based on state
        if self.state == "run":
            # Small noise during run to maintain roughly straight path
            angle_change = np.random.normal(0, self.worm_turn_noise)
            self.angle += angle_change
        else:  # tumble
            # Large random turn during tumble
            self.angle = np.random.uniform(0, 2 * np.pi)
        
        # Normalize angle to [0, 2pi]
        self.angle = self.angle % (2 * np.pi)

    def __update_state(self):
        # Update state timer and switch state if duration elapsed
        self.state_timer += 1
        
        if self.state == "run":
            if self.state_timer >= self.run_duration:
                self.state = "tumble"
                self.state_timer = 0
                self.tumble_duration = self.__sample_tumble_duration()
        else:  # tumble
            if self.state_timer >= self.tumble_duration:
                self.state = "run"
                self.state_timer = 0
                self.run_duration = self.__sample_run_duration()

    def __drop_bacteria(self, environment):
        if not self.bacteria_enabled:
            return
        if self.timestep >= self.next_drop_timestep:
            environment.add_bacteria_source(self.x, self.y, self.bacteria_amount)
            self.next_drop_timestep += int(self.bacteria_drop_interval)

    def step(self, environment):
        # Single time step update, update angle and move
        self.__update_angle()
        self.__update_movement(environment)
        self.__update_state()
        self.__drop_bacteria(environment)
        self.timestep += 1