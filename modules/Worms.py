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
    
    def __check_arena_boundary(self, environment, coord):
        # Check if coordinate is within bounds
        return environment.x_min < coord < environment.x_max
    
    def __update_movement(self, environment):
        # Update worm position based on current angle and speed
        # Calculate intended new position
        next_x = self.x + self.worm_step_size * np.cos(self.angle)
        next_y = self.y + self.worm_step_size * np.sin(self.angle)

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
        # Update heading with random turn noise
        # Add Gausian noise to current angle
        angle_change = np.random.normal(0, self.worm_turn_noise)
        self.angle += angle_change

        # Normalize angle to [0, 2pi]
        self.angle = self.angle % (2 * np.pi)

    def step(self, environment):
        # Single time step update, update angle and move
        self.__update_angle()
        self.__update_movement(environment)
        self.timestep += 1