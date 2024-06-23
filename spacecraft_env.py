import numpy as np
import gymnasium as gym
from gymnasium import spaces
from orbital_mechanics import celestial_bodies, gravitational_force, G, update_positions_and_velocities

class SpacecraftEnv(gym.Env):
    def __init__(self, initial_positions, initial_velocities, destination, time_step=3600):
        super(SpacecraftEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        self.initial_positions = initial_positions
        self.initial_velocities = initial_velocities
        self.positions = initial_positions.copy()
        self.velocities = initial_velocities.copy()
        self.destination = destination
        self.spacecraft_mass = 1000  # kg
        self.max_fuel = 1000  # kg
        self.fuel = self.max_fuel
        self.time_step = time_step
        self.max_steps = 8760  # 1 year
        self.trajectory = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.positions = self.initial_positions.copy()
        self.velocities = self.initial_velocities.copy()
        self.positions["Spacecraft"] = self.initial_positions.copy()  # Initialize spacecraft's position
        self.velocities["Spacecraft"] = self.initial_velocities.copy()  # Initialize spacecraft's velocity
        self.fuel = self.max_fuel
        self.step_count = 0
        self.trajectory = [self.positions["Spacecraft"].copy()]
        return self._get_observation(), {}


    def step(self, action):
        self.step_count += 1
        
        # Apply thrust to spacecraft
        thrust = action * 1e-3  # Scale down the action
        self.velocities["Spacecraft"] += thrust * self.time_step / self.spacecraft_mass
        self.fuel -= np.linalg.norm(thrust) * self.time_step * 1e-3  # Simple fuel consumption model
        
        # Update positions and velocities of all bodies
        new_positions, new_velocities = update_positions_and_velocities(self.positions, self.velocities, self.time_step)
    
        self.positions = new_positions
        self.velocities = new_velocities
    
        self.trajectory.append(self.positions["Spacecraft"].copy())
    
        # Calculate reward
        distance_to_target = np.linalg.norm(self.positions["Spacecraft"] - self.positions[self.destination])
        reward = -distance_to_target / 1e9  # Convert to billions of km for nicer numbers
        
        # Check if done
        done = (distance_to_target < celestial_bodies[self.destination]["radius"] * 10 or self.fuel <= 0 or self.step_count >= self.max_steps)
    
        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        observation = np.concatenate([
            self.positions["Spacecraft"],
            self.velocities["Spacecraft"],
            self.positions[self.destination],
            self.velocities[self.destination],
            [self.fuel]
        ])
        return observation

    def render(self, mode='human'):
        pass
