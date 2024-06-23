import numpy as np

G = 6.67430e-11  # Gravitational constant

celestial_bodies = {
    "Sun": {"mass": 1.989e30, "radius": 696340e3, "color": "yellow"},
    "Mercury": {"mass": 3.3011e23, "radius": 2439.7e3, "color": "gray"},
    "Venus": {"mass": 4.8675e24, "radius": 6051.8e3, "color": "orange"},
    "Earth": {"mass": 5.97237e24, "radius": 6371.0e3, "color": "blue"},
    "Mars": {"mass": 6.4171e23, "radius": 3389.5e3, "color": "red"},
    "Jupiter": {"mass": 1.8982e27, "radius": 69911e3, "color": "brown"},
    "Saturn": {"mass": 5.6834e26, "radius": 58232e3, "color": "gold"},
    "Uranus": {"mass": 8.6810e25, "radius": 25362e3, "color": "lightblue"},
    "Neptune": {"mass": 1.02413e26, "radius": 24622e3, "color": "blue"}
}

def initial_conditions():
    positions = {
        "Sun": np.array([0, 0, 0]),
        "Mercury": np.array([57.9e9, 0, 0]),
        "Venus": np.array([108.2e9, 0, 0]),
        "Earth": np.array([149.6e9, 0, 0]),
        "Mars": np.array([227.9e9, 0, 0]),
        "Jupiter": np.array([778.5e9, 0, 0]),
        "Saturn": np.array([1.434e12, 0, 0]),
        "Uranus": np.array([2.871e12, 0, 0]),
        "Neptune": np.array([4.495e12, 0, 0])
    }
    
    velocities = {name: np.array([0, np.sqrt(G * celestial_bodies["Sun"]["mass"] / r[0]), 0]) 
                  for name, r in positions.items() if name != "Sun"}
    velocities["Sun"] = np.array([0, 0, 0])
    
    return positions, velocities

def gravitational_force(pos1, pos2, mass1, mass2):
    r = pos2 - pos1
    distance = np.linalg.norm(r)
    force_magnitude = G * mass1 * mass2 / (distance ** 2)
    force_direction = r / distance
    return force_magnitude * force_direction

def update_positions_and_velocities(positions, velocities, dt):
    new_positions = {}
    new_velocities = {}
    
    for body1 in celestial_bodies:
        total_force = np.zeros(3)
        for body2 in celestial_bodies:
            if body1 != body2:
                force = gravitational_force(positions[body1], positions[body2], 
                                            celestial_bodies[body1]["mass"], 
                                            celestial_bodies[body2]["mass"])
                total_force += force
        
        acceleration = total_force / celestial_bodies[body1]["mass"]
        new_velocities[body1] = velocities[body1] + acceleration * dt
        new_positions[body1] = positions[body1] + new_velocities[body1] * dt
    
    return new_positions, new_velocities