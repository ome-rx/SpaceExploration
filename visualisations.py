import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from orbital_mechanics import celestial_bodies

class SolarSystemVisualizer:
    def __init__(self, time_step=86400):  # 1 day time step
        self.time_step = time_step
        self.fig, self.ax = self.setup_plot()
        self.bodies = self.setup_bodies()
        self.trails = self.setup_trails()
        self.positions = {body: np.zeros(3) for body in celestial_bodies.keys()}
        self.positions["Spacecraft"] = np.zeros(3)

    def setup_plot(self):
        plt.style.use('default')
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        
        max_distance = 5e12  # Approximate distance to Neptune
        ax.set_xlim(-max_distance, max_distance)
        ax.set_ylim(-max_distance, max_distance)
        ax.set_zlim(-max_distance, max_distance)
        
        ax.set_box_aspect((1, 1, 0.5))  # Adjust the aspect ratio
        return fig, ax

    def setup_bodies(self):
        bodies = {}
        for body, data in celestial_bodies.items():
            size = np.log(data['radius']) * 2
            color = data['color']
            bodies[body] = self.ax.plot([], [], [], 'o', markersize=size, color=color, label=body)[0]

        bodies["Spacecraft"] = self.ax.plot([], [], [], 'o', markersize=5, color='white', label='Spacecraft')[0]
        self.ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        return bodies

    def setup_trails(self):
        trails = {}
        for body in celestial_bodies:
            if body != "Sun":
                trails[body] = self.ax.plot([], [], [], '-', linewidth=0.5, color=celestial_bodies[body]['color'], alpha=0.3)[0]
        trails["Spacecraft"] = self.ax.plot([], [], [], '-', linewidth=0.5, color='white', alpha=0.3)[0]
        return trails

    def update(self, frame):
        for body, plot in self.bodies.items():
            pos = self.positions[body]
            plot.set_data([pos[0]], [pos[1]])
            plot.set_3d_properties([pos[2]])
            
            if body != "Sun":
                trail = self.trails[body]
                x, y, z = trail.get_data_3d()
                x = np.append(x, pos[0])[-100:]  # Keep only last 100 points
                y = np.append(y, pos[1])[-100:]
                z = np.append(z, pos[2])[-100:]
                trail.set_data_3d(x, y, z)
    
        return list(self.bodies.values()) + list(self.trails.values())

    def animate(self, frames=365):
        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(self.fig, self.update, frames=frames, interval=50, blit=True)
        return anim