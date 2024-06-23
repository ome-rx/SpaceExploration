import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from stable_baselines3 import PPO
from orbital_mechanics import celestial_bodies, initial_conditions
from spacecraft_env import SpacecraftEnv
from visualisations import SolarSystemVisualizer

def update_training(frame, visualizer, env, model):
    if frame == 0:
        obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    visualizer.positions = env.positions
    artists = visualizer.update(frame)
    
    plt.title(f'Training - Step: {frame}, Reward: {reward:.2f}, Fuel: {env.fuel:.2f}')
    
    if done:
        obs, _ = env.reset()
    
    return artists

def update_final_path(frame, visualizer, best_trajectory):
    if frame < len(best_trajectory):
        visualizer.positions["Spacecraft"] = best_trajectory[frame]
    artists = visualizer.update(frame)
    
    plt.title(f'Best Path - Step: {frame}')
    
    return artists

def main():
    # Initialize the solar system
    initial_pos, initial_vel = initial_conditions()
    
    # Ask user for destination
    destinations = list(celestial_bodies.keys())
    destinations.remove("Sun")
    destinations.remove("Earth")
    print("Choose a destination:")
    for i, dest in enumerate(destinations):
        print(f"{i+1}. {dest}")
    choice = int(input("Enter the number of your choice: ")) - 1
    destination = destinations[choice]

    # Create the environment
    env = SpacecraftEnv(initial_pos, initial_vel, destination)

    # Set up the visualization
    visualizer = SolarSystemVisualizer()
    visualizer.positions = env.positions

    # Create and train the model
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Visualize training process
    fig = plt.figure(figsize=(12, 8))
    training_anim = FuncAnimation(fig, update_training, frames=1000, 
                                  fargs=(visualizer, env, model), interval=50, blit=True)
    plt.show()

    # Continue training without visualization
    model.learn(total_timesteps=100000)

    # Find the best path
    best_reward = -np.inf
    best_trajectory = None

    for _ in range(10):  # Run 10 episodes to find the best path
        obs, _ = env.reset()
        done = False
        episode_trajectory = []
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_trajectory.append(env.positions["Spacecraft"].copy())
            episode_reward += reward
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_trajectory = episode_trajectory

    # Visualize the best path
    fig = plt.figure(figsize=(12, 8))
    final_anim = FuncAnimation(fig, update_final_path, frames=len(best_trajectory), 
                               fargs=(visualizer, best_trajectory), interval=50, blit=True)
    plt.show()

    # Run some test episodes
    for episode in range(5):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Test Episode {episode + 1}: Total Reward: {total_reward}")

if __name__ == "__main__":
    main()