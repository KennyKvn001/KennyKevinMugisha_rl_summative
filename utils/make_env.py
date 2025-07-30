import gymnasium as gym
from custom_env.map_navigation_env import MapNavigationEnv


def create_env():
    """
    Initializes the MapNavigationEnv with default config and returns a Gym-compatible environment.
    """
    env = MapNavigationEnv(grid_size=(15, 15), render_mode="human", training_mode=False)
    return env
