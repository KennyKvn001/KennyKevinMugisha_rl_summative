import gymnasium as gym
from custom_env.map_navigation_env import MapNavigationEnv


def create_env(training_mode=True):
    """
    Initializes the MapNavigationEnv with default config and returns a Gym-compatible environment.
    """
    if training_mode:
        # For training: no rendering, training mode enabled
        env = MapNavigationEnv(grid_size=(15, 15), render_mode=None, training_mode=True)
    else:
        # For evaluation: with rendering, training mode disabled
        env = MapNavigationEnv(grid_size=(15, 15), render_mode="human", training_mode=False)
    return env
