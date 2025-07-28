import numpy as np


def generate_map(grid_size):
    landmarks = {
        "hospital": np.random.randint(0, grid_size[0], size=2),
        "school": np.random.randint(0, grid_size[0], size=2),
        "market": np.random.randint(0, grid_size[0], size=2),
    }
    return np.zeros(grid_size), landmarks
