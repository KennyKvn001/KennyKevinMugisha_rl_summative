import gymnasium as gym
from gymnasium import spaces
import numpy as np

from custom_env.map_generator import generate_map
from custom_env.renderer import Renderer


class CityEnv(gym.Env):
    def __init__(self, grid_size=(10, 10), render_mode="human"):
        super(CityEnv, self).__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.renderer = Renderer(grid_size) if render_mode == "human" else None

        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(grid_size[0], grid_size[1], 3),  # 3 channels: agent, landmarks, goal
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        self.map, self.destinations = generate_map(self.grid_size)
        self.agent_pos = np.array([0, 0])
        self.goal_type = np.random.choice(list(self.destinations.keys()))
        self.goal_pos = self.destinations[self.goal_type]

        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros((*self.grid_size, 3), dtype=np.float32)
        obs[self.agent_pos[0], self.agent_pos[1], 0] = 1  # agent
        for loc in self.destinations.values():
            obs[loc[0], loc[1], 1] = 1  # destinations
        obs[self.goal_pos[0], self.goal_pos[1], 2] = 1  # goal
        return obs

    def step(self, action):
        self.steps += 1
        move = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}  # up, down, left, right
        new_pos = self.agent_pos + move.get(action, [0, 0])

        if self._valid_pos(new_pos):
            self.agent_pos = new_pos

        done = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 10 if done else -1
        if self.steps >= 100:
            done = True

        return self._get_obs(), reward, done, False, {}

    def _valid_pos(self, pos):
        return 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]

    def render(self):
        if self.renderer:
            self.renderer.render(self.agent_pos, self.destinations, self.goal_pos)

    def close(self):
        if self.renderer:
            self.renderer.close()
