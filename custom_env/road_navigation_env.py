"""
Road Navigation Environment
Agent must follow roads to reach destinations
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from custom_env.map_generator import create_road_network_map, is_valid_position
from custom_env.renderer import Renderer


class RoadNavigationEnv(gym.Env):
    def __init__(self, grid_size=(15, 15), render_mode="human"):
        super(RoadNavigationEnv, self).__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.renderer = Renderer(grid_size) if render_mode == "human" else None

        # Action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        # Observation space: road map + agent position + goal position
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(*grid_size, 3),  # 3 channels: map, agent, goal
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        """Reset environment and generate new road network"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Generate road network map
        self.road_map, self.destinations, self.road_positions = create_road_network_map(
            self.grid_size
        )

        # Place agent at a random destination
        self.agent_pos = np.array(random.choice(list(self.destinations.values())))

        # Choose a different destination as goal
        available_goals = [
            name
            for name, pos in self.destinations.items()
            if not np.array_equal(pos, self.agent_pos)
        ]
        self.current_goal = random.choice(available_goals)
        self.goal_pos = np.array(self.destinations[self.current_goal])

        self.steps = 0
        self.max_steps = 300  # Increased due to complex paths

        # Initialize distance tracking for reward calculation
        self.previous_distance = np.abs(self.agent_pos[0] - self.goal_pos[0]) + np.abs(
            self.agent_pos[1] - self.goal_pos[1]
        )

        return self._get_obs(), {}

    def _get_obs(self):
        """Get observation with map, agent position, and goal position"""
        obs = np.zeros((*self.grid_size, 3), dtype=np.float32)

        # Channel 0: Road map (0=obstacle, 1=road, 2=destination)
        obs[:, :, 0] = self.road_map.astype(np.float32)

        # Channel 1: Agent position
        obs[self.agent_pos[0], self.agent_pos[1], 1] = 1.0

        # Channel 2: Goal position
        obs[self.goal_pos[0], self.goal_pos[1], 2] = 1.0

        return obs

    def step(self, action):
        """Take a step in the environment"""
        self.steps += 1

        # Define movement directions
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right

        if action in moves:
            delta = moves[action]
            new_pos = self.agent_pos + delta

            # Check if new position is valid (on road or destination)
            if is_valid_position(self.road_map, tuple(new_pos)):
                self.agent_pos = new_pos
            # If invalid move, agent stays in place (no movement)

        # Check if goal reached
        done = np.array_equal(self.agent_pos, self.goal_pos)

        # Calculate reward with distance-based incentives
        if done:
            # Success reward based on efficiency
            efficiency_bonus = max(0, (self.max_steps - self.steps) // 10)
            reward = 10 + efficiency_bonus  # Base success + efficiency bonus
            # Success message handled by main script
        else:
            # Distance-based reward to encourage progress
            current_distance = np.abs(self.agent_pos[0] - self.goal_pos[0]) + np.abs(
                self.agent_pos[1] - self.goal_pos[1]
            )

            # Base step penalty
            reward = -0.5

            # Distance-based incentive (very small to avoid greedy behavior)
            if hasattr(self, "previous_distance"):
                if current_distance < self.previous_distance:
                    reward += 1  # Small reward for getting closer
                elif current_distance > self.previous_distance:
                    reward -= 1  # Small penalty for moving away

            self.previous_distance = current_distance

            # Penalty for hitting walls (trying invalid moves)
            if action in moves:
                delta = moves[action]
                attempted_pos = self.agent_pos + delta
                if not is_valid_position(self.road_map, tuple(attempted_pos)):
                    reward = -5  # Penalty for trying to move into obstacle

        # Episode ends if goal reached or max steps exceeded
        if self.steps >= self.max_steps:
            done = True
            if not np.array_equal(self.agent_pos, self.goal_pos):
                reward = -10  # Penalty for not reaching goal in time
                # Timeout message handled by main script

        return (
            self._get_obs(),
            reward,
            done,
            False,
            {
                "goal": self.current_goal,
                "steps": self.steps,
                "agent_pos": tuple(self.agent_pos),
                "goal_pos": tuple(self.goal_pos),
            },
        )

    def render(self):
        """Render the environment"""
        if self.renderer:
            self.renderer.render(
                self.road_map, self.agent_pos, self.destinations, self.current_goal
            )

    def close(self):
        """Close the environment"""
        if self.renderer:
            self.renderer.close()

    def get_valid_actions(self):
        """Get list of valid actions from current position"""
        valid_actions = []
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        for action, delta in moves.items():
            new_pos = self.agent_pos + delta
            if is_valid_position(self.road_map, tuple(new_pos)):
                valid_actions.append(action)

        return valid_actions
