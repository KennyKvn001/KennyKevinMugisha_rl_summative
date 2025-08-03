#!/usr/bin/env python3
"""
Clean Environment Wrapper
Provides consistent observation format for different algorithm types
"""

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import sys
import os

# Add parent directory to path to import custom environment
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_env.map_navigation_env import MapNavigationEnv
from config.environment_config import get_environment_config, get_reward_config


class CleanEnvironmentWrapper(gym.Wrapper):
    """
    Clean wrapper that provides consistent environment behavior
    without conflicting reward systems
    """

    def __init__(self, env):
        super().__init__(env)
        self.reward_config = get_reward_config()

        # Override base environment rewards with consistent system
        self.step_penalty = self.reward_config["step_penalty"]
        self.goal_reward = self.reward_config["goal_reward"]
        self.mission_complete_reward = self.reward_config["mission_complete_reward"]
        self.invalid_move_penalty = self.reward_config["invalid_move_penalty"]
        self.timeout_penalty = self.reward_config["timeout_penalty"]
        self.efficiency_bonus_scale = self.reward_config["efficiency_bonus_scale"]

        # Distance rewards (optional)
        self.distance_reward_enabled = self.reward_config["distance_reward_enabled"]
        self.distance_progress_reward = self.reward_config["distance_progress_reward"]
        self.distance_regression_penalty = self.reward_config[
            "distance_regression_penalty"
        ]

        self.previous_distance = None

    def reset(self, **kwargs):
        """Reset environment and initialize tracking"""
        obs, info = self.env.reset(**kwargs)

        # Initialize distance tracking if enabled
        if (
            self.distance_reward_enabled
            and hasattr(self.env, "agent_pos")
            and hasattr(self.env, "goal_pos")
        ):
            self.previous_distance = self._calculate_distance()

        return obs, info

    def step(self, action):
        """Step with consistent reward system"""
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        # Replace original reward with consistent reward system
        reward = self._calculate_reward(
            action, original_reward, terminated, truncated, info
        )

        return obs, reward, terminated, truncated, info

    def _calculate_distance(self):
        """Calculate Manhattan distance to current goal"""
        if hasattr(self.env, "agent_pos") and hasattr(self.env, "goal_pos"):
            return abs(self.env.agent_pos[0] - self.env.goal_pos[0]) + abs(
                self.env.agent_pos[1] - self.env.goal_pos[1]
            )
        return 0

    def _calculate_reward(self, action, original_reward, terminated, truncated, info):
        """Calculate consistent reward based on configured system"""
        reward = 0.0

        # 1. Base step penalty to encourage efficiency
        reward += self.step_penalty

        # 2. Goal completion rewards
        if info.get("goal_reached", False):
            if info.get("mission_complete", False):
                # Mission completely finished
                efficiency_bonus = max(
                    0,
                    (self.env.max_steps - self.env.steps) * self.efficiency_bonus_scale,
                )
                reward += self.mission_complete_reward + efficiency_bonus
            else:
                # Intermediate goal reached
                reward += self.goal_reward

        # 3. Invalid move penalty
        if info.get("invalid_move", False):
            reward += self.invalid_move_penalty

        # 4. Timeout penalty
        if truncated and not terminated:
            reward += self.timeout_penalty

        # 5. Distance-based rewards (optional)
        if self.distance_reward_enabled:
            current_distance = self._calculate_distance()

            if self.previous_distance is not None:
                if current_distance < self.previous_distance:
                    # Reward for getting closer
                    progress = self.previous_distance - current_distance
                    reward += self.distance_progress_reward * progress
                elif current_distance > self.previous_distance:
                    # Penalty for moving away
                    regression = current_distance - self.previous_distance
                    reward -= self.distance_regression_penalty * regression

            self.previous_distance = current_distance

        return reward


def create_dqn_environment(training_mode=True):
    """
    Create environment for DQN (CNN-based)
    Returns environment with (84, 84, 3) image observations
    """
    env_config = get_environment_config()

    # Create base environment
    env = MapNavigationEnv(
        grid_size=env_config["grid_size"],
        render_mode=(
            env_config["training_render_mode"]
            if training_mode
            else env_config["evaluation_render_mode"]
        ),
        training_mode=training_mode,
    )

    # Apply clean wrapper for consistent rewards
    env = CleanEnvironmentWrapper(env)

    # DQN expects image observations - no flattening
    return env


def create_policy_gradient_environment(training_mode=True):
    """
    Create environment for Policy Gradient methods (MLP-based)
    Returns environment with flattened observations
    """
    env_config = get_environment_config()

    # Create base environment
    env = MapNavigationEnv(
        grid_size=env_config["grid_size"],
        render_mode=(
            env_config["training_render_mode"]
            if training_mode
            else env_config["evaluation_render_mode"]
        ),
        training_mode=training_mode,
    )

    # Apply clean wrapper for consistent rewards
    env = CleanEnvironmentWrapper(env)

    # Policy Gradient methods expect flattened observations for MLP
    env = FlattenObservation(env)

    return env


def test_environments():
    """Test both environment types"""
    print("🧪 Testing Environment Wrappers")
    print("=" * 50)

    # Test DQN environment (image observations)
    print("Testing DQN Environment (Image observations):")
    dqn_env = create_dqn_environment(training_mode=True)

    obs, info = dqn_env.reset()
    print(f"DQN observation shape: {obs.shape}")
    print(f"DQN observation type: {type(obs)}")
    print(f"DQN observation min/max: {obs.min()}/{obs.max()}")

    action = dqn_env.action_space.sample()
    obs, reward, terminated, truncated, info = dqn_env.step(action)
    print(f"DQN step reward: {reward}")
    print(f"DQN step info keys: {list(info.keys())}")

    dqn_env.close()

    # Test Policy Gradient environment (flattened observations)
    print(f"\nTesting Policy Gradient Environment (Flattened observations):")
    pg_env = create_policy_gradient_environment(training_mode=True)

    obs, info = pg_env.reset()
    print(f"PG observation shape: {obs.shape}")
    print(f"PG observation type: {type(obs)}")
    print(f"PG observation min/max: {obs.min()}/{obs.max()}")

    action = pg_env.action_space.sample()
    obs, reward, terminated, truncated, info = pg_env.step(action)
    print(f"PG step reward: {reward}")
    print(f"PG step info keys: {list(info.keys())}")

    pg_env.close()

    print("✅ Environment wrapper test completed successfully!")


if __name__ == "__main__":
    test_environments()
