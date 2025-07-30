#!/usr/bin/env python3

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import FlattenObservation

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import TrainingCallback


def create_environment(render_mode=None):
    """Create the custom MapNavigationEnv."""
    from utils.make_env import create_env

    # Create the custom environment
    env = create_env()

    # Update render mode if specified
    if render_mode is not None:
        env.render_mode = render_mode

    # Flatten observations for MlpPolicy (15, 15, 3) -> (675,)
    env = FlattenObservation(env)

    return env


def train_agent(
    env,
    policy_type="CnnPolicy",
    hyperparams=None,
    timesteps=2000000,
    experiment_name="default",
):
    """Train a DQN agent with the specified policy and hyperparameters."""
    # Create log directory
    log_dir = f"logs/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Wrap environment with Monitor
    env = Monitor(env, log_dir)

    # Default hyperparameters
    default_params = {
        "learning_rate": 1e-4,
        "buffer_size": 500000,
        "learning_starts": 10000,
        "batch_size": 32,
        "gamma": 0.99,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "train_freq": 8,
        "gradient_steps": 2,
        "target_update_interval": 2000,
    }

    # Update with provided hyperparameters
    if hyperparams:
        default_params.update(hyperparams)

    print(f"Training DQN agent with {policy_type} policy")
    print(f"Hyperparameters: {default_params}")

    # Create the DQN agent
    model = DQN(
        policy_type,
        env,
        verbose=1,
        tensorboard_log=f"./tensorboard_logs/{experiment_name}/",
        **default_params,
    )

    # Create callback
    callback = TrainingCallback(plot_interval=10)

    # Train the agent
    model.learn(total_timesteps=timesteps, callback=callback)

    # Save the trained model
    model_path = f"model_{experiment_name}"
    model.save(model_path)
    print(f"Model saved as {model_path}.zip")

    # Print final training statistics
    final_stats = {}
    if callback.rewards:
        avg_reward = (
            np.mean(callback.rewards[-100:])
            if len(callback.rewards) >= 100
            else np.mean(callback.rewards)
        )
        avg_length = (
            np.mean(callback.episode_lengths[-100:])
            if len(callback.episode_lengths) >= 100
            else np.mean(callback.episode_lengths)
        )
        print(f"Final training statistics:")
        print(f"Total episodes: {len(callback.rewards)}")
        print(f"Average reward over last 100 episodes: {avg_reward:.2f}")
        print(f"Average episode length over last 100 episodes: {avg_length:.2f}")

        final_stats = {
            "avg_reward": avg_reward,
            "avg_length": avg_length,
            "total_episodes": len(callback.rewards),
            "rewards": callback.rewards,
            "episode_lengths": callback.episode_lengths,
        }

    return model, final_stats


def main():
    # Create the custom environment
    env = create_environment()

    # Use MlpPolicy for the custom environment (flatten the Box observation space)
    policy_type = "MlpPolicy"

    print(f"Using {policy_type} for custom MapNavigationEnv")

    # Train the agent (adjust timesteps as needed)
    model, training_stats = train_agent(
        env,
        policy_type=policy_type,
        experiment_name="dqn",
    )


if __name__ == "__main__":
    main()
