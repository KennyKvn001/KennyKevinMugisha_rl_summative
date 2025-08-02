#!/usr/bin/env python3

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import TrainingCallback


def create_environment(render_mode=None):
    """Create the custom MapNavigationEnv with dense rewards and flattened observations."""
    from utils.make_env import create_env
    from dense_reward_wrapper import DenseRewardWrapper
    from gymnasium.wrappers import FlattenObservation

    # Create the custom environment for training
    env = create_env(training_mode=True)
    
    # Add dense rewards for better learning signals
    env = DenseRewardWrapper(env, dense_reward_scale=0.2)  # Conservative scale for A2C
    
    # Flatten observations for MLP policy (84x84x3 -> 21168 vector)
    env = FlattenObservation(env)

    # Update render mode if specified
    if render_mode is not None:
        env.render_mode = render_mode

    return env


def train_agent(
    env,
    policy_type="MlpPolicy",
    hyperparams=None,
    timesteps=300000,
    experiment_name="default",
):
    """Train an A2C agent with the specified policy and hyperparameters."""
    # Create log directory
    log_dir = f"logs/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Wrap environment with Monitor
    env = Monitor(env, log_dir)

    # Optimized hyperparameters for A2C with dense rewards and MLP
    default_params = {
        "learning_rate": 3e-4,      # Good learning rate for MLP and dense rewards
        "n_steps": 1024,            # Longer rollouts for better estimates
        "gamma": 0.99,              # Standard discount factor
        "gae_lambda": 0.95,         # Use GAE smoothing instead of raw returns
        "ent_coef": 0.01,           # Conservative entropy for stability
        "vf_coef": 0.5,             # Standard value function weight
        "max_grad_norm": 0.5,       # Gradient clipping
        "rms_prop_eps": 1e-5,       # RMSprop epsilon
        "use_rms_prop": True,       # Use RMSprop optimizer
    }

    # Update with provided hyperparameters
    if hyperparams:
        default_params.update(hyperparams)

    print(f"Training A2C agent with {policy_type} policy")
    print(f"Hyperparameters: {default_params}")

    # Create the A2C agent
    model = A2C(
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

    # Use MlpPolicy for flattened observations (faster training, less complex)
    policy_type = "MlpPolicy"

    print(f"Using {policy_type} with Dense Rewards for custom MapNavigationEnv")

    # Train the agent (adjust timesteps as needed)
    model, training_stats = train_agent(
        env,
        policy_type=policy_type,
        experiment_name="a2c_dense_mlp",
    )


if __name__ == "__main__":
    main()
