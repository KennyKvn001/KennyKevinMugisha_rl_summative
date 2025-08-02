#!/usr/bin/env python3

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import TrainingCallback


class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE algorithm."""

    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.fc(x)


def create_environment(render_mode=None):
    """Create the custom MapNavigationEnv."""
    from utils.make_env import create_env

    # Create the custom environment for training
    env = create_env(training_mode=True)

    # Update render mode if specified
    if render_mode is not None:
        env.render_mode = render_mode

    return env


def train_agent(
    env,
    hyperparams=None,
    episodes=10000,
    experiment_name="default",
):
    """Train a REINFORCE agent with the specified hyperparameters."""
    # Create log directory
    log_dir = f"logs/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Get environment dimensions - custom env has Box observation space
    obs_shape = env.observation_space.shape  # Should be (84, 84, 3)
    obs_space = np.prod(obs_shape)  # Flatten for policy network (will use CNN later)
    action_space = env.action_space.n
    
    print(f"Observation shape: {obs_shape}")
    print(f"Flattened observation size: {obs_space}")
    print(f"Action space: {action_space}")

    # Default hyperparameters
    default_params = {
        "learning_rate": 1e-4,
        "gamma": 0.99,
    }

    # Update with provided hyperparameters
    if hyperparams:
        default_params.update(hyperparams)

    print(f"Training REINFORCE agent")
    print(f"Hyperparameters: {default_params}")

    # Create the policy network
    policy = PolicyNetwork(obs_space, action_space)
    optimizer = optim.Adam(policy.parameters(), lr=default_params["learning_rate"])

    # Training statistics
    episode_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Flatten the observation for the policy network
            state_flattened = state.flatten()
            state_tensor = torch.tensor(state_flattened, dtype=torch.float32)
            probs = policy(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            total_reward += reward
            steps += 1

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + default_params["gamma"] * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Normalize

        # Compute loss
        loss = 0
        for log_prob, R in zip(log_probs, returns):
            loss -= log_prob * R

        # Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}: Average Reward (last 50) = {avg_reward:.2f}")
        else:
            print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    # Save the trained model
    model_path = f"model_{experiment_name}.pth"
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hyperparams": default_params,
        },
        model_path,
    )
    print(f"Model saved as {model_path}")

    # Print final training statistics
    final_stats = {}
    if episode_rewards:
        avg_reward = (
            np.mean(episode_rewards[-100:])
            if len(episode_rewards) >= 100
            else np.mean(episode_rewards)
        )
        avg_length = (
            np.mean(episode_lengths[-100:])
            if len(episode_lengths) >= 100
            else np.mean(episode_lengths)
        )
        print(f"Final training statistics:")
        print(f"Total episodes: {len(episode_rewards)}")
        print(f"Average reward over last 100 episodes: {avg_reward:.2f}")
        print(f"Average episode length over last 100 episodes: {avg_length:.2f}")

        final_stats = {
            "avg_reward": avg_reward,
            "avg_length": avg_length,
            "total_episodes": len(episode_rewards),
            "rewards": episode_rewards,
            "episode_lengths": episode_lengths,
        }

    env.close()
    return policy, final_stats


def main():
    # Create the custom environment
    env = create_environment()

    print(f"Using REINFORCE algorithm for custom MapNavigationEnv")

    # Train the agent
    model, training_stats = train_agent(
        env,
        experiment_name="reinforce",
        episodes=1000,
    )


if __name__ == "__main__":
    main()
