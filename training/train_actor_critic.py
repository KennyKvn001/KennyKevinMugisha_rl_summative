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


class ActorCritic(nn.Module):
    """Actor-Critic network architecture."""

    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor = nn.Sequential(nn.Linear(128, output_dim), nn.Softmax(dim=-1))

        # Critic head (value function)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        shared_features = self.shared(x)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value


def create_environment(render_mode=None):
    """Create the custom MapNavigationEnv."""
    from utils.make_env import create_env

    # Create the custom environment
    env = create_env()

    # Update render mode if specified
    if render_mode is not None:
        env.render_mode = render_mode

    return env


def train_agent(
    env,
    hyperparams=None,
    episodes=1000,
    experiment_name="default",
):
    """Train an Actor-Critic agent with the specified hyperparameters."""
    # Create log directory
    log_dir = f"logs/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Get environment dimensions - custom env has Box observation space
    obs_shape = env.observation_space.shape  # Should be (grid_size, grid_size, 3)
    obs_space = np.prod(obs_shape)  # Flatten for actor-critic network
    action_space = env.action_space.n

    # Default hyperparameters
    default_params = {
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
    }

    # Update with provided hyperparameters
    if hyperparams:
        default_params.update(hyperparams)

    print(f"Training Actor-Critic agent")
    print(f"Hyperparameters: {default_params}")

    # Create the actor-critic network
    actor_critic = ActorCritic(obs_space, action_space)
    optimizer = optim.Adam(
        actor_critic.parameters(), lr=default_params["learning_rate"]
    )

    # Training statistics
    episode_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        # Storage for episode data
        log_probs = []
        values = []
        rewards = []
        entropies = []

        while not done:
            # Flatten the observation for the actor-critic network
            state_flattened = state.flatten()
            state_tensor = torch.tensor(state_flattened, dtype=torch.float32)
            action_probs, value = actor_critic(state_tensor)

            # Sample action
            dist = Categorical(action_probs)
            action = dist.sample()

            # Calculate log probability and entropy
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # Store data
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)

            state = next_state
            total_reward += reward
            steps += 1

        # Calculate returns and advantages
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + default_params["gamma"] * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(values).squeeze()
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        # Calculate advantages
        advantages = returns - values

        # Calculate losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = -entropies.mean()

        # Total loss
        total_loss = (
            actor_loss
            + default_params["value_loss_coef"] * critic_loss
            + default_params["entropy_coef"] * entropy_loss
        )

        # Update network
        optimizer.zero_grad()
        total_loss.backward()
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
    model_path = f"actor_critic_model_{experiment_name}.pth"
    torch.save(
        {
            "model_state_dict": actor_critic.state_dict(),
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
    return actor_critic, final_stats


def main():
    # Create the custom environment
    env = create_environment()

    print(f"Using Actor-Critic algorithm for custom MapNavigationEnv")

    # Train the agent
    model, training_stats = train_agent(
        env,
        experiment_name="default_run",
        episodes=1000,
    )


if __name__ == "__main__":
    main()
