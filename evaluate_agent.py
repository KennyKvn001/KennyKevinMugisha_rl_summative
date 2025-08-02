#!/usr/bin/env python3

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from custom_env.map_navigation_env import MapNavigationEnv
from stable_baselines3 import DQN, PPO, A2C

# No longer need FlattenObservation - using CNN-compatible observations
import numpy as np
import time
import torch
import torch.nn as nn
from torch.distributions import Categorical


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


class SimpleCNN(nn.Module):
    """Simple CNN for REINFORCE to preserve spatial information"""
    
    def __init__(self, input_channels=3, output_dim=4):
        super(SimpleCNN, self).__init__()
        
        # Simple CNN layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=8, stride=4),  # 84x84 -> 20x20
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),              # 20x20 -> 9x9
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),              # 9x9 -> 7x7
            nn.ReLU(),
        )
        
        # Calculate flattened size
        self.conv_output_size = 32 * 7 * 7  # 1568
        
        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(self.conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        # Input: (batch, height, width, channels) -> (batch, channels, height, width)
        if len(x.shape) == 3:  # Single observation
            x = x.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
        else:  # Batch of observations
            x = x.permute(0, 3, 1, 2)
        
        x = x.float() / 255.0  # Normalize to [0, 1]
        
        conv_out = self.conv(x)
        flattened = conv_out.view(conv_out.size(0), -1)
        policy_out = self.policy(flattened)
        
        return policy_out


class REINFORCEWrapper:
    """Wrapper to make REINFORCE model compatible with stable-baselines3 interface"""
    
    def __init__(self, policy, obs_shape, model_type='mlp'):
        self.policy = policy
        self.obs_shape = obs_shape
        self.model_type = model_type
        
    def predict(self, obs, deterministic=True):
        """Predict action from observation"""
        with torch.no_grad():
            if self.model_type == 'cnn':
                # Use CNN directly on image observations
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                probs = self.policy(obs_tensor)
            else:
                # Flatten observation for MLP policy network
                obs_flattened = obs.flatten()
                obs_tensor = torch.tensor(obs_flattened, dtype=torch.float32)
                probs = self.policy(obs_tensor)
            
            if deterministic:
                # Take the action with highest probability
                action = torch.argmax(probs).item()
            else:
                # Sample from the distribution
                dist = Categorical(probs)
                action = dist.sample().item()
                
        return action, None  # Return None for state (not used in evaluation)


def evaluate_agent(model, num_episodes=5, render_mode="human"):
    """Evaluate a trained agent."""
    try:
        # Create environment with rendering - now uses 84x84 CNN-compatible observations
        env = MapNavigationEnv(render_mode=render_mode, training_mode=False)

        rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            truncated = False

            print(f"Starting episode {episode+1}/{num_episodes}")

            while not (done or truncated):
                # Get action from the trained model
                action, _ = model.predict(obs, deterministic=True)

                # Execute action in the environment
                obs, reward, done, truncated, info = env.step(action)

                # Update episode statistics
                episode_reward += reward
                episode_length += 1

                # Render the environment
                if render_mode == "human":
                    env.render()
                    time.sleep(0.1)

            rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            print(
                f"Episode {episode+1} finished with reward: {episode_reward}, length: {episode_length}"
            )

        # Print evaluation statistics
        print("\nEvaluation Results:")
        print(f"Average reward over {num_episodes} episodes: {np.mean(rewards):.2f}")
        print(f"Average episode length: {np.mean(episode_lengths):.2f}")
        print(f"Rewards per episode: {rewards}")

        env.close()
        return rewards, episode_lengths

    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Falling back to non-rendering evaluation")

        # Try again without rendering
        if render_mode == "human":
            return evaluate_agent(model, num_episodes, render_mode=None)
        return [], []


def load_model(algorithm, model_path):
    """Load the appropriate model based on algorithm."""
    if algorithm == "dqn":
        return DQN.load(model_path)
    elif algorithm == "ppo":
        return PPO.load(model_path)
    elif algorithm == "a2c":
        return A2C.load(model_path)
    elif algorithm == "reinforce":
        # Load PyTorch REINFORCE model
        if not model_path.endswith('.pth'):
            model_path += '.pth'
            
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract model parameters
        obs_shape = checkpoint['obs_shape']
        action_space = checkpoint['action_space']
        model_type = checkpoint.get('model_type', 'mlp')  # Default to MLP for old models
        
        # Recreate policy network based on type
        if model_type == 'cnn':
            policy = SimpleCNN(input_channels=3, output_dim=action_space)
        else:
            # Old MLP model
            obs_space = np.prod(obs_shape)
            policy = PolicyNetwork(obs_space, action_space)
        
        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy.eval()
        
        # Return wrapped model
        return REINFORCEWrapper(policy, obs_shape, model_type)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: dqn, ppo, a2c, reinforce")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL models")
    parser.add_argument("--algo", choices=["dqn", "ppo", "a2c", "reinforce"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no_render", action="store_true")

    args = parser.parse_args()

    model = load_model(args.algo, args.model_path)
    render_mode = None if args.no_render else "human"

    rewards, lengths = evaluate_agent(
        model, num_episodes=args.episodes, render_mode=render_mode
    )


if __name__ == "__main__":
    main()
