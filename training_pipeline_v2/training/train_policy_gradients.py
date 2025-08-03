#!/usr/bin/env python3
"""
Policy Gradient Training Script - Clean Implementation
Trains PPO, A2C, and REINFORCE with MLP on flattened observations
"""

import os
import sys
import time
import torch
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from torch.distributions import Categorical

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.policy_gradient_config import (
    get_ppo_config,
    get_a2c_config,
    get_reinforce_config,
)
from config.environment_config import get_environment_config
from environment.env_wrapper import create_policy_gradient_environment
from models.policy_gradient_mlp import create_policy_mlp


class PolicyGradientCallback(BaseCallback):
    """Enhanced callback for Policy Gradient training monitoring"""

    def __init__(self, algorithm_name, log_interval=100, verbose=0):
        super().__init__(verbose)
        self.algorithm_name = algorithm_name
        self.log_interval = log_interval

        # Tracking
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_goals = []
        self.training_start_time = time.time()

        # Best performance tracking
        self.best_mean_reward = float("-inf")
        self.best_episode = 0

    def _on_step(self) -> bool:
        # Check if episode is done
        if self.locals.get("dones", [False])[0]:
            info = self.locals.get("infos", [{}])[0]

            if "episode" in info:
                episode_reward = info["episode"]["r"]
                episode_length = info["episode"]["l"]

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_count += 1

                # Extract goals completed
                goals_completed = info.get("goals_completed", 0)
                self.episode_goals.append(goals_completed)

                # Log progress
                if self.episode_count % self.log_interval == 0:
                    self._log_progress()

        return True

    def _log_progress(self):
        """Log training progress"""
        recent_rewards = (
            self.episode_rewards[-50:]
            if len(self.episode_rewards) >= 50
            else self.episode_rewards
        )
        recent_goals = (
            self.episode_goals[-50:]
            if len(self.episode_goals) >= 50
            else self.episode_goals
        )
        recent_lengths = (
            self.episode_lengths[-50:]
            if len(self.episode_lengths) >= 50
            else self.episode_lengths
        )

        avg_reward = np.mean(recent_rewards)
        avg_goals = np.mean(recent_goals)
        avg_length = np.mean(recent_lengths)

        elapsed_time = time.time() - self.training_start_time

        print(f"{self.algorithm_name} Episode {self.episode_count}:")
        print(f"  Avg Reward (50ep): {avg_reward:.2f}")
        print(f"  Avg Goals (50ep): {avg_goals:.2f}")
        print(f"  Avg Length (50ep): {avg_length:.1f}")
        print(f"  Training Time: {elapsed_time/60:.1f}min")

        # Success rate
        if recent_goals:
            success_rate = (
                sum(1 for g in recent_goals if g >= 3) / len(recent_goals) * 100
            )
            print(f"  Success Rate: {success_rate:.1f}%")

        # Update best performance
        if len(self.episode_rewards) >= 100:
            mean_reward = np.mean(self.episode_rewards[-100:])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.best_episode = self.episode_count
                print(f"  🏆 New best mean reward: {mean_reward:.2f}")

        print()

    def get_training_stats(self):
        """Get comprehensive training statistics"""
        return {
            "algorithm": self.algorithm_name,
            "total_episodes": self.episode_count,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_goals": self.episode_goals,
            "best_mean_reward": self.best_mean_reward,
            "best_episode": self.best_episode,
            "training_time": time.time() - self.training_start_time,
        }


def train_ppo(experiment_name="ppo_clean"):
    """Train PPO agent with MLP architecture"""
    print("🚀 TRAINING PPO WITH MLP ARCHITECTURE")
    print("=" * 60)

    # Load configuration
    ppo_config = get_ppo_config()
    env_config = get_environment_config()

    print("PPO Configuration:")
    for key, value in ppo_config.items():
        print(f"  {key}: {value}")
    print()

    # Create environment
    env = create_policy_gradient_environment(training_mode=True)
    print(f"Environment: Observation shape {env.observation_space.shape}")

    # Create log directory
    log_dir = f"logs/policy_gradients/ppo/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)

    # Create PPO model
    model = PPO(
        ppo_config["policy_type"],
        env,
        learning_rate=ppo_config["learning_rate"],
        n_steps=ppo_config["n_steps"],
        batch_size=ppo_config["batch_size"],
        n_epochs=ppo_config["n_epochs"],
        gamma=ppo_config["gamma"],
        gae_lambda=ppo_config["gae_lambda"],
        clip_range=ppo_config["clip_range"],
        ent_coef=ppo_config["ent_coef"],
        vf_coef=ppo_config["vf_coef"],
        max_grad_norm=ppo_config["max_grad_norm"],
        target_kl=ppo_config["target_kl"],
        verbose=ppo_config["verbose"],
        tensorboard_log=f"logs/policy_gradients/tensorboard/ppo/{experiment_name}/",
        device="auto",
    )

    # Create callback
    callback = PolicyGradientCallback("PPO", log_interval=ppo_config["log_interval"])

    # Train
    print("Starting PPO training...")
    model.learn(total_timesteps=ppo_config["total_timesteps"], callback=callback)

    # Save
    save_path = (
        f"logs/policy_gradients/ppo/{experiment_name}/model_ppo_{experiment_name}.zip"
    )
    model.save(save_path)
    print(f"💾 PPO model saved to: {save_path}")

    env.close()
    return model, callback.get_training_stats()


def train_a2c(experiment_name="a2c_clean"):
    """Train A2C agent with MLP architecture"""
    print("🚀 TRAINING A2C WITH MLP ARCHITECTURE")
    print("=" * 60)

    # Load configuration
    a2c_config = get_a2c_config()

    print("A2C Configuration:")
    for key, value in a2c_config.items():
        print(f"  {key}: {value}")
    print()

    # Create environment
    env = create_policy_gradient_environment(training_mode=True)
    print(f"Environment: Observation shape {env.observation_space.shape}")

    # Create log directory
    log_dir = f"logs/policy_gradients/a2c/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)

    # Create A2C model
    model = A2C(
        a2c_config["policy_type"],
        env,
        learning_rate=a2c_config["learning_rate"],
        n_steps=a2c_config["n_steps"],
        gamma=a2c_config["gamma"],
        gae_lambda=a2c_config["gae_lambda"],
        ent_coef=a2c_config["ent_coef"],
        vf_coef=a2c_config["vf_coef"],
        max_grad_norm=a2c_config["max_grad_norm"],
        use_rms_prop=a2c_config["use_rms_prop"],
        rms_prop_eps=a2c_config["rms_prop_eps"],
        verbose=a2c_config["verbose"],
        tensorboard_log=f"logs/policy_gradients/tensorboard/a2c/{experiment_name}/",
        device="auto",
    )

    # Create callback
    callback = PolicyGradientCallback("A2C", log_interval=a2c_config["log_interval"])

    # Train
    print("Starting A2C training...")
    model.learn(total_timesteps=a2c_config["total_timesteps"], callback=callback)

    # Save
    save_path = (
        f"logs/policy_gradients/a2c/{experiment_name}/model_a2c_{experiment_name}.zip"
    )
    model.save(save_path)
    print(f"💾 A2C model saved to: {save_path}")

    env.close()
    return model, callback.get_training_stats()


def train_reinforce(experiment_name="reinforce_clean"):
    """Train REINFORCE agent with custom MLP implementation"""
    print("🚀 TRAINING REINFORCE WITH MLP ARCHITECTURE")
    print("=" * 60)

    # Load configuration
    reinforce_config = get_reinforce_config()
    env_config = get_environment_config()

    print("REINFORCE Configuration:")
    for key, value in reinforce_config.items():
        print(f"  {key}: {value}")
    print()

    # Create environment
    env = create_policy_gradient_environment(training_mode=True)
    print(f"Environment: Observation shape {env.observation_space.shape}")

    # Create log directory
    log_dir = f"logs/policy_gradients/reinforce/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Create policy network
    obs_dim = env.observation_space.shape[0]  # Flattened observation
    action_dim = env.action_space.n
    policy = create_policy_mlp(input_dim=obs_dim, output_dim=action_dim)

    # Optimizer
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=reinforce_config["learning_rate"],
        weight_decay=reinforce_config["weight_decay"],
    )

    # Training tracking
    episode_rewards = []
    episode_lengths = []
    episode_goals = []
    training_start_time = time.time()

    print("Starting REINFORCE training...")
    print("-" * 60)

    for episode in range(reinforce_config["episodes"]):
        # Collect episode
        obs, _ = env.reset()
        log_probs = []
        rewards = []
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < reinforce_config["max_episode_steps"]:
            # Get action
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action_probs = policy(obs_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            # Store
            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward
            steps += 1
            obs = next_obs

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + reinforce_config["gamma"] * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Compute loss
        policy_loss = 0
        for log_prob, R in zip(log_probs, returns):
            policy_loss -= log_prob * R

        # Update policy
        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy.parameters(), reinforce_config["gradient_clipping"]
        )
        optimizer.step()

        # Track statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        goals_completed = info.get("goals_completed", 0)
        episode_goals.append(goals_completed)

        # Log progress
        if (episode + 1) % reinforce_config["log_interval"] == 0:
            recent_rewards = (
                episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
            )
            recent_goals = (
                episode_goals[-50:] if len(episode_goals) >= 50 else episode_goals
            )

            avg_reward = np.mean(recent_rewards)
            avg_goals = np.mean(recent_goals)
            elapsed_time = time.time() - training_start_time

            print(f"REINFORCE Episode {episode + 1}:")
            print(f"  Avg Reward (50ep): {avg_reward:.2f}")
            print(f"  Avg Goals (50ep): {avg_goals:.2f}")
            print(f"  Training Time: {elapsed_time/60:.1f}min")

            if recent_goals:
                success_rate = (
                    sum(1 for g in recent_goals if g >= 3) / len(recent_goals) * 100
                )
                print(f"  Success Rate: {success_rate:.1f}%")
            print()

        # Save model periodically
        if (episode + 1) % reinforce_config["save_interval"] == 0:
            save_path = f"{log_dir}/model_reinforce_{experiment_name}_ep{episode+1}.pth"
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "episode": episode + 1,
                    "episode_rewards": episode_rewards,
                    "episode_goals": episode_goals,
                    "config": reinforce_config,
                },
                save_path,
            )

    # Final save
    final_save_path = f"{log_dir}/model_reinforce_{experiment_name}_final.pth"
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": reinforce_config["episodes"],
            "episode_rewards": episode_rewards,
            "episode_goals": episode_goals,
            "episode_lengths": episode_lengths,
            "config": reinforce_config,
            "training_time": time.time() - training_start_time,
        },
        final_save_path,
    )

    print(f"💾 REINFORCE model saved to: {final_save_path}")

    # Training stats
    stats = {
        "algorithm": "REINFORCE",
        "total_episodes": reinforce_config["episodes"],
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_goals": episode_goals,
        "training_time": time.time() - training_start_time,
    }

    env.close()
    return policy, stats


def train_all_policy_gradients(experiment_name="pg_clean"):
    """Train all Policy Gradient methods"""
    print("🎯 TRAINING ALL POLICY GRADIENT METHODS")
    print("=" * 80)

    all_stats = {}

    # Train PPO
    try:
        ppo_model, ppo_stats = train_ppo(f"{experiment_name}_ppo")
        all_stats["PPO"] = ppo_stats
        print("✅ PPO training completed")
    except Exception as e:
        print(f"❌ PPO training failed: {e}")

    print("\n" + "=" * 60 + "\n")

    # Train A2C
    try:
        a2c_model, a2c_stats = train_a2c(f"{experiment_name}_a2c")
        all_stats["A2C"] = a2c_stats
        print("✅ A2C training completed")
    except Exception as e:
        print(f"❌ A2C training failed: {e}")

    print("\n" + "=" * 60 + "\n")

    # Train REINFORCE
    try:
        reinforce_model, reinforce_stats = train_reinforce(
            f"{experiment_name}_reinforce"
        )
        all_stats["REINFORCE"] = reinforce_stats
        print("✅ REINFORCE training completed")
    except Exception as e:
        print(f"❌ REINFORCE training failed: {e}")

    # Summary
    print(f"\n🎉 ALL POLICY GRADIENT TRAINING COMPLETE!")
    print("=" * 60)
    for algo, stats in all_stats.items():
        if stats.get("episode_rewards"):
            final_performance = (
                np.mean(stats["episode_rewards"][-100:])
                if len(stats["episode_rewards"]) >= 100
                else np.mean(stats["episode_rewards"])
            )
            print(f"{algo}: Final Performance = {final_performance:.2f}")

    return all_stats


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description="Train Policy Gradient agents")
    parser.add_argument(
        "--algorithm",
        choices=["ppo", "a2c", "reinforce", "all"],
        default="all",
        help="Algorithm to train",
    )
    parser.add_argument("--experiment", default="pg_clean", help="Experiment name")
    args = parser.parse_args()

    if args.algorithm == "ppo":
        train_ppo(f"{args.experiment}_ppo")
    elif args.algorithm == "a2c":
        train_a2c(f"{args.experiment}_a2c")
    elif args.algorithm == "reinforce":
        train_reinforce(f"{args.experiment}_reinforce")
    else:
        train_all_policy_gradients(args.experiment)


if __name__ == "__main__":
    main()
