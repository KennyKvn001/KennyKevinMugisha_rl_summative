#!/usr/bin/env python3
"""
DQN Training Script - Clean Implementation
Trains DQN agent using CNN on image observations
"""

import os
import sys
import time
import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.dqn_config import get_dqn_config
from config.environment_config import get_environment_config
from environment.env_wrapper import create_dqn_environment


class DQNTrainingCallback(BaseCallback):
    """Enhanced callback for DQN training monitoring"""

    def __init__(
        self, log_interval=100, eval_interval=10000, eval_episodes=5, verbose=0
    ):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes

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

                # Evaluation
                if (
                    self.episode_count % (self.eval_interval // 2048) == 0
                ):  # Approximate episodes
                    self._evaluate_performance()

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

        print(f"DQN Episode {self.episode_count}:")
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
        print()

    def _evaluate_performance(self):
        """Evaluate current performance"""
        if len(self.episode_rewards) >= 100:
            mean_reward = np.mean(self.episode_rewards[-100:])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.best_episode = self.episode_count
                print(
                    f"🏆 New best mean reward: {mean_reward:.2f} at episode {self.episode_count}"
                )

    def get_training_stats(self):
        """Get comprehensive training statistics"""
        return {
            "algorithm": "DQN",
            "total_episodes": self.episode_count,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_goals": self.episode_goals,
            "best_mean_reward": self.best_mean_reward,
            "best_episode": self.best_episode,
            "training_time": time.time() - self.training_start_time,
        }


def train_dqn(experiment_name="dqn_clean", save_path=None):
    """
    Train DQN agent with clean configuration

    Args:
        experiment_name: Name for this training run
        save_path: Path to save trained model (optional)

    Returns:
        model: Trained DQN model
        stats: Training statistics
    """
    print("🚀 TRAINING DQN WITH CNN ARCHITECTURE")
    print("=" * 60)

    # Load configurations
    dqn_config = get_dqn_config()
    env_config = get_environment_config()

    print("Configuration:")
    print(f"  Policy Type: {dqn_config['policy_type']}")
    print(f"  Learning Rate: {dqn_config['learning_rate']}")
    print(f"  Buffer Size: {dqn_config['buffer_size']:,}")
    print(f"  Total Timesteps: {dqn_config['total_timesteps']:,}")
    print(f"  Grid Size: {env_config['grid_size']}")
    print(f"  Max Episode Steps: {env_config['max_episode_steps']}")
    print()

    # Create environment
    env = create_dqn_environment(training_mode=True)
    print(f"Environment created:")
    print(f"  Observation shape: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n}")
    print()

    # Create log directory
    log_dir = f"logs/dqn/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Wrap with Monitor for logging
    env = Monitor(env, log_dir)

    # Create DQN model
    print("Creating DQN model...")
    model = DQN(
        dqn_config["policy_type"],
        env,
        learning_rate=dqn_config["learning_rate"],
        buffer_size=dqn_config["buffer_size"],
        learning_starts=dqn_config["learning_starts"],
        batch_size=dqn_config["batch_size"],
        gamma=dqn_config["gamma"],
        exploration_fraction=dqn_config["exploration_fraction"],
        exploration_initial_eps=dqn_config["exploration_initial_eps"],
        exploration_final_eps=dqn_config["exploration_final_eps"],
        train_freq=dqn_config["train_freq"],
        gradient_steps=dqn_config["gradient_steps"],
        target_update_interval=dqn_config["target_update_interval"],
        verbose=dqn_config["verbose"],
        tensorboard_log=f"logs/dqn/tensorboard/{experiment_name}/",
        device="auto",
    )

    print(f"Model created successfully!")
    print(f"Device: {model.device}")
    print()

    # Create callback
    callback = DQNTrainingCallback(
        log_interval=dqn_config["log_interval"],
        eval_interval=dqn_config["eval_freq"],
        eval_episodes=dqn_config["eval_episodes"],
    )

    # Train model
    print("Starting training...")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=dqn_config["total_timesteps"],
            callback=callback,
            progress_bar=True,
        )

        print("✅ Training completed successfully!")

    except KeyboardInterrupt:
        print("⏹️ Training interrupted by user")

    # Save model
    if save_path is None:
        save_path = f"logs/dqn/{experiment_name}/model_dqn_{experiment_name}.zip"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"💾 Model saved to: {save_path}")

    # Get training statistics
    stats = callback.get_training_stats()

    # Print final statistics
    print(f"\n📊 TRAINING SUMMARY")
    print("=" * 40)
    print(f"Total Episodes: {stats['total_episodes']}")
    print(
        f"Best Mean Reward: {stats['best_mean_reward']:.2f} (Episode {stats['best_episode']})"
    )
    print(f"Training Time: {stats['training_time']/60:.1f} minutes")

    if stats["episode_rewards"]:
        final_performance = (
            np.mean(stats["episode_rewards"][-100:])
            if len(stats["episode_rewards"]) >= 100
            else np.mean(stats["episode_rewards"])
        )
        print(f"Final Performance: {final_performance:.2f}")

    if stats["episode_goals"]:
        final_success_rate = (
            sum(1 for g in stats["episode_goals"][-100:] if g >= 3)
            / min(100, len(stats["episode_goals"]))
            * 100
        )
        print(f"Final Success Rate: {final_success_rate:.1f}%")

    # Close environment
    env.close()

    return model, stats


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description="Train DQN agent")
    parser.add_argument("--experiment", default="dqn_clean", help="Experiment name")
    parser.add_argument("--save-path", help="Path to save model")
    args = parser.parse_args()

    # Train DQN
    model, stats = train_dqn(experiment_name=args.experiment, save_path=args.save_path)

    print(f"\n🎯 DQN Training Complete!")
    print(f"Model and logs saved in: logs/dqn/{args.experiment}/")


if __name__ == "__main__":
    main()
