#!/usr/bin/env python3
"""
Optimized training for policy gradient algorithms with dense rewards and comparison plots
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import json
from datetime import datetime
import time

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.make_env import create_env
from utils.dense_reward_wrapper import DenseRewardWrapper


class DetailedTrainingLogger(BaseCallback):
    """Enhanced logging for training progress"""

    def __init__(self, algorithm_name, log_interval=100, verbose=0):
        super(DetailedTrainingLogger, self).__init__(verbose)
        self.algorithm_name = algorithm_name
        self.log_interval = log_interval

        # Tracking variables
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_goals = []
        self.training_start_time = time.time()

        # Performance tracking
        self.best_reward = float("-inf")
        self.best_goals = 0
        self.improvement_episodes = []

    def _on_step(self) -> bool:
        if self.locals.get("dones", [False])[0]:
            info = self.locals.get("infos", [{}])[0]
            if "episode" in info:
                episode_reward = info["episode"]["r"]
                episode_length = info["episode"]["l"]

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_count += 1

                # Try to get goals completed
                goals = info.get("goals_completed", 0)
                self.episode_goals.append(goals)

                # Track improvements
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    self.improvement_episodes.append(self.episode_count)

                if goals > self.best_goals:
                    self.best_goals = goals

                # Print progress
                if self.episode_count % self.log_interval == 0:
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

                    avg_reward = np.mean(recent_rewards)
                    avg_goals = np.mean(recent_goals)
                    avg_length = (
                        np.mean(self.episode_lengths[-50:])
                        if len(self.episode_lengths) >= 50
                        else np.mean(self.episode_lengths)
                    )

                    elapsed_time = time.time() - self.training_start_time

                    print(f"{self.algorithm_name} - Episode {self.episode_count}:")
                    print(
                        f"  Avg Reward: {avg_reward:.2f} | Avg Goals: {avg_goals:.2f} | Avg Length: {avg_length:.1f}"
                    )
                    print(
                        f"  Best Reward: {self.best_reward:.2f} | Best Goals: {self.best_goals}"
                    )
                    print(f"  Time Elapsed: {elapsed_time/60:.1f}min")

                    # Success rate
                    if len(recent_goals) > 0:
                        success_rate = (
                            sum(1 for g in recent_goals if g >= 3)
                            / len(recent_goals)
                            * 100
                        )
                        print(f"  Mission Success Rate: {success_rate:.1f}%")

                    print()

        return True

    def get_training_stats(self):
        """Get comprehensive training statistics"""
        return {
            "algorithm": self.algorithm_name,
            "total_episodes": self.episode_count,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_goals": self.episode_goals,
            "best_reward": self.best_reward,
            "best_goals": self.best_goals,
            "improvement_episodes": self.improvement_episodes,
            "final_avg_reward": (
                np.mean(self.episode_rewards[-100:])
                if len(self.episode_rewards) >= 100
                else np.mean(self.episode_rewards) if self.episode_rewards else 0
            ),
            "final_avg_goals": (
                np.mean(self.episode_goals[-100:])
                if len(self.episode_goals) >= 100
                else np.mean(self.episode_goals) if self.episode_goals else 0
            ),
            "training_time": time.time() - self.training_start_time,
        }


def train_ppo_optimized():
    """Train PPO with optimal hyperparameters and dense rewards"""
    print("🚀 Training PPO with Dense Rewards...")

    # Create environment with dense rewards
    base_env = create_env(training_mode=True)
    env = DenseRewardWrapper(
        base_env, dense_reward_scale=0.5
    )  # Scale down dense rewards
    env = Monitor(env, "logs/ppo_dense")

    # Optimal PPO hyperparameters for navigation with dense rewards
    ppo_params = {
        "learning_rate": 3e-4,  # Higher learning rate for dense rewards
        "n_steps": 4096,  # Longer rollouts for better estimates
        "batch_size": 256,  # Larger batches for stability
        "n_epochs": 8,  # More epochs to use dense data
        "gamma": 0.995,  # High discount for long-term planning
        "gae_lambda": 0.95,  # Standard GAE
        "clip_range": 0.2,  # Standard clipping
        "ent_coef": 0.02,  # Moderate exploration (dense rewards help)
        "vf_coef": 0.5,  # Standard value function weight
        "max_grad_norm": 0.5,  # Gradient clipping
        "target_kl": 0.02,  # Allow reasonable policy updates
    }

    print("PPO Optimal Hyperparameters:")
    for key, value in ppo_params.items():
        print(f"  {key}: {value}")

    model = PPO(
        "CnnPolicy",
        env,
        verbose=0,
        tensorboard_log="./tensorboard_logs/ppo_dense/",
        **ppo_params,
    )

    callback = DetailedTrainingLogger("PPO", log_interval=50)
    model.learn(total_timesteps=500000, callback=callback)

    model.save("model_ppo_dense")
    print("✅ PPO Dense model saved as model_ppo_dense.zip")

    env.close()
    return model, callback.get_training_stats()


def train_a2c_optimized():
    """Train A2C with optimal hyperparameters and dense rewards"""
    print("\n🚀 Training A2C with Dense Rewards...")

    # Create environment with dense rewards
    base_env = create_env(training_mode=True)
    env = DenseRewardWrapper(base_env, dense_reward_scale=0.3)  # Lower scale for A2C
    env = Monitor(env, "logs/a2c_dense")

    # Optimal A2C hyperparameters for navigation with dense rewards
    a2c_params = {
        "learning_rate": 1e-3,  # Higher learning rate for dense rewards
        "n_steps": 512,  # Longer rollouts (much better than 15)
        "gamma": 0.99,  # Standard discount
        "gae_lambda": 0.95,  # Use GAE smoothing
        "ent_coef": 0.05,  # Higher exploration for navigation
        "vf_coef": 0.25,  # Lower value function weight
        "max_grad_norm": 0.5,  # Gradient clipping
        "rms_prop_eps": 1e-5,  # RMSprop epsilon
        "use_rms_prop": True,  # Use RMSprop optimizer
    }

    print("A2C Optimal Hyperparameters:")
    for key, value in a2c_params.items():
        print(f"  {key}: {value}")

    model = A2C(
        "CnnPolicy",
        env,
        verbose=0,
        tensorboard_log="./tensorboard_logs/a2c_dense/",
        **a2c_params,
    )

    callback = DetailedTrainingLogger("A2C", log_interval=50)
    model.learn(total_timesteps=500000, callback=callback)

    model.save("model_a2c_dense")
    print("✅ A2C Dense model saved as model_a2c_dense.zip")

    env.close()
    return model, callback.get_training_stats()


class OptimizedCNN(nn.Module):
    """Optimized CNN for REINFORCE with dense rewards"""

    def __init__(self, input_channels=3, output_dim=4):
        super(OptimizedCNN, self).__init__()

        # Optimized CNN architecture
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Calculate flattened size
        self.conv_output_size = 64 * 7 * 7

        # Policy head with dropout for regularization
        self.policy = nn.Sequential(
            nn.Linear(self.conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        else:
            x = x.permute(0, 3, 1, 2)

        x = x.float() / 255.0

        conv_out = self.conv(x)
        flattened = conv_out.view(conv_out.size(0), -1)
        policy_out = self.policy(flattened)

        return policy_out


def train_reinforce_optimized():
    """Train REINFORCE with optimal setup and dense rewards"""
    print("\n🚀 Training REINFORCE with Dense Rewards...")

    # Create environment with dense rewards
    base_env = create_env(training_mode=True)
    env = DenseRewardWrapper(
        base_env, dense_reward_scale=0.2
    )  # Lower scale for REINFORCE

    # Optimal REINFORCE hyperparameters
    learning_rate = 3e-4
    gamma = 0.99
    episodes = 5000

    print("REINFORCE Optimal Hyperparameters:")
    print(f"  learning_rate: {learning_rate}")
    print(f"  gamma: {gamma}")
    print(f"  episodes: {episodes}")
    print(f"  policy: OptimizedCNN with dropout")
    print(f"  dense_reward_scale: 0.2")

    # Create optimized policy
    policy = OptimizedCNN(input_channels=3, output_dim=4)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Training tracking
    episode_rewards = []
    episode_goals = []
    episode_lengths = []
    best_reward = float("-inf")
    best_goals = 0

    training_start_time = time.time()

    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 300:
            # Get action from policy
            state_tensor = torch.tensor(state, dtype=torch.float32)
            probs = policy(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            total_reward += reward
            steps += 1

        # Compute returns with baseline
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            # Use baseline to reduce variance
            baseline = returns.mean()
            returns = returns - baseline
            returns = returns / (returns.std() + 1e-9)

        # Compute loss
        loss = 0
        for log_prob, R in zip(log_probs, returns):
            loss -= log_prob * R

        # Update policy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)  # Gradient clipping
        optimizer.step()

        # Track progress
        episode_rewards.append(total_reward)
        goals = info.get("goals_completed", 0)
        episode_goals.append(goals)
        episode_lengths.append(steps)

        # Update best scores
        if total_reward > best_reward:
            best_reward = total_reward
        if goals > best_goals:
            best_goals = goals

        # Print progress
        if (episode + 1) % 100 == 0:
            recent_rewards = episode_rewards[-50:]
            recent_goals = episode_goals[-50:]

            avg_reward = np.mean(recent_rewards)
            avg_goals = np.mean(recent_goals)
            avg_length = np.mean(episode_lengths[-50:])

            elapsed_time = time.time() - training_start_time
            success_rate = (
                sum(1 for g in recent_goals if g >= 3) / len(recent_goals) * 100
            )

            print(f"REINFORCE - Episode {episode + 1}:")
            print(
                f"  Avg Reward: {avg_reward:.2f} | Avg Goals: {avg_goals:.2f} | Avg Length: {avg_length:.1f}"
            )
            print(f"  Best Reward: {best_reward:.2f} | Best Goals: {best_goals}")
            print(f"  Mission Success Rate: {success_rate:.1f}%")
            print(f"  Time Elapsed: {elapsed_time/60:.1f}min")
            print()

    # Save model
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hyperparams": {"learning_rate": learning_rate, "gamma": gamma},
            "obs_shape": env.observation_space.shape,
            "action_space": env.action_space.n,
            "model_type": "optimized_cnn",
        },
        "model_reinforce_dense.pth",
    )

    print("✅ REINFORCE Dense model saved as model_reinforce_dense.pth")

    env.close()

    # Return training stats
    return policy, {
        "algorithm": "REINFORCE",
        "total_episodes": episodes,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_goals": episode_goals,
        "best_reward": best_reward,
        "best_goals": best_goals,
        "final_avg_reward": (
            np.mean(episode_rewards[-100:])
            if len(episode_rewards) >= 100
            else np.mean(episode_rewards)
        ),
        "final_avg_goals": (
            np.mean(episode_goals[-100:])
            if len(episode_goals) >= 100
            else np.mean(episode_goals)
        ),
        "training_time": time.time() - training_start_time,
    }


def create_comparison_plots(training_stats):
    """Create comprehensive comparison plots"""
    print("\n📊 Creating comparison plots...")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Policy Gradient Algorithms Performance Comparison",
        fontsize=16,
        fontweight="bold",
    )

    algorithms = list(training_stats.keys())
    colors = ["blue", "red", "green"]

    # 1. Episode Rewards Over Time
    ax1 = axes[0, 0]
    for i, (algo, stats) in enumerate(training_stats.items()):
        rewards = stats["episode_rewards"]
        # Smooth the curve
        window = min(50, len(rewards) // 10)
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax1.plot(
                range(window - 1, len(rewards)),
                smoothed,
                label=algo,
                color=colors[i],
                linewidth=2,
            )
        else:
            ax1.plot(rewards, label=algo, color=colors[i], linewidth=2)

    ax1.set_title("Episode Rewards (Smoothed)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Goals Completed Over Time
    ax2 = axes[0, 1]
    for i, (algo, stats) in enumerate(training_stats.items()):
        goals = stats["episode_goals"]
        # Smooth the curve
        window = min(50, len(goals) // 10)
        if len(goals) >= window:
            smoothed = np.convolve(goals, np.ones(window) / window, mode="valid")
            ax2.plot(
                range(window - 1, len(goals)),
                smoothed,
                label=algo,
                color=colors[i],
                linewidth=2,
            )
        else:
            ax2.plot(goals, label=algo, color=colors[i], linewidth=2)

    ax2.set_title("Goals Completed (Smoothed)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Goals Completed")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Episode Lengths
    ax3 = axes[0, 2]
    for i, (algo, stats) in enumerate(training_stats.items()):
        lengths = stats["episode_lengths"]
        window = min(50, len(lengths) // 10)
        if len(lengths) >= window:
            smoothed = np.convolve(lengths, np.ones(window) / window, mode="valid")
            ax3.plot(
                range(window - 1, len(lengths)),
                smoothed,
                label=algo,
                color=colors[i],
                linewidth=2,
            )
        else:
            ax3.plot(lengths, label=algo, color=colors[i], linewidth=2)

    ax3.set_title("Episode Lengths (Smoothed)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Steps per Episode")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Final Performance Comparison
    ax4 = axes[1, 0]
    final_rewards = [stats["final_avg_reward"] for stats in training_stats.values()]
    bars = ax4.bar(algorithms, final_rewards, color=colors)
    ax4.set_title("Final Average Reward (Last 100 Episodes)")
    ax4.set_ylabel("Average Reward")

    # Add value labels on bars
    for bar, value in zip(bars, final_rewards):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(final_rewards) * 0.01,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 5. Final Goals Comparison
    ax5 = axes[1, 1]
    final_goals = [stats["final_avg_goals"] for stats in training_stats.values()]
    bars = ax5.bar(algorithms, final_goals, color=colors)
    ax5.set_title("Final Average Goals (Last 100 Episodes)")
    ax5.set_ylabel("Average Goals Completed")
    ax5.set_ylim(0, 3)

    # Add value labels on bars
    for bar, value in zip(bars, final_goals):
        ax5.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 6. Training Time Comparison
    ax6 = axes[1, 2]
    training_times = [
        stats["training_time"] / 60 for stats in training_stats.values()
    ]  # Convert to minutes
    bars = ax6.bar(algorithms, training_times, color=colors)
    ax6.set_title("Training Time")
    ax6.set_ylabel("Time (minutes)")

    # Add value labels on bars
    for bar, value in zip(bars, training_times):
        ax6.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(training_times) * 0.01,
            f"{value:.1f}m",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"policy_gradient_comparison_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"📈 Comparison plot saved as {plot_filename}")

    plt.show()


def save_training_logs(training_stats):
    """Save detailed training logs to JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_logs_{timestamp}.json"

    # Convert numpy arrays to lists for JSON serialization
    json_stats = {}
    for algo, stats in training_stats.items():
        json_stats[algo] = {
            "algorithm": stats["algorithm"],
            "total_episodes": stats["total_episodes"],
            "best_reward": float(stats["best_reward"]),
            "best_goals": int(stats["best_goals"]),
            "final_avg_reward": float(stats["final_avg_reward"]),
            "final_avg_goals": float(stats["final_avg_goals"]),
            "training_time_minutes": float(stats["training_time"] / 60),
            "episode_rewards": [float(r) for r in stats["episode_rewards"]],
            "episode_goals": [int(g) for g in stats["episode_goals"]],
            "episode_lengths": [int(l) for l in stats["episode_lengths"]],
        }

    with open(log_filename, "w") as f:
        json.dump(json_stats, f, indent=2)

    print(f"📝 Training logs saved as {log_filename}")

    # Print summary
    print(f"\n📋 TRAINING SUMMARY")
    print("=" * 50)
    for algo, stats in training_stats.items():
        print(f"{algo}:")
        print(f"  Final Avg Reward: {stats['final_avg_reward']:.2f}")
        print(f"  Final Avg Goals: {stats['final_avg_goals']:.2f}")
        print(f"  Best Reward: {stats['best_reward']:.2f}")
        print(f"  Best Goals: {stats['best_goals']}")
        print(f"  Training Time: {stats['training_time']/60:.1f} minutes")
        print()


def main():
    """Train all policy gradient algorithms with dense rewards and create comparisons"""
    print("🎯 OPTIMIZED POLICY GRADIENT TRAINING WITH DENSE REWARDS")
    print("=" * 60)
    print("Features:")
    print("• Dense reward wrapper for better learning signals")
    print("• Optimal hyperparameters for each algorithm")
    print("• Comprehensive logging and monitoring")
    print("• Comparison plots and performance analysis")
    print()

    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)

    training_stats = {}

    try:
        # Train PPO
        ppo_model, ppo_stats = train_ppo_optimized()
        training_stats["PPO"] = ppo_stats

        # Train A2C
        a2c_model, a2c_stats = train_a2c_optimized()
        training_stats["A2C"] = a2c_stats

        # Train REINFORCE
        reinforce_model, reinforce_stats = train_reinforce_optimized()
        training_stats["REINFORCE"] = reinforce_stats

        # Create comparison plots
        create_comparison_plots(training_stats)

        # Save training logs
        save_training_logs(training_stats)

        print(f"\n🎉 ALL TRAINING COMPLETE!")
        print("=" * 40)
        print("Models saved:")
        print("• model_ppo_dense.zip")
        print("• model_a2c_dense.zip")
        print("• model_reinforce_dense.pth")
        print()
        print("Evaluation commands:")
        print(
            "python evaluate_agent.py --algo ppo --model_path model_ppo_dense.zip --episodes 5"
        )
        print(
            "python evaluate_agent.py --algo a2c --model_path model_a2c_dense.zip --episodes 5"
        )
        print(
            "python evaluate_agent.py --algo reinforce --model_path model_reinforce_dense.pth --episodes 5"
        )

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        if training_stats:
            print("Creating plots for completed training...")
            create_comparison_plots(training_stats)
            save_training_logs(training_stats)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
