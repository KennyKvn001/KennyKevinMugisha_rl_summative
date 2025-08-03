#!/usr/bin/env python3
"""
Unified Evaluation System for All Algorithms
Evaluates DQN, PPO, A2C, and REINFORCE models consistently
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.environment_config import get_environment_config
from environment.env_wrapper import (
    create_dqn_environment,
    create_policy_gradient_environment,
)
from models.policy_gradient_mlp import create_policy_mlp


class UnifiedEvaluator:
    """Unified evaluation system for all algorithms"""

    def __init__(self, num_episodes=100, render=False):
        self.num_episodes = num_episodes
        self.render = render
        self.results = {}

    def evaluate_dqn(self, model_path, experiment_name="dqn_evaluation"):
        """Evaluate DQN model"""
        print(f"🔬 Evaluating DQN: {model_path}")

        # Create DQN environment
        env = create_dqn_environment(training_mode=False)

        try:
            # Load model
            model = DQN.load(model_path, env=env, device="auto")

            # Evaluate
            episode_rewards, episode_lengths = evaluate_policy(
                model,
                env,
                n_eval_episodes=self.num_episodes,
                return_episode_rewards=True,
                render=self.render,
            )

            # Detailed evaluation
            detailed_stats = self._detailed_evaluation(model, env, "DQN")

            # Store results
            self.results["DQN"] = {
                "mean_reward": np.mean(episode_rewards),
                "std_reward": np.std(episode_rewards),
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
                "success_rate": detailed_stats["success_rate"],
                "avg_goals": detailed_stats["avg_goals"],
                "completion_time": detailed_stats["completion_time"],
                "model_path": model_path,
            }

            print(
                f"✅ DQN - Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
            )

        except Exception as e:
            print(f"❌ DQN evaluation failed: {e}")
            self.results["DQN"] = {"error": str(e)}

        env.close()

    def evaluate_ppo(self, model_path, experiment_name="ppo_evaluation"):
        """Evaluate PPO model"""
        print(f"🔬 Evaluating PPO: {model_path}")

        # Create Policy Gradient environment
        env = create_policy_gradient_environment(training_mode=False)

        try:
            # Load model
            model = PPO.load(model_path, env=env, device="auto")

            # Evaluate
            episode_rewards, episode_lengths = evaluate_policy(
                model,
                env,
                n_eval_episodes=self.num_episodes,
                return_episode_rewards=True,
                render=self.render,
            )

            # Detailed evaluation
            detailed_stats = self._detailed_evaluation(model, env, "PPO")

            # Store results
            self.results["PPO"] = {
                "mean_reward": np.mean(episode_rewards),
                "std_reward": np.std(episode_rewards),
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
                "success_rate": detailed_stats["success_rate"],
                "avg_goals": detailed_stats["avg_goals"],
                "completion_time": detailed_stats["completion_time"],
                "model_path": model_path,
            }

            print(
                f"✅ PPO - Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
            )

        except Exception as e:
            print(f"❌ PPO evaluation failed: {e}")
            self.results["PPO"] = {"error": str(e)}

        env.close()

    def evaluate_a2c(self, model_path, experiment_name="a2c_evaluation"):
        """Evaluate A2C model"""
        print(f"🔬 Evaluating A2C: {model_path}")

        # Create Policy Gradient environment
        env = create_policy_gradient_environment(training_mode=False)

        try:
            # Load model
            model = A2C.load(model_path, env=env, device="auto")

            # Evaluate
            episode_rewards, episode_lengths = evaluate_policy(
                model,
                env,
                n_eval_episodes=self.num_episodes,
                return_episode_rewards=True,
                render=self.render,
            )

            # Detailed evaluation
            detailed_stats = self._detailed_evaluation(model, env, "A2C")

            # Store results
            self.results["A2C"] = {
                "mean_reward": np.mean(episode_rewards),
                "std_reward": np.std(episode_rewards),
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
                "success_rate": detailed_stats["success_rate"],
                "avg_goals": detailed_stats["avg_goals"],
                "completion_time": detailed_stats["completion_time"],
                "model_path": model_path,
            }

            print(
                f"✅ A2C - Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
            )

        except Exception as e:
            print(f"❌ A2C evaluation failed: {e}")
            self.results["A2C"] = {"error": str(e)}

        env.close()

    def evaluate_reinforce(self, model_path, experiment_name="reinforce_evaluation"):
        """Evaluate REINFORCE model"""
        print(f"🔬 Evaluating REINFORCE: {model_path}")

        # Create Policy Gradient environment
        env = create_policy_gradient_environment(training_mode=False)

        try:
            # Load model
            checkpoint = torch.load(model_path, map_location="cpu")

            # Recreate policy
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            policy = create_policy_mlp(input_dim=obs_dim, output_dim=action_dim)
            policy.load_state_dict(checkpoint["policy_state_dict"])
            policy.eval()

            # Evaluate
            episode_rewards = []
            episode_lengths = []
            goals_completed = []
            episode_times = []

            for episode in range(self.num_episodes):
                obs, _ = env.reset()
                total_reward = 0
                steps = 0
                done = False
                start_time = time.time()

                while not done:
                    with torch.no_grad():
                        obs_tensor = torch.tensor(obs, dtype=torch.float32)
                        action_probs = policy(obs_tensor)
                        action = torch.argmax(action_probs).item()

                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    steps += 1

                episode_time = time.time() - start_time
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
                goals_completed.append(info.get("goals_completed", 0))
                episode_times.append(episode_time)

            # Calculate statistics
            success_rate = (
                sum(1 for g in goals_completed if g >= 3) / len(goals_completed) * 100
            )
            avg_goals = np.mean(goals_completed)
            completion_time = np.mean(episode_times)

            # Store results
            self.results["REINFORCE"] = {
                "mean_reward": np.mean(episode_rewards),
                "std_reward": np.std(episode_rewards),
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
                "success_rate": success_rate,
                "avg_goals": avg_goals,
                "completion_time": completion_time,
                "model_path": model_path,
            }

            print(
                f"✅ REINFORCE - Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
            )

        except Exception as e:
            print(f"❌ REINFORCE evaluation failed: {e}")
            self.results["REINFORCE"] = {"error": str(e)}

        env.close()

    def _detailed_evaluation(self, model, env, algorithm_name):
        """Perform detailed evaluation with additional metrics"""
        goals_completed = []
        episode_times = []

        for episode in range(
            min(50, self.num_episodes)
        ):  # Detailed evaluation on subset
            obs, _ = env.reset()
            done = False
            start_time = time.time()

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            episode_time = time.time() - start_time
            goals_completed.append(info.get("goals_completed", 0))
            episode_times.append(episode_time)

        # Calculate detailed statistics
        success_rate = (
            sum(1 for g in goals_completed if g >= 3) / len(goals_completed) * 100
        )
        avg_goals = np.mean(goals_completed)
        completion_time = np.mean(episode_times)

        return {
            "success_rate": success_rate,
            "avg_goals": avg_goals,
            "completion_time": completion_time,
        }

    def generate_comparison_report(
        self, save_path="logs/evaluation/comparison_report.txt"
    ):
        """Generate comprehensive comparison report"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("UNIFIED ALGORITHM COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Summary table
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 50 + "\n")
            f.write(
                f"{'Algorithm':<12} {'Mean Reward':<12} {'Success Rate':<12} {'Avg Goals':<10}\n"
            )
            f.write("-" * 50 + "\n")

            for algo in ["DQN", "PPO", "A2C", "REINFORCE"]:
                if algo in self.results and "error" not in self.results[algo]:
                    stats = self.results[algo]
                    f.write(
                        f"{algo:<12} {stats['mean_reward']:<12.2f} {stats['success_rate']:<12.1f}% {stats['avg_goals']:<10.2f}\n"
                    )
                else:
                    f.write(f"{algo:<12} {'ERROR':<12} {'ERROR':<12} {'ERROR':<10}\n")

            f.write("\n" + "=" * 80 + "\n\n")

            # Detailed statistics
            for algo, stats in self.results.items():
                if "error" not in stats:
                    f.write(f"{algo} DETAILED RESULTS\n")
                    f.write("-" * 30 + "\n")
                    f.write(
                        f"Mean Reward: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}\n"
                    )
                    f.write(f"Success Rate: {stats['success_rate']:.1f}%\n")
                    f.write(f"Average Goals: {stats['avg_goals']:.2f}\n")
                    f.write(
                        f"Average Episode Length: {np.mean(stats['episode_lengths']):.1f}\n"
                    )
                    f.write(
                        f"Average Completion Time: {stats['completion_time']:.3f}s\n"
                    )
                    f.write(f"Model Path: {stats['model_path']}\n")
                    f.write("\n")

        print(f"📊 Comparison report saved to: {save_path}")

    def create_comparison_plots(self, save_dir="logs/evaluation/plots"):
        """Create comprehensive comparison plots"""
        os.makedirs(save_dir, exist_ok=True)

        # Filter successful results
        successful_results = {k: v for k, v in self.results.items() if "error" not in v}

        if not successful_results:
            print("❌ No successful evaluations to plot")
            return

        # Set style
        plt.style.use("default")
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        # 1. Mean Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Mean Reward Comparison
        algorithms = list(successful_results.keys())
        mean_rewards = [successful_results[algo]["mean_reward"] for algo in algorithms]
        std_rewards = [successful_results[algo]["std_reward"] for algo in algorithms]

        axes[0, 0].bar(
            algorithms,
            mean_rewards,
            yerr=std_rewards,
            capsize=5,
            color=colors[: len(algorithms)],
        )
        axes[0, 0].set_title("Mean Reward Comparison")
        axes[0, 0].set_ylabel("Mean Reward")
        axes[0, 0].grid(True, alpha=0.3)

        # Success Rate Comparison
        success_rates = [
            successful_results[algo]["success_rate"] for algo in algorithms
        ]
        axes[0, 1].bar(algorithms, success_rates, color=colors[: len(algorithms)])
        axes[0, 1].set_title("Success Rate Comparison")
        axes[0, 1].set_ylabel("Success Rate (%)")
        axes[0, 1].grid(True, alpha=0.3)

        # Average Goals Comparison
        avg_goals = [successful_results[algo]["avg_goals"] for algo in algorithms]
        axes[1, 0].bar(algorithms, avg_goals, color=colors[: len(algorithms)])
        axes[1, 0].set_title("Average Goals Completed")
        axes[1, 0].set_ylabel("Average Goals")
        axes[1, 0].grid(True, alpha=0.3)

        # Episode Length Comparison
        avg_lengths = [
            np.mean(successful_results[algo]["episode_lengths"]) for algo in algorithms
        ]
        axes[1, 1].bar(algorithms, avg_lengths, color=colors[: len(algorithms)])
        axes[1, 1].set_title("Average Episode Length")
        axes[1, 1].set_ylabel("Average Steps")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{save_dir}/performance_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. Reward Distribution Comparison
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, (algo, stats) in enumerate(successful_results.items()):
            ax.hist(
                stats["episode_rewards"],
                bins=30,
                alpha=0.7,
                label=f"{algo} (μ={stats['mean_reward']:.2f})",
                color=colors[i % len(colors)],
            )

        ax.set_xlabel("Episode Reward")
        ax.set_ylabel("Frequency")
        ax.set_title("Reward Distribution Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.savefig(f"{save_dir}/reward_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 3. Performance Radar Chart
        categories = ["Mean Reward", "Success Rate", "Avg Goals", "Efficiency"]

        # Normalize values to 0-1 scale for radar chart
        all_mean_rewards = [
            stats["mean_reward"] for stats in successful_results.values()
        ]
        all_success_rates = [
            stats["success_rate"] for stats in successful_results.values()
        ]
        all_avg_goals = [stats["avg_goals"] for stats in successful_results.values()]
        all_avg_lengths = [
            np.mean(stats["episode_lengths"]) for stats in successful_results.values()
        ]

        max_reward = max(all_mean_rewards) if all_mean_rewards else 1
        max_success = max(all_success_rates) if all_success_rates else 1
        max_goals = max(all_avg_goals) if all_avg_goals else 1
        min_length = (
            min(all_avg_lengths) if all_avg_lengths else 1
        )  # Lower is better for efficiency

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        for i, (algo, stats) in enumerate(successful_results.items()):
            values = [
                stats["mean_reward"] / max_reward,
                stats["success_rate"] / max_success,
                stats["avg_goals"] / max_goals,
                min_length
                / np.mean(stats["episode_lengths"]),  # Inverse for efficiency
            ]
            values += values[:1]  # Complete the circle

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                label=algo,
                color=colors[i % len(colors)],
            )
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title("Performance Radar Chart", size=16, y=1.1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        plt.savefig(f"{save_dir}/performance_radar.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📈 Comparison plots saved to: {save_dir}")

    def get_summary(self):
        """Get summary of evaluation results"""
        successful_results = {k: v for k, v in self.results.items() if "error" not in v}

        if not successful_results:
            return "No successful evaluations"

        # Find best performer
        best_algo = max(
            successful_results.keys(),
            key=lambda x: successful_results[x]["mean_reward"],
        )

        summary = {
            "total_algorithms": len(self.results),
            "successful_evaluations": len(successful_results),
            "best_performer": best_algo,
            "best_mean_reward": successful_results[best_algo]["mean_reward"],
            "best_success_rate": successful_results[best_algo]["success_rate"],
            "results": successful_results,
        }

        return summary


def find_latest_models():
    """Find the latest trained models for each algorithm"""
    models = {}

    # DQN model
    dqn_paths = [
        "logs/dqn/dqn_clean/model_dqn_dqn_clean.zip",
        "logs/dqn/model_dqn.zip",
        "../model_dqn.zip",  # Original model
    ]
    for path in dqn_paths:
        if os.path.exists(path):
            models["DQN"] = path
            break

    # PPO model
    ppo_paths = [
        "logs/policy_gradients/ppo/pg_clean_ppo/model_ppo_pg_clean_ppo.zip",
        "logs/ppo/model_ppo.zip",
        "../model_ppo.zip",  # Original model
    ]
    for path in ppo_paths:
        if os.path.exists(path):
            models["PPO"] = path
            break

    # A2C model
    a2c_paths = [
        "logs/policy_gradients/a2c/pg_clean_a2c/model_a2c_pg_clean_a2c.zip",
        "logs/a2c/model_a2c.zip",
        "../model_a2c_dense_mlp.zip",  # Original model
    ]
    for path in a2c_paths:
        if os.path.exists(path):
            models["A2C"] = path
            break

    # REINFORCE model
    reinforce_paths = [
        "logs/policy_gradients/reinforce/pg_clean_reinforce/model_reinforce_pg_clean_reinforce_final.pth",
        "logs/reinforce/model_reinforce.pth",
        "../models/model_reinforce_dense.pth",  # Original model
    ]
    for path in reinforce_paths:
        if os.path.exists(path):
            models["REINFORCE"] = path
            break

    return models


def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate all trained models")
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--algorithm",
        choices=["dqn", "ppo", "a2c", "reinforce", "all"],
        default="all",
        help="Algorithm to evaluate",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render evaluation episodes"
    )
    args = parser.parse_args()

    print("🎯 UNIFIED ALGORITHM EVALUATION")
    print("=" * 60)

    # Create evaluator
    evaluator = UnifiedEvaluator(num_episodes=args.episodes, render=args.render)

    # Find models
    models = find_latest_models()
    print(f"Found models: {list(models.keys())}")

    # Evaluate based on argument
    if args.algorithm == "all":
        for algo, model_path in models.items():
            if algo == "DQN":
                evaluator.evaluate_dqn(model_path)
            elif algo == "PPO":
                evaluator.evaluate_ppo(model_path)
            elif algo == "A2C":
                evaluator.evaluate_a2c(model_path)
            elif algo == "REINFORCE":
                evaluator.evaluate_reinforce(model_path)
    else:
        algo = args.algorithm.upper()
        if algo in models:
            if algo == "DQN":
                evaluator.evaluate_dqn(models[algo])
            elif algo == "PPO":
                evaluator.evaluate_ppo(models[algo])
            elif algo == "A2C":
                evaluator.evaluate_a2c(models[algo])
            elif algo == "REINFORCE":
                evaluator.evaluate_reinforce(models[algo])
        else:
            print(f"❌ No model found for {algo}")

    # Generate reports and plots
    evaluator.generate_comparison_report()
    evaluator.create_comparison_plots()

    # Print summary
    summary = evaluator.get_summary()
    if isinstance(summary, dict):
        print(f"\n🏆 EVALUATION SUMMARY")
        print(f"Best Performer: {summary['best_performer']}")
        print(f"Best Mean Reward: {summary['best_mean_reward']:.2f}")
        print(f"Best Success Rate: {summary['best_success_rate']:.1f}%")
        print(
            f"Successful Evaluations: {summary['successful_evaluations']}/{summary['total_algorithms']}"
        )


if __name__ == "__main__":
    main()
