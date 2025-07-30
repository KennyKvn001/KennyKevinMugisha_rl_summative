#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class TrainingCallback(BaseCallback):
    """
    Custom callback for monitoring training progress and plotting results.
    """

    def __init__(self, plot_interval=10, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.plot_interval = plot_interval
        self.rewards = []
        self.episode_lengths = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if episode is done
        if self.locals.get("dones", [False])[0]:
            # Get episode info
            info = self.locals.get("infos", [{}])[0]
            if "episode" in info:
                episode_reward = info["episode"]["r"]
                episode_length = info["episode"]["l"]

                self.rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_count += 1

                # Print progress every episode
                print(
                    f"Episode {self.episode_count}: Reward = {episode_reward:.2f}, Length = {episode_length}"
                )

                # Plot results periodically
                if self.episode_count % self.plot_interval == 0:
                    self._plot_results()

        return True

    def plot_training_progress(self):
        """Plot the training progress."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot rewards
        ax1.plot(self.rewards)
        ax1.set_title("Episode Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")

        # Plot moving average of rewards
        window_size = min(20, len(self.rewards))
        moving_avg = np.convolve(
            self.rewards, np.ones(window_size) / window_size, mode="valid"
        )
        ax1.plot(
            range(window_size - 1, len(self.rewards)),
            moving_avg,
            "r-",
            label=f"{window_size}-episode moving average",
        )
        ax1.legend()

        # Plot episode lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title("Episode Lengths")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Length")

        plt.tight_layout()
        plt.show()

    def _plot_results(self):
        """Plot training progress."""
        if len(self.rewards) < 2:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot rewards
        ax1.plot(self.rewards)
        ax1.set_title("Episode Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")

        # Plot moving average if we have enough data
        if len(self.rewards) >= 10:
            window_size = min(10, len(self.rewards))
            moving_avg = np.convolve(
                self.rewards, np.ones(window_size) / window_size, mode="valid"
            )
            ax1.plot(
                range(window_size - 1, len(self.rewards)),
                moving_avg,
                "r--",
                label=f"Moving Avg ({window_size})",
            )
            ax1.legend()

        # Plot episode lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title("Episode Lengths")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps")

        plt.tight_layout()
        plt.pause(0.1)  # Brief pause to update the plot

    def get_statistics(self):
        """Get training statistics."""
        if not self.rewards:
            return {}

        return {
            "total_episodes": len(self.rewards),
            "mean_reward": np.mean(self.rewards),
            "std_reward": np.std(self.rewards),
            "min_reward": np.min(self.rewards),
            "max_reward": np.max(self.rewards),
            "mean_length": np.mean(self.episode_lengths),
            "std_length": np.std(self.episode_lengths),
            "rewards": self.rewards.copy(),
            "episode_lengths": self.episode_lengths.copy(),
        }
