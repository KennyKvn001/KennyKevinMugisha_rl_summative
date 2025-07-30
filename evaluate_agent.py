#!/usr/bin/env python3

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.train_dqn import create_environment
from stable_baselines3 import DQN
import numpy as np
import time


def evaluate_agent(model, num_episodes=5, render_mode="human"):
    """Evaluate a trained agent."""
    try:
        # Create environment with rendering
        env = create_environment(render_mode=render_mode)

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

                # Slow down rendering to make it viewable
                if render_mode == "human":
                    time.sleep(0.01)

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


def main():
    model_name = "dqn_model_default_run"
    model = DQN.load(model_name)
    rewards, lengths = evaluate_agent(model, num_episodes=5, render_mode="human")


if __name__ == "__main__":
    main()
