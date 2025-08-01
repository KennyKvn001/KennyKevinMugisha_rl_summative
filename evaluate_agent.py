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
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL models")
    parser.add_argument("--algo", choices=["dqn", "ppo", "a2c"], required=True)
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
