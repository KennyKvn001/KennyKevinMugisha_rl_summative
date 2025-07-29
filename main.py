#!/usr/bin/env python3
"""
Road Navigation Game - Main Entry Point
"""
import time
import numpy as np
from custom_env import RoadNavigationEnv


def main():
    """Main road navigation game"""
    print(" MAP NAVIGATION GAME")
    print("=" * 40)

    # Create environment with larger grid for complex road network
    env = RoadNavigationEnv(grid_size=(20, 20), render_mode="human")

    # Reset environment
    obs, info = env.reset()

    print(f" Mission: Reach {env.current_goal.upper()}")
    print(f" Start: {tuple(env.agent_pos)} → Goal: {tuple(env.goal_pos)}")
    print(f"  Max steps: {env.max_steps}")

    # Show initial state
    env.render()
    print("\n Starting navigation...")
    print(" Format: Step | Action | Position | Reward | Status")
    print("-" * 55)

    time.sleep(2)

    # Auto-navigation demo with clean action/reward display
    total_reward = 0
    for step in range(30):
        # Get valid actions from current position
        valid_actions = env.get_valid_actions()

        if valid_actions:
            # Try to move towards goal (simple pathfinding)
            goal_dir = env.goal_pos - env.agent_pos
            best_action = None

            # Prioritize directions that get us closer to goal
            if goal_dir[0] < 0 and 0 in valid_actions:  # Need to go up
                best_action = 0
            elif goal_dir[0] > 0 and 1 in valid_actions:  # Need to go down
                best_action = 1
            elif goal_dir[1] < 0 and 2 in valid_actions:  # Need to go left
                best_action = 2
            elif goal_dir[1] > 0 and 3 in valid_actions:  # Need to go right
                best_action = 3
            else:
                # If can't move directly towards goal, pick any valid action
                best_action = np.random.choice(valid_actions)

            action_names = ["↑", "↓", "←", "→"]
            action_symbol = action_names[best_action]
        else:
            best_action = 0
            action_symbol = "✗"

        obs, reward, done, truncated, info = env.step(best_action)
        env.render()

        total_reward += reward

        # Clean action/reward display
        distance = abs(env.agent_pos[0] - env.goal_pos[0]) + abs(
            env.agent_pos[1] - env.goal_pos[1]
        )
        status = f"Dist:{distance:2d}"

        print(
            f"{step+1:3d}   | {action_symbol:4s} | {str(tuple(env.agent_pos)):8s} | {reward:+6.2f} | {status}"
        )

        if done:
            if reward > 0:
                efficiency_bonus = max(0, (env.max_steps - env.steps) // 10)
                print(
                    f"\n SUCCESS! Reached {env.current_goal.upper()} in {info['steps']} steps!"
                )
                print(
                    f" Total Reward: {total_reward:.1f} (includes +{efficiency_bonus} efficiency bonus)"
                )
            else:
                print(f"\n TIME OUT! Goal not reached in {env.max_steps} steps.")
                print(f" Total Reward: {total_reward:.1f}")
            break

        time.sleep(0.4)

    if not done:
        print(f"\n Demo ended. Agent at {tuple(env.agent_pos)}")
        print(f" Goal {env.current_goal.upper()} at {tuple(env.goal_pos)}")
        print(f" Total Reward so far: {total_reward:.1f}")

    print(f"\n Performance Summary:")
    if done and reward > 0:
        print(" Mission completed successfully")
        print(f" Efficiency: {info['steps']}/{env.max_steps} steps")
        print(f" Score: {total_reward:.1f} points")
    else:
        print(" Training opportunity - complex navigation required")
        print(f" Current progress: {env.steps}/{env.max_steps} steps")
        print(f" Distance to goal: {distance} tiles")

    # Keep window open
    print(f"\nKeeping window open for 3 seconds...")
    time.sleep(3)

    env.close()
    print(" Game complete!")


if __name__ == "__main__":
    main()
