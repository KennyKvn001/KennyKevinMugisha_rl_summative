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

    print(f" MULTI-DESTINATION MISSION:")
    print(f" Route: {' → '.join([g.upper() for g in env.mission_goals])}")
    print(f" Start: {tuple(env.agent_pos)} | First Goal: {env.current_goal.upper()}")
    print(f" Max steps: {env.max_steps}")

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

        # Clean action/reward display with mission progress
        distance = abs(env.agent_pos[0] - env.goal_pos[0]) + abs(
            env.agent_pos[1] - env.goal_pos[1]
        )
        mission_progress = f"{info['goals_completed']}/{info['total_goals']}"
        status = (
            f"Goal:{env.current_goal[:4].upper()} {mission_progress} D:{distance:2d}"
        )

        print(
            f"{step+1:3d}   | {action_symbol:4s} | {str(tuple(env.agent_pos)):8s} | {reward:+6.2f} | {status}"
        )

        # Check for goal completion message
        if info.get("goal_reached", False) and not done:
            next_goal_index = info["goals_completed"]
            if next_goal_index < len(info["mission_goals"]):
                next_goal = info["mission_goals"][next_goal_index]
                print(f"    🎯 GOAL REACHED! → Next: {next_goal.upper()}")
            else:
                print(f"    🎯 GOAL REACHED!")

        if done:
            if reward > 0:
                efficiency_bonus = max(0, (env.max_steps - env.steps) // 20)
                print(f"\n 🏆 MISSION COMPLETE! All destinations visited!")
                print(
                    f" Route completed: {' → '.join([g.upper() for g in info['mission_goals']])}"
                )
                print(
                    f" Total steps: {info['steps']} | Efficiency bonus: +{efficiency_bonus}"
                )
                print(f" Total Reward: {total_reward:.1f}")
            else:
                print(
                    f"\n TIME OUT! Mission incomplete ({info['goals_completed']}/{info['total_goals']} goals)"
                )
                print(f" Total Reward: {total_reward:.1f}")
            break

        time.sleep(0.4)

    if not done:
        print(f"\n Demo ended. Agent at {tuple(env.agent_pos)}")
        print(f" Current Goal: {env.current_goal.upper()} at {tuple(env.goal_pos)}")
        print(
            f" Mission Progress: {info['goals_completed']}/{info['total_goals']} destinations"
        )
        print(f" Total Reward so far: {total_reward:.1f}")

    print(f"\n Performance Summary:")
    if done and reward > 0:
        print(" 🏆 FULL MISSION SUCCESS - All destinations reached!")
        print(f" Destinations visited: {info['goals_completed']}/{info['total_goals']}")
        print(f" Efficiency: {info['steps']}/{env.max_steps} steps")
        print(f" Final Score: {total_reward:.1f} points")
    else:
        print(" 📈 Training opportunity - multi-destination navigation required")
        print(
            f" Mission progress: {info.get('goals_completed', 0)}/{info.get('total_goals', 3)} destinations"
        )
        print(f" Current progress: {env.steps}/{env.max_steps} steps")
        print(f" Distance to current goal: {distance} tiles")

    # Keep window open
    print(f"\nKeeping window open for 3 seconds...")
    time.sleep(3)

    env.close()
    print(" Game complete!")


if __name__ == "__main__":
    main()
