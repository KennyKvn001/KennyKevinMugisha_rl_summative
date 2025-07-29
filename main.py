#!/usr/bin/env python3
"""
Road Navigation Game - Main Entry Point
"""
import time
import numpy as np
from custom_env import RoadNavigationEnv


def main():
    """Main road navigation game"""
    print("ROAD NAVIGATION GAME")
    print("=" * 40)

    # Create environment
    env = RoadNavigationEnv(grid_size=(15, 15), render_mode="human")

    print("Game Overview:")
    print("  • Navigate through city using road network")
    print("  • Agent (red circle) can ONLY move on roads (gray) or destinations (blue)")
    print("  • Dark gray areas are obstacles - agent cannot enter them")
    print("  • Reach the yellow highlighted goal destination")
    print("  • Places: HOSPITAL, HOME, MARKET, BANK, PARK")

    # Reset environment
    obs, info = env.reset()

    print(f"\nCity Places:")
    for name, pos in env.destinations.items():
        print(f"{name.upper()}: {pos}")

    print(f"\nMission: Reach {env.current_goal.upper()}")
    print(f"🚶 Starting from: {tuple(env.agent_pos)}")

    # Show initial state
    env.render()
    print("\n Visual Guide:")
    print("  • Dark Gray = Obstacles (impassable)")
    print("  • Light Gray = Roads (passable)")
    print("  • Blue = Places (HOSPITAL, HOME, MARKET, BANK, PARK)")
    print("  • Yellow = Current Goal")
    print("  • Red Circle = Agent")

    print(f"\nControls: ↑↓←→ or Actions 0-3")
    print("Watching agent navigate for 15 seconds...")
    time.sleep(3)

    # Auto-navigation demo
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
            print(
                f"Step {step+1}: {action_names[best_action]} → Moving towards {env.current_goal.upper()}"
            )
        else:
            best_action = 0
            print(f"Step {step+1}: No valid moves available")

        obs, reward, done, truncated, info = env.step(best_action)
        env.render()

        if done:
            if reward > 0:
                print(
                    f"\n SUCCESS! Reached {env.current_goal.upper()} in {info['steps']} steps!"
                )
            else:
                print(f"\n Time limit reached!")
            break

        time.sleep(0.5)

    if not done:
        print(f"\n Demo complete. Agent at {tuple(env.agent_pos)}")
        print(f"Goal {env.current_goal.upper()} is at {tuple(env.goal_pos)}")

    # Keep window open
    print(f"\nKeeping window open for 5 seconds...")
    time.sleep(1)

    env.reset()
    print(" Game complete!")


if __name__ == "__main__":
    main()
