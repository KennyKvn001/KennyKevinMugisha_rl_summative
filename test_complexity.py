#!/usr/bin/env python3
"""
Test script to demonstrate the increased complexity of the road network
"""
import time
import numpy as np
from custom_env import RoadNavigationEnv


def test_complexity():
    """Test the complexity of the new road network"""
    print("🧩 ROAD NETWORK COMPLEXITY TEST")
    print("=" * 50)

    # Create environment with complex road network
    env = RoadNavigationEnv(grid_size=(20, 20), render_mode="human")

    print("🗺️  Complex Road Network Features:")
    print("✓ Maze-like patterns around destinations")
    print("✓ Dead ends and detours")
    print("✓ Multiple alternate routes")
    print("✓ Strategic bottlenecks and gaps")
    print("✓ Random connecting roads")
    print("✓ Varied road widths (highways vs streets)")

    # Reset and analyze the map
    obs, info = env.reset()

    print(f"\n📊 Map Analysis:")
    print(f"Grid Size: {env.grid_size}")
    print(f"Total Cells: {env.grid_size[0] * env.grid_size[1]}")

    # Count different cell types
    total_cells = env.grid_size[0] * env.grid_size[1]
    road_cells = np.sum(env.road_map == 1)
    destination_cells = np.sum(env.road_map == 2)
    obstacle_cells = np.sum(env.road_map == 0)

    print(f"🛣️  Road Cells: {road_cells} ({road_cells/total_cells*100:.1f}%)")
    print(
        f"🏢 Destination Cells: {destination_cells} ({destination_cells/total_cells*100:.1f}%)"
    )
    print(
        f"🚧 Obstacle Cells: {obstacle_cells} ({obstacle_cells/total_cells*100:.1f}%)"
    )

    # Calculate path complexity
    print(f"\n🎯 MULTI-DESTINATION MISSION:")
    print(f"Route: {' → '.join([g.upper() for g in env.mission_goals])}")
    print(f"Start: {tuple(env.agent_pos)} | Current Goal: {env.current_goal.upper()}")
    print(
        f"Manhattan Distance: {np.abs(env.agent_pos[0] - env.goal_pos[0]) + np.abs(env.agent_pos[1] - env.goal_pos[1])}"
    )

    # Render the environment
    env.render()

    print("\n🔍 Complexity Features Visible:")
    print("• Gray roads with varying widths")
    print("• Complex L-shaped and zigzag access roads")
    print("• Dead-end branches in each quadrant")
    print("• Detour loops requiring strategic navigation")
    print("• Gaps in main arterials forcing alternate routes")

    print(f"\n⏱️  Max Steps: {env.max_steps} (increased for complexity)")
    print("📈 Enhanced Reward System:")
    print("  • Distance-based progress rewards")
    print("  • Efficiency bonuses for faster completion")
    print("  • Reduced penalties for better learning")

    # Test a few random moves to show complexity
    print(f"\n🎮 Testing Navigation Complexity...")
    print("📊 Step | Action | Valid Actions | Reward")
    print("-" * 40)

    for step in range(10):
        valid_actions = env.get_valid_actions()

        if valid_actions:
            action = np.random.choice(valid_actions)
            obs, reward, done, truncated, info = env.step(action)
            env.render()

            action_names = ["↑", "↓", "←", "→"]
            print(
                f"{step+1:4d} | {action_names[action]:6s} | {str(valid_actions):13s} | {reward:+6.3f}"
            )

            if done:
                print("     | GOAL  | REACHED!      |")
                break
        else:
            print(f"{step+1:4d} | ✗     | No moves      | 0.000")
            break

        time.sleep(0.6)

    print(f"\n🎯 Training Implications:")
    print("✅ Prevents overfitting - Multiple paths to each destination")
    print("✅ Requires exploration - Dead ends teach path planning")
    print("✅ Strategic thinking - Must choose optimal routes")
    print("✅ Generalization - Randomized elements each episode")
    print("✅ Scalable difficulty - Complex enough for deep RL")

    # Keep window open
    print(f"\nKeeping visualization open for 8 seconds...")
    time.sleep(8)

    env.close()
    print("🏁 Complexity test complete!")


if __name__ == "__main__":
    test_complexity()
