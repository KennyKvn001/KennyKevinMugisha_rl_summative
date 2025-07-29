#!/usr/bin/env python3
"""
Quick demo of the new road navigation system
"""
from custom_env import RoadNavigationEnv


def main():
    print("🛣️  ROAD NAVIGATION ENVIRONMENT")
    print("Creating environment...")

    # Create the road navigation environment
    env = RoadNavigationEnv(grid_size=(12, 12), render_mode="human")

    # Reset and show the initial state
    obs, info = env.reset()

    print(f"\n📍 5 Places: {list(env.destinations.keys())}")
    print(f"🎯 Goal: Reach '{env.current_goal.upper()}'")
    print(f"🔴 Agent starts at: {tuple(env.agent_pos)}")

    print("\n🎮 What you'll see:")
    print("  • Dark Gray = Obstacles (agent cannot enter)")
    print("  • Light Gray = Roads (agent can move on)")
    print("  • Blue Squares = Places (HOSPITAL, HOME, MARKET, BANK, PARK)")
    print("  • Yellow Square = Current goal destination")
    print("  • Red Circle = Agent")

    print("\n🏁 Rules:")
    print("  • Agent can ONLY move on roads or destinations")
    print("  • Agent cannot enter dark gray obstacles")
    print("  • Use the road network to navigate between places")

    # Render the environment
    env.render()

    input("\nPress Enter to close...")
    env.close()

    print("✅ Demo complete!")


if __name__ == "__main__":
    main()
