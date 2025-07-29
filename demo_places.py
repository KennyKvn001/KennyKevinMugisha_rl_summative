#!/usr/bin/env python3
"""
Demo showing meaningful place names in the road navigation environment
"""
import time
from custom_env import RoadNavigationEnv


def show_places():
    print("🏙️  CITY PLACES NAVIGATION")
    print("=" * 40)

    env = RoadNavigationEnv(grid_size=(14, 14), render_mode="human")
    obs, info = env.reset()

    print("🏢 Available Places:")
    for place_name, position in env.destinations.items():
        print(f"  📍 {place_name.upper()}: {position}")

    print(f"\n🎯 Current Mission: Go to {env.current_goal.upper()}")
    print(f"🚶 Starting from: {tuple(env.agent_pos)}")

    print("\n🗺️  City Layout:")
    print("  🏥 HOSPITAL - Medical services (top-left)")
    print("  🏠 HOME - Residential area (top-right)")
    print("  🛒 MARKET - Shopping center (center)")
    print("  🏦 BANK - Financial services (bottom-left)")
    print("  🌳 PARK - Recreation area (bottom-right)")

    env.render()

    print("\n💡 Navigation Tips:")
    print("  • Follow the gray roads to reach your destination")
    print("  • Each place is labeled with its name")
    print("  • Goal destination is highlighted in yellow")
    print("  • Dark areas are obstacles - use roads to go around them")

    print("\nDisplaying for 8 seconds...")
    time.sleep(8)

    env.close()
    print("✅ Places demo complete!")


if __name__ == "__main__":
    show_places()
