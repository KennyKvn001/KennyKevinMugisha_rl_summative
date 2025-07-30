#!/usr/bin/env python3
"""
Create a GIF demonstration of the Map Navigation Environment
"""

import numpy as np
from PIL import Image
import os
import time
import pygame
from custom_env.map_navigation_env import MapNavigationEnv


def create_demo_gif():
    """Create a GIF showing the agent navigating the environment with enhanced visuals"""

    print("🎬 Creating Enhanced Map Navigation Demo GIF...")
    print("=" * 50)

    # Create environment with human mode to get the enhanced pygame visuals
    env = MapNavigationEnv(grid_size=(18, 18), render_mode="human", training_mode=False)
    obs, info = env.reset()

    print(f"✅ Environment initialized")
    print(f"   Grid size: {env.grid_size}")
    print(f"   Mission: {' → '.join([g.upper() for g in env.mission_goals])}")
    print(f"   Starting position: {tuple(env.agent_pos)}")
    print(f"   First goal: {env.current_goal.upper()} at {tuple(env.goal_pos)}")

    frames = []
    total_reward = 0
    step_count = 0

    # Capture initial frame from pygame surface
    env.render()  # Update pygame display
    time.sleep(0.1)  # Let initial animations render

    # Get pygame surface and convert to PIL Image
    pygame_surface = env.renderer.screen
    frame_array = pygame.surfarray.array3d(pygame_surface)
    frame_array = np.transpose(
        frame_array, (1, 0, 2)
    )  # Pygame uses (width, height, channels)

    print(f"   Enhanced frame shape: {frame_array.shape}, dtype: {frame_array.dtype}")
    frames.append(Image.fromarray(frame_array.astype(np.uint8)))

    print(f"\n🎮 Running demo (capturing frames)...")

    # Run for up to 150 steps or until mission complete
    for step in range(150):
        step_count += 1

        # Get valid actions
        valid_actions = env.get_valid_actions()

        if valid_actions:
            # Simple pathfinding towards goal
            goal_dir = env.goal_pos - env.agent_pos
            best_action = None

            # Choose action that gets us closer to goal
            if goal_dir[0] < 0 and 0 in valid_actions:  # Up
                best_action = 0
            elif goal_dir[0] > 0 and 1 in valid_actions:  # Down
                best_action = 1
            elif goal_dir[1] < 0 and 2 in valid_actions:  # Left
                best_action = 2
            elif goal_dir[1] > 0 and 3 in valid_actions:  # Right
                best_action = 3
            else:
                # Random valid action if can't move toward goal
                best_action = np.random.choice(valid_actions)
        else:
            best_action = 0  # Stay in place

        # Take action
        obs, reward, done, truncated, info = env.step(best_action)
        total_reward += reward

        # Capture frame from enhanced pygame visuals
        env.render()  # Update pygame display
        time.sleep(0.1)  # Let animations render

        # Get pygame surface and convert to PIL Image
        pygame_surface = env.renderer.screen
        frame_array = pygame.surfarray.array3d(pygame_surface)
        frame_array = np.transpose(
            frame_array, (1, 0, 2)
        )  # Pygame uses (width, height, channels)

        frames.append(Image.fromarray(frame_array.astype(np.uint8)))

        # Print progress every 10 steps
        if step % 10 == 0:
            distance = abs(env.agent_pos[0] - env.goal_pos[0]) + abs(
                env.agent_pos[1] - env.goal_pos[1]
            )
            progress = f"{info['goals_completed']}/{info['total_goals']}"
            print(
                f"   Step {step+1:3d}: Goal {env.current_goal[:4].upper()} {progress} | Distance: {distance:2d} | Reward: {total_reward:+6.1f}"
            )

        # Check for goal completion
        if info.get("goal_reached", False):
            print(f"   🎯 Goal reached: {env.current_goal.upper()}!")

        if done:
            if info.get("mission_complete", False):
                print(f"   🏆 MISSION COMPLETE! All destinations visited!")
            else:
                print(f"   ⏰ Timeout reached")
            break

    print(f"\n📊 Demo Statistics:")
    print(f"   Total steps: {step_count}")
    print(f"   Goals completed: {info['goals_completed']}/{info['total_goals']}")
    print(f"   Final reward: {total_reward:.1f}")
    print(f"   Frames captured: {len(frames)}")

    # Save as GIF
    gif_path = "media/enhanced_navigation_demo.gif"
    print(f"\n💾 Saving Enhanced GIF: {gif_path}")

    # Create GIF with appropriate duration
    frame_duration = 200  # 200ms per frame (5 FPS)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,  # Infinite loop
    )

    # Also save final frame as static image
    final_frame_path = "media/enhanced_final_state.png"
    frames[-1].save(final_frame_path)

    print(f"✅ GIF created successfully!")
    print(f"   File: {gif_path}")
    print(f"   Size: {len(frames)} frames")
    print(f"   Duration: ~{len(frames) * frame_duration / 1000:.1f} seconds")
    print(f"   Final state saved: {final_frame_path}")

    env.close()
    return gif_path


if __name__ == "__main__":
    try:
        gif_file = create_demo_gif()
        print(f"\n🎉 Demo GIF ready: {gif_file}")
        print("   You can now view the GIF to see the agent navigating!")

    except Exception as e:
        print(f"❌ Error creating GIF: {e}")
        import traceback

        traceback.print_exc()
