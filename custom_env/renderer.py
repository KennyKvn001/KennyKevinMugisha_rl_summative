"""
Renderer for road navigation game
"""

import pygame
import numpy as np


class Renderer:
    def __init__(self, grid_size, tile_size=40):
        pygame.init()
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.width = grid_size[1] * tile_size
        self.height = grid_size[0] * tile_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Road Navigation Game")

        # Colors
        self.colors = {
            "obstacle": (60, 60, 60),  # Dark gray for obstacles/walls
            "road": (200, 200, 200),  # Light gray for roads
            "destination": (0, 150, 255),  # Blue for destinations
            "agent": (255, 50, 50),  # Red for agent
            "goal": (255, 255, 0),  # Yellow for current goal
            "grid": (150, 150, 150),  # Grid lines
            "background": (255, 255, 255),  # White background
        }

    def render(self, road_map, agent_pos, destinations, current_goal=None):
        """Render the road navigation game"""
        self.screen.fill(self.colors["background"])

        # Draw map tiles
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                rect = pygame.Rect(
                    col * self.tile_size,
                    row * self.tile_size,
                    self.tile_size,
                    self.tile_size,
                )

                # Choose color based on map value
                if road_map[row, col] == 0:  # Obstacle
                    color = self.colors["obstacle"]
                elif road_map[row, col] == 1:  # Road
                    color = self.colors["road"]
                elif road_map[row, col] == 2:  # Destination
                    color = self.colors["destination"]
                else:
                    color = self.colors["background"]

                pygame.draw.rect(self.screen, color, rect)

                # Draw grid lines
                pygame.draw.rect(self.screen, self.colors["grid"], rect, 1)

        # Highlight current goal destination
        if current_goal and current_goal in destinations:
            goal_pos = destinations[current_goal]
            goal_rect = pygame.Rect(
                goal_pos[1] * self.tile_size,
                goal_pos[0] * self.tile_size,
                self.tile_size,
                self.tile_size,
            )
            pygame.draw.rect(self.screen, self.colors["goal"], goal_rect)

            # Add goal label
            font = pygame.font.Font(None, 18)  # Smaller font for longer names
            display_name = current_goal.upper()
            text = font.render(display_name, True, (0, 0, 0))
            text_rect = text.get_rect(center=goal_rect.center)
            self.screen.blit(text, text_rect)

        # Draw destination labels
        font = pygame.font.Font(None, 16)  # Smaller font for longer names
        for dest_name, dest_pos in destinations.items():
            if dest_name != current_goal:  # Don't label the goal twice
                dest_rect = pygame.Rect(
                    dest_pos[1] * self.tile_size,
                    dest_pos[0] * self.tile_size,
                    self.tile_size,
                    self.tile_size,
                )
                # Use uppercase for better visibility
                display_name = dest_name.upper()
                text = font.render(display_name, True, (255, 255, 255))
                text_rect = text.get_rect(center=dest_rect.center)
                self.screen.blit(text, text_rect)

        # Draw agent
        agent_center = (
            agent_pos[1] * self.tile_size + self.tile_size // 2,
            agent_pos[0] * self.tile_size + self.tile_size // 2,
        )
        pygame.draw.circle(
            self.screen, self.colors["agent"], agent_center, self.tile_size // 3
        )

        pygame.display.flip()

    def close(self):
        pygame.quit()
