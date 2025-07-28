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
        pygame.display.set_caption("City Env")

    def render(self, agent_pos, destinations, goal_pos):
        self.screen.fill((255, 255, 255))

        # Draw grid
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                rect = pygame.Rect(
                    y * self.tile_size,
                    x * self.tile_size,
                    self.tile_size,
                    self.tile_size,
                )
                pygame.draw.rect(self.screen, (220, 220, 220), rect, 1)

        # Draw destinations
        for key, pos in destinations.items():
            color = {
                "hospital": (255, 0, 0),
                "school": (0, 255, 0),
                "market": (0, 0, 255),
            }[key]
            pygame.draw.rect(
                self.screen,
                color,
                pygame.Rect(
                    pos[1] * self.tile_size,
                    pos[0] * self.tile_size,
                    self.tile_size,
                    self.tile_size,
                ),
            )

        # Draw goal
        pygame.draw.rect(
            self.screen,
            (255, 255, 0),
            pygame.Rect(
                goal_pos[1] * self.tile_size,
                goal_pos[0] * self.tile_size,
                self.tile_size,
                self.tile_size,
            ),
        )

        # Draw agent
        pygame.draw.circle(
            self.screen,
            (0, 0, 0),
            (
                agent_pos[1] * self.tile_size + self.tile_size // 2,
                agent_pos[0] * self.tile_size + self.tile_size // 2,
            ),
            self.tile_size // 3,
        )

        pygame.display.flip()

    def close(self):
        pygame.quit()
