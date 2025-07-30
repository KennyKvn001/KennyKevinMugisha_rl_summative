"""
Enhanced Renderer for road navigation game with visual polish and effects
"""

import pygame
import numpy as np
import math
import time
import random
from typing import Dict, List, Tuple, Optional


class Particle:
    """Simple particle for effects"""

    def __init__(self, x, y, vx, vy, color, life=1.0, size=3):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.life = life
        self.max_life = life
        self.size = size

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        self.vy += 200 * dt  # Gravity

    def is_alive(self):
        return self.life > 0

    def get_alpha(self):
        return int(255 * (self.life / self.max_life))


class Renderer:
    def __init__(self, grid_size, tile_size=40):
        pygame.init()
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.width = grid_size[1] * tile_size + 200  # Extra space for UI panel
        self.height = grid_size[0] * tile_size + 100  # Extra space for header
        self.game_width = grid_size[1] * tile_size
        self.game_height = grid_size[0] * tile_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("🚗 Road Navigation - Enhanced Edition")

        # Enhanced color palette
        self.colors = {
            # Map colors with gradients
            "obstacle": (45, 45, 50),
            "obstacle_shadow": (25, 25, 30),
            "road": (220, 220, 225),
            "road_lines": (180, 180, 185),
            "destination": (70, 130, 230),
            "destination_glow": (120, 180, 255),
            # Agent colors
            "agent": (255, 65, 65),
            "agent_glow": (255, 120, 120),
            "agent_trail": (255, 100, 100),
            # Goal colors
            "goal": (255, 215, 0),
            "goal_glow": (255, 255, 150),
            "goal_pulse": (255, 165, 0),
            # UI colors
            "ui_bg": (40, 42, 54),
            "ui_text": (248, 248, 242),
            "ui_accent": (80, 250, 123),
            "ui_progress": (255, 121, 198),
            # Background
            "background": (250, 250, 255),
            "grid": (200, 200, 210),
        }

        # Animation state
        self.time = 0
        self.agent_smooth_pos = None
        self.agent_trail = []
        self.particles = []
        self.last_goal_celebration = 0
        self.screen_shake = 0
        self.goal_pulse = 0

        # Fonts
        try:
            self.font_large = pygame.font.Font(None, 32)
            self.font_medium = pygame.font.Font(None, 24)
            self.font_small = pygame.font.Font(None, 18)
        except:
            self.font_large = pygame.font.Font(None, 32)
            self.font_medium = pygame.font.Font(None, 24)
            self.font_small = pygame.font.Font(None, 18)

        # Game offset for UI
        self.game_offset_x = 10
        self.game_offset_y = 60

    def render(self, road_map, agent_pos, destinations, current_goal=None, info=None):
        """Enhanced render with visual effects and animations"""
        dt = 1 / 60  # Assume 60 FPS
        self.time += dt

        # Update smooth agent position
        self._update_agent_animation(agent_pos)

        # Update particles
        self._update_particles(dt)

        # Clear screen with background
        self.screen.fill(self.colors["background"])

        # Apply screen shake if active
        shake_offset = self._get_screen_shake()

        # Draw game area background
        game_rect = pygame.Rect(
            self.game_offset_x, self.game_offset_y, self.game_width, self.game_height
        )
        pygame.draw.rect(self.screen, (240, 240, 245), game_rect)
        pygame.draw.rect(self.screen, self.colors["grid"], game_rect, 2)

        # Draw enhanced map
        self._draw_enhanced_map(road_map, shake_offset)

        # Draw destinations with glow effects
        self._draw_destinations(destinations, current_goal, shake_offset)

        # Draw current goal with pulsing effect
        if current_goal and current_goal in destinations:
            self._draw_goal_highlight(
                destinations[current_goal], current_goal, shake_offset
            )

        # Draw agent trail
        self._draw_agent_trail(shake_offset)

        # Draw enhanced agent
        self._draw_enhanced_agent(shake_offset)

        # Draw particles
        self._draw_particles()

        # Draw UI panel
        self._draw_ui_panel(info, current_goal)

        # Draw header
        self._draw_header(info)

        pygame.display.flip()

    def _update_agent_animation(self, agent_pos):
        """Update smooth agent position for animations"""
        target_x = (
            agent_pos[1] * self.tile_size + self.tile_size // 2 + self.game_offset_x
        )
        target_y = (
            agent_pos[0] * self.tile_size + self.tile_size // 2 + self.game_offset_y
        )

        if self.agent_smooth_pos is None:
            self.agent_smooth_pos = [target_x, target_y]
        else:
            # Smooth interpolation
            lerp_speed = 0.3
            self.agent_smooth_pos[0] += (
                target_x - self.agent_smooth_pos[0]
            ) * lerp_speed
            self.agent_smooth_pos[1] += (
                target_y - self.agent_smooth_pos[1]
            ) * lerp_speed

        # Add to trail
        self.agent_trail.append(
            (self.agent_smooth_pos[0], self.agent_smooth_pos[1], self.time)
        )

        # Limit trail length
        if len(self.agent_trail) > 15:
            self.agent_trail.pop(0)

    def _update_particles(self, dt):
        """Update particle system"""
        self.particles = [p for p in self.particles if p.is_alive()]
        for particle in self.particles:
            particle.update(dt)

    def _get_screen_shake(self):
        """Calculate screen shake offset"""
        if self.screen_shake > 0:
            self.screen_shake -= 0.1
            intensity = min(self.screen_shake, 1.0)
            return (
                int(math.sin(self.time * 50) * intensity * 3),
                int(math.cos(self.time * 60) * intensity * 3),
            )
        return (0, 0)

    def _draw_enhanced_map(self, road_map, shake_offset):
        """Draw map with enhanced visuals"""
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                x = col * self.tile_size + self.game_offset_x + shake_offset[0]
                y = row * self.tile_size + self.game_offset_y + shake_offset[1]
                rect = pygame.Rect(x, y, self.tile_size, self.tile_size)

                if road_map[row, col] == 0:  # Obstacle
                    # Draw obstacle with shadow effect
                    shadow_rect = pygame.Rect(
                        x + 2, y + 2, self.tile_size, self.tile_size
                    )
                    pygame.draw.rect(
                        self.screen, self.colors["obstacle_shadow"], shadow_rect
                    )
                    pygame.draw.rect(self.screen, self.colors["obstacle"], rect)

                elif road_map[row, col] == 1:  # Road
                    # Draw road with center lines
                    pygame.draw.rect(self.screen, self.colors["road"], rect)

                    # Add road markings
                    if col % 3 == 0:  # Vertical lines
                        line_x = x + self.tile_size // 2
                        pygame.draw.line(
                            self.screen,
                            self.colors["road_lines"],
                            (line_x, y),
                            (line_x, y + self.tile_size),
                            1,
                        )
                    if row % 3 == 0:  # Horizontal lines
                        line_y = y + self.tile_size // 2
                        pygame.draw.line(
                            self.screen,
                            self.colors["road_lines"],
                            (x, line_y),
                            (x + self.tile_size, line_y),
                            1,
                        )

                # Subtle grid
                pygame.draw.rect(self.screen, self.colors["grid"], rect, 1)

    def _draw_destinations(self, destinations, current_goal, shake_offset):
        """Draw destinations with glow effects"""
        for dest_name, dest_pos in destinations.items():
            if dest_name == current_goal:
                continue  # Skip current goal, drawn separately

            x = dest_pos[1] * self.tile_size + self.game_offset_x + shake_offset[0]
            y = dest_pos[0] * self.tile_size + self.game_offset_y + shake_offset[1]
            center = (x + self.tile_size // 2, y + self.tile_size // 2)

            # Draw glow effect
            for i in range(3):
                glow_color = (*self.colors["destination_glow"], 50 - i * 15)
                glow_surface = pygame.Surface(
                    (self.tile_size + i * 4, self.tile_size + i * 4), pygame.SRCALPHA
                )
                pygame.draw.circle(
                    glow_surface,
                    glow_color,
                    (self.tile_size // 2 + i * 2, self.tile_size // 2 + i * 2),
                    self.tile_size // 2 + i * 2,
                )
                self.screen.blit(glow_surface, (x - i * 2, y - i * 2))

            # Draw main destination
            pygame.draw.circle(
                self.screen, self.colors["destination"], center, self.tile_size // 3
            )

            # Draw icon/label
            text = self.font_small.render(dest_name.upper()[:4], True, (255, 255, 255))
            text_rect = text.get_rect(center=center)
            self.screen.blit(text, text_rect)

    def _draw_goal_highlight(self, goal_pos, goal_name, shake_offset):
        """Draw current goal with pulsing effect"""
        x = goal_pos[1] * self.tile_size + self.game_offset_x + shake_offset[0]
        y = goal_pos[0] * self.tile_size + self.game_offset_y + shake_offset[1]
        center = (x + self.tile_size // 2, y + self.tile_size // 2)

        # Pulsing effect
        pulse = abs(math.sin(self.time * 4)) * 0.3 + 0.7
        glow_radius = int((self.tile_size // 2 + 8) * pulse)

        # Draw multiple glow rings
        for i in range(5):
            alpha = int(80 * pulse * (1 - i / 5))
            glow_color = (*self.colors["goal_glow"], alpha)
            glow_surface = pygame.Surface(
                (glow_radius * 2, glow_radius * 2), pygame.SRCALPHA
            )
            pygame.draw.circle(
                glow_surface,
                glow_color,
                (glow_radius, glow_radius),
                glow_radius - i * 2,
            )
            self.screen.blit(
                glow_surface, (center[0] - glow_radius, center[1] - glow_radius)
            )

        # Draw goal circle
        pygame.draw.circle(
            self.screen, self.colors["goal"], center, self.tile_size // 3
        )
        pygame.draw.circle(
            self.screen, self.colors["goal_pulse"], center, self.tile_size // 3, 3
        )

        # Draw label with emphasis
        text = self.font_medium.render(f"🎯 {goal_name.upper()}", True, (0, 0, 0))
        text_rect = text.get_rect(center=(center[0], center[1] - 5))
        self.screen.blit(text, text_rect)

    def _draw_agent_trail(self, shake_offset):
        """Draw agent movement trail"""
        if len(self.agent_trail) < 2:
            return

        for i, (x, y, timestamp) in enumerate(self.agent_trail):
            age = self.time - timestamp
            if age > 1.0:  # Trail fades after 1 second
                continue

            alpha = int(255 * (1 - age) * 0.5)
            size = int((self.tile_size // 6) * (1 - age))

            if size > 1:
                trail_color = (*self.colors["agent_trail"], alpha)
                trail_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(trail_surface, trail_color, (size, size), size)
                self.screen.blit(
                    trail_surface,
                    (x - size + shake_offset[0], y - size + shake_offset[1]),
                )

    def _draw_enhanced_agent(self, shake_offset):
        """Draw agent with enhanced effects"""
        if self.agent_smooth_pos is None:
            return

        x, y = self.agent_smooth_pos
        center = (int(x + shake_offset[0]), int(y + shake_offset[1]))

        # Agent glow
        glow_radius = self.tile_size // 2
        glow_color = (*self.colors["agent_glow"], 100)
        glow_surface = pygame.Surface(
            (glow_radius * 2, glow_radius * 2), pygame.SRCALPHA
        )
        pygame.draw.circle(
            glow_surface, glow_color, (glow_radius, glow_radius), glow_radius
        )
        self.screen.blit(
            glow_surface, (center[0] - glow_radius, center[1] - glow_radius)
        )

        # Main agent circle
        pygame.draw.circle(
            self.screen, self.colors["agent"], center, self.tile_size // 3
        )
        pygame.draw.circle(self.screen, (255, 255, 255), center, self.tile_size // 3, 2)

        # Direction indicator (simple arrow)
        arrow_size = self.tile_size // 6
        pygame.draw.circle(self.screen, (255, 255, 255), center, arrow_size)

    def _draw_particles(self):
        """Draw particle effects"""
        for particle in self.particles:
            alpha = particle.get_alpha()
            if alpha > 0:
                color = (*particle.color, alpha)
                particle_surface = pygame.Surface(
                    (particle.size * 2, particle.size * 2), pygame.SRCALPHA
                )
                pygame.draw.circle(
                    particle_surface,
                    color,
                    (particle.size, particle.size),
                    particle.size,
                )
                self.screen.blit(
                    particle_surface,
                    (particle.x - particle.size, particle.y - particle.size),
                )

    def _draw_ui_panel(self, info, current_goal):
        """Draw information panel"""
        panel_x = self.game_width + self.game_offset_x + 10
        panel_y = self.game_offset_y
        panel_width = 180
        panel_height = self.game_height

        # Panel background
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, self.colors["ui_bg"], panel_rect)
        pygame.draw.rect(self.screen, self.colors["ui_accent"], panel_rect, 2)

        y_offset = panel_y + 20

        # Current Goal
        if current_goal:
            text = self.font_medium.render(
                "Current Goal:", True, self.colors["ui_text"]
            )
            self.screen.blit(text, (panel_x + 10, y_offset))
            y_offset += 25

            goal_text = self.font_small.render(
                f"🎯 {current_goal.upper()}", True, self.colors["ui_accent"]
            )
            self.screen.blit(goal_text, (panel_x + 10, y_offset))
            y_offset += 35

        # Mission Progress
        if info:
            text = self.font_medium.render(
                "Mission Progress:", True, self.colors["ui_text"]
            )
            self.screen.blit(text, (panel_x + 10, y_offset))
            y_offset += 25

            completed = info.get("goals_completed", 0)
            total = info.get("total_goals", 0)

            # Progress bar
            bar_width = 150
            bar_height = 15
            bar_rect = pygame.Rect(panel_x + 10, y_offset, bar_width, bar_height)
            pygame.draw.rect(self.screen, (60, 60, 60), bar_rect)

            if total > 0:
                progress = completed / total
                progress_width = int(bar_width * progress)
                progress_rect = pygame.Rect(
                    panel_x + 10, y_offset, progress_width, bar_height
                )
                pygame.draw.rect(self.screen, self.colors["ui_progress"], progress_rect)

            pygame.draw.rect(self.screen, self.colors["ui_text"], bar_rect, 1)

            progress_text = self.font_small.render(
                f"{completed}/{total}", True, self.colors["ui_text"]
            )
            self.screen.blit(progress_text, (panel_x + 10, y_offset + 20))
            y_offset += 50

        # Legend
        text = self.font_medium.render("Legend:", True, self.colors["ui_text"])
        self.screen.blit(text, (panel_x + 10, y_offset))
        y_offset += 25

        legend_items = [
            ("🚗 Agent", self.colors["agent"]),
            ("🎯 Goal", self.colors["goal"]),
            ("🏢 Places", self.colors["destination"]),
            ("🛣️ Roads", self.colors["road"]),
        ]

        for item_text, color in legend_items:
            # Color square
            color_rect = pygame.Rect(panel_x + 10, y_offset, 12, 12)
            pygame.draw.rect(self.screen, color, color_rect)
            pygame.draw.rect(self.screen, (255, 255, 255), color_rect, 1)

            # Text
            text = self.font_small.render(item_text, True, self.colors["ui_text"])
            self.screen.blit(text, (panel_x + 30, y_offset))
            y_offset += 20

    def _draw_header(self, info):
        """Draw header with game title and stats"""
        # Title
        title = self.font_large.render(
            "🚗 Road Navigation - Enhanced Edition", True, self.colors["ui_bg"]
        )
        title_rect = title.get_rect(center=(self.width // 2, 25))
        self.screen.blit(title, title_rect)

        # Stats line
        if info:
            completed = info.get("goals_completed", 0)
            total = info.get("total_goals", 0)
            stats_text = f"Progress: {completed}/{total} goals completed"
            stats = self.font_small.render(stats_text, True, self.colors["ui_bg"])
            stats_rect = stats.get_rect(center=(self.width // 2, 45))
            self.screen.blit(stats, stats_rect)

    def add_celebration_particles(self, x, y):
        """Add celebration particles at position"""
        # Define color choices for particles
        color_choices = [
            self.colors["goal"],
            self.colors["ui_accent"],
            self.colors["ui_progress"],
        ]

        for _ in range(20):
            vx = (np.random.random() - 0.5) * 200
            vy = (np.random.random() - 0.5) * 200 - 100

            # Use random.choice instead of np.random.choice for tuples
            color = random.choice(color_choices)

            particle = Particle(
                x, y, vx, vy, color, life=2.0, size=np.random.randint(2, 6)
            )
            self.particles.append(particle)

        # Add screen shake
        self.screen_shake = 1.0

    def trigger_goal_celebration(self, goal_pos):
        """Trigger celebration effects when goal is reached"""
        world_x = (
            goal_pos[1] * self.tile_size + self.tile_size // 2 + self.game_offset_x
        )
        world_y = (
            goal_pos[0] * self.tile_size + self.tile_size // 2 + self.game_offset_y
        )
        self.add_celebration_particles(world_x, world_y)

    def close(self):
        pygame.quit()
