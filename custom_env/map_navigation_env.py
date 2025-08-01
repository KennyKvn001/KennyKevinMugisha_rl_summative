"""
Map Navigation Environment
Agent must follow roads to reach destinations
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from custom_env.map_generator import create_road_network_map, is_valid_position
from custom_env.renderer import Renderer


class MapNavigationEnv(gym.Env):
    def __init__(self, grid_size=(15, 15), render_mode="human", training_mode=False):
        super(MapNavigationEnv, self).__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.training_mode = training_mode

        # Enhanced renderer with visual effects
        if render_mode == "human" and not training_mode:
            self.renderer = Renderer(grid_size)
        else:
            self.renderer = None

        # Action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        # Observation space: scaled image for CNN compatibility (Atari-like)
        # Scale from 15x15 to 84x84 (standard Atari size)
        self.obs_size = 84
        self.tile_size = self.obs_size // max(grid_size)  # Calculate scaling factor

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.obs_size, self.obs_size, 3),  # 3 channels: map, agent, goal
            dtype=np.uint8,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        """Reset environment and generate new road network"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Generate road network map
        self.road_map, self.destinations, self.road_positions = create_road_network_map(
            self.grid_size
        )

        # Place agent at a random destination
        self.agent_pos = np.array(random.choice(list(self.destinations.values())))

        # Create a sequence of goals to visit (3-4 destinations)
        all_destinations = list(self.destinations.keys())
        start_dest = None
        for name, pos in self.destinations.items():
            if np.array_equal(pos, self.agent_pos):
                start_dest = name
                break

        # Remove starting destination and create goal sequence
        available_goals = [name for name in all_destinations if name != start_dest]

        # Create mission: visit 3 destinations in sequence
        self.mission_goals = random.sample(
            available_goals, min(3, len(available_goals))
        )
        self.current_goal_index = 0
        self.goals_completed = 0

        # Set first goal
        self.current_goal = self.mission_goals[self.current_goal_index]
        self.goal_pos = np.array(self.destinations[self.current_goal])

        self.steps = 0
        self.max_steps = 300  # Increased due to complex paths

        # Initialize distance tracking for reward calculation
        self.previous_distance = np.abs(self.agent_pos[0] - self.goal_pos[0]) + np.abs(
            self.agent_pos[1] - self.goal_pos[1]
        )

        return self._get_obs(), {}

    def _get_obs(self):
        """Get observation with tile-based scaling for CNN compatibility"""
        # Create base observation at original grid size
        base_obs = np.zeros((*self.grid_size, 3), dtype=np.uint8)

        # Channel 0: Road map (0=obstacle, 128=road, 255=destination)
        base_obs[:, :, 0] = self.road_map.astype(np.uint8) * 128
        for dest_pos in self.destinations.values():
            base_obs[dest_pos[0], dest_pos[1], 0] = 255

        # Channel 1: Agent position
        base_obs[self.agent_pos[0], self.agent_pos[1], 1] = 255

        # Channel 2: Goal position
        base_obs[self.goal_pos[0], self.goal_pos[1], 2] = 255

        # Scale up using tile-based scaling
        scaled_obs = self._scale_observation(base_obs)

        return scaled_obs

    def _scale_observation(self, base_obs):
        """Scale observation from grid_size to obs_size using tile-based scaling"""
        scaled_obs = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)

        # Calculate scaling factors
        scale_h = self.obs_size / self.grid_size[0]
        scale_w = self.obs_size / self.grid_size[1]

        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                # Calculate pixel boundaries for this grid cell
                start_row = int(row * scale_h)
                end_row = int((row + 1) * scale_h)
                start_col = int(col * scale_w)
                end_col = int((col + 1) * scale_w)

                # Fill the tile with the grid cell value
                scaled_obs[start_row:end_row, start_col:end_col, :] = base_obs[
                    row, col, :
                ]

        return scaled_obs

    def step(self, action):
        """Take a step in the environment"""
        self.steps += 1

        # Convert action to int if it's a numpy array (from model.predict())
        if hasattr(action, "item"):
            action = action.item()
        action = int(action)

        # Initialize info dictionary
        info = {
            "goal": self.current_goal,
            "steps": self.steps,
            "agent_pos": tuple(self.agent_pos),
            "goal_pos": tuple(self.goal_pos),
            "goals_completed": self.goals_completed,
            "total_goals": len(self.mission_goals),
            "mission_goals": self.mission_goals,
            "goal_reached": False,
        }

        # Define movement directions
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right

        # Initialize reward for this step
        reward = -0.1  # Small penalty for each step to encourage efficiency

        if action in moves:
            delta = moves[action]
            new_pos = self.agent_pos + delta

            # Check if new position is valid (on road or destination)
            if is_valid_position(self.road_map, tuple(new_pos)):
                # Valid move - update agent position
                self.agent_pos = new_pos
            else:
                # Invalid move - agent hits obstacle/wall
                reward -= 2.0  # Penalty for hitting obstacles
                info["invalid_move"] = True
                # Agent stays in current position

        # Check if current goal reached
        current_goal_reached = np.array_equal(self.agent_pos, self.goal_pos)
        done = False

        # Calculate reward with distance-based incentives
        if current_goal_reached:
            # Trigger celebration effects
            if self.renderer:
                self.renderer.trigger_goal_celebration(self.goal_pos)

            # Goal completed! Give reward and check if mission complete
            self.goals_completed += 1
            goal_reward = 20  # Reward for reaching a destination

            # Check if all goals in mission are completed
            if self.current_goal_index < len(self.mission_goals) - 1:
                # Move to next goal in sequence
                self.current_goal_index += 1
                self.current_goal = self.mission_goals[self.current_goal_index]
                self.goal_pos = np.array(self.destinations[self.current_goal])

                # Update distance tracking for new goal
                self.previous_distance = np.abs(
                    self.agent_pos[0] - self.goal_pos[0]
                ) + np.abs(self.agent_pos[1] - self.goal_pos[1])

                reward = goal_reward  # Intermediate goal reward
                info["goal_reached"] = True
            else:
                # Mission complete! All goals visited
                efficiency_bonus = max(0, (self.max_steps - self.steps) // 10)
                reward = 100 + efficiency_bonus  # Large completion bonus
                done = True
                info["mission_complete"] = True
        else:
            # Distance-based reward to encourage progress
            current_distance = np.abs(self.agent_pos[0] - self.goal_pos[0]) + np.abs(
                self.agent_pos[1] - self.goal_pos[1]
            )

            # Distance-based incentive (modify existing reward from beginning)
            if hasattr(self, "previous_distance"):
                if current_distance < self.previous_distance:
                    reward += 0.5  # Small reward for getting closer
                elif current_distance > self.previous_distance:
                    reward -= 0.2  # Small penalty for moving away

            self.previous_distance = current_distance

        # Episode ends if goal reached or max steps exceeded
        if self.steps >= self.max_steps:
            done = True
            if not np.array_equal(self.agent_pos, self.goal_pos):
                reward = -10  # Penalty for not reaching goal in time
                # Timeout message handled by main script

        # Validate agent position (safety check)
        if not is_valid_position(self.road_map, tuple(self.agent_pos)):
            print(
                f"WARNING: Agent at invalid position {tuple(self.agent_pos)}! Map value: {self.road_map[self.agent_pos[0], self.agent_pos[1]]}"
            )
            # Force agent back to a valid position (nearest destination)
            self.agent_pos = np.array(random.choice(list(self.destinations.values())))
            reward -= 10  # Heavy penalty for position error
            info["position_corrected"] = True

        # Update info dictionary with final values
        info.update(
            {
                "agent_pos": tuple(self.agent_pos),
                "goal_pos": tuple(self.goal_pos),
                "goals_completed": self.goals_completed,
                "goal_reached": current_goal_reached,
            }
        )

        return (
            self._get_obs(),
            reward,
            done,
            False,
            info,
        )

    def render(self):
        """Render the environment with enhanced visual effects"""
        if self.renderer:
            # Prepare info for enhanced UI
            info = {
                "goals_completed": self.goals_completed,
                "total_goals": len(self.mission_goals),
                "mission_goals": self.mission_goals,
                "current_step": self.steps,
                "max_steps": self.max_steps,
            }
            self.renderer.render(
                self.road_map,
                self.agent_pos,
                self.destinations,
                self.current_goal,
                info,
            )

    def close(self):
        """Close the environment"""
        if self.renderer:
            self.renderer.close()

    def get_valid_actions(self):
        """Get list of valid actions from current position"""
        valid_actions = []
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        for action, delta in moves.items():
            new_pos = self.agent_pos + delta
            if is_valid_position(self.road_map, tuple(new_pos)):
                valid_actions.append(action)

        return valid_actions
