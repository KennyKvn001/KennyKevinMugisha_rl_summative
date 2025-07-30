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

        # Observation space: road map + agent position + goal position
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(*grid_size, 3),  # 3 channels: map, agent, goal
            dtype=np.float32,
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
        """Get observation with map, agent position, and goal position"""
        obs = np.zeros((*self.grid_size, 3), dtype=np.float32)

        # Channel 0: Road map (0=obstacle, 1=road, 2=destination)
        obs[:, :, 0] = self.road_map.astype(np.float32)

        # Channel 1: Agent position
        obs[self.agent_pos[0], self.agent_pos[1], 1] = 1.0

        # Channel 2: Goal position
        obs[self.goal_pos[0], self.goal_pos[1], 2] = 1.0

        return obs

    def step(self, action):
        """Take a step in the environment"""
        self.steps += 1

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
        reward = -0.5  # Small penalty for each step to encourage efficiency

        if action in moves:
            delta = moves[action]
            new_pos = self.agent_pos + delta

            # Check if new position is valid (on road or destination)
            if is_valid_position(self.road_map, tuple(new_pos)):
                # Valid move - update agent position
                self.agent_pos = new_pos
            else:
                # Invalid move - agent hits obstacle/wall
                reward -= 5.0  # Penalty for hitting obstacles
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
            goal_reward = 15  # Reward for reaching a destination

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
                efficiency_bonus = max(0, (self.max_steps - self.steps) // 20)
                reward = 50 + efficiency_bonus  # Large completion bonus
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
                    reward += 1  # Small reward for getting closer
                elif current_distance > self.previous_distance:
                    reward -= 1  # Small penalty for moving away

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
