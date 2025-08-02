#!/usr/bin/env python3
"""
Dense Reward Wrapper for Policy Gradient Algorithms
Adds dense rewards without changing the original reward structure
"""

import gymnasium as gym
import numpy as np
from collections import deque


class DenseRewardWrapper(gym.RewardWrapper):
    """
    Adds dense rewards for policy gradients while preserving original rewards
    """
    
    def __init__(self, env, dense_reward_scale=1.0):
        super(DenseRewardWrapper, self).__init__(env)
        self.dense_reward_scale = dense_reward_scale
        
        # Tracking variables
        self.previous_distance = None
        self.visited_positions = set()
        self.waypoints_reached = set()
        self.best_distance = float('inf')
        self.steps_since_progress = 0
        self.last_goal_pos = None
        
    def reset(self, **kwargs):
        """Reset tracking variables"""
        obs, info = self.env.reset(**kwargs)
        
        # Reset tracking
        self.visited_positions.clear()
        self.waypoints_reached.clear()
        self.best_distance = float('inf')
        self.steps_since_progress = 0
        self.last_goal_pos = None
        
        # Initialize distance tracking
        if hasattr(self.env, 'agent_pos') and hasattr(self.env, 'goal_pos'):
            self.previous_distance = self._calculate_distance()
            self.best_distance = self.previous_distance
            self.last_goal_pos = tuple(self.env.goal_pos)
        
        return obs, info
    
    def _calculate_distance(self):
        """Calculate Manhattan distance to goal"""
        return abs(self.env.agent_pos[0] - self.env.goal_pos[0]) + abs(self.env.agent_pos[1] - self.env.goal_pos[1])
    
    def reward(self, reward):
        """Add dense rewards to original reward"""
        if not hasattr(self.env, 'agent_pos') or not hasattr(self.env, 'goal_pos'):
            return reward
        
        dense_reward = 0.0
        current_distance = self._calculate_distance()
        current_pos = tuple(self.env.agent_pos)
        current_goal_pos = tuple(self.env.goal_pos)
        
        # Check if goal changed (new goal in sequence)
        if self.last_goal_pos != current_goal_pos:
            self.previous_distance = current_distance
            self.best_distance = current_distance
            self.waypoints_reached.clear()
            self.steps_since_progress = 0
            self.last_goal_pos = current_goal_pos
        
        # 1. Progress Reward - Moving closer to goal
        if self.previous_distance is not None:
            if current_distance < self.previous_distance:
                # Reward for getting closer (scaled by progress amount)
                progress = self.previous_distance - current_distance
                dense_reward += 2.0 * progress  # Bigger reward for bigger progress
                self.steps_since_progress = 0
            elif current_distance > self.previous_distance:
                # Penalty for moving away
                regression = current_distance - self.previous_distance
                dense_reward -= 1.0 * regression
                self.steps_since_progress += 1
            else:
                # No progress
                self.steps_since_progress += 1
        
        # 2. Best Distance Reward - Reaching new best distance
        if current_distance < self.best_distance:
            improvement = self.best_distance - current_distance
            dense_reward += 5.0 * improvement  # Big reward for new best
            self.best_distance = current_distance
        
        # 3. Waypoint Rewards - Milestone distances
        distance_waypoints = [20, 15, 10, 7, 5, 3, 1]
        for waypoint in distance_waypoints:
            if current_distance <= waypoint and waypoint not in self.waypoints_reached:
                dense_reward += 10.0 + (20 - waypoint)  # Bigger reward for closer waypoints
                self.waypoints_reached.add(waypoint)
        
        # 4. Exploration Reward - Visiting new positions
        if current_pos not in self.visited_positions:
            dense_reward += 0.5  # Small reward for exploration
            self.visited_positions.add(current_pos)
        
        # 5. Direction Reward - Moving in right general direction
        if self.previous_distance is not None and current_distance != self.previous_distance:
            goal_direction = np.sign(self.env.goal_pos - self.env.agent_pos)
            # Reward for any movement toward goal
            if current_distance < self.previous_distance:
                dense_reward += 1.0  # Good direction
        
        # 6. Stagnation Penalty - Encourage action when stuck
        if self.steps_since_progress > 20:
            dense_reward -= 0.1 * (self.steps_since_progress - 20)  # Increasing penalty
        
        # 7. Goal Proximity Bonus - Higher rewards when very close
        if current_distance <= 3:
            dense_reward += (4 - current_distance) * 2.0  # Exponential bonus when close
        
        # Update tracking
        self.previous_distance = current_distance
        
        # Scale dense rewards and add to original
        total_dense_reward = dense_reward * self.dense_reward_scale
        
        # Return original reward + dense reward
        return reward + total_dense_reward


class PolicyGradientEnvWrapper:
    """
    Complete environment wrapper for policy gradient algorithms
    """
    
    def __init__(self, base_env_creator, use_dense_rewards=True, dense_scale=1.0):
        self.base_env_creator = base_env_creator
        self.use_dense_rewards = use_dense_rewards
        self.dense_scale = dense_scale
    
    def create_env(self):
        """Create environment optimized for policy gradients"""
        # Create base environment
        env = self.base_env_creator()
        
        # Add dense rewards if requested
        if self.use_dense_rewards:
            env = DenseRewardWrapper(env, dense_reward_scale=self.dense_scale)
        
        # Add monitoring
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(env)
        
        return env


def test_dense_wrapper():
    """Test the dense reward wrapper"""
    from utils.make_env import create_env
    
    print("Testing Dense Reward Wrapper...")
    
    # Create wrapped environment
    base_env = create_env(training_mode=True)
    wrapped_env = DenseRewardWrapper(base_env, dense_reward_scale=1.0)
    
    obs, info = wrapped_env.reset()
    total_original_reward = 0
    total_dense_reward = 0
    
    print(f"Initial distance to goal: {wrapped_env._calculate_distance()}")
    
    for step in range(50):
        # Take random action
        action = wrapped_env.action_space.sample()
        obs, reward, done, truncated, info = wrapped_env.step(action)
        
        # Get original reward (would need to track this separately in real implementation)
        # For testing, we'll estimate
        distance = wrapped_env._calculate_distance()
        
        print(f"Step {step+1}: Action={action}, Distance={distance}, Total Reward={reward:.2f}")
        
        if done or truncated:
            break
    
    print("Dense wrapper test completed!")


if __name__ == "__main__":
    test_dense_wrapper()