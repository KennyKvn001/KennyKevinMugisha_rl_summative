#!/usr/bin/env python3
"""
Environment Configuration
Consistent environment settings for all algorithms
"""

# Environment settings that work for all algorithms
ENVIRONMENT_CONFIG = {
    # Grid size - consistent across all training
    "grid_size": (15, 15),
    # Episode settings
    "max_episode_steps": 300,
    # Observation settings
    "observation_size": 84,  # Scaled to 84x84 for CNN compatibility
    # Training/evaluation modes
    "training_render_mode": None,
    "evaluation_render_mode": "human",
    # Reproducibility
    "random_seed": 42,
}

# Reward settings - single consistent reward system
REWARD_CONFIG = {
    # Base rewards
    "step_penalty": -0.1,  # Small penalty per step
    "goal_reward": 50.0,  # Reward for reaching intermediate goal
    "mission_complete_reward": 100.0,  # Large reward for completing mission
    "invalid_move_penalty": -2.0,  # Penalty for hitting walls
    "timeout_penalty": -10.0,  # Penalty for timeout
    # Efficiency bonus
    "efficiency_bonus_scale": 0.1,  # Bonus for completing quickly
    # Distance-based progress (optional, can be disabled)
    "distance_reward_enabled": True,
    "distance_progress_reward": 0.5,  # Reward for getting closer
    "distance_regression_penalty": 0.2,  # Penalty for moving away
}


def get_environment_config():
    """Get environment configuration"""
    return ENVIRONMENT_CONFIG.copy()


def get_reward_config():
    """Get reward configuration"""
    return REWARD_CONFIG.copy()


def print_config():
    """Print current configuration"""
    print("🏗️ ENVIRONMENT CONFIGURATION")
    print("=" * 50)
    for key, value in ENVIRONMENT_CONFIG.items():
        print(f"{key:25}: {value}")

    print(f"\n💰 REWARD CONFIGURATION")
    print("=" * 50)
    for key, value in REWARD_CONFIG.items():
        print(f"{key:25}: {value}")


if __name__ == "__main__":
    print_config()
