#!/usr/bin/env python3
"""
DQN Configuration
Hyperparameters and settings specifically for DQN with CNN architecture
"""

# DQN Hyperparameters - optimized for CNN and image observations
DQN_CONFIG = {
    # Learning parameters
    "learning_rate": 1e-4,
    "gamma": 0.99,
    # Experience replay
    "buffer_size": 50000,
    "learning_starts": 10000,
    "batch_size": 32,
    # Exploration
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    # Training frequency
    "train_freq": 8,
    "gradient_steps": 2,
    "target_update_interval": 2000,
    # Network architecture
    "policy_type": "CnnPolicy",  # Use CNN for image processing
    # Training settings
    "total_timesteps": 200000,
    "verbose": 1,
    # Logging
    "log_interval": 100,
    "eval_freq": 10000,
    "eval_episodes": 5,
}

# CNN Architecture settings for DQN
DQN_CNN_CONFIG = {
    "input_channels": 3,  # RGB channels
    "conv_layers": [
        {"filters": 32, "kernel_size": 8, "stride": 4},  # 84x84 -> 20x20
        {"filters": 64, "kernel_size": 4, "stride": 2},  # 20x20 -> 9x9
        {"filters": 64, "kernel_size": 3, "stride": 1},  # 9x9 -> 7x7
    ],
    "fc_layers": [512],  # Dense layers after convolution
    "output_dim": 4,  # Action space size
}


def get_dqn_config():
    """Get DQN hyperparameters"""
    return DQN_CONFIG.copy()


def get_dqn_cnn_config():
    """Get DQN CNN architecture config"""
    return DQN_CNN_CONFIG.copy()


def print_dqn_config():
    """Print DQN configuration"""
    print("🤖 DQN CONFIGURATION")
    print("=" * 50)
    for key, value in DQN_CONFIG.items():
        print(f"{key:25}: {value}")

    print(f"\n🧠 DQN CNN ARCHITECTURE")
    print("=" * 50)
    for key, value in DQN_CNN_CONFIG.items():
        if key == "conv_layers":
            print(f"{key:25}:")
            for i, layer in enumerate(value):
                print(f"  Layer {i+1:15}: {layer}")
        else:
            print(f"{key:25}: {value}")


if __name__ == "__main__":
    print_dqn_config()
