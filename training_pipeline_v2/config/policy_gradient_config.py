#!/usr/bin/env python3
"""
Policy Gradient Configuration
Hyperparameters and settings for PPO, A2C, and REINFORCE with MLP architecture
"""

# PPO Hyperparameters - optimized for MLP and flattened observations
PPO_CONFIG = {
    # Learning parameters
    "learning_rate": 3e-4,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    # Policy optimization
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.02,
    # Training batches
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    # Network architecture
    "policy_type": "MlpPolicy",  # Use MLP for flattened observations
    # Training settings
    "total_timesteps": 500000,
    "verbose": 1,
    # Logging
    "log_interval": 100,
    "eval_freq": 20000,
    "eval_episodes": 5,
}

# A2C Hyperparameters - optimized for MLP and flattened observations
A2C_CONFIG = {
    # Learning parameters
    "learning_rate": 7e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    # Policy optimization
    "ent_coef": 0.01,
    "vf_coef": 0.25,
    "max_grad_norm": 0.5,
    # Training batches
    "n_steps": 256,
    # Network architecture
    "policy_type": "MlpPolicy",  # Use MLP for flattened observations
    # Optimizer settings
    "use_rms_prop": True,
    "rms_prop_eps": 1e-5,
    # Training settings
    "total_timesteps": 500000,
    "verbose": 1,
    # Logging
    "log_interval": 100,
    "eval_freq": 20000,
    "eval_episodes": 5,
}

# REINFORCE Hyperparameters - custom implementation with MLP
REINFORCE_CONFIG = {
    # Learning parameters
    "learning_rate": 3e-4,
    "gamma": 0.99,
    # Training settings
    "episodes": 5000,
    "max_episode_steps": 300,
    # Optimization
    "gradient_clipping": 0.5,
    "weight_decay": 1e-4,
    # Logging
    "log_interval": 100,
    "plot_interval": 500,
    # Model saving
    "save_interval": 1000,
}

# MLP Architecture settings for Policy Gradients
PG_MLP_CONFIG = {
    "input_dim": 84 * 84 * 3,  # Flattened observation size
    "hidden_layers": [256, 256],  # Hidden layer sizes
    "activation": "relu",
    "dropout": 0.0,  # No dropout for better performance
    "output_dim": 4,  # Action space size
}


def get_ppo_config():
    """Get PPO hyperparameters"""
    return PPO_CONFIG.copy()


def get_a2c_config():
    """Get A2C hyperparameters"""
    return A2C_CONFIG.copy()


def get_reinforce_config():
    """Get REINFORCE hyperparameters"""
    return REINFORCE_CONFIG.copy()


def get_pg_mlp_config():
    """Get Policy Gradient MLP architecture config"""
    return PG_MLP_CONFIG.copy()


def print_pg_configs():
    """Print all Policy Gradient configurations"""
    print("🚀 PPO CONFIGURATION")
    print("=" * 50)
    for key, value in PPO_CONFIG.items():
        print(f"{key:25}: {value}")

    print(f"\n⚡ A2C CONFIGURATION")
    print("=" * 50)
    for key, value in A2C_CONFIG.items():
        print(f"{key:25}: {value}")

    print(f"\n🎯 REINFORCE CONFIGURATION")
    print("=" * 50)
    for key, value in REINFORCE_CONFIG.items():
        print(f"{key:25}: {value}")

    print(f"\n🧠 POLICY GRADIENT MLP ARCHITECTURE")
    print("=" * 50)
    for key, value in PG_MLP_CONFIG.items():
        print(f"{key:25}: {value}")


if __name__ == "__main__":
    print_pg_configs()
