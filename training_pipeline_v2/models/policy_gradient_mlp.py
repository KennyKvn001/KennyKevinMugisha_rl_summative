#!/usr/bin/env python3
"""
MLP Architecture for Policy Gradient Methods
Optimized for processing flattened (84*84*3,) observations from PPO, A2C, and REINFORCE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyGradientMLP(nn.Module):
    """
    MLP architecture for Policy Gradient methods
    Processes flattened observations and outputs action probabilities
    """

    def __init__(
        self, input_dim=84 * 84 * 3, hidden_layers=[256, 256], output_dim=4, dropout=0.0
    ):
        super(PolicyGradientMLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softmax(dim=-1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass
        Input: (batch, input_dim) or (input_dim,) flattened observations
        Output: (batch, action_dim) action probabilities
        """
        # Handle single observation vs batch
        if len(x.shape) == 3:  # Image format (H, W, C) - flatten it
            x = x.flatten()
        elif len(x.shape) == 4:  # Batch of images (B, H, W, C) - flatten each
            x = x.view(x.size(0), -1)
        elif len(x.shape) == 1:  # Single flattened observation
            x = x.unsqueeze(0)  # Add batch dimension

        # Normalize if input appears to be image data (0-255 range)
        if x.max() > 1.0:
            x = x.float() / 255.0

        # Forward pass through network
        action_probs = self.network(x)

        return action_probs

    def get_action(self, observation, deterministic=False):
        """
        Get action from observation

        Args:
            observation: Environment observation
            deterministic: If True, take argmax action; if False, sample

        Returns:
            action: Selected action
            log_prob: Log probability of action (for training)
        """
        with torch.no_grad():
            action_probs = self.forward(observation)

            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
                log_prob = torch.log(action_probs.gather(-1, action.unsqueeze(-1)))
            else:
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

        return action.item(), log_prob


class ValueNetwork(nn.Module):
    """
    Value network for Actor-Critic methods
    Processes flattened observations and outputs state values
    """

    def __init__(self, input_dim=84 * 84 * 3, hidden_layers=[256, 256], dropout=0.0):
        super(ValueNetwork, self).__init__()

        self.input_dim = input_dim

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (single value)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass
        Input: (batch, input_dim) or (input_dim,) flattened observations
        Output: (batch, 1) state values
        """
        # Handle single observation vs batch
        if len(x.shape) == 3:  # Image format (H, W, C) - flatten it
            x = x.flatten()
        elif len(x.shape) == 4:  # Batch of images (B, H, W, C) - flatten each
            x = x.view(x.size(0), -1)
        elif len(x.shape) == 1:  # Single flattened observation
            x = x.unsqueeze(0)  # Add batch dimension

        # Normalize if input appears to be image data (0-255 range)
        if x.max() > 1.0:
            x = x.float() / 255.0

        # Forward pass through network
        value = self.network(x)

        return value


def create_policy_mlp(
    input_dim=84 * 84 * 3, hidden_layers=[256, 256], output_dim=4, dropout=0.0
):
    """
    Factory function to create Policy MLP

    Args:
        input_dim: Size of flattened observation (default: 84*84*3)
        hidden_layers: List of hidden layer sizes
        output_dim: Number of actions (default: 4 for navigation)
        dropout: Dropout rate (default: 0.0)

    Returns:
        PolicyGradientMLP model
    """
    return PolicyGradientMLP(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        output_dim=output_dim,
        dropout=dropout,
    )


def create_value_mlp(input_dim=84 * 84 * 3, hidden_layers=[256, 256], dropout=0.0):
    """
    Factory function to create Value MLP

    Args:
        input_dim: Size of flattened observation (default: 84*84*3)
        hidden_layers: List of hidden layer sizes
        dropout: Dropout rate (default: 0.0)

    Returns:
        ValueNetwork model
    """
    return ValueNetwork(
        input_dim=input_dim, hidden_layers=hidden_layers, dropout=dropout
    )


def test_policy_gradient_mlp():
    """Test the Policy Gradient MLP architecture"""
    print("🧪 Testing Policy Gradient MLP Architecture")
    print("=" * 50)

    # Create models
    policy = create_policy_mlp()
    value_net = create_value_mlp()

    print(f"Policy network created successfully")
    print(f"Value network created successfully")

    # Test with single observation (image format)
    single_obs = torch.randint(0, 255, (84, 84, 3), dtype=torch.uint8)
    print(f"Single observation shape: {single_obs.shape}")

    with torch.no_grad():
        action_probs = policy(single_obs)
        state_value = value_net(single_obs)

        print(f"Action probabilities shape: {action_probs.shape}")
        print(f"Action probabilities: {action_probs.numpy()}")
        print(f"State value shape: {state_value.shape}")
        print(f"State value: {state_value.numpy()}")

    # Test action selection
    action, log_prob = policy.get_action(single_obs, deterministic=False)
    print(f"Sampled action: {action}")
    print(f"Log probability: {log_prob.numpy()}")

    # Test with batch
    batch_obs = torch.randint(0, 255, (4, 84, 84, 3), dtype=torch.uint8)
    print(f"\nBatch observation shape: {batch_obs.shape}")

    with torch.no_grad():
        batch_probs = policy(batch_obs)
        batch_values = value_net(batch_obs)

        print(f"Batch action probabilities shape: {batch_probs.shape}")
        print(f"Batch state values shape: {batch_values.shape}")

    # Model summary
    policy_params = sum(p.numel() for p in policy.parameters())
    value_params = sum(p.numel() for p in value_net.parameters())

    print(f"\nModel Summary:")
    print(f"Policy network parameters: {policy_params:,}")
    print(f"Value network parameters: {value_params:,}")
    print(f"Total parameters: {policy_params + value_params:,}")

    print("✅ Policy Gradient MLP test completed successfully!")


if __name__ == "__main__":
    test_policy_gradient_mlp()
