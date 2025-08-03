#!/usr/bin/env python3
"""
CNN Architecture for DQN
Optimized for processing (84, 84, 3) image observations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_CNN(nn.Module):
    """
    CNN architecture optimized for DQN on navigation environment
    Processes (84, 84, 3) images and outputs Q-values for 4 actions
    """

    def __init__(self, input_channels=3, output_dim=4):
        super(DQN_CNN, self).__init__()

        # Convolutional layers - designed for 84x84 input
        self.conv1 = nn.Conv2d(
            input_channels, 32, kernel_size=8, stride=4
        )  # 84x84 -> 20x20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # 20x20 -> 9x9
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # 9x9 -> 7x7

        # Calculate flattened size: 64 * 7 * 7 = 3136
        self.conv_output_size = 64 * 7 * 7

        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 512)
        self.fc2 = nn.Linear(512, output_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass
        Input: (batch, height, width, channels) or (height, width, channels)
        Output: (batch, action_dim) Q-values
        """
        # Handle single observation vs batch
        if len(x.shape) == 3:  # Single observation (H, W, C)
            x = x.permute(2, 0, 1).unsqueeze(0)  # -> (1, C, H, W)
        elif len(x.shape) == 4:  # Batch of observations (B, H, W, C)
            x = x.permute(0, 3, 1, 2)  # -> (B, C, H, W)

        # Normalize pixel values to [0, 1]
        x = x.float() / 255.0

        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten for fully connected layers
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)

        return q_values

    def get_feature_size(self):
        """Get the size of flattened features after conv layers"""
        return self.conv_output_size


def create_dqn_cnn(input_channels=3, output_dim=4):
    """
    Factory function to create DQN CNN

    Args:
        input_channels: Number of input channels (default: 3 for RGB)
        output_dim: Number of actions (default: 4 for navigation)

    Returns:
        DQN_CNN model
    """
    return DQN_CNN(input_channels=input_channels, output_dim=output_dim)


def test_dqn_cnn():
    """Test the DQN CNN architecture"""
    print("🧪 Testing DQN CNN Architecture")
    print("=" * 40)

    # Create model
    model = create_dqn_cnn()
    print(f"Model created successfully")
    print(f"Feature size: {model.get_feature_size()}")

    # Test with single observation
    single_obs = torch.randint(0, 255, (84, 84, 3), dtype=torch.uint8)
    print(f"Single observation shape: {single_obs.shape}")

    with torch.no_grad():
        q_values = model(single_obs)
        print(f"Single Q-values shape: {q_values.shape}")
        print(f"Q-values: {q_values.numpy()}")

    # Test with batch
    batch_obs = torch.randint(0, 255, (4, 84, 84, 3), dtype=torch.uint8)
    print(f"\nBatch observation shape: {batch_obs.shape}")

    with torch.no_grad():
        batch_q_values = model(batch_obs)
        print(f"Batch Q-values shape: {batch_q_values.shape}")

    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("✅ DQN CNN test completed successfully!")


if __name__ == "__main__":
    test_dqn_cnn()
