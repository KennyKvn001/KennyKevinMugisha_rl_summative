#!/usr/bin/env python3
"""
Quick Test Script for New Training Pipeline
Tests all components to ensure everything works correctly
"""

import os
import sys
import torch

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_configurations():
    """Test configuration loading"""
    print("🧪 Testing Configurations...")

    try:
        from config.environment_config import get_environment_config
        from config.dqn_config import get_dqn_config
        from config.policy_gradient_config import (
            get_ppo_config,
            get_a2c_config,
            get_reinforce_config,
        )

        env_config = get_environment_config()
        dqn_config = get_dqn_config()
        ppo_config = get_ppo_config()
        a2c_config = get_a2c_config()
        reinforce_config = get_reinforce_config()

        print("✅ All configurations loaded successfully")
        print(f"  Environment grid size: {env_config['grid_size']}")
        print(f"  DQN learning rate: {dqn_config['learning_rate']}")
        print(f"  PPO total timesteps: {ppo_config['total_timesteps']}")

        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_environments():
    """Test environment creation"""
    print("\n🧪 Testing Environments...")

    try:
        from environment.env_wrapper import (
            create_dqn_environment,
            create_policy_gradient_environment,
        )

        # Test DQN environment
        dqn_env = create_dqn_environment(training_mode=True)
        print(
            f"✅ DQN Environment: {dqn_env.observation_space.shape} observations, {dqn_env.action_space.n} actions"
        )

        # Test episode
        obs, _ = dqn_env.reset()
        action = dqn_env.action_space.sample()
        next_obs, reward, terminated, truncated, info = dqn_env.step(action)
        print(f"  Sample step: reward={reward}, done={terminated or truncated}")
        dqn_env.close()

        # Test Policy Gradient environment
        pg_env = create_policy_gradient_environment(training_mode=True)
        print(
            f"✅ PG Environment: {pg_env.observation_space.shape} observations, {pg_env.action_space.n} actions"
        )

        # Test episode
        obs, _ = pg_env.reset()
        action = pg_env.action_space.sample()
        next_obs, reward, terminated, truncated, info = pg_env.step(action)
        print(f"  Sample step: reward={reward}, done={terminated or truncated}")
        pg_env.close()

        return True
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False


def test_models():
    """Test model creation"""
    print("\n🧪 Testing Models...")

    try:
        from models.dqn_cnn import DQN_CNN
        from models.policy_gradient_mlp import PolicyGradientMLP, create_policy_mlp

        # Test DQN CNN - use HWC format like the environment produces
        dqn_model = DQN_CNN(input_channels=3, output_dim=4)
        test_input = torch.randn(
            1, 84, 84, 3
        )  # (batch, height, width, channels) - like environment
        output = dqn_model(test_input)
        print(f"✅ DQN CNN: Input {test_input.shape} -> Output {output.shape}")

        # Count parameters
        total_params = sum(p.numel() for p in dqn_model.parameters())
        print(f"  Total parameters: {total_params:,}")

        # Test Policy Gradient MLP
        pg_model = PolicyGradientMLP(input_dim=21168, output_dim=4)
        test_input = torch.randn(1, 21168)

        # Test policy output directly
        policy_output = pg_model(test_input)
        print(
            f"✅ Policy MLP: Input {test_input.shape} -> Policy Output {policy_output.shape}"
        )

        # Test separate value network
        from models.policy_gradient_mlp import ValueNetwork

        value_model = ValueNetwork(input_dim=21168)
        value_output = value_model(test_input)
        print(
            f"✅ Value MLP: Input {test_input.shape} -> Value Output {value_output.shape}"
        )

        # Count parameters
        policy_params = sum(p.numel() for p in pg_model.parameters())
        value_params = sum(p.numel() for p in value_model.parameters())
        total_params = policy_params + value_params
        print(f"  Policy parameters: {policy_params:,}")
        print(f"  Value parameters: {value_params:,}")
        print(f"  Total parameters: {total_params:,}")

        # Test simple policy creation
        simple_policy = create_policy_mlp(input_dim=21168, output_dim=4)
        policy_output = simple_policy(test_input)
        print(
            f"✅ Simple Policy: Input {test_input.shape} -> Output {policy_output.shape}"
        )

        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def test_stable_baselines_integration():
    """Test Stable Baselines3 integration"""
    print("\n🧪 Testing Stable Baselines3 Integration...")

    try:
        from stable_baselines3 import DQN, PPO, A2C
        from environment.env_wrapper import (
            create_dqn_environment,
            create_policy_gradient_environment,
        )
        from config.dqn_config import get_dqn_config
        from config.policy_gradient_config import get_ppo_config, get_a2c_config

        # Test DQN
        dqn_env = create_dqn_environment(training_mode=True)
        dqn_config = get_dqn_config()

        dqn_model = DQN(
            dqn_config["policy_type"],
            dqn_env,
            learning_rate=dqn_config["learning_rate"],
            buffer_size=1000,  # Small for testing
            verbose=0,
        )
        print("✅ DQN model created successfully")
        dqn_env.close()

        # Test PPO
        ppo_env = create_policy_gradient_environment(training_mode=True)
        ppo_config = get_ppo_config()

        ppo_model = PPO(
            ppo_config["policy_type"],
            ppo_env,
            learning_rate=ppo_config["learning_rate"],
            n_steps=64,  # Small for testing
            verbose=0,
        )
        print("✅ PPO model created successfully")
        ppo_env.close()

        # Test A2C
        a2c_env = create_policy_gradient_environment(training_mode=True)
        a2c_config = get_a2c_config()

        a2c_model = A2C(
            a2c_config["policy_type"],
            a2c_env,
            learning_rate=a2c_config["learning_rate"],
            n_steps=32,  # Small for testing
            verbose=0,
        )
        print("✅ A2C model created successfully")
        a2c_env.close()

        return True
    except Exception as e:
        print(f"❌ Stable Baselines3 integration test failed: {e}")
        return False


def test_quick_training():
    """Test a quick training run to ensure everything works"""
    print("\n🧪 Testing Quick Training...")

    try:
        from stable_baselines3 import PPO
        from environment.env_wrapper import create_policy_gradient_environment
        from config.policy_gradient_config import get_ppo_config

        # Create environment
        env = create_policy_gradient_environment(training_mode=True)
        ppo_config = get_ppo_config()

        # Create model
        model = PPO(
            ppo_config["policy_type"],
            env,
            learning_rate=ppo_config["learning_rate"],
            n_steps=32,  # Very small for quick test
            verbose=0,
        )

        # Quick training
        print("  Running quick 1000-step training...")
        model.learn(total_timesteps=1000)

        # Test evaluation
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        print(f"✅ Quick training completed! Test action: {action}, reward: {reward}")
        env.close()

        return True
    except Exception as e:
        print(f"❌ Quick training test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 TESTING NEW TRAINING PIPELINE")
    print("=" * 60)

    tests = [
        test_configurations,
        test_environments,
        test_models,
        test_stable_baselines_integration,
        test_quick_training,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Pipeline is ready for training.")
        print("\nNext steps:")
        print("1. Run: python training/train_dqn.py")
        print("2. Run: python training/train_policy_gradients.py")
        print("3. Run: python evaluation/evaluate_all.py")
    else:
        print("❌ Some tests failed. Please fix the issues before training.")

    return passed == total


if __name__ == "__main__":
    main()
