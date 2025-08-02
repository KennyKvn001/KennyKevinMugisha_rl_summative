# Custom Atari RL - Map Navigation Environment

A custom reinforcement learning environment for map navigation with multiple algorithms implementation using Stable-Baselines3 and PyTorch.

## Project Overview

This project implements a custom map navigation environment where an agent must navigate through a grid world to reach multiple destinations. The environment features:

- **Multi-destination navigation**: Agent must visit multiple goals in sequence
- **Dynamic map generation**: Procedurally generated maps with obstacles
- **Enhanced visualization**: Rich visual effects and real-time rendering
- **Multiple RL algorithms**: PPO, DQN, A2C, REINFORCE implementations
- **Dense reward system**: Sophisticated reward shaping for better learning

## Project Structure

```
custom-atari-rl/
├── custom_env/                 # Custom environment implementation
│   ├── __init__.py
│   ├── map_navigation_env.py   # Main environment class
│   ├── map_generator.py        # Procedural map generation
│   └── renderer.py             # Enhanced visualization
├── training/                   # Training scripts for different algorithms
│   ├── train_ppo.py           # Proximal Policy Optimization
│   ├── train_dqn.py           # Deep Q-Network
│   ├── train_a2c.py           # Advantage Actor-Critic
│   ├── train_reinforce.py     # REINFORCE algorithm
│   └── train_policy_gradients_optimized.py  # Optimized policy gradients
├── utils/                      # Utility modules
│   ├── helpers.py              # Training utilities and callbacks
│   ├── make_env.py             # Environment factory
│   └── dense_reward_wrapper.py # Reward shaping wrapper
├── models/                     # Trained model files
├── logs/                       # Training logs
├── tensorboard_logs/           # TensorBoard logs
├── media/                      # Generated media files
├── main.py                     # Demo script
├── evaluate_agent.py           # Model evaluation script
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
└── uv.lock                    # UV lock file
```

## Quick Start

### Prerequisites

- Python 3.11 or higher
- [UV package manager](https://docs.astral.sh/uv/) installed

### Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd custom-atari-rl
   ```

2. **Install dependencies using UV**:

   ```bash
   uv sync
   ```

3. **Activate the virtual environment**:
   ```bash
   uv shell
   ```

### Running the Demo

To see the environment in action:

```bash
uv run python main.py
```

This will run a demo showing the agent navigating through the map with enhanced visuals.

## Environment Features

### Map Navigation Environment

- **Grid Size**: Configurable grid size (default: 18x18)
- **Multi-Goals**: Agent must visit multiple destinations in sequence
- **Obstacles**: Procedurally generated obstacles and walls
- **Reward System**: Dense rewards with efficiency bonuses
- **Visualization**: Rich visual effects with Pygame

### Action Space

- 0: Move Up
- 1: Move Down
- 2: Move Left
- 3: Move Right

### Observation Space

- 84x84x3 RGB image representation of the environment

## 🤖 Training Algorithms

The project includes implementations of several reinforcement learning algorithms:

### 1. PPO (Proximal Policy Optimization)

```bash
uv run python training/train_ppo.py
```

**Features**:

- CNN policy for spatial understanding
- Configurable hyperparameters
- TensorBoard logging
- Early stopping with KL divergence

### 2. DQN (Deep Q-Network)

```bash
uv run python training/train_dqn.py
```

**Features**:

- Experience replay buffer
- Target network updates
- Epsilon-greedy exploration

### 3. A2C (Advantage Actor-Critic)

```bash
uv run python training/train_a2c.py
```

**Features**:

- Separate actor and critic networks
- Advantage estimation
- Parallel environment sampling

### 4. REINFORCE

```bash
uv run python training/train_reinforce.py
```

**Features**:

- Policy gradient implementation
- Baseline subtraction
- Multiple policy architectures (MLP/CNN)

### 5. Optimized Policy Gradients

```bash
uv run python training/train_policy_gradients_optimized.py
```

**Features**:

- Advanced optimization techniques
- Multiple policy architectures
- Comprehensive hyperparameter tuning

## 📊 Training Configuration

### Common Training Parameters

All training scripts support the following parameters:

- **Policy Type**: `CnnPolicy` (default) or `MlpPolicy`
- **Timesteps**: Total training timesteps (default: 500,000)
- **Learning Rate**: Configurable learning rates
- **Batch Size**: Mini-batch size for updates
- **Gamma**: Discount factor for future rewards

### Hyperparameter Examples

```python
# PPO Example
hyperparams = {
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.1,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.01,
}

# DQN Example
hyperparams = {
    "learning_rate": 1e-4,
    "buffer_size": 100000,
    "learning_starts": 1000,
    "batch_size": 32,
    "gamma": 0.99,
    "target_update_interval": 1000,
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
}
```

## Model Evaluation

Evaluate trained models using the evaluation script:

```bash
uv run python evaluate_agent.py --algorithm ppo --model_path models/model_ppo.zip --episodes 10
```

**Supported algorithms**:

- `ppo`: Proximal Policy Optimization
- `dqn`: Deep Q-Network
- `a2c`: Advantage Actor-Critic
- `reinforce`: REINFORCE algorithm

**Evaluation options**:

- `--episodes`: Number of episodes to evaluate (default: 5)
- `--render_mode`: Visualization mode (`human`, `rgb_array`, `none`)
- `--model_path`: Path to the trained model file

## Monitoring Training

### TensorBoard Logs

Monitor training progress with TensorBoard:

```bash
uv run tensorboard --logdir tensorboard_logs/
```

Then open your browser to `http://localhost:6006`

### Training Callbacks

The training scripts include custom callbacks that provide:

- Real-time reward plotting
- Training statistics
- Model checkpointing
- Early stopping

## Customization

### Environment Configuration

Modify environment parameters in `custom_env/map_navigation_env.py`:

```python
# Grid size
grid_size = (18, 18)

# Number of goals
num_goals = 3

# Maximum steps per episode
max_steps = 200

# Reward configuration
goal_reward = 100
step_penalty = -0.1
efficiency_bonus = True
```

### Reward Shaping

Customize rewards using the dense reward wrapper in `utils/dense_reward_wrapper.py`:

```python
# Distance-based rewards
distance_reward = -0.1 * distance_to_goal

# Progress rewards
progress_reward = 10.0 * goals_completed

# Efficiency bonuses
efficiency_bonus = max(0, (max_steps - current_steps) // 20)
```

## 🐛 Troubleshooting

### Common Issues

1. **Pygame Display Issues**:

   ```bash
   export DISPLAY=:0  # For Linux/WSL
   ```

2. **Memory Issues**:

   - Reduce batch size in training scripts
   - Use smaller grid sizes
   - Reduce number of parallel environments

3. **CUDA/GPU Issues**:
   - Install PyTorch with CUDA support: `uv add torch --extra-index-url https://download.pytorch.org/whl/cu118`
   - Or use CPU-only: `uv add torch --extra-index-url https://download.pytorch.org/whl/cpu`

### Performance Tips

- Use `CnnPolicy` for better spatial understanding
- Increase `n_steps` for PPO for more stable training
- Use larger replay buffers for DQN
- Enable efficiency bonuses for faster convergence

## Dependencies

Key dependencies managed by UV:

- **PyTorch**: Deep learning framework
- **Stable-Baselines3**: RL algorithms implementation
- **Gymnasium**: RL environment interface
- **Pygame**: Visualization and rendering
- **NumPy**: Numerical computations
- **Matplotlib**: Plotting and visualization
- **TensorBoard**: Training monitoring

## Acknowledgments

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for RL algorithm implementations
- [Gymnasium](https://gymnasium.farama.org/) for environment interface
- [PyTorch](https://pytorch.org/) for deep learning framework
