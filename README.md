# Welcome to Atari Games

---

## Task

The challenge is to build ML agents that learn to play Atari video games. Instead of telling the agentwhat to do, we let it figure out the optimal strategy on its own through trial, error, and reward - reinforcement learning.

The core difficulty lies in the complexity of the game environments: raw pixel frames as input, sparse and delayed rewards, and a huge state space that makes traditional Q-tables completely impractical. To solve this I used Deep Q-Networks (DQN) — neural networks that approximate the optimal action-value function directly from screen observations.

Three games were tackled, each with increasing complexity:

- **CartPole** - a classic control problem to get started with reinforcement learning
- **Space Invaders** — a pixel-based Atari game requiring visual understanding
- **Pac-Man** — a more complex environment with ghosts, mazes, and long-horizon planning

---

## Description

All three models are trained using **Reinforcement Learning**, specifically the **DQN** algorithm (and its variants), powered by the `stable-baselines3` library on top of OpenAI Gymnasium environments.

### CartPole

CartPole is a classic control problem where a pole must be balanced on a moving cart. This served as the entry point into reinforcement learning — no pixels, just a 4-dimensional state vector (position, velocity, angle, angular velocity). A simple DQN agent learns to keep the pole upright by pushing the cart left or right.

### Space Invaders

Space Invaders uses raw pixel frames as input. A Convolutional Neural Network (CNN) processes the game screen and outputs Q-values for each possible action (move left, move right, shoot). Key techniques used:
- **Experience Replay** — storing past transitions in a buffer and sampling randomly to break correlations
- **Target Network** — a periodically updated copy of the main network for stable training
- **Frame Stacking** — stacking multiple frames together to give the agent a sense of motion

### Pac-Man

Pac-Man is the hardest of the three. The agent must navigate a maze, eat pellets, avoid ghosts, and manage power-ups. Hyperparameter tuning was done manually, and results across multiple experiment configurations are tracked in `pacman_experiments_params.csv`.

---

## Installation

**Prerequisites:** Python 3.9+, pip.

```bash
# Clone the repository
git clone https://github.com/NickKirilov/atari-games.git (or the repo on Gitea, they are identical)
cd atari-games

# Install dependencies
pip install -r requirements.txt

# Install Atari ROMs (required for Space Invaders and Pac-Man)
pip install ale-py
python -m ale_py.roms
```

> On macOS, you may need `brew install swig` before installing some packages.

---

## Usage

Each game is implemented as a self-contained Jupyter Notebook.

```bash
# Launch Jupyter
jupyter notebook
```

Then open the notebook for the game you want to run:

| Notebook | Game |
|---|---|
| `cart_pole.ipynb` | CartPole-v1 |
| `space_invaders.ipynb` | SpaceInvadersNoFrameskip-v4 |
| `pacman.ipynb` | MsPacmanNoFrameskip-v4 |

Pre-trained model weights are saved in the `models/` directory and are loaded automatically when running inference cells in each notebook.

To run a quick demo with a pre-trained model:

```python
# Example: load and render the CartPole model
from stable_baselines3 import DQN
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
chosen_model = "dqn_cartpole.zip"
model = DQN.load(f"models/cart_pole/{chosen_model}")

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
env.close()
```

Helper utilities shared across notebooks are in `my_atari_helpers.py`.

---

### The Core Team

*Made at [Qwasar SV -- Software Engineering School](https://qwasar.io) By Nikolay Kirilov, student at Amsterdam Tech*

<img alt="Qwasar SV -- Software Engineering School's Logo" src="https://storage.googleapis.com/qwasar-public/qwasar-logo_50x50.png" width="20px" />
