# Training ML agent to Play Atari: A Reinforcement Learning Journey

*From a balancing pole to dodging ghosts — how I built three game-playing agents using Deep Q-Networks.*

---

## The Mission

A few months ago I was handed a challenge that sounded almost too fun to be a real project: *build an AI that plays Atari video games*. Not just any games — three of them, each more complex than the last. Cart Pole, Space Invaders, and Pac-Man.

However, there was a challenge. The agent had to figure everything out on its own, learning purely from experience — just like a human would when picking up a new game for the first time.

This post walks through how I approached it, what I built, and what I learned along the way.

---

## What is Reinforcement Learning?

Before diving into code, it helps to understand the framework. Reinforcement learning (RL) is a type of machine learning where an **agent** interacts with an **environment** by taking **actions**, receiving **rewards**, and updating its strategy to maximize long-term cumulative reward.

Think of it like training a dog. The dog (agent) does something. You reward or correct it (reward signal). Over time, the dog learns which behaviors lead to treats.

The formal backbone is the **Markov Decision Process (MDP)**: at each timestep, the agent observes a state `s`, takes an action `a`, receives a reward `r`, and transitions to a new state `s'`. The goal is to learn a **policy** — a mapping from states to actions — that maximizes expected future reward.

---

## Why I did not use Q-Tables for Atari?

The classic RL approach uses a **Q-table**: a giant lookup table mapping every possible (state, action) pair to a value representing how good that action is in that state. It works great for small problems like Tic-Tac-Toe or a simple gridworld.

But Atari games render at **210×160 pixels with 128 colors**. The number of possible game screens is astronomically large — far beyond any table. We need a way to *generalize* across similar states rather than memorizing each one individually.

---

## Why I used DQN?

DQN, introduced by DeepMind in 2015, replaces the Q-table with a **neural network** that takes a game state as input and outputs Q-values for every possible action. The network learns to approximate the optimal Q-function directly from raw pixels.

Two clever tricks make training stable:

**Experience Replay** — Instead of learning from each transition immediately, the agent stores experiences `(s, a, r, s')` in a memory buffer and samples random mini-batches to train on. This breaks the temporal correlation between consecutive frames, which would otherwise make the network overfit to recent experience.

**Target Network** — A second neural network with frozen weights is used to compute the training targets. Its weights are updated periodically (not every step), preventing the "chasing a moving target" instability that would otherwise arise when both the predictions and targets shift simultaneously.

---

## Game 1: CartPole — Getting Comfortable with RL

CartPole is where every RL practitioner starts. A pole is attached to a cart that moves along a track. The agent must push the cart left or right to keep the pole from falling over.

The state is simple: just 4 numbers (cart position, cart velocity, pole angle, pole angular velocity). No images needed. This makes it a perfect environment to verify that the training loop, reward shaping, and model architecture are all working before scaling up.

I used `stable-baselines3`'s DQN implementation with default hyperparameters. The agent trained to the maximum episode length (500 steps) reliably within a few hundred episodes. Watching it balance the pole indefinitely after training was deeply satisfying — the first real proof that the approach works.

**Key insight:** CartPole teaches you what a "solved" RL agent looks like. Everything after this is about scaling up that success.

---

## Game 2: Space Invaders — Learning from Pixels

Space Invaders is where things get genuinely interesting. The input is no longer a clean state vector — it's a **raw pixel frame** from the game screen. The agent must learn to associate visual patterns (where the invaders are, where the bullets are) with the right actions (move left, move right, fire).

### Architecture

The network uses a **Convolutional Neural Network (CNN)** backbone — the same basic architecture DeepMind used in the original DQN paper. Convolutional layers are ideal for spatial data because they detect local features (edges, shapes, objects) that are translation-invariant.

### Frame Stacking

A single frame doesn't tell you which direction things are moving. To give the agent a sense of motion and velocity, I stacked **4 consecutive frames** as a single input. Now the network can implicitly "see" the trajectory of bullets and enemies, not just their static positions.

### Preprocessing

Raw Atari frames are downscaled to **84×84 grayscale** before being fed to the network. This reduces computation while retaining the information the agent actually needs.

### Training

Training on pixel input is significantly heavier than CartPole — this is where GPUs become non-negotiable (I still trained it on my personal laptop, and yes it took a lot of time :)). The agent runs for millions of environment steps, gradually improving its score. Early in training it mostly fires randomly. After enough experience, it learns to aim, dodge, and prioritize which invaders to target first.

**Key insight:** Vision-based RL requires patience and compute. The reward signal is sparse — the agent doesn't know it made a mistake until several steps later when it gets hit.

---

## Game 3: Pac-Man — The Hardest Challenge

Pac-Man is the most complex of the three. The agent must navigate a maze, eat pellets, pursue power pellets strategically, avoid ghosts, and chase them when powered up. The optimal strategy requires long-horizon planning — decisions made now affect outcomes many seconds later.

### Hyperparameter Tuning

With more complexity comes more sensitivity to hyperparameters. Learning rate, batch size, discount factor (gamma), exploration schedule — each of these can make or break training.

Each trial runs for a fixed number of timesteps and reports the mean episode reward. I used the results to suggest more promising configurations for the next trial. All trial configurations and their results are logged in `pacman_experiments_params.csv` for reproducibility and analysis.

### What the Agent Learned

The trained Pac-Man agent demonstrates some genuinely interesting emergent behaviors. It learns to clear pellets efficiently, use power pellets to turn the tables on ghosts, and avoid corners where escape routes are limited. It's not perfect — ghost avoidance is the hardest skill to fully acquire — but it consistently outperforms random play and reaches respectable scores.

**Key insight:** Hyperparameter tuning is not optional for complex environments. A poorly tuned agent can train for hours and learn nothing, while the right configuration unlocks rapid improvement.

---

## Tools and Stack

The entire project runs on Python with the following core libraries:

- **Gymnasium** — the standard RL environment interface (successor to OpenAI Gym)
- **stable-baselines3** — battle-tested RL algorithm implementations
- **PyTorch** — the underlying deep learning framework
- **Matplotlib / Pandas** — visualization and experiment tracking

---

## Results and Reflections

| Game | Approach | Outcome |
|---|---|---|
| CartPole | DQN, state vector input | Solved (500 steps consistently) |
| Space Invaders | DQN + CNN + frame stacking | Strong performance, above human baseline |
| Pac-Man | DQN + CNN + frame stacking + hyperparameter tuning | Solid play, effective pellet clearing |

The jump from CartPole to Space Invaders felt enormous — not just in compute requirements, but in the sheer patience required to watch an agent improve from random flailing to deliberate strategy. Pac-Man added another layer of appreciation for how hard it is to encode long-term planning into a reward signal.

The thing that struck me most across all three projects is how little the agent "knows" yet how much it ultimately achieves. There's no explicit representation of a ghost, no map, no concept of a bullet. Just pixel values and reward signals. And yet, given enough experience, the network learns to act as if it understands all of it.

That's what makes deep reinforcement learning so compelling — and so worth exploring.

---

## What's Next

There are several directions worth exploring from here:

- **Prioritized Experience Replay** — sampling more informative transitions more frequently
- **Dueling DQN** — separating the value and advantage functions for more stable learning
- **PPO or A3C** — policy gradient methods that often outperform DQN on complex Atari games
- **Transfer Learning** — pretrain on simpler games and fine-tune on harder ones

The code, notebooks, and pre-trained models are all available on [GitHub](https://github.com/NickKirilov/atari-games) (or in the corresponfing repo on Gitea). If you're just starting out with RL, CartPole is the perfect entry point — get that working first, then scale up from there.

---

*Made at [Qwasar SV — Software Engineering School](https://qwasar.io), by Nikolay Kirilov, student at Amsterdam Tech*
