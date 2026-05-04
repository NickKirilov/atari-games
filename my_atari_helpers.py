import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

def plot_training(returns, window=20):
    returns = np.array(returns, dtype=float)
    plt.figure()
    plt.plot(returns, label="episode return")
    if len(returns) >= window:
        cumsum = np.cumsum(np.insert(returns, 0, 0))
        rolling = (cumsum[window:] - cumsum[:-window]) / window
        x = np.arange(window-1, len(returns))
        plt.plot(x, rolling, label=f"rolling mean ({window})")
    plt.title("DQN on Space Invaders-v4 - Episode Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.show()

def make_env(env_id: str, seed: int, n_envs: int = 4):
    env = make_atari_env(env_id, n_envs=n_envs, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    
    return env

class EpisodicReturnCallback(BaseCallback):
    def __init__(self, verbose=0) -> None:
        super().__init__(verbose)
        self.episode_returns = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info: 
                ep = info["episode"]
                self.episode_returns.append(float(ep["r"]))
                self.episode_lengths.append(int(ep["l"]))
        return True