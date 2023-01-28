import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

import gym
import numpy as np
from minatar import Environment

from core.utils.torch_utils import random_seed


class MinAtar:
    def __init__(self, env_name, seed=np.random.randint(int(1e5))):
        random_seed(seed)
        self.env = gym.make(env_name)
        self.env.seed(seed)
        self.env.game._max_episode_steps = np.inf # control timeout setting in agent
        self.state = None
        self.state_dim = self.env.game.state_shape()
        self.action_dim = self.env.game.num_actions()

    def reset(self):
        return self.env.reset()

    def step(self, a):
        state, reward, done, info = self.env.step(a[0])
        self.state = state
        self.env.render()
        return np.asarray(state), np.asarray(reward), np.asarray(done), info

    def get_visualization_segment(self):
        raise NotImplementedError

    def get_useful(self, state=None):
        if state:
            return state
        else:
            return np.array(self.env.state)

    def info(self, key):
        return

    def hack_step(self, current_s, action):
        raise NotImplementedError


class Asterix(MinAtar):
    def __init__(self, seed):
        super().__init__('MinAtar/Asterix-v0', seed)


class Freeway(MinAtar):
    def __init__(self, seed):
        super().__init__('MinAtar/Freeway-v0', seed)


class SpaceInvaders(MinAtar):
    def __init__(self, seed):
        super().__init__('MinAtar/SpaceInvaders-v0', seed)


class Seaquest(MinAtar):
    def __init__(self, seed):
        super().__init__('MinAtar/Seaquest-v0', seed)


class Breakout(MinAtar):
    def __init__(self, seed):
        super().__init__('MinAtar/Breakout-v0', seed)

    