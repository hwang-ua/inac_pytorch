import math

import numpy as np
import gym
import copy

from utils.torch_utils import random_seed

def arcradians(cos, sin):
    if cos > 0 and sin > 0:
        return np.arccos(cos)
    elif cos > 0 and sin < 0:
        return np.arcsin(sin)
    elif cos < 0 and sin > 0:
        return np.arccos(cos)
    elif cos < 0 and sin < 0:
        return -1 * np.arccos(cos)

class Acrobot:
    def __init__(self, seed=np.random.randint(int(1e5))):
        random_seed(seed)
        self.state_dim = (6,)
        self.action_dim = 3
        self.env = gym.make('Acrobot-v1')
        self.env._seed = seed
        self.env._max_episode_steps = np.inf # control timeout setting in agent
        self.state = None

    def generate_state(self, coords):
        return coords

    def reset(self):
        self.state = np.asarray(self.env.reset())
        return self.state

    def step(self, a):
        state, reward, done, info = self.env.step(a[0])
        self.state = state
        # self.env.render()
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
        env = copy.deepcopy(self.env).unwrapped
        env.reset()
        # acos0 = np.arccos(current_s[0])
        # asin0 = np.arcsin(current_s[1])
        # acos1 = np.arccos(current_s[2])
        # asin1 = np.arcsin(current_s[3])
        
        env.state = np.array([arcradians(current_s[0], current_s[1]),
                              arcradians(current_s[2], current_s[3]),
                              current_s[4],
                              current_s[5]])
        # print(env.state)
        state, reward, done, info = env.step(action)
        # print(env.state)
        # print(np.sin(env.state[0]), np.cos(env.state[0]), np.sin(env.state[1]), np.cos(env.state[1]))
        # print(state)
        # print()
        return np.asarray(state), np.asarray(reward), np.asarray(done), info
