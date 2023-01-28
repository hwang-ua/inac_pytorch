import numpy as np
import gym
import copy

from core.utils.torch_utils import random_seed


class Cartpole:
    def __init__(self, seed=np.random.randint(int(1e5))):
        random_seed(seed)
        self.state_dim = (4,)
        self.action_dim = 2
        self.env = gym.make('CartPole-v1')
        self.env._seed = seed
        self.env._max_episode_steps = np.inf # control timeout setting in agent

    def generate_state(self, coords):
        return coords

    def reset(self):
        return np.asarray(self.env.reset())

    def step(self, a):
        state, reward, done, info = self.env.step(a[0])
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
        env.state = current_s
        state, reward, done, info = env.step(action)
        return np.asarray(state), np.asarray(reward), np.asarray(done), info
