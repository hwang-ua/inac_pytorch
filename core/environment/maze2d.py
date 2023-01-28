import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

import gym
import d4rl
import numpy as np

from core.utils.torch_utils import random_seed

class Maze2d:
    def __init__(self, seed=np.random.randint(int(1e5))):
        random_seed(seed)
        self.state_dim = (4,)
        self.action_dim = 2
        self.state = None
        self.env = None

    def reset(self):
        return self.env.reset()

    def step(self, a):
        ret = self.env.step(a[0])
        state, reward, done, info = ret
        self.state = state
        # self.env.env.render()
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



class Maze2dUmaze(Maze2d):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super(Maze2dUmaze, self).__init__(seed)
        self.env = gym.make('maze2d-umaze-v1')
        self.env.metadata['render_modes'] = ['human']
        self.env.metadata['render_fps'] = 100
        self.env.unwrapped.seed(seed)
        self.env._max_episode_steps = np.inf # control timeout setting in agent
        

class Maze2dMed(Maze2d):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super(Maze2dMed, self).__init__(seed)
        self.env = gym.make('maze2d-medium-v1')
        self.env.metadata['render_modes'] = ['human']
        self.env.metadata['render_fps'] = 100
        self.env.unwrapped.seed(seed)
        self.env._max_episode_steps = np.inf # control timeout setting in agent


class Maze2dLarge(Maze2d):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super(Maze2dLarge, self).__init__(seed)
        self.env = gym.make('maze2d-large-v1')
        self.env.metadata['render_modes'] = ['human']
        self.env.metadata['render_fps'] = 100
        self.env.unwrapped.seed(seed)
        self.env._max_episode_steps = np.inf # control timeout setting in agent

        # d = self.env.get_dataset()
        # print(d['observations'].shape)
        # img = d['rewards'].reshape(2000, 2000)
        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()
        # exit()
