import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

import gym
import d4rl
import numpy as np

from core.utils.torch_utils import random_seed

class AntMaze:
    def __init__(self, seed=np.random.randint(int(1e5))):
        random_seed(seed)
        self.state_dim = (29,)
        self.action_dim = 8
        self.state = None
        self.env = None
        
    def set_env(self, seed):
        self.env.metadata['render_modes'] = ['human']
        self.env.metadata['render_fps'] = 100
        self.env.unwrapped.seed(seed)
        self.env._max_episode_steps = np.inf # control timeout setting in agent

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



class AntUmaze(AntMaze):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super(AntUmaze, self).__init__(seed)
        self.env = gym.make('antmaze-umaze-v0')
        self.set_env(seed)


class AntUmazeDiverse(AntMaze):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super(AntUmazeDiverse, self).__init__(seed)
        self.env = gym.make('antmaze-umaze-diverse-v0')
        self.set_env(seed)
        

class AntMediumDiverse(AntMaze):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super(AntMediumDiverse, self).__init__(seed)
        self.env = gym.make('antmaze-medium-diverse-v0')
        self.set_env(seed)
        

class AntMediumPlay(AntMaze):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super(AntMediumPlay, self).__init__(seed)
        self.env = gym.make('antmaze-medium-play-v0')
        self.set_env(seed)
        

class AntLargePlay(AntMaze):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super(AntLargePlay, self).__init__(seed)
        self.env = gym.make('antmaze-large-play-v0')
        self.set_env(seed)
        

class AntLargeDiverse(AntMaze):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super(AntLargeDiverse, self).__init__(seed)
        self.env = gym.make('antmaze-large-diverse-v0')
        self.set_env(seed)
        

