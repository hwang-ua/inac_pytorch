import numpy as np
import gym
import copy

from utils.torch_utils import random_seed


class MountainCar:
    def __init__(self, seed=np.random.randint(int(1e5))):
        random_seed(seed)
        self.state_dim = (2,)
        self.action_dim = 3
        self.env = gym.make('MountainCar-v0')
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


class MountainCarRandom(MountainCar):
    def __init__(self, rand_prob=0.1, seed=np.random.randint(int(1e5))):
        super().__init__(seed=seed)
        self.env_rng = np.random.RandomState(seed)
        self.rand_prob = rand_prob

    def step(self, a):
        p = self.env_rng.random()
        if p < self.rand_prob:
            act = self.env_rng.randint(self.action_dim)
        else:
            act = a[0]
        state, reward, done, info = self.env.step(act)
        return np.asarray(state), np.asarray(reward), np.asarray(done), info


class MountainCarSparse(MountainCar):
    def __init__(self, noise_scale=0, seed=np.random.randint(int(1e5))):
        super().__init__(seed=seed)
        self.noise_scale = noise_scale
        self.env_rng = np.random.RandomState(seed)
        
    def step(self, a):
        state, reward, done, info = self.env.step(a[0])
        if done:
            reward = 1
        else:
            reward = 0
        if self.noise_scale != 0:
            reward += self.env_rng.normal(0, self.noise_scale)
        # print(reward, done)
        return np.asarray(state), np.asarray(reward), np.asarray(done), info
