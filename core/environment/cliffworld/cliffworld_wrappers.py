import gym
from collections import deque
import numpy as np
import arguments as args
from cliffworld import cliffworld_env

class PreprocessWrapper():
    def __init__(self, env, r_preprocess=None, s_preprocess=None):
        '''
        reward & state preprocess
        record info like real reward, episode length, etc
        Be careful: when an episode is done: check info['episode'] for information
        '''
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.r_preprocess = r_preprocess
        self.s_preprocess = s_preprocess
        # self.s_preprocess = lambda x:x/255.
        self.rewards = []

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        # state = state.astype('float32') # todo: can change to int8 on atari
        self.rewards.append(reward)
        if done:
            # if no EpisodicLifeEnv_withInfos wrapper, update info here
            if not info.get('EpisodicLife'):
                # return None if there is no EpisodicLife
                eprew = sum(self.rewards)
                eplen = len(self.rewards)
                epinfo = {"r": round(eprew, 6), "l": eplen}
                assert isinstance(info,dict)
                if isinstance(info,dict):
                    info['episode'] = epinfo
                self.rewards = []
        # preprocess reward
        if self.r_preprocess is not None:
            reward = self.r_preprocess(reward)
        # preprocess state
        if self.s_preprocess is not None:
            state = self.s_preprocess(state)
        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        # state = state.astype('float32') # todo: can change to int8 on atari
        # preprocess state
        if self.s_preprocess is not None:
            state = self.s_preprocess(state)
        return state

    def render(self, mode='human'):
        return self.env.render(mode=mode)

class TimeLimit():
    def __init__(self, env, max_episode_steps=None):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class BatchEnvWrapper:
    def __init__(self, envs,planning = False):
        self.envs = envs
        self.observation_space = envs[0].observation_space
        # self.observation_space = [84,84,1]
        self.action_space = envs[0].action_space
        self.epinfobuf = deque(maxlen=100)

    def step(self, actions):
        states = []
        rewards = []
        dones = []
        infos = []
        for i, env in enumerate(self.envs):
            state, reward, done, info = env.step(actions[i])
            if done:
                # print(done,state)
                info['terminal_state'] = state
                if 'TimeLimit.truncated' in info:
                    info['maxsteps_used'] = True
                else:
                    info['maxsteps_used'] = False
                state = env.reset()
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            maybeepinfo = info.get('episode')

            if maybeepinfo:
                self.epinfobuf.append(maybeepinfo)

        # print(infos)
        return states, rewards, dones, infos

    def reset(self):
        return [self.envs[i].reset() for i in range(self.get_num_of_envs())]

    def render(self, mode='human'):
        return self.envs[0].render(mode=mode)

    def close(self):
        self.envs[0].close()

    def get_num_of_envs(self):
        return len(self.envs)

    def get_episode_rewmean(self):
        #print([epinfo['r'] for epinfo in self.epinfobuf])
        #input()
        return round(self.safemean([epinfo['r'] for epinfo in self.epinfobuf]),2)

    def get_episode_rewstd(self):
        #print([epinfo['r'] for epinfo in self.epinfobuf])
        return round(self.safestd([epinfo['r'] for epinfo in self.epinfobuf]),2)

    def get_episode_rewmax(self):
        return round(self.safemax([epinfo['r'] for epinfo in self.epinfobuf]),2)

    def get_list_of_episode(self):
        return [epinfo['r'] for epinfo in self.epinfobuf]

    def get_episode_lenmean(self):
        return round(self.safemean([epinfo['l'] for epinfo in self.epinfobuf]),2)

    def safemean(self,xs):
        return np.nan if len(xs) == 0 else np.mean(xs)

    def safemax(self, xs):
        return np.nan if len(xs) == 0 else np.max(xs)

    def safestd(self,xs):
        return np.nan if len(xs) == 0 else np.array(xs).std()

def Baselines_DummyVecEnv(env_id,num_env,array_obs=True,scalar_obs=False):
    envs = []
    for i in range(num_env):
        env = cliffworld_env.Environment(array_obs=array_obs,scalar_obs=scalar_obs)
        env = TimeLimit(env, max_episode_steps=100)
        env = PreprocessWrapper(env)
        envs.append(env)
    batch_env = BatchEnvWrapper(envs)
    return batch_env