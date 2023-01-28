import pickle
import time
import copy
import numpy as np

import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
import gym
import d4rl
import gzip

EARLYCUTOFF = "EarlyCutOff"

def load_testset(paths, run=0, env_name=None, max_length=1000000):
    if paths is not None:
        testsets = {}
        for name in paths:
            if name == "env":
                env = gym.make(paths['env'])
                try:
                    data = env.get_dataset()
                except:
                    env = env.unwrapped
                    data = env.get_dataset()

                if 'next_observations' not in data.keys():
                    obs = data['observations']
                    data['observations'] = obs[:-1]
                    data['actions'] = data['actions'][:-1]
                    data['rewards'] = data['rewards'][:-1]
                    data['terminals'] = data['terminals'][:-1]
                    data['next_observations'] = obs[1:]
                    # data['timeouts'] = data['timeouts'][:-1]
                testsets[name] = {
                    'states': data['observations'],
                    'actions': data['actions'],
                    'rewards': data['rewards'],
                    'next_states': data['next_observations'],
                    'terminations': data['terminals'],
                    # 'timeouts': data['timeouts'],
                }
            else:
                pth = paths[name]
                with open(pth.format(run), 'rb') as f:
                    testsets[name] = pickle.load(f)
        return testsets
    else:
        return {}

def run_steps(agent, max_steps, log_interval):
    # valuesets = load_true_values(agent.cfg)
    t0 = time.time()
    transitions = []
    agent.populate_returns(initialize=True)
    # agent.random_fill_buffer(agent.cfg.warm_up_step)
    while True:
        if log_interval and not agent.total_steps % log_interval:
            agent.log_file(elapsed_time=log_interval / (time.time() - t0), test=True)
            t0 = time.time()
        if max_steps and agent.total_steps >= max_steps:
            break
        agent.step()
    agent.save()