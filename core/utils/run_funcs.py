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

def load_true_values(cfg):
    if cfg.true_value_paths is not None:
        valuesets = {}
        for name in cfg.true_value_paths:
            pth = cfg.true_value_paths[name]
            with open(pth, 'rb') as f:
                valuesets[name] = pickle.load(f)
        return valuesets
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


def value_iteration(env, gamma):
    max_iter = 10000
    done = False
    iter_count = 0
    eps = 0
    p_matrix, r_matrix, goal_idx, all_states = env.transition_reward_model()
    
    num_states = len(p_matrix)
    num_actions = len(env.actions)
    v_matrix = np.zeros(num_states)

    while not done and iter_count < max_iter:
        v_new = np.zeros(num_states)
        for i in range(num_states):
            for a in range(num_actions):
                cur_val = 0
                for j in np.nonzero(p_matrix[i][a])[0]:
                    cur_val += p_matrix[i][a][j] * v_matrix[j]
                if i == goal_idx:
                    cur_val *= 0.
                else:
                    cur_val *= gamma
                cur_val += r_matrix[i][a]
                v_new[i] = max(v_new[i], cur_val)
        max_diff = 0
        for i in range(num_states):
            max_diff = max(max_diff, abs(v_matrix[i] - v_new[i]))
        
        v_matrix = v_new
        
        iter_count += 1
        if (max_diff <= eps):
            print("state value converged at {}th iteration".format(iter_count))
            done = True
    
    
    q_matrix = np.zeros((num_states, num_actions))
    for i in range(num_states):
        for a in range(num_actions):
            temp = 0
            for j in np.nonzero(p_matrix[i][a])[0]:
                temp += p_matrix[i][a][j] * (r_matrix[i][a] + gamma * v_matrix[j])
            q_matrix[i, a] = temp
            
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, num_actions, figsize=(12, 3))
    # for a in range(num_actions):
    #     check = np.zeros((15, 15))
    #     for idx, s in enumerate(all_states):
    #         x, y = s
    #         check[x, y] = q_matrix[idx, a]
    #     im = axs[a].imshow(check, cmap="Blues", vmin=0.5, vmax=1, interpolation='none')
    # plt.colorbar(im)
    # plt.show()
    
    return q_matrix, all_states