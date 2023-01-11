import pickle
import time
import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
import gym
import d4rl

EARLYCUTOFF = "EarlyCutOff"

def load_testset(env_name, dataset, id):
    if env_name == 'HalfCheetah':
        if dataset == 'expert':
            path = {"env": "halfcheetah-expert-v2"}
        elif dataset == 'medexp':
            path = {"env": "halfcheetah-medium-expert-v2"}
        elif dataset == 'medium':
            path = {"env": "halfcheetah-medium-v2"}
        elif dataset == 'medrep':
            path = {"env": "halfcheetah-medium-replay-v2"}
    elif env_name == 'Walker2d':
        if dataset == 'expert':
            path = {"env": "walker2d-expert-v2"}
        elif dataset == 'medexp':
            path = {"env": "walker2d-medium-expert-v2"}
        elif dataset == 'medium':
            path = {"env": "walker2d-medium-v2"}
        elif dataset == 'medrep':
            path = {"env": "walker2d-medium-replay-v2"}
    elif env_name == 'Hopper':
        if dataset == 'expert':
            path = {"env": "hopper-expert-v2"}
        elif dataset == 'medexp':
            path = {"env": "hopper-medium-expert-v2"}
        elif dataset == 'medium':
            path = {"env": "hopper-medium-v2"}
        elif dataset == 'medrep':
            path = {"env": "hopper-medium-replay-v2"}
    elif env_name == 'Ant':
        if dataset == 'expert':
            path = {"env": "ant-expert-v2"}
        elif dataset == 'medexp':
            path = {"env": "ant-medium-expert-v2"}
        elif dataset == 'medium':
            path = {"env": "ant-medium-v2"}
        elif dataset == 'medrep':
            path = {"env": "ant-medium-replay-v2"}

    # elif env_name == 'Acrobot':
    #     if dataset == 'expert':
    #         path = {"pkl": "data/dataset/acrobot/transitions_50k/train_40k/{}_run.pkl".format(id)}
    #     elif dataset == 'mixed':
    #         path = {"pkl": "data/dataset/acrobot/transitions_50k/train_mixed/{}_run.pkl".format(id)}
    # elif env_name == 'LunarLander':
    #     if dataset == 'expert':
    #         path = {"pkl": "data/dataset/lunar_lander/transitions_50k/train_500k/{}_run.pkl".format(id)}
    #     elif dataset == 'mixed':
    #         path = {"pkl": "data/dataset/lunar_lander/transitions_50k/train_mixed/{}_run.pkl".format(id)}
    # elif env_name == 'MountainCar':
    #     if dataset == 'expert':
    #         path = {"pkl": "data/dataset/mountain_car/transitions_50k/train_60k/{}_run.pkl".format(id)}
    #     elif dataset == 'mixed':
    #         path = {"pkl": "data/dataset/mountain_car/transitions_50k/train_mixed/{}_run.pkl".format(id)}
    
    assert path is not None
    testsets = {}
    for name in path:
        if name == "env":
            env = gym.make(path['env'])
            try:
                data = env.get_dataset()
            except:
                env = env.unwrapped
                data = env.get_dataset()
            testsets[name] = {
                'states': data['observations'],
                'actions': data['actions'],
                'rewards': data['rewards'],
                'next_states': data['next_observations'],
                'terminations': data['terminals'],
            }
        else:
            pth = path[name]
            with open(pth.format(id), 'rb') as f:
                testsets[name] = pickle.load(f)
        
        return testsets
    else:
        return {}

def run_steps(agent,
              warm_up_step=0,
              log_interval=10000,
              eval_interval=10000,
              max_steps=1000000
              ):
    t0 = time.time()
    agent.populate_returns(initialize=True)
    agent.random_fill_buffer(warm_up_step)
    while True:
        if log_interval and not agent.total_steps % log_interval:
            agent.log_file(elapsed_time=log_interval / (time.time() - t0), test=(not agent.total_steps % eval_interval))
            t0 = time.time()
        if max_steps and agent.total_steps >= max_steps:
            break
        seq, loss_dict = agent.step()
    agent.save()