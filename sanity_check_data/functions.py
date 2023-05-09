import numpy as np
from environment import gridworld

def rollout(env, q_table, size, timeout, policy_fn):
    states = []
    actions = []
    rewards = []
    next_states = []
    next_actions = []
    termins = []
    pos = env.reset()
    for count in range(size):
        a = policy_fn(q_table[env.pos_to_state(pos[0], pos[1])])
        if len(next_states) >= 1:
            next_actions.append(a)
        states.append(pos)
        actions.append(a)
        
        pos, reward, termin, _ = env.step(a)
        rewards.append(reward)
        next_states.append(pos)
        termins.append(termin)
        if termin or (len(states) % timeout == 0):
            pos = env.reset()
    if size > 0:
        next_actions.append(a)
    return {'states': np.array(states),
            'actions': np.array(actions),
            'next_states': np.array(next_states),
            'rewards': np.array(rewards),
            'terminations': np.array(termins),
            'next_actions': np.array(next_actions),
            }

def value_iteration(P, r, discount, num_iterations, init_constant=0):
    num_states, num_actions, _ = P.shape
    q = np.zeros((num_states, num_actions)) + init_constant
    for _ in range(num_iterations):
        q = r + discount * (P @ q.max(axis=-1, keepdims=True)).squeeze(-1)
    return q

def optimal_data_collection(env, q_table, size, timeout):
    optimal_policy = lambda q: np.random.choice(np.flatnonzero(q == q.max()))
    return rollout(env, q_table, size, timeout, optimal_policy)


def random_data_collection(env, q_table, size, timeout):
    random_policy = lambda q: np.random.choice(len(q))
    env._random_start = True
    return rollout(env, q_table, size, timeout, random_policy)

def mixed_data_collection(env, q_table, size, timeout, p_opt=0.01):
    optimal_policy = lambda q: np.random.choice(np.flatnonzero(q == q.max()))
    random_policy = lambda q: np.random.choice(len(q))
    opt_data = rollout(env, q_table, int(size * p_opt), timeout, optimal_policy)
    env._random_start = True
    random_data = rollout(env, q_table, int(size * (1 - p_opt)), timeout, random_policy)
    return {'states': np.concatenate([opt_data['states'], random_data['states']], axis=0),
            'actions': np.concatenate([opt_data['actions'], random_data['actions']], axis=0),
            'next_states': np.concatenate([opt_data['next_states'], random_data['next_states']], axis=0),
            'rewards': np.concatenate([opt_data['rewards'], random_data['rewards']], axis=0),
            'terminations': np.concatenate([opt_data['terminations'], random_data['terminations']], axis=0),
            'next_actions': np.concatenate([opt_data['next_actions'], random_data['next_actions']], axis=0),
            }


def remove_action(dataset, removed_state, removed_action):
    states = dataset['states']
    actions = dataset['actions']
    remove_x = np.where(np.logical_and(states[:, 0] >= removed_state[0][0], states[:, 0] <= removed_state[0][1]))
    remove_y = np.where(np.logical_and(states[:, 1] >= removed_state[1][0], states[:, 1] <= removed_state[1][1]))
    remove_a = np.where(actions == removed_action)
    remove_s = np.intersect1d(remove_x, remove_y)
    remove_sa = np.intersect1d(remove_s, remove_a)
    mask = np.ones(len(states), dtype=bool)
    mask[remove_sa] = False
    newset = {
        'states': states[mask],
        'actions': actions[mask],
        'next_states': dataset['next_states'][mask],
        'rewards': dataset['rewards'][mask],
        'terminations': dataset['terminations'][mask],
        'next_actions': dataset['next_actions'][mask]
    }
    return newset

def exp():
    env_fn = lambda: gridworld.GridWorld(random_start=False)
    gw = env_fn()
    discount = 0.95
    policy = 'opt'
    size = 10000
    timeout = 100
  	
    opt_q = value_iteration(gw.P, gw.r, discount, 10000)
    
    rs_gw = gridworld.GridWorld(random_start=True)
    if policy == 'opt':
        dataset = optimal_data_collection(gw, opt_q, size, timeout)
    elif policy == 'mixed':
        dataset = mixed_data_collection(gw, opt_q, size, timeout, p_opt=0.01)
    elif policy == 'random':
        dataset = random_data_collection(rs_gw, opt_q, size, timeout)
    elif policy == 'missing_a':
        dataset = mixed_data_collection(gw, opt_q, size, timeout)
        dataset = remove_action(dataset, [[0, 6], [0, 6]], 2)
    print(dataset)
exp()
