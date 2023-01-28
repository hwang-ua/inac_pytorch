import numpy as np
import matplotlib.pyplot as plt

from core.utils import helpers


def visualize_dataset(dataset, env, slice=None, save=None):
    if not slice:
        states = dataset['states']
        actions = dataset['actions']
    else:
        states = dataset['states'][slice[0]: slice[1]]
        actions = dataset['actions'][slice[0]: slice[1]]
    
    action_names = ['0', '1', '2', '3']
    
    fig, axs = plt.subplots(1, 5, figsize=(12, 3))
    for action in range(4):
        count = np.zeros((env.num_rows, env.num_cols))
        for s, a in zip(states, actions):
            if a == action:
                count[s[0], s[1]] += 1
        axs[action].set_title(action_names[action])
        im = axs[action].imshow(count, cmap="Blues")
        plt.colorbar(im, ax=axs[action], shrink=0.7)
        print("\n------------- Action {} ----------".format(action_names[action]))
        print(count)
    
    states = dataset['states']
    count = np.zeros((env.num_rows, env.num_cols))
    for s in states:
        count[s[0], s[1]] += 1
    print("============= Total =============")
    print(count)
    axs[-1].set_title("Total")
    im = axs[-1].imshow(count, cmap="Blues")
    plt.colorbar(im, ax=axs[-1], shrink=0.7)
    
    fig.tight_layout()
    if save:
        plt.savefig(save+"/dataset_vis.png", dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()
    else:
        plt.show()
        
def visualize_policy(q_table_lst, pi_lst, lable_lst, env, policy_name):
    import matplotlib.cm as cm
    fig, axs = plt.subplots(1, len(lable_lst), figsize=(2 * len(lable_lst), 2.5))
    axs = [axs] if len(lable_lst) == 1 else axs
    # action_list = ['A', '>', 'V', '<']
    action_list = [r"$\uparrow$", r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"]
    
    for idx, (q_table, pi, label) in enumerate(zip(q_table_lst, pi_lst, lable_lst)):
        # value = q_table.max(axis=1).reshape((env.num_rows, env.num_cols))
        # policy = q_table.argmax(axis=1).reshape((env.num_rows, env.num_cols))
        value = q_table[np.arange(len(q_table)), pi].reshape((env.num_rows, env.num_cols))
        policy = pi.reshape((env.num_rows, env.num_cols))
        
        img = axs[idx].imshow(value, cmap="Blues", vmin=0, vmax=12)#vmin=value.min(), vmax=value.max())
        for x in range(env.num_rows):
            for y in range(env.num_cols):
                if env._matrix_mdp[x, y] != -1:
                    axs[idx].text(y - 0.6, x + 0.4, action_list[int(policy[x, y])], color="black")
        axs[idx].set_title(label)
        walls = np.ma.masked_where(env._matrix_mdp != -1, env._matrix_mdp)
        # walls[env._goal_x, env._goal_y] = -1
        axs[idx].imshow(walls, cmap=cm.gray)
        axs[idx].axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig("plot/img/exp_tabular_{}.pdf".format(policy_name), dpi=300, bbox_inches='tight')


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

def rollout_limited(env, q_table, size, timeout, policy_fn, legal_starts):
    def limited_start():
        pos = env.reset()
        while len(np.where((legal_starts == tuple(pos)).all(axis=-1))[0]) == 0:
            pos = env.reset()
        return pos
    states = []
    actions = []
    rewards = []
    next_states = []
    next_actions = []
    termins = []
    pos = limited_start()
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
            pos = limited_start()
    if size > 0:
        next_actions.append(a)
    return {'states': np.array(states),
            'actions': np.array(actions),
            'next_states': np.array(next_states),
            'rewards': np.array(rewards),
            'terminations': np.array(termins),
            'next_actions': np.array(next_actions),
            }


def optimal_data_collection(env, q_table, size, timeout):
    optimal_policy = lambda q: np.random.choice(np.flatnonzero(q == q.max()))
    return rollout(env, q_table, size, timeout, optimal_policy)


def random_data_collection(env, q_table, size, timeout):
    random_policy = lambda q: np.random.choice(len(q))
    env._random_start = True
    return rollout(env, q_table, size, timeout, random_policy)

def adversarial_data_collection(env, q_table, size, timeout, advsr_perc=1):
    random_size = int(size * (1-advsr_perc))
    advsr_size = size - random_size
    random_policy = lambda q: np.random.choice(len(q))
    env._random_start = True
    random_data = rollout(env, q_table, random_size, timeout, random_policy)
    
    # def advsr_policy(q):
    #     choices = np.flatnonzero(q < q.max())
    #     if len(choices) > 0:
    #         return np.random.choice(choices)
    #     else:
    #         return np.random.choice(np.flatnonzero(q == q.max()))
    advsr_policy = lambda q: np.random.choice(np.flatnonzero(q == q.min()))
    env._random_start = True
    left_lower_x = np.arange(7, 12)
    left_lower_y = np.arange(0, 6)
    left_lower = np.array([[x, y] for x in left_lower_x for y in left_lower_y])
    advsr_data = rollout_limited(env, q_table, advsr_size, 1, advsr_policy, left_lower)

    if advsr_size > 0:
        return {'states': np.concatenate([random_data['states'], advsr_data['states']], axis=0),
                'actions': np.concatenate([random_data['actions'], advsr_data['actions']], axis=0),
                'next_states': np.concatenate([random_data['next_states'], advsr_data['next_states']], axis=0),
                'rewards': np.concatenate([random_data['rewards'], advsr_data['rewards']], axis=0),
                'terminations': np.concatenate([random_data['terminations'], advsr_data['terminations']], axis=0),
                'next_actions': np.concatenate([random_data['next_actions'], advsr_data['next_actions']], axis=0),
                }
    else:
        return random_data


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


def remove_selfloop(dataset):
    states = dataset['states']
    next_states = dataset['next_states']
    self_loop = np.equal(states, next_states).sum(axis=-1)
    remove_s = np.where(self_loop == states.shape[1])
    mask = np.ones(len(states), dtype=bool)
    mask[remove_s] = False
    newset = {
        'states': dataset['states'][mask],
        'actions': dataset['actions'][mask],
        'next_states': dataset['next_states'][mask],
        'rewards': dataset['rewards'][mask],
        'terminations': dataset['terminations'][mask],
        'next_actions': dataset['next_actions'][mask]
    }
    return newset


def remove_state(dataset, removed_state):
    states = dataset['states']
    remove_s = []
    for rs in removed_state:
        one_case = np.where((states == tuple(rs)).all(axis=-1))[0]
        remove_s += list(one_case)
    mask = np.ones(len(states), dtype=bool)
    mask[remove_s] = False
    newset = {
        'states': states[mask],
        'actions': dataset['actions'][mask],
        'next_states': dataset['next_states'][mask],
        'rewards': dataset['rewards'][mask],
        'terminations': dataset['terminations'][mask],
        'next_actions': dataset['next_actions'][mask]
    }
    return newset


def behavior_policy(data, env):
    # with open(data_pth, 'rb') as f:
    #   data = pickle.load(f)
    train_s = data['states']
    train_a = data['actions']
    train_sp = data['next_states']
    train_r = data['rewards']
    train_ap = data['next_actions']
    
    beta_as = np.zeros((env.num_states, env.num_actions))
    beta_next = np.zeros((env.num_states, env.num_actions))
    P_mat = np.zeros((env.num_states, env.num_actions, env.num_states))
    R_mat = np.zeros((env.num_states, env.num_actions))
    for x in range(env.num_cols):
        for y in range(env.num_rows):
            temp_s = env.pos_to_state(x, y)
            same_s = helpers.search_same_row(train_s, np.array([x, y]))[0]
            if len(same_s) == 0:
                continue
            for a in range(env.num_actions):
                same_a = np.where(train_a == a)[0]
                same_sa = np.intersect1d(same_s, same_a)
                p = float(len(same_sa)) / float(len(same_s))
                beta_as[temp_s, a] = p
                if len(same_sa) == 0:
                    continue
                xp, yp = train_sp[same_sa[0]]
                r = train_r[same_sa[0]]
                temp_sp = env.pos_to_state(xp, yp)
                P_mat[temp_s, a, temp_sp] = 1
                R_mat[temp_s, a] = r
                
                ap = train_ap[same_sa[0]]
                beta_next[temp_sp, ap] += 1
    bn_mask = np.where(beta_next.sum(axis=-1) > 0)
    beta_next[bn_mask] = beta_next[bn_mask] / beta_next[bn_mask].sum(axis=-1, keepdims=True)
    return beta_as, P_mat, R_mat, beta_next
