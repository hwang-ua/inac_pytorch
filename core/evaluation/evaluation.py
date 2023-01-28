import os
import copy
import torch
import numpy as np
import pickle as pkl
import core.utils.torch_utils as torch_utils


def load_dataset(path, size=1000):
    with open(path, "rb") as f:
        dataset = pkl.load(f)
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    next_states = dataset['next_states']
    terminations = dataset['terminations']
    
    idx_rng = np.random.RandomState(0)
    idxs = np.arange(0, len(states))
    idx_rng.shuffle(idxs)
    taken = idxs[: size]
    states, actions, rewards, next_states, terminations = states[taken], actions[taken], rewards[taken], next_states[taken], terminations[taken]
    return states, actions, rewards, next_states, terminations

def load_value_functions(cfg, is_ac):
    def load_rep_fn(parameters_dir):
        path = os.path.join(cfg.data_root, parameters_dir)
        cfg.rep_net.load_state_dict(torch.load(path, map_location=cfg.device))
    
    def load_val_fn(parameters_dir):
        path = os.path.join(cfg.data_root, parameters_dir)
        cfg.val_net.load_state_dict(torch.load(path, map_location=cfg.device))
        
    def load_actor_fn(parameters_dir):
        path = os.path.join(cfg.data_root, parameters_dir)
        cfg.policy_net.load_state_dict(torch.load(path, map_location=cfg.device))

    def load_critic_fn(parameters_dir):
        path = os.path.join(cfg.data_root, parameters_dir)
        cfg.critic_net.load_state_dict(torch.load(path, map_location=cfg.device))

    if is_ac:
        if 'load_params' in cfg.policy_fn_config and cfg.policy_fn_config['load_params']:
            load_actor_fn(cfg.policy_fn_config['path'])
        if 'load_params' in cfg.critic_fn_config and cfg.critic_fn_config['load_params']:
            load_critic_fn(cfg.critic_fn_config['path'])
    else:
        if 'load_params' in cfg.rep_fn_config and cfg.rep_fn_config['load_params']:
            load_rep_fn(cfg.rep_fn_config['path'])
        if 'load_params' in cfg.val_fn_config and cfg.val_fn_config['load_params']:
            load_val_fn(cfg.val_fn_config['path'])
    return cfg
    
def move_val2rep(cfg):
    original = copy.deepcopy(cfg.val_net)
    cfg.rep_net = original.body
    cfg.val_net = original.head_activation(original.fc_head)
    return cfg

def move_policy2rep(cfg):
    cfg.rep_net = cfg.policy_net.body
    cfg.val_net = cfg.critic_net
    return cfg

def dynamics_awareness(cfg, states, similar_s):
    different_idx = list(range(len(states)))
    cfg.test_rng.shuffle(different_idx)
    states = torch_utils.tensor(cfg.state_normalizer(states), cfg.device)
    similar_s = torch_utils.tensor(cfg.state_normalizer(similar_s), cfg.device)
    with torch.no_grad():
        base_rep = torch_utils.to_np(cfg.rep_net(states))
        similar_rep = torch_utils.to_np(cfg.rep_net(similar_s))
    
    # prop = dist_difference(base_rep, similar_rep, different_idx)
    similar_dist = np.linalg.norm(similar_rep - base_rep, axis=1).mean()
    diff_rep1 = base_rep
    diff_rep2 = base_rep[different_idx]
    diff_dist = np.linalg.norm(diff_rep1 - diff_rep2, axis=1).mean()
    if diff_dist == 0:
        prop = 0
    else:
        prop = (diff_dist - similar_dist) / diff_dist
        if np.isinf(prop) or np.isnan(prop) or prop < 0:
            prop = 0

    fp = os.path.join(cfg.get_warmstart_property_dir(), "dynamics_awareness.npy")
    np.save(fp, prop)
        
def orthogonality(cfg, states):
    states = torch_utils.tensor(cfg.state_normalizer(states), cfg.device)
    with torch.no_grad():
        reps = cfg.rep_net(states)
        reps = torch_utils.to_np(reps.detach())

    random_idx = list(range(len(states)))
    cfg.test_rng.shuffle(random_idx)

    dot_prod = np.multiply(reps, reps[random_idx]).sum(axis=1)
    norm = np.linalg.norm(reps, axis=1).reshape((-1, 1))
    norm_prod = np.multiply(norm, norm[random_idx]).sum(axis=1)
    if len(np.where(norm_prod==0)[0]) != 0:
        norm_prod[np.where(norm_prod==0)] += 1e-05
    normalized = np.abs(np.divide(dot_prod, norm_prod))
    rho = 1 - normalized.mean()

    fp = os.path.join(cfg.get_warmstart_property_dir(), "orthogonality.npy")
    np.save(fp, rho)


def diversity(cfg, states):
    states = torch_utils.tensor(cfg.state_normalizer(states), cfg.device)
    with torch.no_grad():
        phi_s = cfg.rep_net(states)
        values = cfg.val_net(phi_s)

    random_idx = list(range(len(states)))
    cfg.test_rng.shuffle(random_idx)
    
    phi1 = torch_utils.to_np(phi_s)
    phi2 = phi1[random_idx]
    val1 = torch_utils.to_np(values)
    val2 = val1[random_idx]
    
    diff_phi = np.linalg.norm(phi1 - phi2, axis=1)
    diff_val = np.abs(val1.max(axis=1) - val2.max(axis=1))
    
    max_dphi = diff_phi.max()
    max_dv = diff_val.max()
    if max_dv != 0:
        normalized_dv = diff_val / max_dv
    else:
        normalized_dv = diff_val
    if max_dphi != 0:
        normalized_dphi = diff_phi / max_dphi
    else:
        normalized_dphi = diff_phi

    # Removing the indexes with zero value of representation difference
    nonzero_idx = normalized_dphi != 0
    normalized_dv = normalized_dv[nonzero_idx]
    normalized_dphi = normalized_dphi[nonzero_idx]

    divers = 1 - np.clip(normalized_dv / normalized_dphi, 0, 1).mean()  # 1 - specialization

    fp = os.path.join(cfg.get_warmstart_property_dir(), "diversity.npy")
    np.save(fp, divers)


def lipschitz(cfg, states, next_states):
    states = torch_utils.tensor(cfg.state_normalizer(states), cfg.device)
    next_states = torch_utils.tensor(cfg.state_normalizer(next_states), cfg.device)
    with torch.no_grad():
        phi_s = cfg.rep_net(states)
        value_s = cfg.val_net(phi_s)
        phi_ns = cfg.rep_net(next_states)
        value_ns = cfg.val_net(phi_ns)
    states = torch_utils.to_np(states)
    next_states = torch_utils.to_np(next_states)
    value_s = torch_utils.to_np(value_s)
    value_ns = torch_utils.to_np(value_ns)
    diff_s = np.linalg.norm(next_states - states, axis=1)
    # diff_phi = np.linalg.norm(phi_ns - phi_s, axis=1)
    diff_val = np.abs(value_ns.max(axis=1) - value_s.max(axis=1))
    lip = np.divide(diff_val, diff_s)
    conclusion = np.array([lip.max(), lip.mean(), lip.min()])
    # print(conclusion)
    
    fp = os.path.join(cfg.get_warmstart_property_dir(), "lipschitz.npy")
    np.save(fp, conclusion)

def action_diff(cfg, states):
    states = torch_utils.tensor(cfg.state_normalizer(states), cfg.device)
    with torch.no_grad():
        phi_s = cfg.rep_net(states)
        value_s = cfg.val_net(phi_s)
    value_s = torch_utils.to_np(value_s)
    diff_estimation = value_s.max(axis=1) - value_s.min(axis=1)
    # conclusion = np.array([diff_estimation.mean()])
    fp = os.path.join(cfg.get_warmstart_property_dir(), "action_diff.npy")
    np.save(fp, diff_estimation.mean())

def mdp_rank(cfg, states):
    """ From https://arxiv.org/pdf/2207.02099.pdf Appendix A.11"""
    def compute_rank_from_features(feature_matrix, rank_delta=0.01):
        sing_values = np.linalg.svd(feature_matrix, compute_uv=False)
        cumsum = np.cumsum(sing_values)
        nuclear_norm = np.sum(sing_values)
        approximate_rank_threshold = 1.0 - rank_delta
        threshold_crossed = (
            cumsum >= approximate_rank_threshold * nuclear_norm)
        effective_rank = sing_values.shape[0] - np.sum(threshold_crossed) + 1
        return effective_rank
    
    states = torch_utils.tensor(cfg.state_normalizer(states), cfg.device)
    with torch.no_grad():
        phi_s = torch_utils.to_np(cfg.rep_net(states))
    erank = compute_rank_from_features(phi_s)
    # print(erank)
    # print(np.linalg.matrix_rank(phi_s))
    fp = os.path.join(cfg.get_warmstart_property_dir(), "mdp_rank_optTraj.npy")
    np.save(fp, erank)
