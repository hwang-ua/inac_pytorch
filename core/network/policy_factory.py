import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical

from core.network import network_utils, network_bodies
from core.utils import torch_utils
import core.network.representation as representation


class MLPCont(nn.Module):
    def __init__(self, device, obs_dim, act_dim, hidden_sizes, action_range=1.0, rep=None, init_type='xavier', info=None,
                 min_log_std=-6, max_log_std=0):
        super().__init__()
        self.device = device
        if rep is None:
            self.rep = lambda x: x
        else:
            self.rep = rep()
            obs_dim = self.rep.output_dim
        body = network_bodies.FCBody(device, obs_dim, hidden_units=tuple(hidden_sizes), init_type=init_type)
        body_out = obs_dim if hidden_sizes==[] else hidden_sizes[-1]
        self.body = body
        if init_type == "xavier":
            self.mu_layer = network_utils.layer_init_xavier(nn.Linear(body_out, act_dim))
        elif init_type == "uniform":
            self.mu_layer = network_utils.layer_init_uniform(nn.Linear(body_out, act_dim))
        elif init_type == "zeros":
            self.mu_layer = network_utils.layer_init_zero(nn.Linear(body_out, act_dim))
        elif init_type == "constant":
            self.mu_layer = network_utils.layer_init_constant(nn.Linear(body_out, act_dim), const=info)
        else:
            raise ValueError('init_type is not defined: {}'.format(init_type))

        self.log_std_logits = nn.Parameter(torch.zeros(act_dim, requires_grad=True))
        if info is not None:
            self.min_log_std = info.get("min_log_std", -6) #-5
            self.max_log_std = info.get("max_log_std", 0) #2
        else:
            self.min_log_std = -6
            self.max_log_std = 0
        # self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.action_range = action_range


    # def forward(self, obs, deterministic=False, nsample=1):
    def forward(self, obs, deterministic=False):
        """
        https://github.com/hari-sikchi/AWAC/blob/3ad931ec73101798ffe82c62b19313a8607e4f1e/core.py#L91
        """
        if not isinstance(obs, torch.Tensor): obs = torch_utils.tensor(obs, self.device)
        # print("Using the special policy")
        recover_size = False
        if len(obs.size()) == 1:
            recover_size = True
            obs = obs.reshape((1, -1))
        obs = self.rep(obs)
        net_out = self.body(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.action_range

        log_std = torch.sigmoid(self.log_std_logits)

        log_std = self.min_log_std + log_std * (self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        # print("Std: {}".format(std))

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
            # if nsample > 1:
            #     pi_action = pi_distribution.rsample(sample_shape=(self.n_samples,))
            # else:
            #     pi_action = pi_distribution.rsample()

        # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
        # NOTE: The correction formula is a little bit magic. To get an understanding
        # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
        # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
        # Try deriving it yourself as a (very difficult) exercise. :)
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)

        if recover_size:
            pi_action, logp_pi = pi_action[0], logp_pi[0]
        return pi_action, logp_pi

    def get_logprob(self, obs, actions):
        if not isinstance(obs, torch.Tensor): obs = torch_utils.tensor(obs, self.device)
        if not isinstance(actions, torch.Tensor): actions = torch_utils.tensor(actions, self.device)
        obs = self.rep(obs)
        net_out = self.body(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.action_range
        log_std = torch.sigmoid(self.log_std_logits)
        # log_std = self.log_std_layer(net_out)
        log_std = self.min_log_std + log_std * (
            self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        logp_pi = pi_distribution.log_prob(actions).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - actions - F.softplus(-2*actions))).sum(axis=1)
        return logp_pi


class MLPDiscrete(nn.Module):
    def __init__(self, device, obs_dim, act_dim, hidden_sizes, rep=None, init_type='xavier', info=None):
        super().__init__()
        self.device = device
        if rep is None:
            self.rep = lambda x: x
        else:
            self.rep = rep()
            obs_dim = self.rep.output_dim
        body = network_bodies.FCBody(device, obs_dim, hidden_units=tuple(hidden_sizes), init_type=init_type)
        body_out = obs_dim if hidden_sizes==[] else hidden_sizes[-1]
        self.body = body
        if init_type == "xavier":
            self.mu_layer = network_utils.layer_init_xavier(nn.Linear(body_out, act_dim))
        elif init_type == "uniform":
            self.mu_layer = network_utils.layer_init_uniform(nn.Linear(body_out, act_dim))
        elif init_type == "zeros":
            self.mu_layer = network_utils.layer_init_zero(nn.Linear(body_out, act_dim))
        elif init_type == "constant":
            self.mu_layer = network_utils.layer_init_constant(nn.Linear(body_out, act_dim), const=info)
        else:
            raise ValueError('init_type is not defined: {}'.format(init_type))

        # self.body = network_bodies.FCBody(device, obs_dim, hidden_units=tuple(hidden_sizes))
        # self.mu_layer = nn.Linear(body_out, act_dim)
        
        self.log_std_logits = nn.Parameter(torch.zeros(act_dim, requires_grad=True))
        self.min_log_std = -6
        self.max_log_std = 0
        # self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
    
    def forward(self, obs, deterministic=True):
        if not isinstance(obs, torch.Tensor): obs = torch_utils.tensor(obs, self.device)
        # print("Using the special policy")
        recover_size = False
        if len(obs.size()) == 1:
            recover_size = True
            obs = obs.reshape((1, -1))
        obs = self.rep(obs)
        net_out = self.body(obs)
        probs = self.mu_layer(net_out)
        probs = F.softmax(probs, dim=1)
        m = Categorical(probs)
        action = m.sample()
        logp = m.log_prob(action)
        if recover_size:
            action, logp = action[0], logp[0]
        return action, logp
    
    def get_logprob(self, obs, actions):
        if not isinstance(obs, torch.Tensor): obs = torch_utils.tensor(obs, self.device)
        if not isinstance(actions, torch.Tensor): actions = torch_utils.tensor(actions, self.device)
        obs = self.rep(obs)
        net_out = self.body(obs)
        probs = self.mu_layer(net_out)
        probs = F.softmax(probs, dim=1)
        m = Categorical(probs)
        logp_pi = m.log_prob(actions)
        return logp_pi
    

# class PolicyFactory:
#     @classmethod
#     def get_policy_fn(cls, cfg):
#         # if cfg.policy_fn_config['policy_type'] == "gaussian":
#         #     return lambda: GaussianPolicy(cfg.device, np.prod(cfg.policy_fn_config['in_dim']),
#         #                                   cfg.policy_fn_config['hidden_units'],
#         #                                   cfg.action_dim, action_space=cfg.action_range)
#         if cfg.policy_fn_config['policy_type'] == "policy-cont":
#             if cfg.action_range == 1:
#                 action_space = None
#             else:
#                 action_space = {"low": -cfg.action_range, "high": cfg.action_range}
#             # return lambda: GaussianPolicy(cfg.device, np.prod(cfg.policy_fn_config['in_dim']),
#             #                            cfg.action_dim, cfg.policy_fn_config['hidden_units'],
#             #                            action_space=action_space,
#             #                            rep=cfg.rep_fn,
#             #                            init_type=cfg.policy_fn_config.get('init_type', 'xavier'),
#             #                            )
#             return lambda: AwacMLPCont(cfg.device, np.prod(cfg.policy_fn_config['in_dim']),
#                                        cfg.action_dim, cfg.policy_fn_config['hidden_units'],
#                                        action_range=cfg.action_range,
#                                        rep=cfg.rep_fn,
#                                        init_type=cfg.policy_fn_config.get('init_type', 'xavier'),
#                                        info=cfg.policy_fn_config.get('info', None),
#                                        )
#         elif cfg.policy_fn_config['policy_type'] == 'policy-discrete':
#             return lambda: AwacMLPDiscrete(cfg.device, np.prod(cfg.policy_fn_config['in_dim']),
#                                            cfg.action_dim, cfg.policy_fn_config['hidden_units'],
#                                            rep=cfg.rep_fn,
#                                            init_type=cfg.policy_fn_config.get('init_type', 'xavier'),
#                                            info=cfg.policy_fn_config.get('info', None),
#                                            )
#         else:
#             raise NotImplementedError