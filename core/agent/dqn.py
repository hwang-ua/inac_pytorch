import os
# from collections import namedtuple
# import numpy as np
# import matplotlib.pyplot as plt
import torch

from core.agent import base
from core.utils import torch_utils


class DQNAgent(base.ValueBased):
    def __init__(self, cfg):
        super().__init__(cfg)

        if cfg.agent_name == 'DQNAgent' and cfg.load_offline_data:
            self.fill_offline_data_to_buffer()
            
    def update(self, data):
        states, actions, rewards, next_states, terminals = data['obs'], data['act'], data['reward'], data['obs2'], data['done']

        # actions = torch_utils.tensor(actions, self.cfg.device).long()
        if not self.cfg.rep_fn_config['train_params']:
            with torch.no_grad():
                phi = self.rep_net(states)
        else:
            phi = self.rep_net(states)

        if not self.cfg.val_fn_config['train_params']:
            with torch.no_grad():
                q = self.val_net(phi)[self.batch_indices, actions]
        else:
            q = self.val_net(phi)[self.batch_indices, actions]

        # Constructing the target
        with torch.no_grad():
            q_next = self.targets.val_net(self.targets.rep_net(next_states))
            q_next = q_next.max(1)[0]
            terminals = torch_utils.tensor(terminals, self.cfg.device)
            rewards = torch_utils.tensor(rewards, self.cfg.device)
            target = self.cfg.discount * q_next * (1 - terminals).float()
            target.add_(rewards.float())
        
        loss = self.vf_loss(q, target)
        constr = self.constr_fn(phi, q, target)
        
        loss += constr

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.sync_target()
            # self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
            # self.targets.val_net.load_state_dict(self.val_net.state_dict())

    def load_rep_fn(self, parameters_dir):
        path = os.path.join(self.cfg.data_root, parameters_dir)
        if self.cfg.rep_fn_config.get('load_body_only', False):
            # self.rep_net.net.body.load_state_dict(torch.load(path, map_location=self.cfg.device).net.body)
            pretrained_dict = torch.load(path, map_location=self.cfg.device)
            model_dict = self.rep_net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'q1_net.body.' in k}
            name_correction = {}
            ptype = ['weight','bias']
            for count in range(len(model_dict)):
                lidx = count // 2
                ptkey = "q1_net.body.layers.{}.{}".format(lidx, ptype[count % 2])
                if count >= len(model_dict) - 2:
                    mkey = "net.fc_head.{}".format(ptype[count % 2])
                else:
                    mkey = "net.body.layers.{}.{}".format(lidx, ptype[count % 2])
                name_correction[mkey] = pretrained_dict[ptkey]
            self.rep_net.load_state_dict(name_correction)
        else:
            self.rep_net.load_state_dict(torch.load(path, map_location=self.cfg.device))
        self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
        self.cfg.logger.info("Load rep function from {}".format(path))

    def load_val_fn(self, parameters_dir):
        path = os.path.join(self.cfg.data_root, parameters_dir)
        if self.cfg.val_fn_config.get('load_head_only', False):
            # self.val_net.net.fc_head.load_state_dict(torch.load(path, map_location=self.cfg.device).net.fc_head)
            pretrained_dict = torch.load(path, map_location=self.cfg.device)
            name_correction = {
                'fc_head.weight': pretrained_dict['q1_net.fc_head.weight'],
                'fc_head.bias': pretrained_dict['q1_net.fc_head.bias'],
            }
            self.val_net.load_state_dict(name_correction)
        else:
            self.val_net.load_state_dict(torch.load(path, map_location=self.cfg.device))
        self.targets.val_net.load_state_dict(self.val_net.state_dict())
        self.cfg.logger.info("Load value function from {}".format(path))


# """
# NOT Used
# """
# class SlowPolicyDQN(DQNAgent):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#         self.slow_pi_prob = cfg.slow_policy_prob
#
#     def no_grad_target_policy(self, state):
#         with torch.no_grad():
#             phi = self.targets.rep_net(self.cfg.state_normalizer(state))
#             q_target = torch_utils.to_np(self.targets.val_net(phi)).flatten()
#         target_action = self.agent_rng.choice(np.flatnonzero(q_target == q_target.max()))
#         return target_action
#
#     def slow_policy(self, state):
#         q_values = self.no_grad_value(state)
#         action = self.agent_rng.choice(np.flatnonzero(q_values == q_values.max()))
#         target_action = self.no_grad_target_policy(state)
#         if self.agent_rng.rand() <= self.slow_pi_prob:
#             action = target_action
#         return action
#
#     def policy(self, state, eps):
#         if self.agent_rng.rand() < eps:
#             action = self.agent_rng.randint(0, self.cfg.action_dim)
#         else:
#             action = self.slow_policy(state)
#         return action
#
#     def eval_step(self, state):
#         action = self.slow_policy(state)
#         return action