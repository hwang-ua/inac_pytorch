import numpy as np
from collections import namedtuple
from core.agent import base
from core.utils import torch_utils
from core.utils import helpers

import os
import torch

"""
https://arxiv.org/pdf/2106.08909.pdf
"""
class OneStep(base.ActorCritic):
    def __init__(self, cfg):
        super(OneStep, self).__init__(cfg)
        self.offline_param_init()
        self.beta_net = cfg.policy_fn()
        self.beta_optimizer = cfg.policy_optimizer_fn(list(self.beta_net.parameters()))
        self.alpha = cfg.alpha

    def get_data(self):
        return self.get_offline_data()

    def feed_data(self):
        self.update_stats(0, None)
        return

    def compute_loss_beta(self, data):
        states, actions = data['obs'], data['act']
        beh_log_probs = self.beta_net.get_logprob(states, actions)
        beh_loss = -beh_log_probs.mean()
        return beh_loss, beh_log_probs

    def compute_loss_q(self, data):
        states, actions, rewards, next_states, dones, next_actions = data['obs'], data['act'], data['reward'], data['obs2'], data['done'], data['act2']
        with torch.no_grad():
            min_Q, _, _ = self.get_q_value_target(next_states, next_actions)
            q_target = rewards + (1 - dones) * self.cfg.discount * min_Q

        _, q1, q2 = self.get_q_value(states, actions, with_grad=True)
        
        critic1_loss = (0.5* (q_target - q1) ** 2).mean()
        critic2_loss = (0.5* (q_target - q2) ** 2).mean()
        loss_q = (critic1_loss + critic2_loss) * 0.5
        return loss_q
        
    def compute_loss_pi(self, data):
        states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
        act, pi_log_prob = self.ac.pi(states) # self.n_sample
        qval, _, _ = self.get_q_value(states, act, with_grad=False)
        
        beta_log_prob = self.beta_net.get_logprob(states, act)
        kl = (pi_log_prob - beta_log_prob).mean(dim=0)
        loss = (- qval + self.alpha * kl).mean()
        return loss
    
    def update(self, data):
        timeout = data['timeout']
        not_to = np.where(timeout==False, True, False)
        
        in_ = data['obs'][not_to]
        act = data['act'][not_to]
        ns = data['obs2'][not_to]
        t = data['done'][not_to]
        r = data['reward'][not_to]
        nact = data['act2'][not_to]
        data = {
            'obs': in_,
            'act': act,
            'obs2': ns,
            'done': t,
            'reward': r,
            'act2': nact
        }
        
        loss_beta, _ = self.compute_loss_beta(data)
        self.beta_optimizer.zero_grad()
        loss_beta.backward()
        # clip_grad_norm_(self.beh_pi.parameters(), self.clip_grad_param)
        self.beta_optimizer.step()
        
        loss_q = self.compute_loss_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()
    
        loss_pi = self.compute_loss_pi(data)
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.sync_target()

        return loss_pi.item(), loss_q.item()
