import numpy as np
from collections import namedtuple
from core.agent import base
from core.utils import torch_utils
from core.utils import helpers

import os
import torch

class InSample(base.ActorCritic):
    def __init__(self, cfg):
        super(InSample, self).__init__(cfg)
        self.offline_param_init()
        
        self.automatic_tmp_tuning = cfg.automatic_tmp_tuning
        if self.automatic_tmp_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.cfg.action_dim).to(self.cfg.device)).item()
            self.log_tau = torch.zeros(1, requires_grad=True, device=self.cfg.device)
            self.auto_tau_optim = torch.optim.Adam([self.log_tau], lr=self.cfg.learning_rate)
            self.tau = self.log_tau.exp().detach()
        else:
            self.tau = cfg.tau
        self.value_net = cfg.state_value_fn()
        self.value_optimizer = cfg.vs_optimizer_fn(list(self.value_net.parameters()))
        self.beh_pi = cfg.policy_fn()
        self.beh_pi_optimizer = cfg.policy_optimizer_fn(list(self.beh_pi.parameters()))

        self.clip_grad_param = cfg.clip_grad_param
        self.exp_threshold = cfg.exp_threshold
        self.beta_threshold = 1e-3

        self.eq18_v_calculation = cfg.eq18_v_calculation # calculate v with eq18 when update q and pi, instead of using predicted v

        # if self.cfg.pretrain_beta:
        #     for i in range(self.cfg.max_steps):
        #         if i % 10000 == 0:
        #             print(i)
        #         data = self.get_offline_data()
        #         self.update_beta(data)

    def coord2onehot_2d(self, num_cols, xs, ys):
        idxs = xs * num_cols + ys
        return idxs

    def v_equation(self, states, q):
        """
        For discrete action space only
        """
        with torch.no_grad():
            logbeta = []
            for a in range(self.cfg.action_dim):
                actions = torch_utils.tensor(np.ones(len(states)) * a, self.cfg.device)
                logbeta.append(self.beh_pi.get_logprob(states, actions).reshape((-1, 1)))
            logbeta = torch.cat(logbeta, dim=1)
            beta = torch.exp(logbeta)

            beta_zero = torch.where(beta < self.beta_threshold)
            q[beta_zero] = -torch.finfo(torch.float).max # To make sure qmax only take the max over actions having beta(a|s)>0
            beta[beta_zero] = 0
            beta /= beta.sum(axis=1, keepdims=True)
            qmax = q.max(axis=-1, keepdims=True)[0]
            next_v = self.tau * torch.log(torch.clip(
                (beta * torch.exp((q - qmax) / self.tau - logbeta)).sum(axis=-1, keepdims=True),
                self.eps, self.exp_threshold)) + qmax
        return next_v.squeeze(-1), logbeta

    def get_data(self):
        return self.get_offline_data()

    def feed_data(self):
        self.update_stats(0, None)
        return

    def compute_loss_beh_pi(self, data):
        """L_{\omega}, learn behavior policy"""
        states, actions = data['obs'], data['act']
        beh_log_probs = self.beh_pi.get_logprob(states, actions)
        beh_loss = -beh_log_probs.mean()
        return beh_loss, beh_log_probs
    
    def compute_loss_value(self, data):
        """L_{\phi}, learn z for state value, v = tau log z"""
        states, actions = data['obs'], data['act']
        z_phi = self.value_net(states).squeeze(-1)
        min_Q, _, _ = self.get_q_value_target(states, actions)
        with torch.no_grad():
            beh_log_prob = self.beh_pi.get_logprob(states, actions)
            
        # target = torch.exp(min_Q / self.tau - beh_log_prob)
        # # target = torch.clip(torch.exp(min_Q / self.tau - beh_log_prob), self.eps, self.exp_threshold)
        
        # ## Debug
        target = min_Q - self.tau * beh_log_prob

        # ## Debug
        # with torch.no_grad():
        #     q1, q2 = self.ac_targ.q1q2(states)
        #     value, _ = self.v_equation(states, torch.min(q1, q2))
        #     beh_log_prob = self.beh_pi.get_logprob(states, actions)
        #     # target = torch.exp(value / self.tau - beh_log_prob)
        #     target = value - self.tau * beh_log_prob

        value_loss = (0.5 * (z_phi - target)**2).mean()
        return value_loss
    
    def get_state_value(self, state):
        with torch.no_grad():
            # z_phi = self.value_net(state).squeeze(-1)
            # log0 = torch.where(z_phi <= 0)[0]
            # z_phi[log0] = self.eps
            # value = self.tau * torch.log(z_phi)
            
            # ## Debug
            value = self.value_net(state).squeeze(-1)
        return value

    def compute_loss_q(self, data):
        """
        L_{\theta}, learn action value
        Same as IQL, except reparameterizing v_\phi=\tau * log(z_{\phi})
        """
        states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
        with torch.no_grad():
            if self.eq18_v_calculation:
                q1, q2 = self.ac_targ.q1q2(next_states)
                next_v, _ = self.v_equation(next_states, torch.min(q1, q2))
            else:
                next_v = self.get_state_value(next_states)
                
            q_target = rewards + (self.gamma * (1 - dones) * next_v)
            
        _, q1, q2 = self.get_q_value(states, actions, with_grad=True)

        critic1_loss = (0.5* (q_target - q1) ** 2).mean()
        critic2_loss = (0.5* (q_target - q2) ** 2).mean()
        loss_q = (critic1_loss + critic2_loss) * 0.5
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())
        return loss_q, q_info
        
    def compute_loss_pi(self, data):
        """L_{\psi}, extract learned policy"""
        states, actions = data['obs'], data['act']

        log_probs = self.ac.pi.get_logprob(states, actions)
        min_Q, _, _ = self.get_q_value(states, actions, with_grad=False)
        with torch.no_grad():
            if self.eq18_v_calculation:
                q1, q2 = self.ac.q1q2(states)
                value, _ = self.v_equation(states, torch.min(q1, q2))
            else:
                value = self.get_state_value(states)
            beh_log_prob = self.beh_pi.get_logprob(states, actions)

        clipped = torch.clip(torch.exp((min_Q - value) / self.tau - beh_log_prob), self.eps, self.exp_threshold)
        pi_loss = -(clipped * log_probs).mean()
        return pi_loss, ""
    
    def update_tau(self, data):
        with torch.no_grad():
            _, log_pi = self.ac.pi(data['states'])
        tau_loss = -(self.log_tau * (log_pi + self.target_entropy).detach()).mean()
        self.auto_tau_optim.zero_grad()
        tau_loss.backward()
        self.auto_tau_optim.step()
        self.tau = self.log_tau.exp().detach()

    def update_beta(self, data):
        loss_beh_pi, _ = self.compute_loss_beh_pi(data)
        self.beh_pi_optimizer.zero_grad()
        loss_beh_pi.backward()
        # clip_grad_norm_(self.beh_pi.parameters(), self.clip_grad_param)
        self.beh_pi_optimizer.step()
        return loss_beh_pi

    def update(self, data):
        # if not self.cfg.pretrain_beta:
        loss_beta = self.update_beta(data).item()
        
        self.value_optimizer.zero_grad()
        loss_vs = self.compute_loss_value(data)
        loss_vs.backward()
        # clip_grad_norm_(self.val_net.parameters(), self.clip_grad_param)
        self.value_optimizer.step()

        loss_q, _ = self.compute_loss_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        # clip_grad_norm_(self.ac.q1q2.parameters(), self.clip_grad_param)
        self.q_optimizer.step()

        loss_pi, _ = self.compute_loss_pi(data)
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        # clip_grad_norm_(self.ac.pi.parameters(), self.clip_grad_param)
        self.pi_optimizer.step()
        
        if self.automatic_tmp_tuning:
            self.update_tau(data)
            
        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.sync_target()

        return {"beta": loss_beta,
                "actor": loss_pi.item(),
                "critic": loss_q.item(),
                "value": loss_vs.item()}
    # def policy(self, o, eval=False):
    #     o = torch_utils.tensor(self.cfg.state_normalizer(o), self.cfg.device)
    #     with torch.no_grad():
    #         a, _ = self.ac.pi(o)
    #     a = torch_utils.to_np(a)
    #     return a

    def save(self, early=False):
        parameters_dir = self.cfg.get_parameters_dir()
        if early:
            path = os.path.join(parameters_dir, "actor_net_earlystop")
        elif self.cfg.checkpoints:
            path = os.path.join(parameters_dir, "actor_net_{}".format(self.total_steps))
        else:
            path = os.path.join(parameters_dir, "actor_net")
        torch.save(self.ac.pi.state_dict(), path)

        if early:
            path = os.path.join(parameters_dir, "critic_net_earlystop")
        else:
            path = os.path.join(parameters_dir, "critic_net")
        torch.save(self.ac.q1q2.state_dict(), path)

        if early:
            path = os.path.join(parameters_dir, "vs_net_earlystop")
        else:
            path = os.path.join(parameters_dir, "vs_net")
        torch.save(self.value_net.state_dict(), path)


# class InSampleOnline(InSample):
#     def __init__(self, cfg):
#         super(InSampleOnline, self).__init__(cfg)
#         if cfg.agent_name == 'InSampleOnline' and cfg.load_offline_data:
#             self.fill_offline_data_to_buffer()
#         if 'load_params' in self.cfg.val_fn_config and self.cfg.val_fn_config['load_params']:
#             self.load_state_value_fn(cfg.val_fn_config['path'])
#
#     def get_data(self):
#         return base.ActorCritic.get_data(self)
#
#     def feed_data(self):
#         return base.ActorCritic.feed_data(self)
#
#
# class InSampleAC(InSample):
#     def __init__(self, cfg):
#         super(InSampleAC, self).__init__(cfg)
#
#     def get_state_value(self, state):
#         with torch.no_grad():
#             value = self.value_net(state).squeeze(-1)
#         return value
#
#     def compute_loss_q(self, data):
#         states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
#         with torch.no_grad():
#             next_actions, log_probs = self.ac.pi(next_states)
#         min_Q, _, _ = self.get_q_value_target(next_states, next_actions)
#         q_target = rewards + self.gamma * (1 - dones) * (min_Q - self.tau * log_probs)
#
#         _, q1, q2 = self.get_q_value(states, actions, with_grad=True)
#
#         critic1_loss = (0.5 * (q_target - q1) ** 2).mean()
#         critic2_loss = (0.5 * (q_target - q2) ** 2).mean()
#         loss_q = (critic1_loss + critic2_loss) * 0.5
#         q_info = dict(Q1Vals=q1.detach().numpy(),
#                       Q2Vals=q2.detach().numpy())
#         return loss_q, q_info
#
#     def compute_loss_value(self, data):
#         """L_{\phi}, learn z for state value, v = tau log z"""
#         states = data['obs']
#         v_phi = self.value_net(states).squeeze(-1)
#         with torch.no_grad():
#             actions, log_probs = self.ac.pi(states)
#             min_Q, _, _ = self.get_q_value_target(states, actions)
#             # beh_log_prob = self.beh_pi.get_logprob(states, actions)
#             beh_log_prob = self.ac.pi.get_logprob(states, actions)
#         target = min_Q - self.tau * beh_log_prob
#         value_loss = (0.5 * (v_phi - target) ** 2).mean()
#         return value_loss
#
#
# class InSampleACOnline(InSampleAC):
#     def __init__(self, cfg):
#         super(InSampleACOnline, self).__init__(cfg)
#         if cfg.agent_name == 'InSampleACOnline' and cfg.load_offline_data:
#             self.fill_offline_data_to_buffer()
#         if 'load_params' in self.cfg.val_fn_config and self.cfg.val_fn_config['load_params']:
#             self.load_state_value_fn(cfg.val_fn_config['path'])
#
#     def get_data(self):
#         return base.ActorCritic.get_data(self)
#
#     def feed_data(self):
#         return base.ActorCritic.feed_data(self)
#
#
# class InSampleMaxAC(InSampleAC):
#     def __init__(self, cfg):
#         super(InSampleMaxAC, self).__init__(cfg)
#         assert self.eq18_v_calculation == False
#         assert self.cfg.discrete_control == True
#
#     def compute_loss_q(self, data):
#         states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
#
#         if self.cfg.use_true_beta:
#             with torch.no_grad():
#                 q1, q2 = self.ac_targ.q1q2(next_states)
#                 q = torch.min(q1, q2)
#                 logbeta = []
#                 for a in range(self.cfg.action_dim):
#                     temp_a = torch_utils.tensor(np.ones(len(states)) * a, self.cfg.device)
#                     logbeta.append(self.beh_pi.get_logprob(states, temp_a).reshape((-1, 1)))
#                 logbeta = torch.cat(logbeta, dim=1)
#                 beta = torch.exp(logbeta)
#                 beta_zero = torch.where(beta < self.beta_threshold)
#         else:
#             temp_s = self.env.pos_to_state(states[:, 0], states[:, 1])
#             beta = self.cfg.true_beta[temp_s, actions]
#             beta_zero = torch.where(beta == 0)
#
#         q[beta_zero] = -torch.finfo(torch.float).max
#         target_Q = q.max(axis=-1, keepdims=False)[0]
#
#         q_target = rewards + self.gamma * (1 - dones) * target_Q
#
#         _, q1, q2 = self.get_q_value(states, actions, with_grad=True)
#
#         critic1_loss = (0.5 * (q_target - q1) ** 2).mean()
#         critic2_loss = (0.5 * (q_target - q2) ** 2).mean()
#         loss_q = (critic1_loss + critic2_loss) * 0.5
#         q_info = dict(Q1Vals=q1.detach().numpy(),
#                       Q2Vals=q2.detach().numpy())
#         return loss_q, q_info
#
#
# class InSampleSoftMax(InSampleAC):
#     def __init__(self, cfg):
#         super(InSampleSoftMax, self).__init__(cfg)
#         assert self.eq18_v_calculation == False
#         assert self.cfg.discrete_control == True
#
#     def compute_loss_q(self, data):
#         states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
#         if self.cfg.use_true_beta:
#             with torch.no_grad():
#                 q1, q2 = self.ac_targ.q1q2(next_states)
#                 q = torch.min(q1, q2)
#                 logbeta = []
#                 for a in range(self.cfg.action_dim):
#                     temp_a = torch_utils.tensor(np.ones(len(states)) * a, self.cfg.device)
#                     logbeta.append(self.beh_pi.get_logprob(states, temp_a).reshape((-1, 1)))
#                 logbeta = torch.cat(logbeta, dim=1)
#                 beta = torch.exp(logbeta)
#                 beta_zero = torch.where(beta < self.beta_threshold)
#         else:
#             temp_s = self.env.pos_to_state(states[:, 0], states[:, 1])
#             beta = self.cfg.true_beta[temp_s, actions]
#             beta_zero = torch.where(beta == 0)
#
#         q[beta_zero] = -torch.finfo(torch.float).max
#         q_max = q.max(axis=-1, keepdims=True)[0]
#         target_Q = self.tau * torch.log(torch.exp((q - q_max) / self.tau).sum(-1, keepdims=True)) + q_max
#         target_Q = target_Q.squeeze(-1)
#         q_target = rewards + self.gamma * (1 - dones) * target_Q
#
#         _, q1, q2 = self.get_q_value(states, actions, with_grad=True)
#
#         critic1_loss = (0.5 * (q_target - q1) ** 2).mean()
#         critic2_loss = (0.5 * (q_target - q2) ** 2).mean()
#         loss_q = (critic1_loss + critic2_loss) * 0.5
#         q_info = dict(Q1Vals=q1.detach().numpy(),
#                       Q2Vals=q2.detach().numpy())
#         return loss_q, q_info
#
#
# class InSampleOnPiW(InSample):
#     def __init__(self, cfg):
#         super(InSampleOnPiW, self).__init__(cfg)
#
#     def compute_loss_value(self, data):
#         states = data['obs']
#         z_phi = self.value_net(states).squeeze(-1)
#         with torch.no_grad():
#             actions, beh_log_prob = self.beh_pi(states)
#         min_Q, _, _ = self.get_q_value_target(states, actions)
#         target = min_Q - self.tau * beh_log_prob
#         value_loss = (0.5 * (z_phi - target) ** 2).mean()
#         return value_loss
