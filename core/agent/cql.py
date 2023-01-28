import os
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import copy
import torch.nn.functional as F

from core.agent import base
from core.utils import torch_utils



from ml_collections import ConfigDict
# from .model import Scalar, soft_target_update
# from .utils import prefix_metrics


class CQLAgentOffline(base.ValueBased):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.offline_param_init()
        self.alpha = cfg.cql_alpha

    def get_data(self):
        return self.get_offline_data()

    def feed_data(self):
        self.update_stats(0, None)
        return

    def update(self, data):
        in_ = data['obs']
        act = data['act']
        ns = data['obs2']
        t = data['done']
        r = data['reward']
        
        """
        According to https://github.com/BY571/CQL/blob/main/CQL-DQN/agent.py
        def learn(self, experiences):
        """
        q_s = self.val_net(self.rep_net(in_))
        q_s_a = q_s[np.arange(len(in_)), act]
        with torch.no_grad():
            q_tar = r + (self.cfg.discount * (1 - t) * self.targets.val_net(self.targets.rep_net(ns)).max(1)[0])
        loss = self.cql_loss(q_s, q_s_a, q_tar)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.rep_net.parameters()) + list(self.val_net.parameters()), 1)
        self.optimizer.step()

        self.training_loss.append(torch_utils.to_np(loss))
        self.update_stats(0, None)
        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.sync_target()
        return

    def cql_loss(self, q_s, q_s_a, q_tar):
        cql1_loss = torch.logsumexp(q_s, dim=1).mean() - q_s_a.mean()
        bellmann_error = self.vf_loss(q_s_a, q_tar)
        loss = self.alpha * cql1_loss + 0.5 * bellmann_error
        return loss


class Scalar(torch.nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = torch.nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32)
        )
    def forward(self):
        return self.constant
class CQLSACOffline(base.ActorCritic):
    """
    class ConservativeSAC
    https://github.com/young-geng/CQL/blob/934b0e8354ca431d6c083c4e3a29df88d4b0a24d/SimpleSAC/conservative_sac.py
    """

    def __init__(self, cfg):
        super(CQLSACOffline, self).__init__(cfg)
        self.offline_param_init()

    # def __init__(self, config, policy, qf1, qf2, target_qf1, target_qf2):
        config = ConfigDict()
        config.discount = cfg.discount
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.backup_entropy = False
        config.target_entropy = 0.0
        config.policy_lr = 3e-05#cfg.learning_rate
        config.qf_lr = cfg.learning_rate
        config.optimizer_type = 'adam'
        # config.soft_target_update_rate = cfg.polyak
        config.target_update_period = 1
        config.use_cql = True
        config.cql_n_actions = 10
        config.cql_importance_sample = True
        config.cql_lagrange = False
        config.cql_target_action_gap = cfg.target_action_gap
        config.cql_temp = cfg.temperature
        config.cql_min_q_weight = 5.0
        config.cql_max_target_backup = False
        config.cql_clip_diff_min = -np.inf
        config.cql_clip_diff_max = np.inf

        self.config = config
        # self.policy = policy
        # self.qf1 = qf1
        # self.qf2 = qf2
        # self.target_qf1 = target_qf1
        # self.target_qf2 = target_qf2

        optimizer_class = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }[self.config.optimizer_type]

        # self.policy_optimizer = optimizer_class(
        #     self.policy.parameters(), self.config.policy_lr,
        # )
        self.pi_optimizer = torch.optim.Adam(list(self.ac.pi.parameters()), lr=self.config.policy_lr) # use a smaller learning rate as in paper
        # self.qf_optimizer = optimizer_class(
        #     list(self.qf1.parameters()) + list(self.qf2.parameters()), self.config.qf_lr
        # )
        
        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = optimizer_class(
                self.log_alpha.parameters(),
                lr=self.config.policy_lr,
            )
        else:
            self.log_alpha = None

        if self.config.cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            self.alpha_prime_optimizer = optimizer_class(
                self.log_alpha_prime.parameters(),
                lr=self.config.qf_lr,
            )

        # self.update_target_network(1.0)
        # self._total_steps = 0

    # def update_target_network(self, soft_target_update_rate):
    #     soft_target_update(self.qf1, self.target_qf1, soft_target_update_rate)
    #     soft_target_update(self.qf2, self.target_qf2, soft_target_update_rate)
    
    def extend_and_repeat(self, tensor, dim, repeat):
        # Extend and repeast the tensor along dim axie and repeat it
        ones_shape = [1 for _ in range(tensor.ndim + 1)]
        ones_shape[dim] = repeat
        return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)

    def update(self, data, bc=False):
        # self._total_steps += 1

        # observations = batch['observations']
        # actions = batch['actions']
        # rewards = batch['rewards']
        # next_observations = batch['next_observations']
        # dones = batch['dones']
        observations, actions, rewards, next_observations, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
        actions = torch_utils.tensor(actions, self.cfg.device)
        # new_actions, log_pi = self.policy(observations)
        new_actions, log_pi = self.ac.pi(observations)

        if self.config.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (log_pi + self.config.target_entropy).detach()).mean()
            alpha = self.log_alpha().exp() * self.config.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.config.alpha_multiplier)

        """ Policy loss """
        if bc:
            log_probs = self.policy.log_prob(observations, actions)
            policy_loss = (alpha*log_pi - log_probs).mean()
        else:
            # q_new_actions = torch.min(
            #     self.qf1(observations, new_actions),
            #     self.qf2(observations, new_actions),
            # )
            q_new_actions, _, _ = self.get_q_value(observations, new_actions, with_grad=False)
            policy_loss = (alpha*log_pi - q_new_actions).mean()

        """ Q function loss """
        # q1_pred = self.qf1(observations, actions)
        # q2_pred = self.qf2(observations, actions)
        _, q1_pred, q2_pred = self.get_q_value(observations, actions, with_grad=True)

        if self.config.cql_max_target_backup:
            new_next_actions, next_log_pi = self.policy(next_observations, repeat=self.config.cql_n_actions)
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_qf1(next_observations, new_next_actions),
                    self.target_qf2(next_observations, new_next_actions),
                ),
                dim=-1
            )
            next_log_pi = torch.gather(next_log_pi, -1, max_target_indices.unsqueeze(-1)).squeeze(-1)
        else:
            # new_next_actions, next_log_pi = self.policy(next_observations)
            new_next_actions, next_log_pi = self.ac.pi(next_observations)
            # target_q_values = torch.min(
            #     self.target_qf1(next_observations, new_next_actions),
            #     self.target_qf2(next_observations, new_next_actions),
            # )
            target_q_values, _, _ = self.get_q_value(next_observations, new_next_actions, with_grad=False)

        if self.config.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        td_target = rewards + (1. - dones) * self.config.discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred, td_target.detach())
        qf2_loss = F.mse_loss(q2_pred, td_target.detach())


        ### CQL
        if not self.config.use_cql:
            qf_loss = qf1_loss + qf2_loss
        else:
            batch_size = actions.shape[0]
            action_dim = actions.shape[-1]
            # cql_random_actions = actions.new_empty((batch_size, self.config.cql_n_actions, action_dim), requires_grad=False).uniform_(-1, 1)
            cql_random_actions = actions.new_empty((batch_size*self.config.cql_n_actions, action_dim), requires_grad=False).uniform_(-1, 1)
            # cql_current_actions, cql_current_log_pis = self.policy(observations, repeat=self.config.cql_n_actions)
            # cql_next_actions, cql_next_log_pis = self.policy(next_observations, repeat=self.config.cql_n_actions)
            observations = self.extend_and_repeat(observations, 1, self.config.cql_n_actions).reshape((self.batch_size * self.config.cql_n_actions, -1))
            cql_current_actions, cql_current_log_pis = self.ac.pi(observations)
            # cql_current_actions = cql_current_actions.reshape((self.batch_size, self.config.cql_n_actions, -1))
            # cql_current_log_pis = cql_current_log_pis.reshape((self.batch_size, self.config.cql_n_actions, -1))
            next_observations = self.extend_and_repeat(next_observations, 1, self.config.cql_n_actions).reshape((self.batch_size*self.config.cql_n_actions, -1))
            cql_next_actions, cql_next_log_pis = self.ac.pi(next_observations)
            # cql_next_actions = cql_next_actions.reshape((self.batch_size, self.config.cql_n_actions, -1))
            # cql_next_log_pis = cql_next_log_pis.reshape((self.batch_size, self.config.cql_n_actions, -1))
            cql_current_actions, cql_current_log_pis = cql_current_actions.detach(), cql_current_log_pis.detach()
            cql_next_actions, cql_next_log_pis = cql_next_actions.detach(), cql_next_log_pis.detach()

            # cql_q1_rand = self.qf1(observations, cql_random_actions)
            # cql_q2_rand = self.qf2(observations, cql_random_actions)
            _, cql_q1_rand, cql_q2_rand = self.get_q_value(observations, cql_random_actions)
            cql_q1_rand = cql_q1_rand.reshape((self.batch_size, self.config.cql_n_actions, -1))
            cql_q2_rand = cql_q2_rand.reshape((self.batch_size, self.config.cql_n_actions, -1))
            # cql_q1_current_actions = self.qf1(observations, cql_current_actions)
            # cql_q2_current_actions = self.qf2(observations, cql_current_actions)
            _, cql_q1_current_actions, cql_q2_current_actions = self.get_q_value(observations, cql_current_actions)
            cql_q1_current_actions = cql_q1_current_actions.reshape((self.batch_size, self.config.cql_n_actions, -1))
            cql_q2_current_actions = cql_q2_current_actions.reshape((self.batch_size, self.config.cql_n_actions, -1))

            # cql_q1_next_actions = self.qf1(observations, cql_next_actions)
            # cql_q2_next_actions = self.qf2(observations, cql_next_actions)
            _, cql_q1_next_actions, cql_q2_next_actions = self.get_q_value(observations, cql_next_actions)
            cql_q1_next_actions = cql_q1_next_actions.reshape((self.batch_size, self.config.cql_n_actions, -1))
            cql_q2_next_actions = cql_q2_next_actions.reshape((self.batch_size, self.config.cql_n_actions, -1))

            cql_cat_q1 = torch.cat(
                [cql_q1_rand, torch.unsqueeze(q1_pred, 1).unsqueeze(1), cql_q1_next_actions, cql_q1_current_actions], dim=1
            )
            cql_cat_q2 = torch.cat(
                [cql_q2_rand, torch.unsqueeze(q2_pred, 1).unsqueeze(1), cql_q2_next_actions, cql_q2_current_actions], dim=1
            )
            cql_std_q1 = torch.std(cql_cat_q1, dim=1)
            cql_std_q2 = torch.std(cql_cat_q2, dim=1)

            cql_next_log_pis = cql_next_log_pis.reshape((self.batch_size, self.config.cql_n_actions, -1))
            cql_current_log_pis = cql_current_log_pis.reshape((self.batch_size, self.config.cql_n_actions, -1))
            if self.config.cql_importance_sample:
                random_density = np.log(0.5 ** action_dim)
                cql_cat_q1 = torch.cat(
                    [cql_q1_rand - random_density,
                     cql_q1_next_actions - cql_next_log_pis.detach(),
                     cql_q1_current_actions - cql_current_log_pis.detach()],
                    dim=1
                )
                cql_cat_q2 = torch.cat(
                    [cql_q2_rand - random_density,
                     cql_q2_next_actions - cql_next_log_pis.detach(),
                     cql_q2_current_actions - cql_current_log_pis.detach()],
                    dim=1
                )

            cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.config.cql_temp, dim=1) * self.config.cql_temp
            cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.config.cql_temp, dim=1) * self.config.cql_temp

            """Subtract the log likelihood of data"""
            cql_qf1_diff = torch.clamp(
                cql_qf1_ood - q1_pred,
                self.config.cql_clip_diff_min,
                self.config.cql_clip_diff_max,
            ).mean()
            cql_qf2_diff = torch.clamp(
                cql_qf2_ood - q2_pred,
                self.config.cql_clip_diff_min,
                self.config.cql_clip_diff_max,
            ).mean()

            if self.config.cql_lagrange:
                alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0)
                cql_min_qf1_loss = alpha_prime * self.config.cql_min_q_weight * (cql_qf1_diff - self.config.cql_target_action_gap)
                cql_min_qf2_loss = alpha_prime * self.config.cql_min_q_weight * (cql_qf2_diff - self.config.cql_target_action_gap)

                self.alpha_prime_optimizer.zero_grad()
                alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss)*0.5
                alpha_prime_loss.backward(retain_graph=True)
                self.alpha_prime_optimizer.step()
            else:
                cql_min_qf1_loss = cql_qf1_diff * self.config.cql_min_q_weight
                cql_min_qf2_loss = cql_qf2_diff * self.config.cql_min_q_weight
                alpha_prime_loss = observations.new_tensor(0.0)
                alpha_prime = observations.new_tensor(0.0)


            qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss


        if self.config.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # self.policy_optimizer.zero_grad()
        self.pi_optimizer.zero_grad()
        policy_loss.backward()
        # self.policy_optimizer.step()
        self.pi_optimizer.step()

        # self.qf_optimizer.zero_grad()
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        # self.qf_optimizer.step()
        self.q_optimizer.step()

        # if self.total_steps % self.config.target_update_period == 0:
        #     self.update_target_network(
        #         self.config.soft_target_update_rate
        #     )
        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.sync_target()

        # metrics = dict(
        #     log_pi=log_pi.mean().item(),
        #     policy_loss=policy_loss.item(),
        #     qf1_loss=qf1_loss.item(),
        #     qf2_loss=qf2_loss.item(),
        #     alpha_loss=alpha_loss.item(),
        #     alpha=alpha.item(),
        #     average_qf1=q1_pred.mean().item(),
        #     average_qf2=q2_pred.mean().item(),
        #     average_target_q=target_q_values.mean().item(),
        #     total_steps=self.total_steps,
        # )
        #
        # if self.config.use_cql:
        #     metrics.update(prefix_metrics(dict(
        #         cql_std_q1=cql_std_q1.mean().item(),
        #         cql_std_q2=cql_std_q2.mean().item(),
        #         cql_q1_rand=cql_q1_rand.mean().item(),
        #         cql_q2_rand=cql_q2_rand.mean().item(),
        #         cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
        #         cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
        #         cql_qf1_diff=cql_qf1_diff.mean().item(),
        #         cql_qf2_diff=cql_qf2_diff.mean().item(),
        #         cql_q1_current_actions=cql_q1_current_actions.mean().item(),
        #         cql_q2_current_actions=cql_q2_current_actions.mean().item(),
        #         cql_q1_next_actions=cql_q1_next_actions.mean().item(),
        #         cql_q2_next_actions=cql_q2_next_actions.mean().item(),
        #         alpha_prime_loss=alpha_prime_loss.item(),
        #         alpha_prime=alpha_prime.item(),
        #     ), 'cql'))

        return #metrics

    # def torch_to_device(self, device):
    #     for module in self.modules:
    #         module.to(device)

    # @property
    # def modules(self):
    #     modules = [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2]
    #     if self.config.use_automatic_entropy_tuning:
    #         modules.append(self.log_alpha)
    #     if self.config.cql_lagrange:
    #         modules.append(self.log_alpha_prime)
    #     return modules

    # @property
    # def total_steps(self):
    #     return self._total_steps
    
    def get_data(self):
        return self.get_offline_data()

    def feed_data(self):
        self.update_stats(0, None)
        return
