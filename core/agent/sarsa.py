import copy
import os
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import torch

from core.agent import base
from core.utils import torch_utils


class SarsaAgent(base.ValueBased):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.prev_state = None
        self.prev_action = None
        self.state = None
        self.reward = None
        self.done = None

    def step(self):
        need_update = True
        if self.reset is True:
            self.state = self.env.reset()
            self.reset = False
            need_update = False

        action = self.policy(self.state, self.cfg.eps_schedule())
        if need_update:
            self.update(self.prev_state, self.prev_action, self.state, self.reward, self.done, action)

        self.prev_state = self.state
        self.prev_action = action
        self.state, self.reward, self.done, _ = self.env.step([action])

        self.update_stats(self.reward, self.done)
        return self.prev_state, self.prev_action, self.state, self.reward, int(self.done)

    def update(self, prev_state, prev_action, next_state, reward, done, next_action):
        if (not self.cfg.rep_fn_config['train_params']) and (not self.cfg.val_fn_config['train_params']):
            return
        prev_state = torch_utils.tensor(self.cfg.state_normalizer(prev_state), self.cfg.device)
        next_state = torch_utils.tensor(self.cfg.state_normalizer(next_state), self.cfg.device)
        prev_action = torch_utils.tensor(prev_action, self.cfg.device).long()
        if not self.cfg.rep_fn_config['train_params']:
            with torch.no_grad():
                phi = self.rep_net(prev_state)
        else:
            phi = self.rep_net(prev_state)

        if not self.cfg.val_fn_config['train_params']:
            with torch.no_grad():
                q = self.val_net(phi)[prev_action]
        else:
            q = self.val_net(phi)[prev_action]

        # Constructing the target
        with torch.no_grad():
            q_next = self.targets.val_net(self.targets.rep_net(next_state))
            q_next = q_next[next_action]
            terminal = torch_utils.tensor(done, self.cfg.device)
            reward = torch_utils.tensor(reward, self.cfg.device)
            target = self.cfg.discount * q_next * (1 - terminal).float()
            target.add_(reward.float())

        loss = self.vf_loss(q, target)
        constr = self.constr_fn(phi, q, target)
        
        loss += constr

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
            self.targets.val_net.load_state_dict(self.val_net.state_dict())


class SarsaOfflineBatch(base.ValueBased):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.offline_param_init()

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
        nact = data['act2']

        no_na = np.where(nact != -1, True, False)
        not_termin = np.where(t != 1, True, False)
        not_last = np.logical_and(no_na, not_termin)
        in_ = data['obs'][not_last]
        act = data['act'][not_last]
        ns = data['obs2'][not_last]
        t = data['done'][not_last]
        r = data['reward'][not_last]
        nact = data['act2'][not_last]
    
        q = self.val_net(self.rep_net(in_))[np.arange(len(in_)), act]
        with torch.no_grad():
            # # Follow the dataset trajectory
            tar_ = r + self.cfg.discount * (1 - t) * self.targets.val_net(self.targets.rep_net(ns))[np.arange(len(ns)), nact]
    
            # # Based on the current estimation (target)
            # next_q = self.targets.val_net(self.targets.rep_net(ns))
            # _, nact = torch.max(next_q, 1)
            # temp = self.agent_rng.random(size=len(nact))
            # change_idx = np.where(temp < self.cfg.eps_schedule())[0]
            # nact[change_idx] = torch_utils.tensor(self.agent_rng.randint(0, self.cfg.action_dim, size=len(change_idx)), self.cfg.device).type(torch.long)
            # tar_ = r + self.cfg.discount * (1 - t) * next_q[np.arange(len(ns)), nact]

        loss = self.vf_loss(q, tar_)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_loss.append(torch_utils.to_np(loss))
        self.update_stats(0, None)
        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.sync_target()
            # self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
            # self.targets.val_net.load_state_dict(self.val_net.state_dict())
        return

    # def epoch_step(self):
    #     train_s, train_a, train_r, train_ns, train_t, train_na, _, _, _ = self.trainset
    #
    #     self.agent_rng.shuffle(self.training_indexs)
    #     ls_epoch = []
    #     for b in range(int(np.ceil(self.training_size / self.cfg.batch_size))):
    #         idxs = self.training_indexs[b * self.cfg.batch_size: (b + 1) * self.cfg.batch_size]
    #         in_ = torch_utils.tensor(self.cfg.state_normalizer(train_s[idxs]), self.cfg.device)
    #         act = train_a[idxs]
    #         r = torch_utils.tensor(train_r[idxs], self.cfg.device)
    #         ns = torch_utils.tensor(self.cfg.state_normalizer(train_ns[idxs]), self.cfg.device)
    #         t = torch_utils.tensor(train_t[idxs], self.cfg.device)
    #         nact = train_na[idxs] # Will be replaced when using the current estimation
    #
    #         q = self.val_net(self.rep_net(in_))[np.arange(len(in_)), act]
    #         with torch.no_grad():
    #             # # Follow the dataset trajectory
    #             tar_ = r + self.cfg.discount * (1 - t) * self.targets.val_net(self.targets.rep_net(ns))[np.arange(len(ns)), nact]
    #
    #             # # Based on the current estimation (target)
    #             # next_q = self.targets.val_net(self.targets.rep_net(ns))
    #             # _, nact = torch.max(next_q, 1)
    #             # temp = self.agent_rng.random(size=len(nact))
    #             # change_idx = np.where(temp < self.cfg.eps_schedule())[0]
    #             # nact[change_idx] = torch_utils.tensor(self.agent_rng.randint(0, self.cfg.action_dim, size=len(change_idx)), self.cfg.device).type(torch.long)
    #             # tar_ = r + self.cfg.discount * (1 - t) * next_q[np.arange(len(ns)), nact]
    #
    #         loss = self.vf_loss(q, tar_)
    #
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #         ls_epoch.append(torch_utils.to_np(loss))
    #
    #     self.training_loss.append(np.array(ls_epoch).mean())
    #     self.update_stats(0, None)
    #     if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
    #         self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
    #         self.targets.val_net.load_state_dict(self.val_net.state_dict())
    #
    #     return self.test_fn()
    
    # def test_fn(self):
    #     test_s, test_a, test_r, test_sp, test_term, test_ap, _, _, _ = self.testset # test_ap will be replaced if following the current estimation
    #     test_s = torch_utils.tensor(self.cfg.state_normalizer(test_s), self.cfg.device)
    #     test_r = torch_utils.tensor(test_r, self.cfg.device)
    #     test_sp = torch_utils.tensor(self.cfg.state_normalizer(test_sp), self.cfg.device)
    #     test_term = torch_utils.tensor(test_term, self.cfg.device)
    #     with torch.no_grad():
    #         q = self.val_net(self.rep_net(test_s))[np.arange(len(test_s)), test_a]
    #         next_q = self.targets.val_net(self.targets.rep_net(test_sp))
    #
    #         # # Follow the dataset trajectory
    #         next_q = next_q[np.arange(len(test_sp)), test_ap]
    #
    #         # # Based on the current estimation (target)
    #         # _, test_ap = torch.max(next_q, 1)
    #         # temp = self.agent_rng.random(size=len(test_ap))
    #         # change_idx = np.where(temp < self.cfg.eps_schedule())[0]
    #         # test_ap[change_idx] = torch_utils.tensor(self.agent_rng.randint(0, self.cfg.action_dim, size=len(change_idx)), self.cfg.device).type(torch.long)
    #
    #         target = test_r + self.cfg.discount * (1 - test_term) * next_q
    #         tloss = self.vf_loss(q, target).numpy()
    #
    #     if tloss - self.tloss_rec > 0:
    #         self.tloss_increase += 1
    #     else:
    #         self.tloss_increase = 0
    #     self.tloss_rec = tloss
    #     self.test_loss.append(tloss)
    #     if self.tloss_increase > self.cfg.early_cut_threshold:
    #         return "EarlyCutOff"
    #     return

    # def log_file(self, elapsed_time=-1):
    #     if len(self.training_loss) > 0:
    #         training_loss = np.array(self.training_loss)
    #         self.training_loss = []
    #         mean, median, min_, max_ = np.mean(training_loss), np.median(training_loss), np.min(training_loss), np.max(training_loss)
    #
    #         if len(self.test_loss) == 0:
    #             self.test_fn()
    #         test_loss = np.array(self.test_loss)
    #         self.test_loss = []
    #         tmean, tmedian, tmin_, tmax_ = np.mean(test_loss), np.median(test_loss), np.min(test_loss), np.max(test_loss)
    #
    #         log_str = 'TRAIN LOG: epoch %d, ' \
    #                   'training loss %.4f/%.4f/%.4f/%.4f/%d (mean/median/min/max/num), ' \
    #                   'test loss %.4f/%.4f/%.4f/%.4f/%d (mean/median/min/max/num), %.4f steps/s'
    #         self.cfg.logger.info(log_str % (self.total_steps,
    #                                         mean, median, min_, max_, len(training_loss),
    #                                         tmean, tmedian, tmin_, tmax_, len(test_loss),
    #                                         elapsed_time))
    #         self.populate_states, self.populate_actions, self.populate_true_qs = self.populate_returns(log_traj=True)
    #         self.populate_latest = True
    #         mean, median, min_, max_ = self.log_return(self.ep_returns_queue_test, "TEST", elapsed_time)
    #         return tmean, tmedian, tmin_, tmax_
    #     else:
    #         log_str = 'TRAIN LOG: epoch %d, ' \
    #                   'training loss %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), ' \
    #                   'test loss %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), %.2f steps/s'
    #         self.cfg.logger.info(log_str % (self.total_steps,
    #                                         np.nan, np.nan, np.nan, np.nan, 0,
    #                                         np.nan, np.nan, np.nan, np.nan, 0,
    #                                         elapsed_time))
    #         self.populate_states, self.populate_actions, self.populate_true_qs = self.populate_returns(log_traj=True)
    #         self.populate_latest = True
    #         mean, median, min_, max_ = self.log_return(self.ep_returns_queue_test, "TEST", elapsed_time)
    #         return [None] * 4
