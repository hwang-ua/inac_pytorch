import os
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import torch

from core.agent import base
from core.utils import torch_utils

"""
Not Used
"""
class QmaxCloneOffline(base.Agent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rep_net = cfg.rep_fn()
        self.val_net = cfg.val_fn()

        params = list(self.rep_net.parameters()) + list(self.val_net.parameters())
        self.optimizer = cfg.optimizer_fn(params)

        self.env = cfg.env_fn()
        self.vf_loss = cfg.vf_loss_fn()
        self.constr_fn = cfg.constr_fn()

        self.true_q_predictor = self.cfg.tester_fn.get('true_value_estimator', lambda x:None)
        self.state = None
        self.action = None
        self.next_state = None

        self.trainset, self.testset = self.training_set_construction(cfg.offline_data)
        self.training_size = len(self.trainset[0])
        self.training_indexs = np.arange(self.training_size)
        self.eval_set = self.property_evaluation_dataset(cfg.eval_data)
        
        self.training_loss = []
        self.test_loss = []
        self.tloss_increase = 0
        self.tloss_rec = np.inf

    def constraint_fn(self, v, action, qtar):
        return 0

    def step(self):
        train_s, train_a, _, _, _, _, _, _, train_qmax = self.trainset
        idxs = self.agent_rng.randint(0, len(train_s), size=self.cfg.batch_size)
        in_ = torch_utils.tensor(self.cfg.state_normalizer(train_s[idxs]), self.cfg.device)
        act = train_a[idxs]
        qtar = torch_utils.tensor(train_qmax[idxs], self.cfg.device)

        v = self.val_net(self.rep_net(in_))
        q = v[np.arange(len(in_)), act]
        loss = self.vf_loss(q, qtar)
        constraint = self.constraint_fn(v, act, qtar)
        loss += constraint

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_loss.append(torch_utils.to_np(loss))
        self.update_stats(0, None)
        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
            self.targets.val_net.load_state_dict(self.val_net.state_dict())
        return

    def epoch_step(self):
        train_s, train_a, _, _, _, _, _, _, train_qmax = self.trainset
        
        self.agent_rng.shuffle(self.training_indexs)
        ls_epoch = []
        for b in range(int(np.ceil(self.training_size / self.cfg.batch_size))):
            idxs = self.training_indexs[b * self.cfg.batch_size: (b + 1) * self.cfg.batch_size]
            in_ = torch_utils.tensor(self.cfg.state_normalizer(train_s[idxs]), self.cfg.device)
            act = train_a[idxs]
            qtar = torch_utils.tensor(train_qmax[idxs], self.cfg.device)
            
            v = self.val_net(self.rep_net(in_))
            q = v[np.arange(len(in_)), act]
            loss = self.vf_loss(q, qtar)
            constraint = self.constraint_fn(v, act, qtar)
            # print(loss, constraint)
            loss += constraint
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            ls_epoch.append(torch_utils.to_np(loss))
        
        self.training_loss.append(np.array(ls_epoch).mean())
        self.update_stats(0, None)
        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
            self.targets.val_net.load_state_dict(self.val_net.state_dict())
        
        return self.test_fn()
    
    def test_fn(self):
        test_s, test_a, _, _, _, _, _, _, test_qmax = self.testset
        test_s = torch_utils.tensor(self.cfg.state_normalizer(test_s), self.cfg.device)
        test_qmax = torch_utils.tensor(test_qmax, self.cfg.device)
        with torch.no_grad():
            v = self.val_net(self.rep_net(test_s))
            q = v[np.arange(len(test_s)), test_a]
            tloss = self.vf_loss(q, test_qmax)
            tloss += self.constraint_fn(v, test_a, test_qmax)
        tloss = tloss.numpy()
        if tloss - self.tloss_rec > 0:
            self.tloss_increase += 1
        else:
            self.tloss_increase = 0
        self.tloss_rec = tloss
        self.test_loss.append(tloss)
        if self.tloss_increase > self.cfg.early_cut_threshold:
            return "EarlyCutOff"
        return
    
    def eval_step(self, state):
        with torch.no_grad():
            q_values = self.val_net(self.rep_net(self.cfg.state_normalizer(state)))
            q_values = torch_utils.to_np(q_values).flatten()
        return self.agent_rng.choice(np.flatnonzero(q_values == q_values.max()))

    def save(self, early=False):
        parameters_dir = self.cfg.get_parameters_dir()
        if early:
            path = os.path.join(parameters_dir, "rep_net_earlystop")
        else:
            path = os.path.join(parameters_dir, "rep_net")
        torch.save(self.rep_net.state_dict(), path)

        if early:
            path = os.path.join(parameters_dir, "val_net_earlystop")
        else:
            path = os.path.join(parameters_dir, "val_net")
        torch.save(self.val_net.state_dict(), path)

    def load_rep_fn(self, parameters_dir):
        path = os.path.join(self.cfg.data_root, parameters_dir)
        self.rep_net.load_state_dict(torch.load(path, map_location=self.cfg.device))
        self.cfg.logger.info("Load rep function from {}".format(path))

    def load_val_fn(self, parameters_dir):
        path = os.path.join(self.cfg.data_root, parameters_dir)
        self.val_net.load_state_dict(torch.load(path, map_location=self.cfg.device))
        self.cfg.logger.info("Load value function from {}".format(path))

    def log_file(self, elapsed_time=-1):
        if len(self.training_loss) > 0:
            training_loss = np.array(self.training_loss)
            self.training_loss = []
            mean, median, min_, max_ = np.mean(training_loss), np.median(training_loss), np.min(training_loss), np.max(training_loss)
            
            if len(self.test_loss) == 0:
                self.test_fn()
            test_loss = np.array(self.test_loss)
            self.test_loss = []
            tmean, tmedian, tmin_, tmax_ = np.mean(test_loss), np.median(test_loss), np.min(test_loss), np.max(test_loss)
            
            log_str = 'TRAIN LOG: epoch %d, ' \
                      'training loss %.4f/%.4f/%.4f/%.4f/%d (mean/median/min/max/num), ' \
                      'test loss %.4f/%.4f/%.4f/%.4f/%d (mean/median/min/max/num), %.4f steps/s'
            self.cfg.logger.info(log_str % (self.total_steps,
                                            mean, median, min_, max_, len(training_loss),
                                            tmean, tmedian, tmin_, tmax_, len(test_loss),
                                            elapsed_time))
            self.populate_states, self.populate_actions, self.populate_true_qs = self.populate_returns(log_traj=True)
            self.populate_latest = True
            mean, median, min_, max_ = self.log_return(self.ep_returns_queue_test, "TEST", elapsed_time)
            return tmean, tmedian, tmin_, tmax_
        else:
            log_str = 'TRAIN LOG: epoch %d, ' \
                      'training loss %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), ' \
                      'test loss %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), %.2f steps/s'
            self.cfg.logger.info(log_str % (self.total_steps,
                                            np.nan, np.nan, np.nan, np.nan, 0,
                                            np.nan, np.nan, np.nan, np.nan, 0,
                                            elapsed_time))
            self.populate_states, self.populate_actions, self.populate_true_qs = self.populate_returns(log_traj=True)
            self.populate_latest = True
            mean, median, min_, max_ = self.log_return(self.ep_returns_queue_test, "TEST", elapsed_time)
            return [None] * 4


class QmaxConstrOffline(QmaxCloneOffline):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.constr_weight = cfg.constr_weight
    
    def constraint_fn(self, v, action, qtar):
        one = torch.ones(v.size()[1]).reshape((v.size()[1], 1))
        qtar = torch.matmul(one, qtar.reshape((1, qtar.size()[0]))).T
        higher = torch.clamp(v - qtar, min=0)
        higher[np.arange(len(higher)), action] = 0
        # const = torch.square(higher).mean()
        const = higher.sum(axis=1).mean()
        return self.constr_weight * const
