import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from core.agent import base
from core.utils import torch_utils


class MonteCarloAgent(base.ValueBased):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.agent_name == "MonteCarlo":
            assert self.replay.memory_size >= self.cfg.timeout

    def step(self):
        if self.reset is True:
            self.update(None)
            self.state = self.env.reset()
            self.reset = False

        action = self.policy(self.state, self.cfg.eps_schedule())
        next_state, reward, done, _ = self.env.step([action])
        self.replay.feed([self.state, action, reward, next_state, int(done)])
        prev_state = self.state
        self.state = next_state
        self.update_stats(reward, done)
        return prev_state, action, reward, next_state, int(done)


    def update(self, data):
        if self.replay.size() == 0 or ((not self.cfg.rep_fn_config['train_params'])
                                       and (not self.cfg.val_fn_config['train_params'])):
            return
        episode = self.replay.get_buffer()
        ret = 0
        in_ = []
        act_ = []
        tar_ = []
        for t in range(len(episode)-1, -1, -1):
            s, a, r, _, gamma = episode[t]
            ret += r + gamma * ret
            phi = self.cfg.state_normalizer(s)
            in_.append(phi)
            act_.append(a)
            tar_.append(ret)
        in_ = torch_utils.tensor(in_, self.cfg.device)
        tar_ = torch_utils.tensor(tar_, self.cfg.device).reshape((len(tar_), 1))
        pred = self.val_net(self.rep_net(in_))[np.arange(len(act_)), act_].reshape((len(act_), 1))
        loss = self.vf_loss(pred, tar_)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    # No target net in MC
    def load_rep_fn(self, parameters_dir):
        path = os.path.join(self.cfg.data_root, parameters_dir)
        self.rep_net.load_state_dict(torch.load(path, map_location=self.cfg.device))
        self.cfg.logger.info("Load rep function from {}".format(path))

    # No target net in MC
    def load_val_fn(self, parameters_dir):
        path = os.path.join(self.cfg.data_root, parameters_dir)
        self.val_net.load_state_dict(torch.load(path, map_location=self.cfg.device))
        self.cfg.logger.info("Load value function from {}".format(path))

    
class MonteCarloOffline(MonteCarloAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.offline_param_init()
        
    def step(self):
        train_s, train_a, train_r, train_ns, train_t, train_na, train_data_return, _, _ = self.trainset
        idxs = self.agent_rng.randint(0, len(train_s), size=self.cfg.batch_size)
        in_ = torch_utils.tensor(self.cfg.state_normalizer(train_s[idxs]), self.cfg.device)
        act = train_a[idxs]
        tar_ = torch_utils.tensor(train_data_return[idxs], self.cfg.device)  # .reshape((len(idxs), 1))
        pred = self.val_net(self.rep_net(in_))[np.arange(len(act)), act]  # .reshape((len(act), 1))
        loss = self.vf_loss(pred, tar_)
        # print(self.cfg.state_normalizer(train_s[idxs]), train_s[idxs])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_loss.append(torch_utils.to_np(loss))
        self.update_stats(0, None)
        # if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
        #     self.trainset, self.testset = self.training_set_construction(self.cfg.offline_data)
        if self.total_steps // (len(train_s) / self.cfg.batch_size) > (self.total_steps - 1) // (len(train_s) / self.cfg.batch_size):
            self.trainset, self.testset = self.training_set_construction(self.cfg.offline_data)
        return

    # def epoch_step(self): # epoch
    #     train_s, train_a, train_r, train_ns, train_t, train_na, train_data_return, _, _ = self.trainset
    #     self.agent_rng.shuffle(self.training_indexs)
    #     ls_epoch = []
    #     for b in range(int(np.ceil(self.training_size / self.cfg.batch_size))):
    #         idxs = self.training_indexs[b*self.cfg.batch_size: (b+1)*self.cfg.batch_size]
    #         in_ = torch_utils.tensor(self.cfg.state_normalizer(train_s[idxs]), self.cfg.device)
    #         act = train_a[idxs]
    #         tar_ = torch_utils.tensor(train_data_return[idxs], self.cfg.device)#.reshape((len(idxs), 1))
    #         pred = self.val_net(self.rep_net(in_))[np.arange(len(act)), act]#.reshape((len(act), 1))
    #         loss = self.vf_loss(pred, tar_)
    #
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #         ls_epoch.append(torch_utils.to_np(loss))
    #     self.training_loss.append(np.array(ls_epoch).mean())
    #     self.update_stats(0, None)
    #
    #     if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
    #         self.trainset, self.testset = self.training_set_construction(self.cfg.offline_data)
    #     return self.test_fn()
    
    # def test_fn(self):
    #     test_s, test_a, _, _, _, _, test_data_return, _, _ = self.testset
    #     with torch.no_grad():
    #         test_in_ = torch_utils.tensor(self.cfg.state_normalizer(test_s), self.cfg.device)
    #         test_tar_ = torch_utils.tensor(test_data_return, self.cfg.device)  # .reshape((len(test_data_return), 1))
    #         tpred = self.val_net(self.rep_net(test_in_))[np.arange(len(test_a)), test_a]  # .reshape((len(test_a), 1))
    #         tloss = torch_utils.to_np(self.vf_loss(tpred, test_tar_))
    #     if tloss - self.tloss_rec > 0:
    #         self.tloss_increase += 1
    #     else:
    #         self.tloss_increase = 0
    #     self.tloss_rec = tloss
    #     self.test_loss.append(tloss)
    #     if self.tloss_increase > self.cfg.early_cut_threshold:
    #         return "EarlyCutOff"
