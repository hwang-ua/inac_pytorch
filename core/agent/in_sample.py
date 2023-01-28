import copy
import numpy as np
from sklearn.mixture import GaussianMixture
from core.agent import base
import core.network.net_factory as network
import core.network.activations as activations
import core.utils.torch_utils as torch_utils

import os
import torch

class InSampleACOnline(base.ActorCritic):
    def __init__(self, cfg):
        super(InSampleACOnline, self).__init__(cfg)
        if cfg.agent_name == 'InSampleACOnline' and cfg.load_offline_data:
            self.fill_offline_data_to_buffer()
        
        self.tau = cfg.tau
        self.value_net = cfg.state_value_fn()
        if 'load_params' in self.cfg.val_fn_config and self.cfg.val_fn_config['load_params']:
            self.load_state_value_fn(cfg.val_fn_config['path'])

        self.value_optimizer = cfg.vs_optimizer_fn(list(self.value_net.parameters()))
        self.beh_pi = cfg.policy_fn()
        self.beh_pi_optimizer = cfg.policy_optimizer_fn(list(self.beh_pi.parameters()))

        self.clip_grad_param = cfg.clip_grad_param
        self.exp_threshold = cfg.exp_threshold
        # self.beta_threshold = 1e-3
        
    def compute_loss_beh_pi(self, data):
        """L_{\omega}, learn behavior policy"""
        states, actions = data['obs'], data['act']
        beh_log_probs = self.beh_pi.get_logprob(states, actions)
        beh_loss = -beh_log_probs.mean()
        return beh_loss, beh_log_probs
    
    def compute_loss_value(self, data):
        """L_{\phi}, learn z for state value, v = tau log z"""
        states = data['obs']
        v_phi = self.value_net(states).squeeze(-1)
        with torch.no_grad():
            actions, log_probs = self.ac.pi(states)
            min_Q, _, _ = self.get_q_value_target(states, actions)
            # beh_log_prob = self.beh_pi.get_logprob(states, actions)
            # beh_log_prob = self.ac.pi.get_logprob(states, actions)
        target = min_Q - self.tau * log_probs#beh_log_prob
        value_loss = (0.5 * (v_phi - target) ** 2).mean()
        return value_loss, v_phi.detach().numpy(), log_probs.detach().numpy()
    
    def get_state_value(self, state):
        with torch.no_grad():
            value = self.value_net(state).squeeze(-1)
        return value

    def compute_loss_q(self, data):
        states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
        with torch.no_grad():
            next_actions, log_probs = self.ac.pi(next_states)
        min_Q, _, _ = self.get_q_value_target(next_states, next_actions)
        q_target = rewards + self.gamma * (1 - dones) * (min_Q - self.tau * log_probs)
    
        minq, q1, q2 = self.get_q_value(states, actions, with_grad=True)
    
        critic1_loss = (0.5 * (q_target - q1) ** 2).mean()
        critic2_loss = (0.5 * (q_target - q2) ** 2).mean()
        loss_q = (critic1_loss + critic2_loss) * 0.5
        # q_info = dict(Q1Vals=q1.detach().numpy(),
        #               Q2Vals=q2.detach().numpy())
        q_info = minq.detach().numpy()
        return loss_q, q_info

    def compute_loss_pi(self, data):
        """L_{\psi}, extract learned policy"""
        states, actions = data['obs'], data['act']

        log_probs = self.ac.pi.get_logprob(states, actions)
        min_Q, _, _ = self.get_q_value(states, actions, with_grad=False)
        # min_Q, _, _ = self.get_q_value_target(states, actions)
        with torch.no_grad():
            value = self.get_state_value(states)
            beh_log_prob = self.beh_pi.get_logprob(states, actions)

        clipped = torch.clip(torch.exp((min_Q - value) / self.tau - beh_log_prob), self.eps, self.exp_threshold)
        pi_loss = -(clipped * log_probs).mean()
        return pi_loss, ""
    
    def update_beta(self, data):
        loss_beh_pi, _ = self.compute_loss_beh_pi(data)
        self.beh_pi_optimizer.zero_grad()
        loss_beh_pi.backward()
        self.beh_pi_optimizer.step()
        # print(loss_beh_pi)
        return loss_beh_pi

    def update(self, data):
        if not self.cfg.pretrain_beta:
            loss_beta = self.update_beta(data).item()
        else:
            loss_beta = None
        
        self.value_optimizer.zero_grad()
        loss_vs, v_info, logp_info = self.compute_loss_value(data)
        loss_vs.backward()
        self.value_optimizer.step()

        loss_q, qinfo = self.compute_loss_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        loss_pi, _ = self.compute_loss_pi(data)
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()
        
        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.sync_target()

        return {"beta": loss_beta,
                "actor": loss_pi.item(),
                "critic": loss_q.item(),
                "value": loss_vs.item(),
                "q_info": qinfo.mean(),
                "v_info": v_info.mean(),
                "logp_info": logp_info.mean(),
                }


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


class InSampleAC(InSampleACOnline):
    def __init__(self, cfg):
        super(InSampleAC, self).__init__(cfg)
        self.offline_param_init()
        if self.cfg.pretrain_beta:
            for i in range(20000):
                # if i % 10000 == 0:
                #     print(i)
                data = self.get_offline_data()
                self.update_beta(data)

    def get_data(self):
        return self.get_offline_data()

    def feed_data(self):
        self.update_stats(0, None)
        return


class InSamplePiW(InSampleAC):
    """
    Sample from $\pi_\omega$
    """
    def __init__(self, cfg):
        super(InSamplePiW, self).__init__(cfg)

    def compute_loss_pi(self, data):
        """L_{\psi}, extract learned policy"""
        states = data['obs']
        with torch.no_grad():
            value = self.get_state_value(states)
            actions, beh_log_prob = self.beh_pi(states)
    
        log_probs = self.ac.pi.get_logprob(states, actions)
        min_Q, _, _ = self.get_q_value(states, actions, with_grad=False)
    
        clipped = torch.clip(torch.exp((min_Q - value) / self.tau - beh_log_prob), self.eps, self.exp_threshold)
        pi_loss = -(clipped * log_probs).mean()
        return pi_loss, ""


class InSampleBeta(InSampleAC):
    """
    Sample from $true behavior policy$
    """
    def __init__(self, cfg):
        super(InSampleBeta, self).__init__(cfg)
        if type(cfg.true_beh_path) == dict:
            class TrueBehCfg:
                def __init__(self, **entries):
                    self.__dict__.update(entries)
            true_beh_cfg = TrueBehCfg()
            true_beh_cfg.device = cfg.device
            true_beh_cfg.action_dim = cfg.action_dim
            true_beh_cfg.rep_fn_config = cfg.true_beh_structure["rep_fn_config"]
            true_beh_cfg.activation_config = cfg.true_beh_structure["activation_config"]
            true_beh_cfg.val_fn_config = cfg.true_beh_structure["val_fn_config"]
            true_beh_cfg.rep_activation_fn = activations.ActvFactory.get_activation_fn(true_beh_cfg)
            true_beh_cfg.rep_fn = network.NetFactory.get_rep_fn(true_beh_cfg)
            self.true_rep_net = true_beh_cfg.rep_fn()
            rep_path = os.path.join(self.cfg.data_root, cfg.true_beh_path["rep"])
            self.true_rep_net.load_state_dict(torch.load(rep_path, map_location=self.cfg.device))
            true_beh_cfg.val_fn = network.NetFactory.get_val_fn(true_beh_cfg)
            self.true_val_net = true_beh_cfg.val_fn()
            val_path = os.path.join(self.cfg.data_root, cfg.true_beh_path["val"])
            self.true_val_net.load_state_dict(torch.load(val_path, map_location=self.cfg.device))
            self.true_beh = self.value2policy
            self.cfg.logger.info("Load true behavior policy from {}".format(val_path))
        else:
            path = os.path.join(self.cfg.data_root, cfg.true_beh_path)
            self.true_beh = cfg.policy_fn()
            self.true_beh.load_state_dict(torch.load(path, map_location=self.cfg.device))
            self.cfg.logger.info("Load true behavior policy from {}".format(path))

    def value2policy(self, states):
        with torch.no_grad():
            phi = self.true_rep_net(states)
            val = self.true_val_net(phi)
            actions = torch.argmax(val, dim=1)
        log_probs = torch.log(torch.ones(val.size(0)))
        return actions, log_probs
    
    def compute_loss_pi(self, data):
        """L_{\psi}, extract learned policy"""
        states = data['obs']
        with torch.no_grad():
            value = self.get_state_value(states)
            actions, beh_log_prob = self.true_beh(states)
    
        log_probs = self.ac.pi.get_logprob(states, actions)
        min_Q, _, _ = self.get_q_value(states, actions, with_grad=False)
        
        clipped = torch.clip(torch.exp((min_Q - value) / self.tau - beh_log_prob), self.eps, self.exp_threshold)
        pi_loss = -(clipped * log_probs).mean()
        return pi_loss, ""
    

class InSampleNoV(InSampleAC):
    """
    Sample from $\pi_\omega$
    """
    def __init__(self, cfg):
        super(InSampleNoV, self).__init__(cfg)
        delattr(self, "value_net")
        delattr(self, "value_optimizer")

    def get_state_value(self, state):
        with torch.no_grad():
            action, log_probs = self.ac.pi(state)
            min_Q, _, _ = self.get_q_value_target(state, action)
            value = min_Q - self.tau * log_probs
        return value

    def update(self, data):
        self.update_beta(data)
    
        loss_q, _ = self.compute_loss_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()
    
        loss_pi, _ = self.compute_loss_pi(data)
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()
    
        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.sync_target()
    
        return loss_pi.item(), loss_q.item()

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
    

class InSampleEmphatic(InSampleACOnline):
    def __init__(self, cfg):
        super(InSampleEmphatic, self).__init__(cfg)
        self.offline_param_init()
        if self.cfg.pretrain_beta:
            for i in range(20000):
                # if i % 10000 == 0:
                #     print(i)
                data = self.get_offline_data()
                self.update_beta(data)
        self.eta = 0.1
        self.i_fn = self.cfg.i_fn
        self.fhat = copy.deepcopy(self.value_net)
        self.fhat_optimizer = cfg.vs_optimizer_fn(list(self.fhat.parameters()))

    def get_data(self):
        return self.get_offline_data()

    def feed_data(self):
        self.update_stats(0, None)
        return

    def compute_loss_pi(self, data):
        states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
        if self.i_fn==0:
            i_t = torch.zeros(len(dones))
            i_tp1 = torch.zeros(len(dones)) + dones
        elif self.i_fn==1:
            i_t = torch.ones(len(dones))
            i_tp1 = torch.ones(len(dones))
        else:
            raise NotImplementedError
        log_probs = self.ac.pi.get_logprob(states, actions)
        with torch.no_grad():
            # pi_t, pi_log_probs = self.ac.pi(states)
            # mu_log_probs = self.beh_pi.get_logprob(states, pi_t)
            mu_log_probs = self.beh_pi.get_logprob(states, actions)
            rho_t = torch.clip(torch.exp(log_probs - mu_log_probs), -np.inf, 1)
            M_t = (1 - self.eta) * i_t + self.eta * self.fhat(states).squeeze(-1)

            # vs_t = self.value_net(states).squeeze(-1)
            # vs_tp1 = self.value_net(next_states).squeeze(-1)
            # # v_delta = (rewards - self.tau * pi_log_probs) + self.gamma * (1-dones) * vs_tp1 - vs_t
            # v_delta = rewards + self.gamma * (1-dones) * vs_tp1 - vs_t
            f_t = self.fhat(states).squeeze(-1)
            f_targ = i_tp1 + rho_t * self.gamma * (1-dones) * f_t

        # """L_{\psi}, extract learned policy"""
        # states, actions = data['obs'], data['act']
    
        min_Q, _, _ = self.get_q_value(states, actions, with_grad=False)
        with torch.no_grad():
            value = self.get_state_value(states)
            beh_log_prob = self.beh_pi.get_logprob(states, actions)
    
        clipped = torch.clip(torch.exp((min_Q - value) / self.tau - beh_log_prob), self.eps, self.exp_threshold)
        # pi_loss = -(clipped * log_probs).mean()
        # pi_loss = -(rho_t * M_t * clipped * log_probs * v_delta).mean()
        pi_loss = -(rho_t * M_t * clipped * log_probs).mean()
        
        f_tp1 = self.fhat(next_states).squeeze(-1)
        f_loss = (0.5 * (f_tp1 - f_targ) ** 2).mean()
        self.fhat_optimizer.zero_grad()
        f_loss.backward()
        self.fhat_optimizer.step()
        return pi_loss, ""


class InSampleEmphCritic(InSampleACOnline):
    def __init__(self, cfg):
        super(InSampleEmphCritic, self).__init__(cfg)
        self.offline_param_init()
        if self.cfg.pretrain_beta:
            for i in range(20000):
                # if i % 10000 == 0:
                #     print(i)
                data = self.get_offline_data()
                self.update_beta(data)

    def get_data(self):
        return self.get_offline_data()

    def feed_data(self):
        self.update_stats(0, None)
        return

    def compute_loss_q(self, data):
        states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
        with torch.no_grad():
            next_actions, log_probs = self.ac.pi(next_states)
        min_Q, _, _ = self.get_q_value_target(next_states, next_actions)
        q_target = rewards + self.gamma * (1 - dones) * (min_Q - self.tau * log_probs)
        minq, q1, q2 = self.get_q_value(states, actions, with_grad=True)
        
        with torch.no_grad():
            log_pi_phi = self.ac.pi.get_logprob(states, actions)
            log_pi_w = self.beh_pi.get_logprob(states, actions)
        ft = 1.0 / (1.0 - self.gamma*(1-dones) * np.clip(np.exp(log_pi_phi-log_pi_w), -np.inf, 1))
        critic1_loss = (0.5 * (q_target - q1) ** 2 * ft).mean()
        critic2_loss = (0.5 * (q_target - q2) ** 2 * ft).mean()
        loss_q = (critic1_loss + critic2_loss) * 0.5
        q_info = minq.detach().numpy()
        return loss_q, q_info
    # def compute_loss_q(self, data):
    #     states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
    #     with torch.no_grad():
    #         next_actions, log_probs = self.ac.pi(next_states)
    #     min_Q, _, _ = self.get_q_value_target(next_states, next_actions)
    #     q_target = rewards + self.gamma * (1 - dones) * (min_Q - self.tau * log_probs)
    #     minq, q1, q2 = self.get_q_value(states, actions, with_grad=True)
    #
    #     with torch.no_grad():
    #         log_pi_phi = self.ac.pi.get_logprob(states, actions)
    #         log_pi_w = self.beh_pi.get_logprob(states, actions)
    #     ft = 1.0 / (1.0 - self.gamma*(1-dones) * np.clip(np.exp(log_pi_phi-log_pi_w), -np.inf, 1))
    #     critic1_loss = (0.5 * (q_target - q1) ** 2 * ft).mean()
    #     critic2_loss = (0.5 * (q_target - q2) ** 2 * ft).mean()
    #     loss_q = (critic1_loss + critic2_loss) * 0.5
    #     q_info = minq.detach().numpy()
    #     return loss_q, q_info


class InSampleWeighted(InSampleACOnline):
    def __init__(self, cfg):
        super(InSampleWeighted, self).__init__(cfg)
        self.higher_priority_index = self.cfg.higher_priority_index
        self.higher_priority_prob = self.cfg.higher_priority_prob
        self.offline_param_init()
        if self.cfg.pretrain_beta:
            for i in range(20000):
                data = self.get_offline_data()
                self.update_beta(data)

    def get_data(self):
        return self.get_weighted_offline_data(self.higher_priority_index, self.higher_priority_prob)

    def feed_data(self):
        self.update_stats(0, None)
        return


class InSamplePC(InSampleACOnline):
    # prior correction
    def __init__(self, cfg):
        super(InSamplePC, self).__init__(cfg)
        self.offline_param_init()
        assert self.cfg.pretrain_beta == False

    def get_data(self):
        return self.get_offline_traj(traj_len=3)

    def feed_data(self):
        self.update_stats(0, None)
        return

    def compute_loss_q(self, data):
        states_trj, actions_trj, rewards_trj, next_states_trj, dones_trj = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
        with torch.no_grad():
            next_actions, log_probs = self.ac.pi(next_states_trj[0])
        min_Q, _, _ = self.get_q_value_target(next_states_trj[0], next_actions)
        q_target = rewards_trj[0] + self.gamma * (1 - dones_trj[0]) * (min_Q - self.tau * log_probs)
        minq, q1, q2 = self.get_q_value(states_trj[0], actions_trj[0], with_grad=True)
        
        weight = 1
        cutoff = []
        with torch.no_grad():
            for i in range(len(states_trj)):
                a, lp = self.ac.pi(states_trj[i])
                beh_lp = self.beh_pi.get_logprob(states_trj[i], a)
                rho_t = torch.clip(torch.exp(lp - beh_lp), -np.inf, 1)
                co = torch_utils.to_np(torch.where(dones_trj[i] == 1)[0])
                cutoff += list(co)
                rho_t[cutoff] = 1
                weight *= rho_t
        
        critic1_loss = (0.5 * (q_target - q1) ** 2 * weight).mean()
        critic2_loss = (0.5 * (q_target - q2) ** 2 * weight).mean()
        loss_q = (critic1_loss + critic2_loss) * 0.5
        q_info = minq.detach().numpy()
        return loss_q, q_info

    def update(self, data):
        data_t0 = {}
        for k in data.keys():
            data_t0[k] = data[k][0]
        
        loss_beta = self.update_beta(data_t0).item()
    
        self.value_optimizer.zero_grad()
        loss_vs, v_info, logp_info = self.compute_loss_value(data_t0)
        loss_vs.backward()
        self.value_optimizer.step()
    
        loss_q, qinfo = self.compute_loss_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()
    
        loss_pi, _ = self.compute_loss_pi(data_t0)
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()
    
        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.sync_target()


class InSampleUniform(InSampleACOnline):
    def __init__(self, cfg):
        super(InSampleUniform, self).__init__(cfg)
        self.offline_param_init()
        if self.cfg.pretrain_beta:
            for i in range(20000):
                data = self.get_offline_data()
                self.update_beta(data)
        self.probs = self.gaussian_mixer()

    def get_data(self):
        return self.get_uniform_offline_data(self.probs)

    def feed_data(self):
        self.update_stats(0, None)
        return

    def gaussian_mixer(self):
        states = self.trainset[0]
        n_components = self.cfg.gaussian_n_components
        clusters = GaussianMixture(n_components=n_components).fit_predict(states)
        probs = np.zeros(len(states))
        for c in range(n_components):
            same_cluster = np.where(clusters == c)[0]
            probs[same_cluster] = (1.0/n_components) / len(same_cluster)
        return probs