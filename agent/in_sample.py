import copy

from agent import base
from utils import torch_utils

import os
import numpy as np
import torch
from collections import namedtuple

class InSampleAC(base.Agent):
    def __init__(self,
                 project_root,
                 log_dir,
                 logger,
                 device,
                 id_,
                 env,
                 eval_env,
                 policy_fn,
                 critic_fn,
                 state_value_fn,
                 learning_rate=3e-4,
                 temperature=0.1,
                 load_actor_fn_path=None,
                 load_critic_fn_path=None,
                 load_val_fn_path=None,
                 load_offline_data=True,
                 offline_data=None,
                 continuous_action=True,
                 offline_setting=True
                 ):
        super(InSampleAC, self).__init__(
            project_root=project_root,
            log_dir=log_dir,
            logger=logger,
            env=env,
            eval_env=eval_env,
            id_=id_,
            device=device,
            offline_data=offline_data,
            load_offline_data=load_offline_data,
            offline_setting=offline_setting
        )
        q1q2 = critic_fn
        pi = policy_fn
        self.value_net = state_value_fn
        self.beh_pi = policy_fn

        AC = namedtuple('AC', ['q1q2', 'pi'])
        self.ac = AC(q1q2=q1q2, pi=pi)

        q1q2_target = copy.deepcopy(critic_fn)
        pi_target = copy.deepcopy(policy_fn)
        q1q2_target.load_state_dict(q1q2.state_dict())
        pi_target.load_state_dict(pi.state_dict())
        ACTarg = namedtuple('ACTarg', ['q1q2', 'pi'])
        self.ac_targ = ACTarg(q1q2=q1q2_target, pi=pi_target)
        self.ac_targ.q1q2.load_state_dict(self.ac.q1q2.state_dict())
        self.ac_targ.pi.load_state_dict(self.ac.pi.state_dict())

        self.pi_optimizer = torch.optim.Adam(list(self.ac.pi.parameters()), learning_rate)
        self.q_optimizer = torch.optim.Adam(list(self.ac.q1q2.parameters()), learning_rate)
        self.value_optimizer = torch.optim.Adam(list(self.value_net.parameters()), learning_rate)
        self.beh_pi_optimizer = torch.optim.Adam(list(self.beh_pi.parameters()), learning_rate)

        if load_actor_fn_path is not None:
            self.load_actor_fn(load_actor_fn_path)
        if load_critic_fn_path:
            self.load_critic_fn(load_critic_fn_path)
        if load_val_fn_path is not None:
            self.load_state_value_fn(load_val_fn_path)

        if continuous_action:
            self.get_q_value = self.get_q_value_cont
            self.get_q_value_target = self.get_q_value_target_cont
        else:
            self.get_q_value = self.get_q_value_discrete
            self.get_q_value_target = self.get_q_value_target_discrete

        if load_offline_data:
            self.fill_offline_data_to_buffer()

        self.tau = temperature
        
        if offline_setting:
            self.offline_param_init()
            self.feed_data = self.feed_data_offline
        else:
            self.feed_data = self.feed_data_online

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
        target = min_Q - self.tau * log_probs
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

        clipped = torch.clip(torch.exp((min_Q - value) / self.tau - beh_log_prob), self.eps, 10000)
        pi_loss = -(clipped * log_probs).mean()
        return pi_loss, ""
    
    def update(self, data):
        loss_beh_pi, _ = self.compute_loss_beh_pi(data)
        self.beh_pi_optimizer.zero_grad()
        loss_beh_pi.backward()
        self.beh_pi_optimizer.step()
        loss_beta = loss_beh_pi.item()
        
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
        
        if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
            self.sync_target()

        return {"beta": loss_beta,
                "actor": loss_pi.item(),
                "critic": loss_q.item(),
                "value": loss_vs.item(),
                "q_info": qinfo.mean(),
                "v_info": v_info.mean(),
                "logp_info": logp_info.mean(),
                }

    def save(self):
        parameters_dir = self.log_dir + "/parameters"
        if not os.path.exists(parameters_dir): os.makedirs(parameters_dir)
        path = os.path.join(parameters_dir, "actor_net")
        torch.save(self.ac.pi.state_dict(), path)
        path = os.path.join(parameters_dir, "critic_net")
        torch.save(self.ac.q1q2.state_dict(), path)
        path = os.path.join(parameters_dir, "vs_net")
        torch.save(self.value_net.state_dict(), path)

    def feed_data_offline(self):
        self.update_stats(0, None)
        return

    def feed_data_online(self):
        if self.reset is True:
            self.state = self.env.reset()
            self.reset = False
        action = self.policy(self.state, eval=False)
        next_state, reward, done, _ = self.env.step([action])
        self.replay.feed([self.state, action, reward, next_state, int(done)])
        prev_state = self.state
        self.state = next_state
        self.update_stats(reward, done)
        return prev_state, action, reward, next_state, int(done)

    def load_actor_fn(self, parameters_dir):
        path = os.path.join(self.project_root, parameters_dir)
        self.ac.pi.load_state_dict(torch.load(path, map_location=self.device))
        self.ac_targ.pi.load_state_dict(self.ac.pi.state_dict())
        self.logger.info("Load actor function from {}".format(path))

    def load_critic_fn(self, parameters_dir):
        path = os.path.join(self.project_root, parameters_dir)
        self.ac.q1q2.load_state_dict(torch.load(path, map_location=self.device))
        self.ac_targ.q1q2.load_state_dict(self.ac.q1q2.state_dict())
        self.logger.info("Load critic function from {}".format(path))

    def load_state_value_fn(self, parameters_dir):
        path = os.path.join(self.project_root, parameters_dir)
        self.value_net.load_state_dict(torch.load(path, map_location=self.device))
        self.logger.info("Load state value function from {}".format(path))

    def policy(self, o, eval=False):
        o = torch_utils.tensor(self.state_normalizer(o), self.device)
        with torch.no_grad():
            a, _ = self.ac.pi(o, deterministic=eval)
        a = torch_utils.to_np(a)
        return a

    def eval_step(self, state):
        a = self.policy(state, eval=True)
        return a

    def sync_target(self):
        with torch.no_grad():
            for p, p_targ in zip(self.ac.q1q2.parameters(), self.ac_targ.q1q2.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_q_value_discrete(self, o, a, with_grad=False):
        if with_grad:
            q1_pi, q2_pi = self.ac.q1q2(o)
            q1_pi, q2_pi = q1_pi[np.arange(len(a)), a], q2_pi[np.arange(len(a)), a]
            q_pi = torch.min(q1_pi, q2_pi)
        else:
            with torch.no_grad():
                q1_pi, q2_pi = self.ac.q1q2(o)
                q1_pi, q2_pi = q1_pi[np.arange(len(a)), a], q2_pi[np.arange(len(a)), a]
                q_pi = torch.min(q1_pi, q2_pi)
        return q_pi.squeeze(-1), q1_pi.squeeze(-1), q2_pi.squeeze(-1)

    def get_q_value_target_discrete(self, o, a):
        with torch.no_grad():
            q1_pi, q2_pi = self.ac_targ.q1q2(o)
            q1_pi, q2_pi = q1_pi[np.arange(len(a)), a], q2_pi[np.arange(len(a)), a]
            q_pi = torch.min(q1_pi, q2_pi)
        return q_pi.squeeze(-1), q1_pi.squeeze(-1), q2_pi.squeeze(-1)

    def get_q_value_cont(self, o, a, with_grad=False):
        if with_grad:
            q1_pi, q2_pi = self.ac.q1q2(o, a)
            q_pi = torch.min(q1_pi, q2_pi)
        else:
            with torch.no_grad():
                q1_pi, q2_pi = self.ac.q1q2(o, a)
                q_pi = torch.min(q1_pi, q2_pi)
        return q_pi.squeeze(-1), q1_pi.squeeze(-1), q2_pi.squeeze(-1)

    def get_q_value_target_cont(self, o, a):
        with torch.no_grad():
            q1_pi, q2_pi = self.ac_targ.q1q2(o, a)
            q_pi = torch.min(q1_pi, q2_pi)
        return q_pi.squeeze(-1), q1_pi.squeeze(-1), q2_pi.squeeze(-1)

