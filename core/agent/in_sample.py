from core.agent import base

import os
import torch

class InSampleACOnline(base.ActorCritic):
    def __init__(self, cfg):
        super(InSampleACOnline, self).__init__(cfg)
        if cfg.agent_name == 'InSampleACOnline' and cfg.load_offline_data:
            self.fill_offline_data_to_buffer()
        if 'load_params' in self.cfg.val_fn_config and self.cfg.val_fn_config['load_params']:
            self.load_state_value_fn(cfg.val_fn_config['path'])
        
        self.tau = cfg.tau
        self.value_net = cfg.state_value_fn()
        self.value_optimizer = torch.optim.Adam(list(self.value_net.parameters()), cfg.learning_rate)
        self.beh_pi = cfg.policy_fn()
        self.beh_pi_optimizer = torch.optim.Adam(list(self.beh_pi.parameters()), cfg.learning_rate)

        self.clip_grad_param = cfg.clip_grad_param
        self.exp_threshold = cfg.exp_threshold
        
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

    def get_data(self):
        return self.get_offline_data()

    def feed_data(self):
        self.update_stats(0, None)
        return
