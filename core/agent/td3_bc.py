import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.agent import base
from core.utils import torch_utils
"""
https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py
"""

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3BC(base.ActorCritic):
    def __init__(self, cfg):
        super(TD3BC, self).__init__(cfg)

    # def __init__(
    #     self,
    #     state_dim,
    #     action_dim,
    #     max_action,
    #     discount=0.99,
    #     tau=0.005,
    #     policy_noise=0.2,
    #     noise_clip=0.5,
    #     policy_freq=2,
    #     alpha=2.5,
    # ):
        
        # self.actor = self.ac.pi #Actor(state_dim, action_dim, max_action).to(device)
        # self.actor_target = copy.deepcopy(self.actor)
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        # self.critic = Critic(self.cfg.state_dim, self.cfg.action_dim).to(device)
        # self.critic_target = copy.deepcopy(self.critic)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # self.max_action = max_action
        self.discount = self.cfg.discount
        # self.tau = self.cfg.polyak
        # self.policy_noise = policy_noise
        # self.noise_clip = noise_clip
        self.policy_freq = self.cfg.policy_freq
        self.alpha = self.cfg.alpha
        
        # self.total_it = 0
        self.offline_param_init()

    def get_data(self):
        return self.get_offline_data()

    def feed_data(self):
        self.update_stats(0, None)
        return

    # def select_action(self, state):
    #     state = torch.FloatTensor(state.reshape(1, -1)).to(device)
    #     return self.actor(state).cpu().data.numpy().flatten()

    def update(self, data):
    # def train(self, replay_buffer, batch_size=256):
    #     self.total_it += 1
        
        # # Sample replay buffer
        # state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        state, action, reward, next_state, not_done = data['obs'], data['act'], data['reward'], data['obs2'], 1.0 - data['done']
        action = torch_utils.tensor(action, self.cfg.device)
        with torch.no_grad():
            # # Select action according to policy and add clipped noise
            # noise = (
            #     torch.randn_like(action) * self.policy_noise
            # ).clamp(-self.noise_clip, self.noise_clip)
            #
            # next_action = (
            #     self.actor_target(next_state) + noise
            # ).clamp(-self.max_action, self.max_action)
            next_action, _ = self.ac.pi(next_state)
            
            # # Compute the target Q value
            # target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            # target_Q = torch.min(target_Q1, target_Q2)
            target_Q, _, _ = self.get_q_value_target(next_state, next_action)
            target_Q = reward + not_done * self.discount * target_Q
        
        # Get current Q estimates
        # current_Q1, current_Q2 = self.critic(state, action)
        _, current_Q1, current_Q2 = self.get_q_value(state, action, with_grad=True)
        
        # Compute critic loss
        critic_loss = (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
        
        # Optimize the critic
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()
        
        # Delayed policy updates
        # if self.total_it % self.policy_freq == 0:
        if self.total_steps % self.policy_freq == 0:
            
            # Compute actor loss
            # pi = self.actor(state)
            # Q = self.critic.Q1(state, pi)
            pi, _ = self.ac.pi(state)
            Q, _, _ = self.get_q_value(state, pi, with_grad=False)
            lmbda = self.alpha / Q.abs().mean().detach()
            actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)

            # Optimize the actor
            # self.actor_optimizer.zero_grad()
            # actor_loss.backward()
            # self.actor_optimizer.step()
            self.pi_optimizer.zero_grad()
            actor_loss.backward()
            self.pi_optimizer.step()

            # Update the frozen target models
            # for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            #
            # for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
                self.sync_target()


class TD3BCOnline(TD3BC):
    def __init__(self, cfg):
        super(TD3BCOnline, self).__init__(cfg)
        if cfg.agent_name == 'IQLOnline' and cfg.load_offline_data:
            self.fill_offline_data_to_buffer()

    def get_data(self):
        return base.ActorCritic.get_data(self)

    def feed_data(self):
        return base.ActorCritic.feed_data(self)

