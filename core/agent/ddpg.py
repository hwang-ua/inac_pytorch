import os
from collections import namedtuple
import numpy as np
import torch

from core.agent import base
from core.utils import torch_utils
# from core.utils.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

"""
Not Used / Not Checked
"""
class DDPGAgent(base.Agent):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        torch_utils.random_seed(cfg.run)
        
        actor_net = cfg.actor_fn()
        actor_params = list(actor_net.parameters())
        actor_optimizer = cfg.actor_optimizer_fn(actor_params)
        
        critic_net = cfg.critic_fn()
        critic_params = list(critic_net.parameters())
        critic_optimizer = cfg.critic_optimizer_fn(critic_params)
        
        actor_net_target = cfg.actor_fn()
        actor_net_target.load_state_dict(actor_net.state_dict())
        critic_net_target = cfg.critic_fn()
        critic_net_target.load_state_dict(critic_net.state_dict())
        
        TargetNets = namedtuple('TargetNets', ['actor_net', 'critic_net'])
        targets = TargetNets(actor_net=actor_net_target, critic_net=critic_net_target)
        
        Optimizers = namedtuple('Optimizers', ['actor_net', 'critic_net'])
        optimizers = Optimizers(actor_net=actor_optimizer, critic_net=critic_optimizer)
        
        # self.actor_std = cfg.actor_std
        # self.ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.cfg.env_fn().action_dim))
        self.noise = cfg.noise_fn()
        self.tau = cfg.tau
        
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.targets = targets
        self.optimizers = optimizers
        
        # self.actor_loss = cfg.actor_loss_fn()
        self.critic_loss = cfg.critic_loss_fn()
        
        self.env = cfg.env_cls
        # self.env.__init__(self.cfg)
        self.replay = cfg.replay_fn()
        
        self.state = None
        self.action = None
        self.next_state = None
        
        if self.cfg.load_nn:
            self.cfg.logger.info("Loading weight from " + self.cfg.nn_config["load_path"].format(self.cfg.run))
            self.load("data/output/" + self.cfg.nn_config["load_path"].format(self.cfg.run))
        
        if self.cfg.train_nn:
            self.cfg.logger.info("Filling buffer with {} random trajectories.".format(self.cfg.ep_random))
            self.random_fill_buffer()
    
    def step(self, random=False):
        if self.reset is True:
            # self.ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.cfg.env_fn().action_dim),
            #                                              sigma=self.cfg.ou_sigma,
            #                                              theta=self.cfg.ou_theta)
            self.noise.reset()
            self.state = self.env.reset()
            self.cfg.state_normalizer.update(self.state)
            self.reset = False
        
        action, _ = self.policy(self.state, random=random)
        # print("calling", end=" - ")
        next_state, reward, done, _ = self.env.step(action)
        self.cfg.state_normalizer.update(next_state)
        
        mask = 0 if self.ep_steps + 1 == self.env.env._max_episode_steps else float(done)
        self.replay.feed([self.state, action, reward, next_state, mask])
        last_state = self.state
        self.state = next_state
        
        self.update_stats(reward, done)
        if self.cfg.train_nn and self.total_steps % self.cfg.nn_update["learn_freq"] == 0:
            # for _ in range(self.cfg.epoch): self.update()
            self.epoch_update()
        
        return last_state, action, reward, next_state, int(done)
    
    def epoch_update(self):
        for _ in range(self.cfg.nn_update["epoch"]): self.update()
    
    def policy(self, state, random=False):
        if random:
            # action = self.env.env.action_space.sample()
            action = self.env.sample_action()
            return action, action
        
        with torch.no_grad():
            mu = torch_utils.to_np(self.actor_net(self.cfg.state_normalizer(state)))
        # noise = self.agent_rng.normal(loc=0.0, scale=self.cfg.actor_std, size=self.env.action_dim)
        # noise = self.ou_noise()
        # noise = 0
        ns = self.noise()
        action = mu + ns
        # print(mu, noise)
        action = np.clip(action, -1, 1)
        return action, mu
    
    def update(self):
        states, actions, rewards, next_states, terminals = self.replay.sample()
        states = self.cfg.state_normalizer(states)
        rewards = self.cfg.reward_normalizer(rewards)
        next_states = self.cfg.state_normalizer(next_states)
        actions = torch_utils.tensor(actions, self.cfg.device).long()
        
        with torch.no_grad():
            target_mu = self.targets.actor_net(next_states)
            target_q = self.targets.critic_net(next_states, target_mu)
            terminals = torch_utils.tensor(terminals, self.cfg.device).view((-1, 1))
            rewards = torch_utils.tensor(rewards, self.cfg.device).view((-1, 1))
            target = self.cfg.discount * target_q * (1 - terminals).float()
            target.add_(rewards.float())
        
        self.optimizers.critic_net.zero_grad()
        q = self.critic_net(states, actions)
        critic_ls = self.critic_loss(q, target)
        critic_ls.backward()
        self.optimizers.critic_net.step()
        
        # From openAI baseline implementation (https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py)
        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.critic_net.parameters():
            p.requires_grad = False
        
        self.optimizers.actor_net.zero_grad()
        current_mu = self.actor_net(states)
        current_q = self.critic_net(states, current_mu)
        actor_ls = -1 * torch.mean(current_q)
        actor_ls.backward()
        self.optimizers.actor_net.step()
        
        # From openAI baseline implementation (https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py)
        # self.optimizers.critic_net.zero_grad()
        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.critic_net.parameters():
            p.requires_grad = True
        
        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('ddpg/loss/actor_loss', actor_ls.item(), self.total_steps)
            self.cfg.logger.tensorboard_writer.add_scalar('ddpg/loss/critic_loss', critic_ls.item(), self.total_steps)
        
        self.soft_update(self.targets.actor_net, self.actor_net, self.tau)
        self.soft_update(self.targets.critic_net, self.critic_net, self.tau)
        return
    
    def eval_step(self, state):
        with torch.no_grad():
            action = torch_utils.to_np(self.actor_net(self.cfg.state_normalizer(state)))
        action = np.clip(action, -1, 1)
        return action
    
    def save(self):
        parameters_dir = self.cfg.get_parameters_dir()
        path = os.path.join(parameters_dir, "actor_net")
        torch.save(self.actor_net.state_dict(), path)
        
        path = os.path.join(parameters_dir, "critic_net")
        torch.save(self.critic_net.state_dict(), path)
    
    def load(self, parameters_dir):
        path = os.path.join(parameters_dir, "actor_net")
        self.actor_net.load_state_dict(torch.load(path))
        
        path = os.path.join(parameters_dir, "critic_net")
        self.critic_net.load_state_dict(torch.load(path))
        
        self.targets.actor_net.load_state_dict(self.actor_net.state_dict())
        self.targets.critic_net.load_state_dict(self.critic_net.state_dict())