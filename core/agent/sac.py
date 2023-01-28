import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple
from core.agent import base
from core.utils import torch_utils

device = torch.device("cpu")

# def weights_init_(m):
#     if isinstance(m, torch.nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight, gain=1)
#         torch.nn.init.constant_(m.bias, 0)
# LOG_SIG_MAX = 2
# LOG_SIG_MIN = -20
# epsilon = 1e-6
# class GaussianPolicy(torch.nn.Module):
#     def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
#         super(GaussianPolicy, self).__init__()
#
#         self.linear1 = torch.nn.Linear(num_inputs, hidden_dim)
#         self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
#
#         self.mean_linear = torch.nn.Linear(hidden_dim, num_actions)
#         self.log_std_linear = torch.nn.Linear(hidden_dim, num_actions)
#
#         self.apply(weights_init_)
#
#         # action rescaling
#         if action_space is None:
#             self.action_scale = torch.tensor(1.)
#             self.action_bias = torch.tensor(0.)
#         else:
#             self.action_scale = torch.FloatTensor(
#                 (action_space.high - action_space.low) / 2.)
#             self.action_bias = torch.FloatTensor(
#                 (action_space.high + action_space.low) / 2.)
#
#     def forward(self, state):
#         x = F.relu(self.linear1(state))
#         x = F.relu(self.linear2(x))
#         mean = self.mean_linear(x)
#         log_std = self.log_std_linear(x)
#         log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
#         return mean, log_std
#
#     def sample(self, state):
#         mean, log_std = self.forward(state)
#         std = log_std.exp()
#         normal = torch.distributions.Normal(mean, std)
#         x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
#         y_t = torch.tanh(x_t)
#         action = y_t * self.action_scale + self.action_bias
#         log_prob = normal.log_prob(x_t)
#         # Enforcing Action Bound
#         log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
#         # log_prob = log_prob.sum(1, keepdim=True)
#         log_prob = log_prob.sum(1, keepdim=False)
#         mean = torch.tanh(mean) * self.action_scale + self.action_bias
#         return action, log_prob, mean
#
#     def to(self, device):
#         self.action_scale = self.action_scale.to(device)
#         self.action_bias = self.action_bias.to(device)
#         return super(GaussianPolicy, self).to(device)
    
class SAC(base.ActorCritic):
    def __init__(self, cfg):
        super(SAC, self).__init__(cfg)
        self.gamma = cfg.discount
        # self.tau = args.tau
        self.alpha = cfg.ac_alpha

        # self.policy_type = cfg.policy
        # self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = False
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(cfg.action_dim).to(self.cfg.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.cfg.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.learning_rate)
        else:
            self.log_alpha = torch_utils.tensor(np.log(cfg.ac_alpha), self.cfg.device)

        self.device = cfg.device
        # self.policy_net = GaussianPolicy(cfg.policy_fn_config["in_dim"][0], cfg.action_dim, cfg.policy_fn_config["hidden_units"][0]).to(self.device)
        # self.policy_optim = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        
        # self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        # self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        # self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        # hard_update(self.critic_target, self.critic)

        # if self.policy_type == "Gaussian":
        #     # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        #     if self.automatic_entropy_tuning is True:
        #         self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
        #         self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        #         self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
        #
        #     self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        #     self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        #
        # else:
        #     self.alpha = 0
        #     self.automatic_entropy_tuning = False
        #     self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        #     self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    # def select_action(self, state, evaluate=False):
    #     state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
    #     if evaluate is False:
    #         action, _, _ = self.policy.sample(state)
    #     else:
    #         _, _, action = self.policy.sample(state)
    #     return action.detach().cpu().numpy()[0]

    def update(self, data):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data['reward'], data['obs2'], 1-data['done']

        # state_batch = torch.FloatTensor(state_batch).to(self.device)
        # next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # action_batch = torch.FloatTensor(action_batch).to(self.device)
        # reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        # mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            # next_state_action, next_state_log_pi, _ = self.policy_net.sample(next_state_batch)
            next_state_action, next_state_log_pi = self.ac.pi(next_state_batch)
            
            # qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            # min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            min_qf_next_target, qf1_next_target, qf2_next_target = self.get_q_value_target(next_state_batch, next_state_action)
            min_qf_next_target -= self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        # qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        _, qf1, qf2 = self.get_q_value(state_batch, action_batch, with_grad=True)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # pi, log_pi, _ = self.policy_net.sample(state_batch)
        pi, log_pi = self.ac.pi(state_batch)

        # qf1_pi, qf2_pi = self.critic(state_batch, pi)
        # min_qf_pi = torch.min(qf1_pi, qf2_pi)
        min_qf_pi, _, _ = self.get_q_value(state_batch, pi, with_grad=True)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        # self.policy_optim.zero_grad()
        # policy_loss.backward()
        # self.policy_optim.step()
        self.pi_optimizer.zero_grad()
        policy_loss.backward()
        self.pi_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
        #     alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        # else:
        #     alpha_loss = torch.tensor(0.).to(self.device)
        #     alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        # if updates % self.target_update_interval == 0:
        #     soft_update(self.critic_target, self.critic, self.tau)
        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.sync_target()

        # return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

        
# class SAC(base.ActorCritic):
#
#     def __init__(self, cfg):
#         super(SAC, self).__init__(cfg)
#
#         # SAC online fills offline data to buffer
#         if cfg.agent_name == 'SAC' and cfg.load_offline_data:
#             self.fill_offline_data_to_buffer()
#
#         self.fixed_alpha = True
#         self.log_alpha = torch_utils.tensor(np.log(cfg.ac_alpha), self.cfg.device)
#         # self.log_alpha = torch_utils.tensor(np.log(0.1), self.cfg.device)
#         if self.fixed_alpha:
#             self.log_alpha.requires_grad = False
#         else:
#             self.log_alpha.requires_grad = True
#             self.alpha_optimizer = cfg.alpha_optimizer_fn([self.log_alpha])
#
#         self.target_entropy = -1*self.cfg.action_dim
#
#     # Set up function for computing SAC Q-losses
#     def compute_loss_q(self, data):
#         o, a, r, o2, d = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
#
#         # q1, q2 = self.ac.q1q2(o)
#         # q1, q2 = q1[np.arange(len(a)), a], q2[np.arange(len(a)), a]
#         _, q1, q2 = self.get_q_value(o, a, with_grad=True)
#
#         # Bellman backup for Q functions
#         with torch.no_grad():
#             # Target actions come from *current* policy
#             a2, logp_a2 = self.ac.pi(o2)
#
#             # Target Q-values
#             # q1_pi_targ, q2_pi_targ = self.ac_targ.q1q2(o2)
#             # q1_pi_targ, q2_pi_targ = q1_pi_targ[np.arange(len(a2)), a2], q2_pi_targ[np.arange(len(a2)), a2]
#             # q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
#             q_pi_targ, _, _ = self.get_q_value_target(o2, a2)
#             alpha = self.log_alpha.exp().item()
#             backup = r + self.gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)
#         # MSE loss against Bellman backup
#         loss_q1 = ((q1 - backup) ** 2).mean()
#         loss_q2 = ((q2 - backup) ** 2).mean()
#         loss_q = loss_q1 + loss_q2
#
#         # Useful info for logging
#         q_info = dict(Q1Vals=q1.detach().numpy(),
#                       Q2Vals=q2.detach().numpy())
#
#         return loss_q, q_info
#
#     # Set up function for computing SAC pi loss
#     def compute_loss_pi(self, data):
#         o = data['obs']
#         pi, logp_pi = self.ac.pi(o)
#         with torch.no_grad():
#             # q1_pi, q2_pi = self.ac.q1q2(o)
#             # q1_pi, q2_pi = q1_pi[np.arange(len(pi)), pi], q2_pi[np.arange(len(pi)), pi]
#             alpha = self.log_alpha.exp().item()
#         q_pi, _, _ = self.get_q_value(o, pi, with_grad=False)
#         loss_pi = (alpha * logp_pi - q_pi).mean()
#         return loss_pi, logp_pi
#
#     def update(self, data):
#         # if self.total_steps % self.cfg.update_freq == 0:
#         #     for i in range(self.cfg.update_freq):
#         #         data = self.get_data()
#
#         self.q_optimizer.zero_grad()
#         loss_q, q_info = self.compute_loss_q(data)
#         loss_q.backward()
#         self.q_optimizer.step()
#
#         self.pi_optimizer.zero_grad()
#         loss_pi, log_prob = self.compute_loss_pi(data)
#         loss_pi.backward()
#         self.pi_optimizer.step()
#
#         if not self.fixed_alpha:
#             self.alpha_optimizer.zero_grad()
#             # alpha = self.log_alpha.exp()
#             # alpha_loss = (alpha * (-log_prob - self.target_entropy).detach()).mean()
#             alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
#             alpha_loss.backward()
#             self.alpha_optimizer.step()
#
#         if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
#             self.sync_target()


class SACOffline(SAC):
    
    def __init__(self, cfg):
        super(SACOffline, self).__init__(cfg)
        self.offline_param_init()
    
    def get_data(self):
        return self.get_offline_data()

    def feed_data(self):
        self.update_stats(0, None)
        return
