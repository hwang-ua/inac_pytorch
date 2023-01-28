import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from core.network import network_utils, network_bodies
from core.utils import torch_utils


# class LinearNetwork(nn.Module):
#     def __init__(self, device, input_units, output_units, init_type='uniform', bias=False, info=None):
#         super().__init__()
#
#         if init_type == 'xavier':
#             self.fc_head = network_utils.layer_init_xavier(nn.Linear(input_units, output_units, bias=bias), bias=bias)
#         elif init_type == 'uniform':
#             self.fc_head = network_utils.layer_init_uniform(nn.Linear(input_units, output_units, bias=bias), bias=bias)
#         elif init_type == "constant":
#             self.fc_head = network_utils.layer_init_constant(nn.Linear(input_units, output_units, bias=bias), const=info, bias=bias)
#         else:
#             raise ValueError('init_type is not defined: {}'.format(init_type))
#
#         self.to(device)
#         self.device = device
#
#     def forward(self, x):
#         if not isinstance(x, torch.Tensor): x = torch_utils.tensor(x, self.device)
#         if len(x.shape) > 2: x = x.view(x.shape[0], -1)
#         y = self.fc_head(x)
#         return y
#
#     def compute_lipschitz_upper(self):
#         return [np.linalg.norm(self.fc_head.weight.detach().cpu().numpy(), ord=2)]


class FCNetwork(nn.Module):
    def __init__(self, device, input_units, hidden_units, output_units, head_activation=lambda x:x):
        super().__init__()
        body = network_bodies.FCBody(device, input_units, hidden_units=tuple(hidden_units), init_type='xavier')
        self.body = body
        self.fc_head = network_utils.layer_init_xavier(nn.Linear(body.feature_dim, output_units, bias=True), bias=True)
        self.device = device
        self.head_activation = head_activation
        self.to(device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch_utils.tensor(x, self.device)
        if len(x.shape) > 2: x = x.view(x.shape[0], -1)
        y = self.body(x)
        y = self.fc_head(y)
        y = self.head_activation(y)
        return y

    def compute_lipschitz_upper(self):
        lips = self.body.compute_lipschitz_upper()
        lips.append(np.linalg.norm(self.fc_head.weight.detach().cpu().numpy(), ord=2))
        return lips

# class FCInsertInputDiscret(FCNetwork):
#     def __init__(self, device, input_units, hidden_units, output_units, head_activation=lambda x:x):
#         super().__init__(device, input_units, hidden_units, output_units, head_activation=head_activation)
#
#     def forward(self, x, a):
#         y = super().forward(x)
#         return y[a]
#
#
# class FCInsertInputNetwork(nn.Module):
#     def __init__(self, device, input_units, hidden_units, output_units, mid_ipt_dim, mid_ipt_pos, output_activation=None):
#         super().__init__()
#         if mid_ipt_pos > 0 :
#             body1 = network_bodies.FCBody(device, input_units, hidden_units=tuple(hidden_units[: mid_ipt_pos]))
#             body2 = network_bodies.FCBody(device, hidden_units[mid_ipt_pos - 1] + mid_ipt_dim, hidden_units=tuple(hidden_units[mid_ipt_pos:]))
#         else:
#             body1 = lambda x:x
#             body2 = network_bodies.FCBody(device, input_units + mid_ipt_dim, hidden_units=tuple(hidden_units))
#
#         # self.fc_head = network_utils.layer_init_uniform(nn.Linear(body2.feature_dim, output_units))
#         self.fc_head = network_utils.layer_init_xavier(nn.Linear(body2.feature_dim, output_units))
#         self.to(device)
#         self.output_units = output_units
#         self.mid_ipt_dim = mid_ipt_dim
#         self.mid_ipt_pos = mid_ipt_pos
#
#         self.device = device
#         self.before_insert = body1
#         self.after_insert = body2
#
#         if output_activation == "tanh":
#             self.head_activation = torch.tanh
#         elif output_activation == "relu":
#             self.head_activation = functional.relu
#         elif output_activation is None:
#             self.head_activation = lambda x: x
#
#     def forward(self, x, insertion):
#         if not isinstance(x, torch.Tensor): x = torch_utils.tensor(x, self.device)
#         if not isinstance(insertion, torch.Tensor): insertion = torch_utils.tensor(insertion, self.device)
#         if len(x.shape) > 2: x = x.view(x.shape[0], -1)
#
#         before_ist = self.before_insert(x)
#         with_ist = torch.cat((before_ist.float(), insertion.float()), 1)
#         phi = self.after_insert(with_ist)
#         y = self.head_activation(self.fc_head(phi))
#         return y

#
# class ConvNetwork(nn.Module):
#     def __init__(self, device, state_dim, output_units, architecture, head_activation=None):
#         super().__init__()
#
#         self.conv_body = network_bodies.ConvBody(device, state_dim, architecture)
#         if "fc_layers" in architecture:
#             hidden_units = list.copy(architecture["fc_layers"]["hidden_units"])
#             self.fc_body = network_bodies.FCBody(device, self.conv_body.feature_dim, hidden_units=tuple(hidden_units))
#             self.fc_head = network_utils.layer_init_xavier(nn.Linear(self.fc_body.feature_dim, output_units))
#         else:
#             self.fc_body = None
#             self.fc_head = network_utils.layer_init_xavier(nn.Linear(self.conv_body.feature_dim, output_units))
#
#         self.to(device)
#         self.device = device
#         self.head_activation = head_activation
#
#     def forward(self, x):
#         if not isinstance(x, torch.Tensor):
#             x = torch_utils.tensor(x, self.device)
#         phi = self.conv_body(x)
#         if self.fc_body:
#             phi = self.fc_body(phi)
#         phi = self.fc_head(phi)
#         if self.head_activation is not None:
#             phi = self.head_activation(phi)
#         return phi
#

class DoubleCriticDiscrete(nn.Module):
    def __init__(self, device, input_units, hidden_units, output_units):
        super().__init__()
        self.device = device
        self.q1_net = FCNetwork(device, input_units, hidden_units, output_units)
        self.q2_net = FCNetwork(device, input_units, hidden_units, output_units)
    
    # def forward(self, x, a):
    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch_utils.tensor(x, self.device)
        recover_size = False
        if len(x.size()) == 1:
            recover_size = True
            x = x.reshape((1, -1))
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        if recover_size:
            q1 = q1[0]
            q2 = q2[0]
        return q1, q2


class DoubleCriticNetwork(nn.Module):
    def __init__(self, device, num_inputs, num_actions, hidden_units):
        super(DoubleCriticNetwork, self).__init__()
        self.device = device

        # Q1 architecture
        self.body1 = network_bodies.FCBody(device, num_inputs + num_actions, hidden_units=tuple(hidden_units))
        self.head1 = network_utils.layer_init_xavier(nn.Linear(self.body1.feature_dim, 1))
        # Q2 architecture
        self.body2 = network_bodies.FCBody(device, num_inputs + num_actions, hidden_units=tuple(hidden_units))
        self.head2 = network_utils.layer_init_xavier(nn.Linear(self.body2.feature_dim, 1))

    def forward(self, state, action):
        if not isinstance(state, torch.Tensor): state = torch_utils.tensor(state, self.device)
        recover_size = False
        if len(state.shape) > 2:
            state = state.view(state.shape[0], -1)
            action = action.view(action.shape[0], -1)
        elif len(state.shape) == 1:
            state = state.view(1, -1)
            action = action.view(1, -1)
            recover_size = True
        if not isinstance(action, torch.Tensor): action = torch_utils.tensor(action, self.device)

        xu = torch.cat([state, action], 1)

        q1 = self.head1(self.body1(xu))
        q2 = self.head2(self.body2(xu))
        
        if recover_size:
            q1 = q1[0]
            q2 = q2[0]
        return q1, q2
    

# class Constant(nn.Module):
#     def __init__(self, device, out_dim, constant):
#         super().__init__()
#         self.device = device
#         self.constant = torch_utils.tensor([constant]*out_dim, self.device)
#
#     def __call__(self, *args, **kwargs):
#         return self.constant