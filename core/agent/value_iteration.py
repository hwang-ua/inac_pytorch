import os
import torch
import numpy as np
from core.agent import base
from core.utils import torch_utils


class VI2D(base.ValueBased):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.env = cfg.env_fn()
        self.ranges = [self.env.max_x+1, self.env.max_y+1]
        self.q_table = np.zeros((self.ranges+[cfg.action_dim]))
        self.v_table = np.zeros(self.ranges)
        self.goal_x, self.goal_y = self.env.goal_x, self.env.goal_y

    def step(self):
        for x in range(self.ranges[0]):
            for y in range(self.ranges[1]):
                for a in range(self.cfg.action_dim):
                    sp, r, t, _ = self.env.hack_step(self.env.generate_state([x, y]), [a])
                    discount = self.gamma * (1 - t)
                    self.q_table[x, y, a] = r + discount * self.q_table[sp[0], sp[1]].max()
        self.update_stats(0, None)

    def default_value_predictor(self):
        def vp(x):
            if isinstance(x, torch.Tensor): x = torch_utils.to_np(x)
            if len(x.shape) == 1:
                qs = self.q_table[x[0], x[1]]
            else:
                qs = []
                for i in range(len(x)):
                    qs.append(self.q_table[int(x[i][0]), int(x[i][1])])
                qs = np.array(qs)
            return torch_utils.tensor(qs, self.cfg.device)
    
        return vp

    def default_rep_predictor(self):
        return lambda x: x

    def policy(self, state, eps=0):
        q_values = self.q_table[state[0], state[1]]
        action = self.agent_rng.choice(np.flatnonzero(q_values == q_values.max()))
        return action

    def save(self, early=False):
        parameters_dir = self.cfg.get_parameters_dir()
        path = os.path.join(parameters_dir, "q_table")
        torch.save(self.q_table, path)