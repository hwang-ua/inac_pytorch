import os
import numpy as np
import torch

from core.utils.torch_utils import tensor


def compute_lipschitz(cfg, rep_net, val_net, env):
    try:
        _tensor = lambda x: tensor(x, cfg.device)
        states, _, _, _ = env.get_visualization_segment()
        states = cfg.state_normalizer(states)

        with torch.no_grad():
            phi_s = _tensor(rep_net(states))
            values = val_net(phi_s)

        num_states = len(states)
        N = num_states * (num_states - 1) // 2
        diff_v = np.zeros(N)
        diff_phi = np.zeros(N)
        idx = 0
        
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                phi_i, phi_j = phi_s[i], phi_s[j]
                vi, vj = values[i], values[j]
                diff_v[idx] = torch.abs(vi - vj).max().item()
                diff_phi[idx] = np.linalg.norm((phi_i - phi_j).cpu().numpy())
                idx += 1

        ratio_dv_dphi = np.divide(diff_v, diff_phi, out=np.zeros_like(diff_phi), where=diff_phi != 0)
        
        return val_net.compute_lipschitz_upper(), ratio_dv_dphi, np.corrcoef(diff_v, diff_phi)[0][1]
    
    except NotImplementedError:
        return val_net.compute_lipschitz_upper(), 0.0, 0.0
