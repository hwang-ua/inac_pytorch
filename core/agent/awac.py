import numpy as np
import torch
import torch.nn.functional as F
from core.agent import base
from core.utils import torch_utils

device = torch.device("cpu")


"""
Changed based on https://github.com/hari-sikchi/AWAC/blob/master/AWAC/awac.py
"""
class AWACOnline(base.ActorCritic):

    def __init__(self, cfg):
        super(AWACOnline, self).__init__(cfg)
        # AWAC online fills offline data to buffer
        if cfg.agent_name == 'AWACOnline' and cfg.load_offline_data:
            self.fill_offline_data_to_buffer()
        

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        lambda_ = self.cfg.awac_lambda

        o = data['obs']
        pi, _ = self.ac.pi(o)

        # Learned policy
        v_pi, _, _ = self.get_q_value(o, pi, with_grad=False)
        
        # Behavior policy
        a = data['act']
        q_old_actions, _, _ = self.get_q_value(o, a, with_grad=False)

        adv_pi = q_old_actions - v_pi
        beh_logpp = self.ac.pi.get_logprob(o, a)

        if self.cfg.awac_remove_const:
            loss_pi = (-beh_logpp * adv_pi).mean()
            # pi_info = None
        else:
            weights = F.softmax(adv_pi / lambda_, dim=0)
            loss_pi = (-beh_logpp * len(weights) * weights.detach()).mean()
        return loss_pi, beh_logpp
    

class AWACOffline(AWACOnline):

    def __init__(self, cfg):
        super(AWACOffline, self).__init__(cfg)
        # self.true_q_predictor = self.cfg.tester_fn.get('true_value_estimator', lambda x:None)
        self.offline_param_init()

    def get_data(self):
        return self.get_offline_data()

    def feed_data(self):
        self.update_stats(0, None)
        return
