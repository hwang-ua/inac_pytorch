import os
import torch
import core.network.net_factory as network
import core.utils.torch_utils as torch_utils

class TrueValuePredictor:
    def __init__(self, cfg, tester_fn_config):
        tester_cfg = type('tester_cfg', (object,), {})()
        for key in tester_fn_config:
            setattr(tester_cfg, key, tester_fn_config[key])
        setattr(tester_cfg, 'device', cfg.device)
        setattr(tester_cfg, 'action_dim', cfg.action_dim)

        tester_cfg.rep_fn = network.NetFactory.get_rep_fn(tester_cfg)
        tester_cfg.val_fn = network.NetFactory.get_val_fn(tester_cfg)
        self.rep_net = tester_cfg.rep_fn()
        self.val_net = tester_cfg.val_fn()
        
        if tester_cfg.rep_fn_config['load_params']:
            path = os.path.join(cfg.data_root, tester_cfg.rep_fn_config['path'].format(cfg.run))
            self.rep_net.load_state_dict(torch.load(path, map_location=cfg.device))
        if tester_cfg.val_fn_config['load_params']:
            path = os.path.join(cfg.data_root, tester_cfg.val_fn_config['path'].format(cfg.run))
            self.val_net.load_state_dict(torch.load(path, map_location=cfg.device))
        
    
    def __call__(self, x):
        with torch.no_grad():
            prediction = torch_utils.to_np(self.val_net(self.rep_net(x)))
        return prediction
    
class TesterFactory:
    @classmethod
    def get_tester_fn(cls, cfg):
        fns = {}
        for config in cfg.tester_fn_config:
            if config['type'] == 'true_value_estimator':
                fns['true_value_estimator'] = TrueValuePredictor(cfg, config)
            else:
                raise NotImplementedError
        return fns