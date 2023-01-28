import numpy as np

from core.network import network_architectures


class NetFactory:
    
    @classmethod
    def get_double_critic_fn(cls, cfg):
        # Creates a function for constructing the actor network
        if cfg.critic_fn_config['network_type'] == 'fc':
            return lambda: network_architectures.DoubleCriticDiscrete(cfg.device, np.prod(cfg.critic_fn_config['in_dim']),
                                                                      cfg.critic_fn_config['hidden_units'],
                                                                      cfg.critic_fn_config.get('out_dim', cfg.action_dim))
        elif cfg.critic_fn_config['network_type'] == 'fc-insert-input':
            return lambda: network_architectures.DoubleCriticNetwork(cfg.device, np.prod(cfg.critic_fn_config['in_dim']), cfg.action_dim,
                                                                     cfg.critic_fn_config['hidden_units'])
        else:
            raise NotImplementedError
        
    @classmethod
    def get_state_val_fn(cls, cfg):
        if hasattr(cfg, 'val_fn_config'):
            if cfg.val_fn_config['network_type'] == 'fc':
                return lambda: network_architectures.FCNetwork(cfg.device, np.prod(cfg.val_fn_config['in_dim']),
                                                               cfg.val_fn_config['hidden_units'], 1)
            else:
                raise NotImplementedError
        else:
            return None