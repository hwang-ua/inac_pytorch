import torch


class OptFactory:
    @classmethod
    def get_optimizer_fn(cls, learning_rate, opt):
        # opt = cfg.optimizer_type if opt is None else opt
        if opt == 'SGD':
            return lambda params: torch.optim.SGD(params, learning_rate)
        elif opt == 'Adam':
            return lambda params: torch.optim.Adam(params, learning_rate)
        elif opt == 'RMSProp':
            return lambda params: torch.optim.RMSprop(params, learning_rate)
        else:
            raise NotImplementedError
    
    @classmethod
    def get_vf_loss_fn(cls, cfg):
        if cfg.vf_loss == 'mse':
            return torch.nn.MSELoss
        elif cfg.vf_loss == 'huber':
            return torch.nn.SmoothL1Loss
        else:
            raise NotImplementedError
