import torch

class NullConstraint():
    def __init__(self, weight):
        self.weight = weight
        return

    def __call__(self, *argv):
        return 0


class L1Constraint(NullConstraint):
    def __init__(self, weight):
        super().__init__(weight)
    
    def __call__(self, rep, q, target):
        return self.weight * torch.mean(torch.linalg.norm(rep, dim=1))


class ConstraintFactory:
    @classmethod
    def get_constr_fn(cls, cfg):
        if cfg.constraint is None:
            return lambda: NullConstraint(0)
        elif cfg.constraint['type'] == "L1":
            return lambda: L1Constraint(cfg.constraint['weight'])
        else:
            raise NotImplementedError
