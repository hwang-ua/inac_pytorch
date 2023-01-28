import numpy as np


class BaseNormalizer:
    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return


class Identity(BaseNormalizer):
    def __init__(self):
        BaseNormalizer.__init__(self)

    def __call__(self, x):
        return x