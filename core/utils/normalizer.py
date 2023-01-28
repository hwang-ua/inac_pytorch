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


class MinMaxNormalizer(BaseNormalizer):
    def __init__(self, read_only=False, clip=10.0, epsilon=1e-8):
        BaseNormalizer.__init__(self, read_only)
        self.read_only = read_only
        self.maxes = None
        self.mins = None
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x):
        x = np.asarray(x)
        if self.maxes is None:
            self.maxes = np.zeros_like(x)
            self.mins = np.zeros_like(x)
        self.maxes = np.maximum(x, self.maxes)
        self.mins = np.minimum(x, self.mins)
        range = self.maxes - self.mins
        return np.true_divide(x, range, out=np.zeros_like(x), where=range != 0, casting='unsafe')


class Identity(BaseNormalizer):
    def __init__(self):
        BaseNormalizer.__init__(self)

    def __call__(self, x):
        return x


class RescaleNormalizer(BaseNormalizer):
    def __init__(self, coef=1.0):
        BaseNormalizer.__init__(self)
        self.coef = coef

    def __call__(self, x):
        return 2*(1./self.coef)*x - 1

class RangeNormalizer(BaseNormalizer):
    def __init__(self, coef=[-1.0, 1.0]):
        BaseNormalizer.__init__(self)
        self.lower = np.array(coef[0])
        self.lower_for2d = self.lower.reshape((1, -1))
        self.higher = np.array(coef[1])
        self.range_ = self.higher - self.lower
        self.range_for2d = self.range_.reshape((1, -1))

    def __call__(self, x):
        if len(x.shape) == 1:
            return (x - self.lower) / self.range_ * 2.0 - 1.0
        elif len(x.shape) == 2:
            normalized = np.divide(x - np.matmul(np.ones((len(x), 1)), self.lower_for2d), np.matmul(np.ones((len(x), 1)), self.range_for2d)) * 2.0 - 1.0
            return normalized


class StdDevNormalizer(BaseNormalizer):
    def __init__(self):
        BaseNormalizer.__init__(self)
        self.running_stats = RunningStats()

    def __call__(self, x):
        self.running_stats.push(x)
        x = np.asarray(x)
        div = self.running_stats.standard_deviation()
        return np.true_divide(x, self.running_stats.standard_deviation(), out=np.zeros_like(x), where=div != 0, casting='unsafe')


class MeanStdDevNormalizer(BaseNormalizer):
    def __init__(self, read_only=False, state_dim=2):
        BaseNormalizer.__init__(self, read_only)
        self.state_dim = state_dim

        self.running_stats = []
        for k in range(state_dim):
            self.running_stats.append(RunningStats())

    def __call__(self, x):
        means, std_dev = [], []
        for k in range(self.state_dim):
            self.running_stats[k].push(x[k])
            means.append(self.running_stats[k].mean())
            std_dev.append(self.running_stats[k].standard_deviation())
        x = np.asarray(x)
        means = np.array(means)
        std_dev = np.array(std_dev)
        return np.true_divide(x - means, std_dev, out=np.zeros_like(x), where=std_dev != 0, casting='unsafe')


class RunningStats:
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return np.sqrt(self.variance())


class NormalizerFactory:
    @classmethod
    def get_normalizer(cls, cfg):
        if cfg.state_norm_coef == 0:
            state_normalizer = Identity()
            reward_normalizer = Identity()
        elif type(cfg.state_norm_coef) == list and len(cfg.state_norm_coef) == 2:
            state_normalizer = RangeNormalizer(cfg.state_norm_coef)
            reward_normalizer = RescaleNormalizer(cfg.reward_norm_coef)
        else:
            state_normalizer = RescaleNormalizer(cfg.state_norm_coef)
            reward_normalizer = RescaleNormalizer(cfg.reward_norm_coef)
        return state_normalizer, reward_normalizer

