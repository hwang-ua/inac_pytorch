class ConstantEpsilon:
    def __init__(self, val):
        self.val = val

    def __call__(self, steps=1):
        return self.val

    def read_only(self):
        return self.val


class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        if not int(steps):
            self.inc = 0.0
        else:
            self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val

    def read_only(self):
        return self.current


class ScheduleFactory:
    @classmethod
    def get_eps_schedule(cls, cfg):
        if cfg.decay_epsilon:
            return LinearSchedule(cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_schedule_steps)
        else:
            return ConstantEpsilon(cfg.epsilon)
