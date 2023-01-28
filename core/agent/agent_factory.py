import os

from core.agent.in_sample import *


class AgentFactory:
    @classmethod
    def create_agent_fn(cls, cfg):
        if cfg.agent_name == 'InSampleAC':
            return lambda: InSampleAC(cfg)
        else:
            print(cfg.agent_name)
            raise NotImplementedError