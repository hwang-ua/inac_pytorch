import os

from core.agent.iql import IQLOnline, IQLOffline
from core.agent.in_sample import InSampleACOnline, InSampleAC


class AgentFactory:
    @classmethod
    def create_agent_fn(cls, cfg):
        if cfg.agent_name == 'IQLOnline':
            return lambda: IQLOnline(cfg)
        elif cfg.agent_name == 'IQLOffline':
            return lambda: IQLOffline(cfg)
        elif cfg.agent_name == 'InSampleAC':
            return lambda: InSampleAC(cfg)
        elif cfg.agent_name == 'InSampleACOnline':
            return lambda: InSampleACOnline(cfg)
        else:
            print(cfg.agent_name)
            raise NotImplementedError