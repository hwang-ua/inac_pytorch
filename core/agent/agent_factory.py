import os

from core.agent.dqn import DQNAgent#, SlowPolicyDQN
from core.agent.awac import AWACOnline, AWACOffline
from core.agent.sac import SAC, SACOffline
from core.agent.iql import IQLOnline, IQLOffline, IQLOfflineNoV, IQLWeighted
# from core.agent.iql2 import IQL2
from core.agent.td3_bc import *
from core.agent.qrc import QRCOnline, QRCOffline
from core.agent.fqi import FQIAgent
from core.agent.sarsa import *
from core.agent.monte_carlo import *
from core.agent.cql import *
from core.agent.in_sample_old import InSample
from core.agent.in_sample import *
# from core.agent.qmax_clone import *
from core.agent.value_iteration import VI2D
from core.agent.one_step import OneStep


class AgentFactory:
    @classmethod
    def create_agent_fn(cls, cfg):
        if cfg.agent_name == 'DQNAgent':
            return lambda: DQNAgent(cfg)
        elif cfg.agent_name == 'AWACOnline':
            return lambda: AWACOnline(cfg)
        elif cfg.agent_name == 'AWACOffline':
            return lambda: AWACOffline(cfg)
        elif cfg.agent_name == 'SAC':
            return lambda: SAC(cfg)
        elif cfg.agent_name == 'SACOffline':
            return lambda: SACOffline(cfg)
        elif cfg.agent_name == 'IQLOnline':
            return lambda: IQLOnline(cfg)
        elif cfg.agent_name == 'IQLOffline':
            return lambda: IQLOffline(cfg)
            # return lambda: IQL2(cfg)
        elif cfg.agent_name == 'IQLWeighted':
            return lambda: IQLWeighted(cfg)
        elif cfg.agent_name == 'IQLOffline-RemoveV':
            return lambda: IQLOfflineNoV(cfg)
        elif cfg.agent_name == 'QRCOnline':
            return lambda: QRCOnline(cfg)
        elif cfg.agent_name == 'QRCOffline':
            return lambda: QRCOffline(cfg)
        # elif cfg.agent_name == 'SlowPolicyDQN':
        #     return lambda: SlowPolicyDQN(cfg)
        elif cfg.agent_name == 'FQIAgent':
            return lambda: FQIAgent(cfg)
        elif cfg.agent_name == 'SarsaAgent':
            return lambda: SarsaAgent(cfg)
        # elif cfg.agent_name == 'SarsaOffline':
        #     return lambda: SarsaOffline(cfg)
        elif cfg.agent_name == 'SarsaOfflineBatch':
            return lambda: SarsaOfflineBatch(cfg)
        elif cfg.agent_name == 'MonteCarloAgent':
            return lambda: MonteCarloAgent(cfg)
        elif cfg.agent_name == 'MonteCarloOffline':
            return lambda: MonteCarloOffline(cfg)
        elif cfg.agent_name == 'CQLAgentOffline':
            return lambda: CQLAgentOffline(cfg)
        elif cfg.agent_name == 'CQLSACOffline':
            return lambda: CQLSACOffline(cfg)
        elif cfg.agent_name == 'TD3BC':
            return lambda: TD3BC(cfg)
        elif cfg.agent_name == 'TD3BCOnline':
            return lambda: TD3BCOnline(cfg)
        elif cfg.agent_name == 'InSample':
            return lambda: InSample(cfg)
        elif cfg.agent_name == 'InSamplePiW':
            return lambda: InSamplePiW(cfg)
        elif cfg.agent_name == 'InSampleBeta':
            return lambda: InSampleBeta(cfg)
        elif cfg.agent_name == 'InSampleNoV':
            return lambda: InSampleNoV(cfg)
        elif cfg.agent_name == 'InSampleAC':
            return lambda: InSampleAC(cfg)
        elif cfg.agent_name == 'InSampleUniform':
            return lambda: InSampleUniform(cfg)
        elif cfg.agent_name == 'InSamplePC':
            return lambda: InSamplePC(cfg)
        elif cfg.agent_name == 'InSampleEmphatic':
            return lambda: InSampleEmphatic(cfg)
        elif cfg.agent_name == 'InSampleEmphCritic':
            return lambda: InSampleEmphCritic(cfg)
        # elif cfg.agent_name == 'InSampleMaxAC':
        #     return lambda: InSampleMaxAC(cfg)
        # elif cfg.agent_name == 'InSampleSoftMax':
        #     return lambda: InSampleSoftMax(cfg)
        # elif cfg.agent_name == 'InSampleOnline':
        #     return lambda: InSampleOnline(cfg)
        elif cfg.agent_name == 'InSampleACOnline':
            return lambda: InSampleACOnline(cfg)
        elif cfg.agent_name == 'InSampleWeighted':
            return lambda: InSampleWeighted(cfg)
        # elif cfg.agent_name == 'QmaxCloneOffline':
        #     return lambda: QmaxCloneOffline(cfg)
        # elif cfg.agent_name == 'QmaxConstrOffline':
        #     return lambda: QmaxConstrOffline(cfg)
        elif cfg.agent_name == 'VI2D':
            return lambda: VI2D(cfg)
        elif cfg.agent_name == 'OneStep':
            return lambda: OneStep(cfg)
        else:
            print(cfg.agent_name)
            raise NotImplementedError