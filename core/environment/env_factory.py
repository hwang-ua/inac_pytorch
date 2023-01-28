import os

from core.environment.mountaincar import MountainCar, MountainCarRandom, MountainCarSparse
from core.environment.acrobot import Acrobot
from core.environment.cartpole import Cartpole
from core.environment.lunarlander import LunarLander
from core.environment.gridworlds import *
import core.environment.gridworld as gw
import core.environment.cliffworld.cliffworld_env as cliffworld
from core.environment.halfcheetah import HalfCheetah
from core.environment.walker2d import Walker2d
from core.environment.hopper import Hopper
from core.environment.ant import Ant
from core.environment.maze2d import Maze2dUmaze, Maze2dMed, Maze2dLarge
from core.environment.minAtar import *
from core.environment.antmaze import *

class EnvFactory:
    @classmethod
    def create_env_fn(cls, cfg):
        if cfg.env_name == 'MountainCar':
            return lambda: MountainCar(cfg.seed)
        elif cfg.env_name == 'MountainCarRandom':
            return lambda: MountainCarRandom(cfg.env_random_prob, cfg.seed)
        elif cfg.env_name == 'MountainCarSparse':
            return lambda: MountainCarSparse(cfg.env_noise, cfg.seed)
        elif cfg.env_name == 'Acrobot':
            return lambda: Acrobot(cfg.seed)
        elif cfg.env_name == 'Cartpole':
            return lambda: Cartpole(cfg.seed)
        elif cfg.env_name == 'LunarLander':
            return lambda: LunarLander(cfg.seed)
        elif cfg.env_name == 'FourRoom':
            return lambda: FourRoom(cfg.seed, getattr(cfg, 'reward_type', 'sparse'))
        elif cfg.env_name == 'FourRoomFixedStart':
            return lambda: FourRoomFixedStart(cfg.seed, getattr(cfg, 'reward_type', 'sparse'))
        elif cfg.env_name == 'FourRoomNT':
            return lambda: gw.GridWorld(random_start=False)
        elif cfg.env_name == 'Cliff':
            return lambda: cliffworld.Environment()

        elif cfg.env_name == 'HalfCheetah':
            return lambda: HalfCheetah(cfg.seed)
        elif cfg.env_name == 'Walker2d':
            return lambda: Walker2d(cfg.seed)
        elif cfg.env_name == 'Hopper':
            return lambda: Hopper(cfg.seed)
        elif cfg.env_name == 'Ant':
            return lambda: Ant(cfg.seed)
        elif cfg.env_name == 'Maze2dUmaze':
            return lambda: Maze2dUmaze(cfg.seed)
        elif cfg.env_name == 'Maze2dMed':
            return lambda: Maze2dMed(cfg.seed)
        elif cfg.env_name == 'Maze2dLarge':
            return lambda: Maze2dLarge(cfg.seed)
        elif cfg.env_name == 'AntUmaze' or cfg.env_name == 'AntUmazeDense':
            return lambda: AntUmaze(cfg.seed)
        elif cfg.env_name == 'AntUmazeDiverse' or cfg.env_name == 'AntUmazeDiverseDense':
            return lambda: AntUmazeDiverse(cfg.seed)
        elif cfg.env_name == 'AntMediumDiverse' or cfg.env_name == 'AntMediumDiverseDense':
            return lambda: AntMediumDiverse(cfg.seed)
        elif cfg.env_name == 'AntMediumPlay' or cfg.env_name == 'AntMediumPlayDense':
            return lambda: AntMediumPlay(cfg.seed)
        elif cfg.env_name == 'AntLargeDiverse' or cfg.env_name == 'AntLargeDiverseDense':
            return lambda: AntLargeDiverse(cfg.seed)
        elif cfg.env_name == 'AntLargePlay' or cfg.env_name == 'AntLargePlayDense':
            return lambda: AntLargePlay(cfg.seed)

        # elif cfg.env_name == 'Pong':
        #     return lambda: Pong(cfg.seed)
        elif cfg.env_name == 'Asterix':
            return lambda: Asterix(cfg.seed)
        elif cfg.env_name == 'Freeway':
            return lambda: Freeway(cfg.seed)
        elif cfg.env_name == 'SpaceInvaders':
            return lambda: SpaceInvaders(cfg.seed)
        elif cfg.env_name == 'Seaquest':
            return lambda: Seaquest(cfg.seed)
        elif cfg.env_name == 'Breakout':
            return lambda: Breakout(cfg.seed)
        else:
            print(cfg.env_name)
            raise NotImplementedError