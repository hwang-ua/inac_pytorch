import os

from environment.mountaincar import MountainCar
from environment.acrobot import Acrobot
from environment.lunarlander import LunarLander
from environment.halfcheetah import HalfCheetah
from environment.walker2d import Walker2d
from environment.hopper import Hopper
from environment.ant import Ant

def create_env(env_name, seed):
    if env_name == 'MountainCar':
        return MountainCar(seed)
    elif env_name == 'Acrobot':
        return Acrobot(seed)
    elif env_name == 'LunarLander':
        return LunarLander(seed)
    elif env_name == 'HalfCheetah':
        return HalfCheetah(seed)
    elif env_name == 'Walker2d':
        return Walker2d(seed)
    elif env_name == 'Hopper':
        return Hopper(seed)
    elif env_name == 'Ant':
        return Ant(seed)
    else:
        print(env_name)
        raise NotImplementedError