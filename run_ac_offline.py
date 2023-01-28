import os
import argparse

import core.environment.env_factory as environment
import core.agent.agent_factory as agent
import core.network.net_factory as network
import core.network.policy_factory as policy
import core.network.optimizer as optimizer
import core.network.activations as activations
import core.component.replay as replay
import core.component.constraint as constraint
import core.utils.normalizer as normalizer
from core.utils import torch_utils, schedule, logger, run_funcs, format_path
import core.utils.testers as tester
from experiment.sweeper.sweeper import Sweeper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--id', default=0, type=int, help='identifies param_setting number and parameter configuration')
    parser.add_argument('--config-file', default='experiment/config/test_v0/mountain_car/dqn/temp.json')
    parser.add_argument('--device', default=-1, type=int, )
    args = parser.parse_args()

    torch_utils.set_one_thread()

    project_root = os.path.abspath(os.path.dirname(__file__))
    cfg = Sweeper(project_root, args.config_file).parse(args.id)
    cfg.device = torch_utils.select_device(args.device)
    torch_utils.random_seed(cfg.seed)

    cfg.env_fn = environment.EnvFactory.create_env_fn(cfg)
    cfg.offline_data = run_funcs.load_testset(cfg.offline_data_path, cfg.run, cfg.env_name)

    # Setting up the logger
    cfg.logger = logger.Logger(cfg)
    cfg.log_config()

    # Initializing the agent and running the experiment
    agent_obj = agent.AgentFactory.create_agent_fn(cfg)()
    run_funcs.run_steps(agent_obj, cfg.max_steps, cfg.log_interval)