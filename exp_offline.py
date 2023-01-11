import os
import argparse
import torch
import numpy as np

import environment.env_factory as environment
import agent.in_sample as inac_agent
# import agent.iql as iql_agent
import network.network_architectures as network
import network.policy_factory as policy
from utils import torch_utils, logger, run_funcs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--id', default=0, type=int, help='random seed')
    parser.add_argument('--agent', default="InAC")
    parser.add_argument('--env', default='HalfCheetah')
    parser.add_argument('--dataset', default='medium')
    parser.add_argument('--learning-rate', default=3e-4)
    parser.add_argument('--temperature', default=0.1)
    parser.add_argument('--continuous-action', default=True)
    args = parser.parse_args()

    torch_utils.set_one_thread()

    project_root = os.path.abspath(os.path.dirname(__file__))
    log_dir = "{}/output/{}/{}/{}/{}_run".format(project_root, args.env, args.dataset, args.agent, args.id)
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    
    device = torch_utils.select_device(-1)
    torch_utils.random_seed(args.id)

    env = environment.create_env(args.env, args.id)
    eval_env = environment.create_env(args.env, args.id)

    hidden_units = [256, 256]
    if args.continuous_action:
        policy_fn = policy.MLPCont(device, np.prod(env.state_dim), env.action_dim, hidden_units)
        critic_fn = network.DoubleCriticNetwork(device, np.prod(env.state_dim), env.action_dim, hidden_units)
    else:
        policy_fn = policy.MLPDiscrete(device, np.prod(env.state_dim), env.action_dim, hidden_units)
        critic_fn = network.DoubleCriticDiscrete(device, np.prod(env.state_dim), hidden_units, env.action_dim)
    state_value_fn = network.FCNetwork(device, np.prod(env.state_dim), hidden_units, 1)
    
    offline_data = run_funcs.load_testset(args.env, args.dataset, args.id)
    
    # Setting up the logger
    logger = logger.Logger(log_dir)
    for param, value in args.__dict__.items():
        logger.info('{}: {}'.format(param, value))

    # Initializing the agent and running the experiment
    if args.agent == 'InAC':
        agent_obj = inac_agent.InSampleAC(
            project_root,
            log_dir,
            logger,
            device,
            args.id,
            env,
            eval_env,
            policy_fn,
            critic_fn,
            state_value_fn,
            learning_rate=3e-4,
            temperature=0.1,
            load_actor_fn_path=None,
            load_critic_fn_path=None,
            load_val_fn_path=None,
            load_offline_data=True,
            offline_data=offline_data,
            continuous_action=True,
            offline_setting=True
        )
    else:
        raise NotImplementedError
    run_funcs.run_steps(agent_obj)