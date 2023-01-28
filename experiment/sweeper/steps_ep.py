import os
import numpy as np


from sweeper import Sweeper
from visualizer import RunLines, RunLinesIndividual


def parse_steps_log(log_path, max_steps, interval=0):
    """
    Retrieves num_steps at which episodes were completed
    and returns an array where the value at index i represents
    the number of episodes completed until step i
    """
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        rewards_over_time = np.zeros(max_steps // interval + 1)
        num_steps = get_max_steps(lines)
        if num_steps < max_steps:
            return None
        for line in lines:
            if 'total steps' not in line:
                continue
            num_steps = float(line.split("|")[1].split(",")[0].split(" ")[-1])
            if num_steps > max_steps:
                break
            reward = float(line.split("|")[1].split(",")[2].split("/")[0].split(" ")[-1])
            rewards_over_time[int(num_steps//interval)] = reward
        # rewards_over_time[0] = rewards_over_time[1]
        return rewards_over_time
    except:
        return None


def get_max_steps(lines):
    for line in lines[::-1]:
        if 'total steps' in line:
            max_steps = float(line.split("|")[1].split(",")[0].split(" ")[-1])
            return max_steps
    return -1


def draw_lunar_lander_dqn(settings, save_path, interval=10000, title="", ylim=None, ylabel="Average Return"):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_name = "log"
    parser_func = parse_steps_log
    path_formatters = []

    config_files, runs, num_datapoints, labels = [], [], [], []
    for cfg, param_setting, num_runs, num_steps, label in settings:
        config_files.append((cfg, param_setting))
        runs.append(num_runs)
        num_datapoints.append(num_steps)
        labels.append(label)

    for cf, best_setting in config_files:
        swp = Sweeper(os.path.join(project_root, cf))
        cfg = swp.parse(best_setting)      # Creating a cfg with an arbitrary run id
        cfg.data_root = os.path.join(project_root, 'data', 'output')
        logdir_format = cfg.get_logdir_format()
        path_format = os.path.join(logdir_format, log_name
                                   )
        path_formatters.append(path_format)

    v = RunLines(path_formatters, runs, num_datapoints,
                 labels, parser_func=parser_func,
                 save_path=save_path, xlabel="Number of steps in 10000s", ylabel=ylabel,
                 interval=interval, title=title, ylim=ylim)
    v.draw()


def draw_lunar_lander_dqn_linestyle(settings, save_path, interval=10000, title="", ylim=None, ylabel="Average Return"):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_name = "log"
    parser_func = parse_steps_log
    path_formatters = []

    config_files, runs, num_datapoints, labels, linestyles, colors = [], [], [], [], [], []
    for cfg, param_setting, num_runs, num_steps, label, linestyle, color in settings:
        config_files.append((cfg, param_setting))
        runs.append(num_runs)
        num_datapoints.append(num_steps)
        labels.append(label)
        linestyles.append(linestyle)
        colors.append(color)

    for cf, best_setting in config_files:
        swp = Sweeper(os.path.join(project_root, cf))
        cfg = swp.parse(best_setting)      # Creating a cfg with an arbitrary run id
        cfg.data_root = os.path.join(project_root, 'data', 'output')
        logdir_format = cfg.get_logdir_format()
        path_format = os.path.join(logdir_format, log_name
                                   )
        path_formatters.append(path_format)

    v = RunLines(path_formatters, runs, num_datapoints,
                 labels, parser_func=parser_func,
                 save_path=save_path, xlabel="Number of steps in 10000s", ylabel=ylabel,
                 interval=interval, title=title, ylim=ylim)
    v.draw_linestyles(linestyles, colors)


def draw_expected_rollout_linestyle(settings, save_path, interval=10000, title="", ylim=None, ylabel="Average Return"):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_name = "log"
    parser_func = parse_expected_rollout_log
    path_formatters = []

    config_files, runs, num_datapoints, labels, linestyles, colors = [], [], [], [], [], []
    for cfg, param_setting, num_runs, num_steps, label, linestyle, color in settings:
        config_files.append((cfg, param_setting))
        runs.append(num_runs)
        num_datapoints.append(num_steps)
        labels.append(label)
        linestyles.append(linestyle)
        colors.append(color)

    for cf, best_setting in config_files:
        swp = Sweeper(os.path.join(project_root, cf))
        cfg = swp.parse(best_setting)      # Creating a cfg with an arbitrary run id
        cfg.data_root = os.path.join(project_root, 'data', 'output')
        logdir_format = cfg.get_logdir_format()
        path_format = os.path.join(logdir_format, log_name
                                   )
        path_formatters.append(path_format)

    v = RunLines(path_formatters, runs, num_datapoints,
                 labels, parser_func=parser_func,
                 save_path=save_path, xlabel="Number of steps in 10000s", ylabel=ylabel,
                 interval=interval, title=title, ylim=ylim)
    v.draw_linestyles(linestyles, colors)


def draw_expected_rollout(settings, save_path, interval=10000, title="", ylim=None, ylabel="Average Return"):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_name = "log"
    parser_func = parse_expected_rollout_log
    path_formatters = []

    config_files, runs, num_datapoints, labels = [], [], [], []
    for cfg, param_setting, num_runs, num_steps, label in settings:
        config_files.append((cfg, param_setting))
        runs.append(num_runs)
        num_datapoints.append(num_steps)
        labels.append(label)

    for cf, best_setting in config_files:
        swp = Sweeper(os.path.join(project_root, cf))
        cfg = swp.parse(best_setting)      # Creating a cfg with an arbitrary run id
        cfg.data_root = os.path.join(project_root, 'data', 'output')
        logdir_format = cfg.get_logdir_format()
        path_format = os.path.join(logdir_format, log_name
                                   )
        path_formatters.append(path_format)

    v = RunLines(path_formatters, runs, num_datapoints,
                 labels, parser_func=parser_func,
                 save_path=save_path, xlabel="Number of steps in 10000s", ylabel=ylabel,
                 interval=interval, title=title, ylim=ylim)
    v.draw()


def parse_expected_rollout_log(log_path, max_steps, interval=0):
    """
    Retrieves num_steps at which episodes were completed
    and returns an array where the value at index i represents
    the number of episodes completed until step i
    """
    print(log_path)
    with open(log_path, "r") as f:
        lines = f.readlines()
    rewards_over_time = np.zeros(max_steps//interval + 1)
    try:
        num_steps = get_max_steps(lines)
        if num_steps < max_steps:
            return None
        for line in lines:
            if 'total steps' not in line:
                continue
            num_steps = float(line.split("|")[1].split(",")[0].split(" ")[-1])
            if num_steps > max_steps:
                break
            reward = float(line.split("|")[1].split(",")[3].split(" ")[2])
            rewards_over_time[int(num_steps//interval)] = reward
        # rewards_over_time[0] = rewards_over_time[1]
        return rewards_over_time
    except:
        return None
