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


if __name__ == '__main__':

    ################################################
    ##### Navigation
    # settings = [("experiment/archive/model_free_v3/dqn_navigation_c.json", 2, 10, 300000, "DQN"),
    #             ("experiment/archive/mve_avg_perfect/mve_navigation.json", 1, 10, 300000, "MVE DQN k=2"),
    #             ("experiment/archive/mve_avg_perfect/mve_navigation.json", 5, 10, 300000, "MVE DQN k=3"),
    #             ("experiment/archive/mve_avg_perfect/mve_navigation.json", 9, 10, 300000, "MVE DQN k=4"),
    #             ("experiment/archive/mve_avg_perfect/mve_navigation.json", 13, 10, 300000, "MVE DQN k=5"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/mve_avg_perfect/navigation/plot_same.png", interval=10000,
    #                       title="MF vs MVE", ylim=(0, 500), ylabel="Average Steps to Goal")
    #
    # settings = [("experiment/archive/model_free_v3/dqn_navigation_c.json", 2, 10, 300000, "DQN"),
    #             ("experiment/archive/zach_perfect_2/zach_navigation.json", 1, 10, 300000, "Zach DQN 16 x 1"),
    #             ("experiment/archive/zach_perfect_2/zach_navigation.json", 5, 10, 300000, "Zach DQN 8 x 2"),
    #             ("experiment/archive/zach_perfect_2/zach_navigation.json", 10, 10, 300000, "Zach DQN 4 x 4"),
    #             ("experiment/archive/zach_perfect_2/zach_navigation.json", 13, 10, 300000, "Zach DQN 2 x 8"),
    #             ("experiment/archive/zach_perfect_2/zach_navigation.json", 17, 10, 300000, "Zach DQN 1 x 16")
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/zach_perfect_2/navigation/plot.png", interval=10000,
    #                       title="MF vs Zach", ylim=(0, 500), ylabel="Average Steps to Goal")
    #
    # settings = [("experiment/archive/model_free_v3/dqn_navigation_c.json", 2, 10, 300000, "DQN"),
    #             ("experiment/archive/mve_avg_learned/navigation.json", 0, 10, 300000, "MVE DQN k=2"),
    #             ("experiment/archive/mve_avg_learned/navigation.json", 2, 10, 300000, "MVE DQN k=3"),
    #             ("experiment/archive/mve_avg_learned/navigation.json", 4, 10, 300000, "MVE DQN k=4"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/mve_avg_learned/navigation/plot.png", interval=10000,
    #                       title="MF vs MVE-Learned", ylim=(0, 500), ylabel="Average Steps to Goal")

    # settings = [("experiment/archive/model_free_v3/dqn_navigation_c.json", 2, 10, 300000, "DQN"),
    #             ("experiment/archive/search_dt_perfect/navigation.json", 1, 10, 300000, "Search-DT k=1"),
    #             ("experiment/archive/search_dt_perfect/navigation.json", 3, 10, 300000, "Search-DT k=2"),
    #             ("experiment/archive/search_dt_perfect/navigation.json", 5, 10, 300000, "Search-DT k=3"),
    #             ("experiment/archive/search_dt_perfect/navigation.json", 7, 10, 300000, "Search-DT k=4"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/search_dt_perfect/navigation/plot_b.png", interval=10000,
    #                       title="MF vs Search-DT (Perfect)", ylim=(0, 500), ylabel="Average Steps to Goal")


    # settings = [("experiment/archive/model_free_v3/dqn_navigation_c.json", 2, 10, 300000, "DQN", "--", "blue"),
    #             ("experiment/archive/mve_avg_learned/navigation.json", 0, 10, 300000, "MVE DQN LM k=2", "--", "green"),
    #             ("experiment/archive/mve_avg_perfect/mve_navigation.json", 1, 10, 300000, "MVE DQN k=2", "-", "green"),
    #             ("experiment/archive/mve_avg_perfect/mve_navigation.json", 5, 10, 300000, "MVE DQN k=3", "-", "cyan"),
    #             ("experiment/archive/mve_avg_learned/navigation.json", 2, 10, 300000, "MVE DQN LM k=3", "--", "cyan"),
    #             ("experiment/archive/mve_avg_perfect/mve_navigation.json", 9, 10, 300000, "MVE DQN k=4", "-", "red"),
    #             ("experiment/archive/mve_avg_learned/navigation.json", 4, 10, 300000, "MVE DQN LM k=4", "--", "red"),
    #             ]
    # draw_lunar_lander_dqn_linestyle(settings, save_path="plots/mve_avg_learned/navigation/plot.png", interval=10000,
    #                       title="MF vs MVE-Learned", ylim=(0, 500), ylabel="Average Steps to Goal")


    # settings = [("experiment/archive/model_free_v3/dqn_navigation_c.json", 2, 10, 300000, "DQN"),
    #             ("experiment/archive/mve_avg_learned/navigation.json", 4, 10, 300000, "MVE DQN k=4 - JuhnyukFC"),
    #             ("experiment/archive/mve_sw_model/navigation.json", 4, 3, 300000, "MVE DQN k=4 - FCModelNetA"),
    #             ("experiment/archive/mve_sw_model/navigation.json", 8, 3, 300000, "MVE DQN k=4 - FCModelNetB"),
    #             ("experiment/archive/mve_sw_model/navigation.json", 16, 3, 300000, "MVE DQN k=4 - FCModelNetC"),
    #             ("experiment/archive/mve_sw_model/navigation.json", 20, 3, 300000, "MVE DQN k=4 - FCModelNetD"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/mve_sw_model/navigation/plot.png", interval=10000,
    #                       title="Model Sweep", ylim=(0, 500), ylabel="Average Steps to Goal")


    # settings = [("experiment/archive/model_free_v3/dqn_navigation_c.json", 2, 10, 300000, "DQN", "--", "blue"),
    #             ("experiment/archive/mve_avg_perfect/mve_navigation.json", 1, 10, 300000, "MVE DQN k=2", "-", "green"),
    #             ("experiment/archive/mve_avg_learnedv2/navigation.json", 0, 10, 300000, "MVE DQN LM k=2", "--","green"),
    #             ("experiment/archive/mve_avg_perfect/mve_navigation.json", 5, 10, 300000, "MVE DQN k=3", "-", "cyan"),
    #             ("experiment/archive/mve_avg_learnedv2/navigation.json", 2, 10, 300000, "MVE DQN LM k=3", "--", "cyan"),
    #             ("experiment/archive/mve_avg_perfect/mve_navigation.json", 9, 10, 300000, "MVE DQN k=4", "-", "red"),
    #             ("experiment/archive/mve_avg_learnedv2/navigation.json", 4, 10, 300000, "MVE DQN LM k=4", "--", "red"),
    #             ]
    # draw_lunar_lander_dqn_linestyle(settings, save_path="plots/sprint_1/mve_avg_learnedv2/navigation/plot_update_13.png", interval=10000,
    #                       title="MF vs MVE-Learned", ylim=(0, 500), ylabel="Average Steps to Goal")

    # settings = [("experiment/archive/model_free_v3/dqn_navigation_c.json", 2, 10, 300000, "DQN", "-", "blue"),
    #             ("experiment/archive/search_dt_perfect/navigation.json", 1, 8, 300000, "Search k=1", "-", "green"),
    #             ("experiment/archive/search_dt_learnedv2/navigation.json", 1, 8, 300000, "Search LM k=1", "--", "green"),
    #             ("experiment/archive/search_dt_perfect/navigation.json", 3, 8, 300000, "Search k=2", "-", "cyan"),
    #             ("experiment/archive/search_dt_learnedv2/navigation.json", 3, 8, 300000, "Search LM k=2", "--", "cyan"),
    #             ("experiment/archive/search_dt_perfect/navigation.json", 5, 8, 300000, "Search k=3", "-", "red"),
    #             ("experiment/archive/search_dt_learnedv2/navigation.json", 5, 8, 300000, "Search LM k=3", "--", "red"),
    #             ]
    # draw_lunar_lander_dqn_linestyle(settings, save_path="plots/sprint_1/search_dt_learnedv2/navigation/plot_update_13.png", interval=10000,
    #                                 title="MF vs Search-DT", ylim=(0, 500), ylabel="Average Steps to Goal")


    # settings = [("experiment/archive/model_free_v3/dqn_navigation_c.json", 2, 10, 300000, "DQN"),
    #             ("experiment/config_files_2/mve_learned/navigation.json", 8, 5, 300000, "MVE HHH"),
    #             ("experiment/config_files_2/mve_learned/navigation.json", 9, 5, 300000, "MVE H16"),
    #             ("experiment/config_files_2/mve_learned/navigation.json", 14, 5, 300000, "MVE H8"),
    #             ("experiment/config_files_2/mve_learned/navigation.json", 11, 5, 300000, "MVE H4"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/mve_learned/navigation/plot.png", interval=10000,
    #                       title="MVE - Model Capacity - K=4", ylim=(0, 500), ylabel="Average Steps to Goal")


    ################################################

    ################################################
    ##### Cartpole

    # settings = [("experiment/archive/model_free_v3/dqn_cartpole.json", 1, 10, 50000, "DQN"),
    #             ("experiment/archive/zach_perfect_2/zach_cartpole.json", 1, 10, 50000, "Zach DQN 16 x 1"),
    #             ("experiment/archive/zach_perfect_2/zach_cartpole.json", 4, 10, 50000, "Zach DQN 8 x 2"),
    #             ("experiment/archive/zach_perfect_2/zach_cartpole.json", 8, 10, 50000, "Zach DQN 4 x 4"),
    #             ("experiment/archive/zach_perfect_2/zach_cartpole.json", 12, 10, 50000, "Zach DQN 2 x 8"),
    #             ("experiment/archive/zach_perfect_2/zach_cartpole.json", 16, 10, 50000, "Zach DQN 1 x 16"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole_b.json", 1, 10, 50000, "Zach DQN 4 x 1"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole_b.json", 5, 10, 50000, "Zach DQN 2 x 2"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole_b.json", 8, 10, 50000, "Zach DQN 1 x 4"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_1/zach_perfect_2/cartpole/plot_update_13.png", interval=2000,
    #                       title="MF vs Zach", ylim=(0, 200))



    # settings = [("experiment/archive/model_free_v3/dqn_cartpole.json", 1, 10, 50000, "DQN"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole.json", 1, 10, 50000, "Zach DQN 16 x 1"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole.json", 4, 10, 50000, "Zach DQN 8 x 2"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole.json", 8, 10, 50000, "Zach DQN 4 x 4"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole.json", 12, 10, 50000, "Zach DQN 2 x 8"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole.json", 16, 10, 50000, "Zach DQN 1 x 16"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole_b.json", 1, 10, 50000, "Zach DQN 4 x 1"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole_b.json", 5, 10, 50000, "Zach DQN 2 x 2"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole_b.json", 8, 10, 50000, "Zach DQN 1 x 4"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/zach_perfect_2/cartpole/plot_b.png", interval=2000,
    #                       title="MF vs Zach")

    # settings = [("experiment/archive/model_free_v3/dqn_cartpole_b.json", 1, 10, 50000, "DQN"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole.json", 1, 10, 50000, "Zach DQN 16 x 1"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole.json", 4, 10, 50000, "Zach DQN 8 x 2"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole.json", 8, 10, 50000, "Zach DQN 4 x 4"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole.json", 12, 10, 50000, "Zach DQN 2 x 8"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole.json", 16, 10, 50000, "Zach DQN 1 x 16"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole_c.json", 1, 10, 50000, "Zach DQN 6 x 1"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole_c.json", 5, 10, 50000, "Zach DQN 3 x 2"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole_c.json", 9, 10, 50000, "Zach DQN 2 x 2"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole_c.json", 13, 10, 50000, "Zach DQN 2 x 3"),
    #             # ("experiment/archive/zach_perfect_2/zach_cartpole_c.json", 17, 10, 50000, "Zach DQN 1 x 6"),
    #             ("experiment/archive/zach_perfect_2/zach_cartpole_d.json", 1, 10, 50000, "Zach DQN 16 x 1"),
    #             ("experiment/archive/zach_perfect_2/zach_cartpole_d.json", 5, 10, 50000, "Zach DQN 8 x 2"),
    #             ("experiment/archive/zach_perfect_2/zach_cartpole_d.json", 9, 10, 50000, "Zach DQN 4 x 4"),
    #             ("experiment/archive/zach_perfect_2/zach_cartpole_d.json", 12, 10, 50000, "Zach DQN 2 x 8"),
    #             ("experiment/archive/zach_perfect_2/zach_cartpole_d.json", 16, 10, 50000, "Zach DQN 1 x 16"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/zach_perfect_2/cartpole/plot_meeting_2.png", interval=2000,
    #                       title="MF vs Zach")
    #

    # settings = [("experiment/archive/model_free_v3/dqn_cartpole.json", 1, 10, 50000, "DQN"),
    #             ("experiment/archive/mve_avg_perfect/mve_cartpole.json", 1, 10, 50000, "MVE DQN k=2"),
    #             ("experiment/archive/mve_avg_perfect/mve_cartpole.json", 5, 10, 50000, "MVE DQN k=3"),
    #             ("experiment/archive/mve_avg_perfect/mve_cartpole.json", 9, 10, 50000, "MVE DQN k=4"),
    #             ("experiment/archive/mve_avg_perfect/mve_cartpole.json", 13, 10, 50000, "MVE DQN k=5"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_1/mve_avg_perfect/cartpole/plot_update_13.png", interval=2000,
    #                       title="MF vs MVE", ylim=(0, 200))
    #
    # settings = [("experiment/archive/model_free_v3/dqn_cartpole.json", 1, 10, 50000, "DQN"),
    #             ("experiment/archive/search_dt_perfect/cartpole.json", 1, 10, 50000, "Search-DT k=1"),
    #             ("experiment/archive/search_dt_perfect/cartpole.json", 4, 10, 50000, "Search-DT k=2"),
    #             ("experiment/archive/search_dt_perfect/cartpole.json", 7, 10, 50000, "Search-DT k=3"),
    #             ("experiment/archive/search_dt_perfect/cartpole.json", 10, 10, 50000, "Search-DT k=4")
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_1/search_dt_perfect/cartpole/plot_update_13.png", interval=2000, ylim=(0, 200))


    # settings = [("experiment/archive/model_free_v3/dqn_cartpole.json", 1, 10, 50000, "DQN"),
    #             ("experiment/archive/mve_avg_perfect/mve_cartpole.json", 1, 10, 50000, "MVE DQN k=2"),
    #             ("experiment/archive/mve_avg_perfect/mve_cartpole.json", 5, 10, 50000, "MVE DQN k=3"),
    #             ("experiment/archive/mve_avg_perfect/mve_cartpole.json", 9, 10, 50000, "MVE DQN k=4"),
    #             ("experiment/archive/mve_avg_perfect/mve_cartpole.json", 13, 10, 50000, "MVE DQN k=5"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/mve_avg_perfect/cartpole/plot.png", interval=2000,
    #                       title="MF vs MVE")

    # settings = [("experiment/archive/model_free_v3/dqn_cartpole.json", 1, 10, 50000, "DQN", "-", "blue"),
    #             ("experiment/archive/mve_avg_perfect/mve_cartpole.json", 1, 10, 50000, "MVE k=2", "-", "green"),
    #             ("experiment/archive/mve_avg_learned/cartpole.json", 1, 10, 50000, "MVE LM k=2", "--", "green"),
    #             ("experiment/archive/mve_avg_perfect/mve_cartpole.json", 5, 10, 50000, "MVE k=3", "-", "cyan"),
    #             ("experiment/archive/mve_avg_learned/cartpole.json", 4, 10, 50000, "MVE LM k=3", "--", "cyan"),
    #             ("experiment/archive/mve_avg_perfect/mve_cartpole.json", 9, 10, 50000, "MVE k=4", "-", "red"),
    #             ("experiment/archive/mve_avg_learned/cartpole.json", 7, 10, 50000, "MVE LM k=4", "--", "red")
    #             ]
    # draw_lunar_lander_dqn_linestyle(settings, save_path="plots/mve_avg_learned/cartpole/plot.png", interval=2000,
    #                                 title="MF vs MVE")

    # settings = [("experiment/archive/model_free_v3/dqn_cartpole.json", 1, 10, 50000, "DQN"),
    #             ("experiment/archive/mve_avg_learned/cartpole.json", 7, 9, 50000, "MVE DQN k=4 JuhnyukFC"),
    #             ("experiment/archive/mve_sw_model/cartpole.json", 5, 9, 50000, "MVE DQN k=4 FCModelNetA"),
    #             ("experiment/archive/mve_sw_model/cartpole.json", 11, 9, 50000, "MVE DQN k=4 FCModelNetB"),
    #             ("experiment/archive/mve_sw_model/cartpole.json", 17, 9, 50000, "MVE DQN k=4 FCModelNetC"),
    #             ("experiment/archive/mve_sw_model/cartpole.json", 23, 9, 50000, "MVE DQN k=4 FCModelNetD"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/mve_sw_model/cartpole/plot.png", interval=2000,
    #                       title="Model Sweep")

    # settings = [("experiment/archive/model_free_v3/dqn_cartpole.json", 1, 10, 50000, "DQN", "-", "blue"),
    #             ("experiment/archive/mve_avg_perfect/mve_cartpole.json", 1, 10, 50000, "MVE k=2", "-", "green"),
    #             ("experiment/archive/mve_avg_learnedv2/cartpole.json", 1, 10, 50000, "MVE LM k=2", "--", "green"),
    #             ("experiment/archive/mve_avg_perfect/mve_cartpole.json", 5, 10, 50000, "MVE k=3", "-", "cyan"),
    #             ("experiment/archive/mve_avg_learnedv2/cartpole.json", 4, 10, 50000, "MVE LM k=3", "--", "cyan"),
    #             ("experiment/archive/mve_avg_perfect/mve_cartpole.json", 9, 10, 50000, "MVE k=4", "-", "red"),
    #             ("experiment/archive/mve_avg_learnedv2/cartpole.json", 7, 10, 50000, "MVE LM k=4", "--", "red")
    #             ]
    # draw_lunar_lander_dqn_linestyle(settings, save_path="plots/sprint_1/mve_avg_learnedv2/cartpole/plot_update_13.png", interval=2000,
    #                                 title="MF vs MVE", ylim=(0, 200))


    # settings = [("experiment/archive/model_free_v3/dqn_cartpole.json", 1, 10, 50000, "DQN", "-", "blue"),
    #             ("experiment/archive/search_dt_perfect/cartpole.json", 1, 10, 50000, "Search k=1", "-", "green"),
    #             ("experiment/archive/search_dt_learnedv2/cartpole.json", 1, 10, 50000, "Search LM k=1", "--", "green"),
    #             ("experiment/archive/search_dt_perfect/cartpole.json", 4, 10, 50000, "Search k=2", "-", "cyan"),
    #             ("experiment/archive/search_dt_learnedv2/cartpole.json", 5, 10, 50000, "Search LM k=2", "--", "cyan"),
    #             ("experiment/archive/search_dt_perfect/cartpole.json", 7, 10, 50000, "Search k=3", "-", "red"),
    #             ("experiment/archive/search_dt_learnedv2/cartpole.json", 8, 10, 50000, "Search LM k=3", "--", "red"),
    #             ("experiment/archive/search_dt_perfect/cartpole.json", 10, 10, 50000, "Search k=4", "-", "yellow"),
    #             ("experiment/archive/search_dt_learnedv2/cartpole.json", 10, 10, 50000, "Search LM k=4", "--", "yellow")
    #             ]
    # draw_lunar_lander_dqn_linestyle(settings, save_path="plots/sprint_1/search_dt_learnedv2/cartpole/plot_update_13.png", interval=2000,
    #                                 title="MF vs MVE", ylim=(0, 200))




    # settings = [("experiment/archive/model_free_v3/dqn_cartpole.json", 1, 10, 50000, "DQN"),
    #             ("experiment/archive/search_dt_perfect/cartpole.json", 1, 10, 50000, "Search-DT k=1"),
    #             ("experiment/archive/search_dt_perfect/cartpole.json", 4, 10, 50000, "Search-DT k=2"),
    #             ("experiment/archive/search_dt_perfect/cartpole.json", 7, 10, 50000, "Search-DT k=3"),
    #             ("experiment/archive/search_dt_perfect/cartpole.json", 10, 10, 50000, "Search-DT k=4")
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_1/search_dt_perfect/cartpole/plot_update_13.png", interval=2000, ylim=(0, 200))

    # settings = [("experiment/archive/model_free_v3/dqn_cartpole.json", 1, 10, 50000, "DQN"),
    #             ("experiment/config_files_2/mve_learned/cartpole.json", 0, 5, 50000, "MVE HHH"),
    #             ("experiment/config_files_2/mve_learned/cartpole.json", 9, 10, 50000, "MVE H16"),
    #             ("experiment/config_files_2/mve_learned/cartpole.json", 10, 10, 50000, "MVE H8"),
    #             ("experiment/config_files_2/mve_learned/cartpole.json", 3, 10, 50000, "MVE H4"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/mve_learned/cartpole/plot.png", interval=2000,
    #                       title="MVE - Model Capacity - K=4", ylim=(0, 200))


    ################################################

    ################################################
    ##### Catcher Raw
    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
    #             ("experiment/archive/mve_avg_perfect/mve_catcher_raw.json", 2, 10, 200000, "MVE k=2"),
    #             ("experiment/archive/mve_avg_perfect/mve_catcher_raw.json", 6, 10, 200000, "MVE k=3"),
    #             ("experiment/archive/mve_avg_perfect/mve_catcher_raw.json", 10, 10, 200000, "MVE k=4"),
    #             ("experiment/archive/mve_avg_perfect/mve_catcher_raw.json", 14, 10, 200000, "MVE k=5"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_1/mve_avg_perfect/catcher_raw/plot_update_13.png", interval=5000,
    #                       ylim=(-5, 60))
    #
    #
    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
    #             ("experiment/archive/zach_perfect_2/zach_catcher_raw.json", 1, 10, 200000, "Zach DQN 16 x 1"),
    #             ("experiment/archive/zach_perfect_2/zach_catcher_raw.json", 5, 10, 200000, "Zach DQN 8 x 2"),
    #             ("experiment/archive/zach_perfect_2/zach_catcher_raw.json", 9, 10, 200000, "Zach DQN 4 x 4"),
    #             ("experiment/archive/zach_perfect_2/zach_catcher_raw.json", 13, 10, 200000, "Zach DQN 2 x 8"),
    #             ("experiment/archive/zach_perfect_2/zach_catcher_raw.json", 17, 10, 200000, "Zach DQN 1 x 16"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_1/zach_perfect_2/catcher_raw/plot_update_13.png", interval=5000,
    #                       ylim=(-5, 60))
    #
    #
    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
    #             ("experiment/archive/zach_perfect_2/zach_catcher_raw_b.json", 1, 10, 200000, "Zach DQN 6 x 1"),
    #             ("experiment/archive/zach_perfect_2/zach_catcher_raw_b.json", 6, 10, 200000, "Zach DQN 3 x 2"),
    #             ("experiment/archive/zach_perfect_2/zach_catcher_raw_b.json", 10, 10, 200000, "Zach DQN 2 x 2"),
    #             ("experiment/archive/zach_perfect_2/zach_catcher_raw_b.json", 13, 10, 200000, "Zach DQN 2 x 3"),
    #             ("experiment/archive/zach_perfect_2/zach_catcher_raw_b.json", 17, 10, 200000, "Zach DQN 1 x 6"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/zach_perfect_2/catcher_raw/plot_b.png", interval=5000)

    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
    #             ("experiment/archive/search_dt_perfect/catcher_raw.json", 1, 10, 200000, "Search-DT k=1"),
    #             ("experiment/archive/search_dt_perfect/catcher_raw.json", 3, 10, 200000, "Search-DT k=2"),
    #             ("experiment/archive/search_dt_perfect/catcher_raw.json", 5, 10, 200000, "Search-DT k=3"),
    #             ("experiment/archive/search_dt_perfect/catcher_raw.json", 7, 10, 200000, "Search-DT k=4")
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_1/search_dt_perfect/catcher_raw/plot_update_13.png", interval=5000,
    #                       ylim=(-5, 60))

    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN", "-", "blue"),
    #             ("experiment/archive/mve_avg_perfect/mve_catcher_raw.json", 2, 10, 200000, "MVE k=2", "-", "green"),
    #             ("experiment/archive/mve_avg_learned/catcher_raw.json", 1, 10, 200000, "MVE LM k=2", "--", "green"),
    #             ("experiment/archive/mve_avg_perfect/mve_catcher_raw.json", 6, 10, 200000, "MVE k=3", "-", "cyan"),
    #             ("experiment/archive/mve_avg_learned/catcher_raw.json", 3, 10, 200000, "MVE LM k=3", "--", "cyan"),
    #             ("experiment/archive/mve_avg_perfect/mve_catcher_raw.json", 10, 10, 200000, "MVE k=4", "-", "red"),
    #             ("experiment/archive/mve_avg_learned/catcher_raw.json", 5, 10, 200000, "MVE LM k=4", "--", "red")
    #             ]
    # draw_lunar_lander_dqn_linestyle(settings, save_path="plots/mve_avg_learned/catcher_raw/plot.png", interval=5000)


    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN", "-", "blue"),
    #             ("experiment/archive/mve_avg_perfect/mve_catcher_raw.json", 2, 10, 200000, "MVE k=2", "-", "green"),
    #             ("experiment/archive/mve_avg_learned/catcher_raw.json", 1, 10, 200000, "MVE LM k=2", "--", "green"),
    #             ("experiment/archive/mve_error_aware/catcher_raw.json", 4, 10, 200000, "MVE LM EW k=2", ":", "green"),
    #             ("experiment/archive/mve_avg_perfect/mve_catcher_raw.json", 10, 10, 200000,  "MVE k=4", "-", "red"),
    #             ("experiment/archive/mve_avg_learned/catcher_raw.json", 5, 10, 200000, "MVE LM k=4", "--", "red"),
    #             ("experiment/archive/mve_error_aware/catcher_raw.json", 5, 10, 200000, "MVE LM EW k=4", ":", "red")
    #             ]
    # draw_lunar_lander_dqn_linestyle(settings, save_path="plots/mve_error_aware/catcher_raw/plot.png", interval=5000)

    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
                # ("experiment/archive/mve_error_aware/catcher_raw.json", 0, 10, 200000, "MVE LM EW k=2 tau 100.0"),
                # ("experiment/archive/mve_error_aware/catcher_raw.json", 2, 10, 200000, "MVE LM EW k=2 tau 10.0"),
                # ("experiment/archive/mve_error_aware/catcher_raw.json", 4, 10, 200000, "MVE LM EW k=2 tau 1.0"),
                # ("experiment/archive/mve_error_aware/catcher_raw.json", 6, 10, 200000, "MVE LM EW k=2 tau 0.1"),

                # ("experiment/archive/mve_error_aware/catcher_raw.json", 1, 10, 200000, "MVE LM EW k=4 tau 100.0"),
                # ("experiment/archive/mve_error_aware/catcher_raw.json", 3, 10, 200000, "MVE LM EW k=4 tau 10.0"),
                # ("experiment/archive/mve_error_aware/catcher_raw.json", 5, 10, 200000, "MVE LM EW k=4 tau 1.0"),
                # ("experiment/archive/mve_error_aware/catcher_raw.json", 7, 10, 200000, "MVE LM EW k=4 tau 0.1"),
                # ]
    # draw_lunar_lander_dqn(settings, save_path="plots/mve_error_aware/catcher_raw/tau_plot_k4.png", interval=5000)



    # settings = [# ("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
    #             ("experiment/archive/mve_error_aware/catcher_raw.json", 0, 10, 200000, "MVE LM EW k=2 tau 100.0"),
    #             ("experiment/archive/mve_error_aware/catcher_raw.json", 2, 10, 200000, "MVE LM EW k=2 tau 10.0"),
    #             ("experiment/archive/mve_error_aware/catcher_raw.json", 4, 10, 200000, "MVE LM EW k=2 tau 1.0"),
    #             ("experiment/archive/mve_error_aware/catcher_raw.json", 6, 10, 200000, "MVE LM EW k=2 tau 0.1"),
    #
    #             # ("experiment/archive/mve_error_aware/catcher_raw.json", 1, 10, 200000, "MVE LM EW k=4 tau 100.0"),
    #             # ("experiment/archive/mve_error_aware/catcher_raw.json", 3, 10, 200000, "MVE LM EW k=4 tau 10.0"),
    #             # ("experiment/archive/mve_error_aware/catcher_raw.json", 5, 10, 200000, "MVE LM EW k=4 tau 1.0"),
    #             # ("experiment/archive/mve_error_aware/catcher_raw.json", 7, 10, 200000, "MVE LM EW k=4 tau 0.1"),
    #
    #
    #             ]
    # draw_expected_rollout(settings, save_path="plots/mve_error_aware/catcher_raw/rollout_length_plot_k2.png", interval=5000)


    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN-h512-1e-3"),
    #             ("experiment/archive/model_free_v4/catcher_raw.json", 0, 10, 200000, "DQN | h: 256 | 1e-3"),
    #             ("experiment/archive/model_free_v4/catcher_raw.json", 4, 10, 200000, "DQN | h: 256-256 | 1e-3"),
    #             ("experiment/archive/model_free_v4/catcher_raw.json", 8, 10, 200000,"DQN | h: 256-256-256 | 1e-3"),
    #             ("experiment/archive/model_free_v4/catcher_raw.json", 12, 10, 200000, "DQN | h: 256-256-256-256 | 1e-3"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/model_free_v4/catcher_raw/plot_e.png", interval=5000)

    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
    #             ("experiment/archive/search_dt_learned/catcher_raw.json", 1, 10, 200000, "Search-DT k=1"),
    #             ("experiment/archive/search_dt_learned/catcher_raw.json", 3, 10, 200000, "Search-DT k=2"),
    #             ("experiment/archive/search_dt_learned/catcher_raw.json", 5, 10, 200000, "Search-DT k=3"),
    #             ("experiment/archive/search_dt_learned/catcher_raw.json", 7, 10, 200000, "Search-DT k=4")
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/search_dt_learned/catcher_raw/plot.png", interval=5000)
    #
    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
    #             ("experiment/archive/search_dt_learnedv2/catcher_raw.json", 1, 10, 200000, "Search-DT k=1"),
    #             ("experiment/archive/search_dt_learnedv2/catcher_raw.json", 3, 10, 200000, "Search-DT k=2"),
    #             ("experiment/archive/search_dt_learnedv2/catcher_raw.json", 5, 10, 200000, "Search-DT k=3"),
    #             ("experiment/archive/search_dt_learnedv2/catcher_raw.json", 7, 10, 200000, "Search-DT k=4")
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/search_dt_learnedv2/catcher_raw/plot.png", interval=5000)


    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN", "-", "blue"),
    #             ("experiment/archive/mve_avg_perfect/mve_catcher_raw.json", 2, 10, 200000, "MVE k=2", "-", "green"),
    #             ("experiment/archive/mve_avg_learnedv2/catcher_raw.json", 1, 10, 200000, "MVE LM k=2", "--", "green"),
    #             ("experiment/archive/mve_avg_perfect/mve_catcher_raw.json", 6, 10, 200000, "MVE k=3", "-", "cyan"),
    #             ("experiment/archive/mve_avg_learnedv2/catcher_raw.json", 3, 10, 200000, "MVE LM k=3", "--", "cyan"),
    #             ("experiment/archive/mve_avg_perfect/mve_catcher_raw.json", 10, 10, 200000, "MVE k=4", "-", "red"),
    #             ("experiment/archive/mve_avg_learnedv2/catcher_raw.json", 5, 10, 200000, "MVE LM k=4", "--", "red")
    #             ]
    # draw_lunar_lander_dqn_linestyle(settings, save_path="plots/sprint_1/mve_avg_learnedv2/catcher_raw/plot_update_13.png", interval=5000,
    #                                 ylim=(-5, 60))

    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN-h512-1e-3"),
    #             ("experiment/archive/model_free_v4/catcher_raw.json", 1, 10, 200000, "DQN | h: 256 | 3e-4"),
    #             ("experiment/archive/model_free_v4/catcher_raw.json", 6, 10, 200000, "DQN | h: 256-256 | 1e-4"),
    #             ("experiment/archive/model_free_v4/catcher_raw.json",11, 10, 200000,"DQN | h: 256-256-256 | 3e-5"),
    #             ("experiment/archive/model_free_v4/catcher_raw_smaller_lr.json", 10, 10, 200000, "DQN | h: 256-256-256-256 | 1e-5"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/model_free_v4/catcher_raw/plot_smaller_lr.png", interval=5000)

    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
    #             ("experiment/archive/mve_avg_learned/catcher_raw.json", 5, 10, 200000, "MVE k=4 Juhn"),
    #             ("experiment/archive/mve_sw_model/catcher_raw.json", 5, 3, 200000, "MVE k=4 A"),
    #             ("experiment/archive/mve_sw_model/catcher_raw.json", 11, 3, 200000, "MVE k=4 B"),
    #             ("experiment/archive/mve_sw_model/catcher_raw.json", 17, 3, 200000, "MVE k=4 C"),
    #             ("experiment/archive/mve_sw_model/catcher_raw.json", 23, 3, 200000, "MVE k=4 D"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/mve_sw_model/catcher_raw/plot.png", interval=5000)



    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN", "-", "blue"),
    #
    #             ("experiment/archive/mve_avg_perfect/mve_catcher_raw.json", 2, 10, 200000, "MVE k=2", "-", "green"),
    #             ("experiment/archive/mve_avg_learnedv2/catcher_raw.json", 1, 10, 200000, "MVE LM k=2", "--", "green"),
    #             ("experiment/archive/mve_error_awarev2/catcher_raw.json", 4, 10, 200000, "MVE LM EW k=2", ":", "green"),
    #
    #             ("experiment/archive/mve_avg_perfect/mve_catcher_raw.json", 10, 10, 200000,  "MVE k=4", "-", "red"),
    #             ("experiment/archive/mve_avg_learnedv2/catcher_raw.json", 5, 10, 200000, "MVE LM k=4", "--", "red"),
    #             ("experiment/archive/mve_error_awarev2/catcher_raw.json", 5, 10, 200000, "MVE LM EW k=4", ":", "red")
    #             ]
    # draw_lunar_lander_dqn_linestyle(settings, save_path="plots/mve_error_awarev2/catcher_raw/plot.png", interval=5000)


    # settings = [# ("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
                # ("experiment/archive/mve_error_awarev2/catcher_raw.json", 0, 10, 200000, "MVE LM EW k=2 tau 100.0"),
                # ("experiment/archive/mve_error_awarev2/catcher_raw.json", 2, 10, 200000, "MVE LM EW k=2 tau 10.0"),
                # ("experiment/archive/mve_error_awarev2/catcher_raw.json", 4, 10, 200000, "MVE LM EW k=2 tau 1.0"),
                # ("experiment/archive/mve_error_awarev2/catcher_raw.json", 6, 10, 200000, "MVE LM EW k=2 tau 0.1"),

                # ("experiment/archive/mve_error_awarev2/catcher_raw.json", 1, 10, 200000, "MVE LM EW k=4 tau 100.0"),
                # ("experiment/archive/mve_error_awarev2/catcher_raw.json", 3, 10, 200000, "MVE LM EW k=4 tau 10.0"),
                # ("experiment/archive/mve_error_awarev2/catcher_raw.json", 5, 10, 200000, "MVE LM EW k=4 tau 1.0"),
                # ("experiment/archive/mve_error_awarev2/catcher_raw.json", 7, 10, 200000, "MVE LM EW k=4 tau 0.1"),


                # ]
    # draw_expected_rollout(settings, save_path="plots/mve_error_awarev2/catcher_raw/rollout_length_plot_k4.png", interval=5000)

    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
    #             ("experiment/config_files_2/mve_learned/catcher_raw.json", 20, 5, 200000, "MVE k=4 Juhn"),
    #             ("experiment/config_files_2/mve_learned/catcher_raw.json", 21, 5, 200000, "MVE k=4 Juhnv1"),
    #             ("experiment/config_files_2/mve_learned/catcher_raw.json", 22, 5, 200000, "MVE k=4 Juhnv3"),
    #             ("experiment/config_files_2/mve_learned/catcher_raw.json", 23, 5, 200000, "MVE k=4 FCModelNetv1"),
    #             ("experiment/config_files_2/mve_learned/catcher_raw.json", 24, 5, 200000, "MVE k=4 FCModelNetv2")]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/mve_learned/catcher_raw/plot.png", interval=5000)
    #
    #
    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
    #             ("experiment/config_files_2/search_learned/catcher_raw.json", 30, 5, 200000, "Search k=3 Juhn"),
    #             ("experiment/config_files_2/search_learned/catcher_raw.json", 26, 5, 200000, "Search k=3 Juhnv1"),
    #             ("experiment/config_files_2/search_learned/catcher_raw.json", 27, 5, 200000, "Search k=3 Juhnv3"),
    #             ("experiment/config_files_2/search_learned/catcher_raw.json", 23, 5, 200000, "Search k=3 FCModelNetv1"),
    #             ("experiment/config_files_2/search_learned/catcher_raw.json", 24, 5, 200000, "Search k=3 FCModelNetv2")]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/search_learned/catcher_raw/plot.png", interval=5000)
    #
    #
    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
    #             ("experiment/config_files_2/mve_learned/catcher_raw_v2.json", 18, 5, 200000, "MVE k=4 Juhnv3"),
    #             ("experiment/config_files_2/mve_learned/catcher_raw_v2.json", 19, 5, 200000, "MVE k=4 FCModelNetv3"),
    #             ("experiment/config_files_2/mve_learned/catcher_raw_v2.json", 20, 5, 200000, "MVE k=4 FCModelNetv4"),
    #             ("experiment/config_files_2/mve_learned/catcher_raw_v2.json", 21, 5, 200000, "MVE k=4 FCModelNetv5"),
    #             ("experiment/config_files_2/mve_learned/catcher_raw_v2.json", 22, 5, 200000, "MVE k=4 FCModelNetv6"),
    #             ("experiment/config_files_2/mve_learned/catcher_raw_v2.json", 29, 5, 200000, "MVE k=4 FCModelNetv7")]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/mve_learned/catcher_raw/plot_v2_mse.png", interval=5000)
    #
    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
    #             ("experiment/config_files_2/mve_learned/catcher_raw_v3.json", 18, 5, 200000, "MVE k=4 Juhnv3") ,
    #             ("experiment/config_files_2/mve_learned/catcher_raw_v3.json", 19, 5, 200000, "MVE k=4 FCModelNetv3") ,
    #             ("experiment/config_files_2/mve_learned/catcher_raw_v3.json", 20, 5, 200000, "MVE k=4 FCModelNetv4") ,
    #             ("experiment/config_files_2/mve_learned/catcher_raw_v3.json", 21, 5, 200000, "MVE k=4 FCModelNetv5") ,
    #             ("experiment/config_files_2/mve_learned/catcher_raw_v3.json", 22, 5, 200000, "MVE k=4 FCModelNetv6") ,
    #             ("experiment/config_files_2/mve_learned/catcher_raw_v3.json", 23, 5, 200000, "MVE k=4 FCModelNetv7") ]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/mve_learned/catcher_raw/plot_v3_mse.png", interval=5000)
    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
    #             ("experiment/config_files_2/mve_learned/catcher_raw_v4.json", 12, 5, 200000, "MVE k=4 Juhnv3") ,
    #             ("experiment/config_files_2/mve_learned/catcher_raw_v4.json", 7, 5, 200000, "MVE k=4 FCModelNetv3") ,
    #             ("experiment/config_files_2/mve_learned/catcher_raw_v4.json", 2, 5, 200000, "MVE k=4 FCModelNetv4") ,
    #             ("experiment/config_files_2/mve_learned/catcher_raw_v4.json", 9, 5, 200000, "MVE k=4 FCModelNetv5") ,
    #             ("experiment/config_files_2/mve_learned/catcher_raw_v4.json", 22, 5, 200000, "MVE k=4 FCModelNetv4") ,
    #             ("experiment/config_files_2/mve_learned/catcher_raw_v4.json", 5, 5, 200000, "MVE k=4 FCModelNetv7") ]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/mve_learned/catcher_raw/plot_v4_mse.png", interval=5000)
    #
    #
    #
    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
    #             ("experiment/config_files_2/search_learned/catcher_raw_v2.json", 30, 5, 200000, "Search k=3 Juhnv3") ,
    #             ("experiment/config_files_2/search_learned/catcher_raw_v2.json", 31, 5, 200000, "Search k=3 FCModelNetv3") ,
    #             ("experiment/config_files_2/search_learned/catcher_raw_v2.json", 32, 5, 200000, "Search k=3 FCModelNetv4" ),
    #             ("experiment/config_files_2/search_learned/catcher_raw_v2.json", 21, 5, 200000, "Search k=3 FCModelNetv5") ,
    #             ("experiment/config_files_2/search_learned/catcher_raw_v2.json", 22, 5, 200000, "Search k=3 FCModelNetv6") ,
    #             ("experiment/config_files_2/search_learned/catcher_raw_v2.json", 29, 5, 200000, "Search k=3 FCModelNetv7") ]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/search_learned/catcher_raw/plot_v2.png", interval=5000)
    #
    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
    #             ("experiment/config_files_2/search_learned/catcher_raw_v3.json", 24, 5, 200000, "Search k=3 Juhnv3") ,
    #             ("experiment/config_files_2/search_learned/catcher_raw_v3.json", 19, 5, 200000, "Search k=3 FCModelNetv3") ,
    #             ("experiment/config_files_2/search_learned/catcher_raw_v3.json", 26, 5, 200000, "Search k=3 FCModelNetv4" ) ,
    #             ("experiment/config_files_2/search_learned/catcher_raw_v3.json", 21, 5, 200000, "Search k=3 FCModelNetv5") ,
    #             ("experiment/config_files_2/search_learned/catcher_raw_v3.json", 22, 5, 200000, "Search k=3 FCModelNetv6") ,
    #             ("experiment/config_files_2/search_learned/catcher_raw_v3.json", 29, 5, 200000, "Search k=3 FCModelNetv7")]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/search_learned/catcher_raw/plot_v3.png", interval=5000)
    #

    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
    #             ("experiment/config_files_2/search_learned/catcher_raw_v4.json", 12, 5, 200000, "Search k=3 Juhnv3") ,
    #             ("experiment/config_files_2/search_learned/catcher_raw_v4.json", 1, 5, 200000, "Search k=3 FCModelNetv3") ,
    #             ("experiment/config_files_2/search_learned/catcher_raw_v4.json", 14, 5, 200000, "Search k=3 FCModelNetv4" ) ,
    #             ("experiment/config_files_2/search_learned/catcher_raw_v4.json", 15, 5, 200000, "Search k=3 FCModelNetv5") ,
    #             ("experiment/config_files_2/search_learned/catcher_raw_v4.json", 16, 5, 200000, "Search k=3 FCModelNetv6") ,
    #             ("experiment/config_files_2/search_learned/catcher_raw_v4.json", 17, 5, 200000, "Search k=3 FCModelNetv7") ]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/search_learned/catcher_raw/plot_v4.png", interval=5000)

    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "h=512"),
    #             ("experiment/config_files_2/model_free/catcher_raw.json", 25, 9, 200000, "h=256"),
    #             ("experiment/config_files_2/model_free/catcher_raw.json", 21, 9, 200000, "h=128"),
    #             ("experiment/config_files_2/model_free/catcher_raw.json", 16, 10, 200000, "h=64"),
    #             ("experiment/config_files_2/model_free/catcher_raw.json", 12, 10, 200000, "h=32"),
    #             ("experiment/config_files_2/model_free/catcher_raw.json", 8, 10, 200000, "h=16"),
    #             ("experiment/config_files_2/model_free/catcher_raw.json", 4, 10, 200000, "h=8"),
    #             ("experiment/config_files_2/model_free/catcher_raw.json", 1, 10, 200000, "h=4"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/model_free/catcher_raw/plot.png", interval=5000,
    #                       ylim=(-5, 60))

    # settings = [("experiment/config_files_2/model_free/catcher_raw_huber.json", 25, 9, 200000, "h=256"),
    #             ("experiment/config_files_2/model_free/catcher_raw_huber.json", 21, 9, 200000, "h=128"),
    #             ("experiment/config_files_2/model_free/catcher_raw_huber.json", 17, 10, 200000, "h=64"),
    #             ("experiment/config_files_2/model_free/catcher_raw_huber.json", 12, 10, 200000, "h=32"),
    #             ("experiment/config_files_2/model_free/catcher_raw_huber.json", 8, 10, 200000, "h=16"),
    #             ("experiment/config_files_2/model_free/catcher_raw_huber.json", 4, 10, 200000, "h=8"),
    #             ("experiment/config_files_2/model_free/catcher_raw_huber.json", 1, 10, 200000, "h=4"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/model_free/catcher_raw/plot_huber.png", interval=5000,
    #                       ylim=(-5, 60))

    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "h=512"),
    #             ("experiment/config_files_2/model_free/catcher_raw.json", 16, 10, 200000, "h=64"),
    #             ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 215, 10, 200000, "Vh=64 Mh=8"),
    #             ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 201, 10, 200000, "Vh=64 Mh=16"),
    #             ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 217, 10, 200000, "Vh=64 Mh=32"),
    #             ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 228, 10, 200000, "Vh=64 Mh=64"),
    #             # ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 188, 10, 200000, "Vh=64 Mh=64"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/search_learned/catcher_raw/capacity_plot_64_b.png", interval=5000,
    #                       ylim=(-5, 60))
    #
    #
    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "h=512"),
    #             ("experiment/config_files_2/model_free/catcher_raw.json", 21, 9, 200000, "h=128"),
    #             ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 295, 10, 200000, "Vh=128 Mh=8"),
    #             ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 296, 10, 200000, "Vh=128 Mh=16"),
    #             ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 297, 10, 200000, "Vh=128 Mh=32"),
    #             ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 293, 10, 200000, "Vh=128 Mh=64"),
    #             ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 294, 10, 200000, "Vh=128 Mh=128"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/search_learned/catcher_raw/capacity_plot_128.png", interval=5000,
    #                       ylim=(-5, 60))
    #
    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "h=512"),
    #             ("experiment/config_files_2/model_free/catcher_raw.json", 8, 10, 200000, "Vh=16"),
    #             ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 95, 10, 200000, "Vh=16 Mh=8") ,
    #             ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 101, 10, 200000, "Vh=16 Mh=16") ,
    #             # ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 102, 10, 200000, "Vh=16 Mh=32") ,
    #             # ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 103, 10, 200000, "Vh=16 Mh=64") ,
    #             # ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 104, 10, 200000, "Vh=16 Mh=128"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/search_learned/catcher_raw/capacity_plot_16.png", interval=5000,
    #                       ylim=(-5, 60))
    #
    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "h=512"),
    #             ("experiment/config_files_2/model_free/catcher_raw.json", 4, 10, 200000, "Vh=8"),
    #             ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 40, 10, 200000, "Vh=8 Mh=8"),
    #             # ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 11, 10, 200000, "Vh=8 Mh=16"),
    #             # ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 37, 10, 200000, "Vh=8 Mh=32"),
    #             # ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 28, 10, 200000, "Vh=8 Mh=64") ,
    #             # ("experiment/config_files_2/search_learned/capacity_catcher_raw.json", 34, 10, 200000, "Vh=8 Mh=128"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/search_learned/catcher_raw/capacity_plot_8.png", interval=5000,
    #                       ylim=(-5, 60))

    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "h=512"),
    #             ("experiment/config_files_2/model_free/catcher_raw_huber.json", 4, 10, 200000, "Vh=8"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_8.json", 0, 10, 200000, "Vh=8 Mh=4"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_8.json", 1, 10, 200000, "Vh=8 Mh=8"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/search_learned/catcher_raw_huber/capacity_plot_8.png", interval=5000,
    #                       ylim=(-5, 60))

    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "h=512"),
    #             ("experiment/config_files_2/model_free/catcher_raw_huber.json", 8, 10, 200000, "Vh=16"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_16.json", 0, 10, 200000, "Vh=16 Mh=4"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_16.json", 1, 10, 200000, "Vh=16 Mh=8"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_16.json", 2, 10, 200000, "Vh=16 Mh=16"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/search_learned/catcher_raw_huber/capacity_plot_16.png", interval=5000,
    #                       ylim=(-5, 60))

    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "h=512"),
    #             ("experiment/config_files_2/model_free/catcher_raw_huber.json", 13, 10, 200000, "Vh=32"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_32.json", 0, 10, 200000, "Vh=32 Mh=4"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_32.json", 1, 10, 200000, "Vh=32 Mh=8"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_32.json", 2, 10, 200000, "Vh=32 Mh=16"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_32.json", 3, 10, 200000, "Vh=32 Mh=32"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/search_learned/catcher_raw_huber/capacity_plot_32.png", interval=5000,
    #                       ylim=(-5, 60))

    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "h=512"),
    #             ("experiment/config_files_2/model_free/catcher_raw_huber.json", 17, 10, 200000, "Vh=64"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_64.json", 0, 10, 200000, "Vh=64 Mh=8"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_64.json", 1, 10, 200000, "Vh=64 Mh=16"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_64.json", 2, 10, 200000, "Vh=64 Mh=32"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_64.json", 3, 10, 200000, "Vh=64 Mh=64"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/search_learned/catcher_raw_huber/capacity_plot_64.png", interval=5000,
    #                       ylim=(-5, 60))

    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "h=512"),
    #             ("experiment/config_files_2/model_free/catcher_raw_huber.json", 21, 10, 200000, "Vh=128"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_128.json", 0, 10, 200000, "Vh=128 Mh=16"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_128.json", 1, 10, 200000, "Vh=128 Mh=32"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_128.json", 2, 10, 200000, "Vh=128 Mh=64"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_128.json", 3, 10, 200000, "Vh=128 Mh=128"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/search_learned/catcher_raw_huber/capacity_plot_128.png", interval=5000,
    #                       ylim=(-5, 60))

    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "h=512"),
    #             ("experiment/config_files_2/model_free/catcher_raw_huber.json", 25, 10, 200000, "Vh=256"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_128.json", 0, 10, 200000, "Vh=256 Mh=32"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_128.json", 1, 10, 200000, "Vh=256 Mh=64"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_128.json", 2, 10, 200000, "Vh=256 Mh=128"),
    #             ("experiment/config_files_2/search_learned/capacity/catcher_raw_128.json", 3, 10, 200000, "Vh=256 Mh=256"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/search_learned/catcher_raw_huber/capacity_plot_256.png", interval=5000,
    #                       ylim=(-5, 60))




    ################################################




    ################################################
    ##### Acrobot
    # settings = [("experiment/archive/model_free_v3/dqn_test.json", 0, 10, 100000, "DQN-Acrobot"),
    #             ("experiment/archive/mve_perfect/mve_acrobot.json", 0, 10, 100000, "MVE-Acrobot K=2"),
    #             ("experiment/archive/mve_perfect/mve_acrobot.json", 4, 10, 100000, "MVE-Acrobot K=3"),
    #             ("experiment/archive/mve_perfect/mve_acrobot.json", 8, 10, 100000, "MVE-Acrobot K=4"),
    #             ("experiment/archive/mve_perfect/mve_acrobot.json", 12, 10, 100000, "MVE-Acrobot K=5"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/mve_perfect/acrobot/plot.png", interval=2000,
    #                       title="MF vs MVE")

    # settings = [("experiment/archive/model_free_v3/dqn_test.json", 0, 10, 100000, "DQN-Acrobot"),
    #             ("experiment/archive/mve_avg_perfect/mve_acrobot.json", 0, 10, 100000, "MVE-Acrobot K=2"),
    #             ("experiment/archive/mve_avg_perfect/mve_acrobot.json", 4, 10, 100000, "MVE-Acrobot K=3"),
    #             ("experiment/archive/mve_avg_perfect/mve_acrobot.json", 8, 10, 100000, "MVE-Acrobot K=4"),
    #             ("experiment/archive/mve_avg_perfect/mve_acrobot.json", 12, 10, 100000, "MVE-Acrobot K=5"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_1/mve_avg_perfect/acrobot/plot_update_13.png", interval=2000,
    #                       title="MF vs MVE Average", ylim=(-550, -50))
    #
    #
    # settings = [("experiment/archive/model_free_v3/dqn_test.json", 0, 10, 100000, "DQN-Acrobot"),
    #             ("experiment/archive/zach_perfect_2/zach_acrobot.json", 2, 10, 100000, "Zach DQN-Acrobot 16 x 1"),
    #             ("experiment/archive/zach_perfect_2/zach_acrobot.json", 6, 10, 100000, "Zach-DQN-Acrobot 8 x 2"),
    #             ("experiment/archive/zach_perfect_2/zach_acrobot.json", 8, 10, 100000, "Zach-DQN-Acrobot 4 x 4"),
    #             ("experiment/archive/zach_perfect_2/zach_acrobot.json", 12, 10, 100000, "Zach-DQN-Acrobot 2 x 8"),
    #             ("experiment/archive/zach_perfect_2/zach_acrobot.json", 16, 10, 100000, "Zach-DQN-Acrobot 1 x 16"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_1/zach_perfect_2/acrobot/plot_update_13.png", interval=2000,
    #                       title="MF vs Zach DQN", ylim=(-550, -50))
    #
    #
    #
    # settings = [("experiment/archive/model_free_v3/dqn_test.json", 0, 10, 100000, "DQN-Acrobot"),
    #            ("experiment/archive/search_dt_perfect/acrobot.json", 0, 10, 100000, "Search-DT k=1"),
    #            ("experiment/archive/search_dt_perfect/acrobot.json", 2, 10, 100000, "Search-DT k=2"),
    #            ("experiment/archive/search_dt_perfect/acrobot.json", 4, 10, 100000, "Search-DT k=3"),
    #            ("experiment/archive/search_dt_perfect/acrobot.json", 6, 10, 100000, "Search-DT k=4")]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_1/search_dt_perfect/acrobot/plot_update_13.png", interval=2000,
    #                       ylim=(-550, -50))


    # settings = [("experiment/archive/model_free_v3/dqn_test.json", 0, 10, 100000, "DQN", "-", "blue"),
    #             ("experiment/archive/mve_avg_perfect/mve_acrobot.json", 0, 10, 100000, "MVE K=2", "-", "green"),
    #             ("experiment/archive/mve_avg_learned/acrobot.json", 0, 10, 100000, "MVE LM K=2", "--", "green"),
    #             ("experiment/archive/mve_avg_perfect/mve_acrobot.json", 4, 10, 100000, "MVE K=3", "-", "cyan"),
    #             ("experiment/archive/mve_avg_learned/acrobot.json", 2, 10, 100000, "MVE LM K=3", "--", "cyan"),
    #             ("experiment/archive/mve_avg_perfect/mve_acrobot.json", 8, 10, 100000, "MVE K=4", "-", "red"),
    #             ("experiment/archive/mve_avg_learned/acrobot.json", 4, 10, 100000, "MVE LM K=4", "--", "red")
    #             ]
    # draw_lunar_lander_dqn_linestyle(settings, save_path="plots/mve_avg_learned/acrobot/plot.png", interval=2000,
    #                       title="Perfect vs Learned")

    # settings = [("experiment/archive/model_free_v3/dqn_test.json", 0, 10, 100000, "DQN", "-", "blue"),
    #             ("experiment/archive/mve_avg_perfect/mve_acrobot.json", 0, 10, 100000, "MVE K=2", "-", "green"),
    #             ("experiment/archive/mve_sw_model/acrobot.json", 19, 10, 100000, "MVE LM K=2", "--", "green"),
    #             ("experiment/archive/mve_avg_perfect/mve_acrobot.json", 4, 10, 100000, "MVE K=3", "-", "cyan"),
    #             ("experiment/archive/mve_avg_learned/acrobot.json", 21, 10, 100000, "MVE LM K=3", "--", "cyan"),
    #             ("experiment/archive/mve_avg_perfect/mve_acrobot.json", 8, 10, 100000, "MVE K=4", "-", "red"),
    #             ("experiment/archive/mve_sw_model/acrobot.json", 23, 10, 100000, "MVE LM K=4", "--", "red")
    #             ]
    # draw_lunar_lander_dqn_linestyle(settings, save_path="plots/mve_avg_learned/acrobot/plot_model_sw.png", interval=2000,
    #                       title="Perfect vs Learned")

    # settings = [("experiment/archive/model_free_v3/dqn_test.json", 0, 10, 100000, "DQN "),
    #             ("experiment/archive/mve_avg_learned/acrobot.json", 4, 10, 100000, "MVE K=4 Juhn"),
    #             ("experiment/archive/mve_sw_model/acrobot.json", 4, 10, 100000, "MVE K=4 A"),
    #             ("experiment/archive/mve_sw_model/acrobot.json", 10, 10, 100000, "MVE K=4 B"),
    #             ("experiment/archive/mve_sw_model/acrobot.json", 16, 10, 100000, "MVE K=4 C"),
    #             ("experiment/archive/mve_sw_model/acrobot.json", 23, 10, 100000, "MVE K=4 D"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/mve_sw_model/acrobot/plot.png", interval=2000,
    #                       title="Model Sweep")
    #
    # settings = [("experiment/archive/model_free_v3/dqn_test.json", 0, 10, 100000, "DQN", "-", "blue"),
    #             ("experiment/archive/mve_avg_perfect/mve_acrobot.json", 0, 10, 100000, "MVE K=2", "-", "green"),
    #             ("experiment/archive/mve_avg_learnedv2/acrobot.json", 0, 10, 100000, "MVE LM K=2", "--", "green"),
    #             ("experiment/archive/mve_avg_perfect/mve_acrobot.json", 4, 10, 100000, "MVE K=3", "-", "cyan"),
    #             ("experiment/archive/mve_avg_learnedv2/acrobot.json", 2, 10, 100000, "MVE LM K=3", "--", "cyan"),
    #             ("experiment/archive/mve_avg_perfect/mve_acrobot.json", 8, 10, 100000, "MVE K=4", "-", "red"),
    #             ("experiment/archive/mve_avg_learnedv2/acrobot.json", 4, 10, 100000, "MVE LM K=4", "--", "red")
    #             ]
    # draw_lunar_lander_dqn_linestyle(settings, save_path="plots/sprint_1/mve_avg_learnedv2/acrobot/plot_update_13.png", interval=2000,
    #                       title="Perfect vs Learned", ylim=(-550, -50))




    # settings = [("experiment/archive/model_free_v3/dqn_test.json", 0, 10, 100000, "DQN", "-", "blue"),
    #             ("experiment/archive/search_dt_perfect/acrobot.json", 0, 10, 100000, "Search k=1", "-", "green"),
    #             ("experiment/archive/search_dt_learnedv2/acrobot.json", 0, 10, 100000, "Search LM k=1", "--", "green"),
    #             ("experiment/archive/search_dt_perfect/acrobot.json", 2, 10, 100000, "Search k=2", "-", "cyan"),
    #             ("experiment/archive/search_dt_learnedv2/acrobot.json", 2, 10, 100000, "Search LM k=2", "--", "cyan"),
    #             ("experiment/archive/search_dt_perfect/acrobot.json", 4, 10, 100000, "Search k=3", "-", "red"),
    #             ("experiment/archive/search_dt_learnedv2/acrobot.json", 5, 10, 100000, "Search LM k=3", "--", "red"),
    #             ("experiment/archive/search_dt_perfect/acrobot.json", 6, 10, 100000, "Search k=4", "-", "yellow"),
    #             ("experiment/archive/search_dt_learnedv2/acrobot.json", 6, 10, 100000, "Search LM k=4", "--", "yellow")
    #             ]
    # draw_lunar_lander_dqn_linestyle(settings, save_path="plots/sprint_1/search_dt_learnedv2/acrobot/plot_update_13.png", interval=2000,
    #                                 title="MF vs MVE", ylim=(-550, -50))

    # settings = [("experiment/archive/model_free_v3/dqn_test.json", 0, 10, 100000, "DQN-Acrobot"),
    #             ("experiment/config_files_2/mve_learned/acrobot.json", 8, 5, 100000, "MVE HHH"),
    #             ("experiment/config_files_2/mve_learned/acrobot.json", 9, 5, 100000, "MVE H16"),
    #             ("experiment/config_files_2/mve_learned/acrobot.json", 10, 5, 100000, "MVE H8"),
    #             ("experiment/config_files_2/mve_learned/acrobot.json", 7, 5, 100000, "MVE H4"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/sprint_2/mve_learned/acrobot/plot.png", interval=2000,
    #                       title="MVE - Model Capacity - K=4", ylim=(-550, -50))



    ################################################


    ################################################
    ##### Catcher Visual
    # settings = [("experiment/archive/model_free_v2/dqn_catcher_visual.json", 6, 5, 500000, "DQN"),
    #             ("experiment/archive/mve_avg_perfect/mve_catcher_visual.json", 0, 6, 500000, "MVE DQN k=2"),
    #             ("experiment/archive/mve_avg_perfect/mve_catcher_visual.json", 3, 5, 500000, "MVE DQN k=3"),
    #             ("experiment/archive/mve_avg_perfect/mve_catcher_visual.json", 6, 4, 500000, "MVE DQN k=4"),
    #             ("experiment/archive/mve_avg_perfect/mve_catcher_visual.json", 9, 4, 500000, "MVE DQN k=5"),
    #
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/mve_avg_perfect/catcher_visual/plot.png", interval=10000)
    #
    # settings = [("experiment/archive/model_free_v2/dqn_catcher_visual.json", 6, 5, 500000, "DQN"),
    #             ("experiment/archive/zach_perfect_2/zach_catcher_visual_b.json", 0, 6, 500000, "Zach DQN-Acrobot 16 x 1"),
    #             ("experiment/archive/zach_perfect_2/zach_catcher_visual_b.json", 3, 8, 500000, "Zach DQN-Acrobot 8 x 2"),
    #             ("experiment/archive/zach_perfect_2/zach_catcher_visual_b.json", 6, 8, 500000, "Zach DQN-Acrobot 4 x 4"),
    #             ("experiment/archive/zach_perfect_2/zach_catcher_visual_b.json", 9, 6, 500000, "Zach DQN-Acrobot 2 x 8"),
    #             ("experiment/archive/zach_perfect_2/zach_catcher_visual_b.json", 12, 5, 500000, "Zach DQN-Acrobot 1 x 16"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/zach_perfect_2/catcher_visual/plot.png", interval=10000)

    ################################################

    # settings = [("experiment/archive/model_free/dqn_catcher_raw.json", 1, 5, 400000, "DQN-128"),
    #             ("experiment/archive/model_free/dqn_catcher_raw.json", 5, 5, 400000, "DQN-256"),
    #             ("experiment/archive/model_free/dqn_catcher_raw.json", 10, 5, 400000, "DQN-512"),
    #             ("experiment/archive/model_free/dqn_catcher_raw_2h.json", 1, 5, 400000, "DQN-2x128"),
    #             ("experiment/archive/model_free/dqn_catcher_raw_2h.json", 3, 5, 400000, "DQN-2x256"),
    #             ("experiment/archive/model_free/dqn_catcher_raw_2h.json", 5, 5, 400000, "DQN-2x512"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/plot.pdf", interval=10000)
    #
    # settings = [("experiment/archive/model_free/dqn_catcher_visual_gray.json", 3, 6, 350000, "DQN-Wes")]
    # draw_lunar_lander_dqn(settings, save_path="plots/plot_visual_gray.png", interval=10000)

    # settings = [("experiment/archive/model_free/dqn_catcher_visual_gray.json", 3, 6, 200000, "DQN-Wes-3"),
    #             ("experiment/archive/model_free/dqn_catcher_visual_gray.json", 2, 6, 200000, "DQN-Wes-2"),
    #             ("experiment/archive/model_free/dqn_catcher_visual_gray.json", 1, 6, 200000, "DQN-Wes-1"),
    #             ("experiment/archive/model_free/dqn_catcher_visual_gray.json", 0, 6, 200000, "DQN-Wes-0")]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/plot_visual_gray_sw.pdf", interval=10000)

    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 0, 10, 200000, "DQN 0.001"),
    #             ("experiment/archive/model_free_v2/dqn_catcher_raw.json", 1, 10, 200000, "DQN 0.0003"),
    #             ("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN 0.0001"),
    #             ("experiment/archive/model_free_v2/dqn_catcher_raw.json", 3, 10, 200000, "DQN 0.00003")]
    # draw_lunar_lander_dqn(settings, save_path="plots/model_free_v2/catcher_raw/plot.png", interval=5000)

    # settings = [
    #
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite.json", 0, 5, 400000, "DQN 0.001"),
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite.json", 1, 5, 400000, "DQN 0.0003"),
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite.json", 2, 5, 400000, "DQN 0.0001"),
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite.json", 3, 5, 400000, "DQN 3e-05"),
    #
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite.json", 4, 5, 400000, "DQN 0.001"),
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite.json", 5, 5, 400000, "DQN 0.0003"),
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite.json", 6, 5, 400000, "DQN 0.0001"),
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite.json", 7, 5, 400000, "DQN 3e-05"),
    #
    #             ("experiment/archive/model_free_v2/dqn_catcher_lite.json", 8, 5, 400000, "DQN 0.001"),
    #             ("experiment/archive/model_free_v2/dqn_catcher_lite.json", 9, 5, 400000, "DQN 0.0003"),
    #             ("experiment/archive/model_free_v2/dqn_catcher_lite.json", 10, 5, 400000, "DQN 0.0001"),
    #             ("experiment/archive/model_free_v2/dqn_catcher_lite.json", 11, 5, 400000, "DQN 3e-05"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/model_free_v2/catcher_lite_c.png", interval=10000)

    # settings = [
    #             ("experiment/archive/model_free_v2/dqn_catcher_visual.json", 3, 5, 500000, "DQN Wes Big"),
    #             ("experiment/archive/model_free_v2/dqn_catcher_visual.json", 0, 5, 500000, "DQN Wes Small"),
    #             ("experiment/archive/model_free_v2/dqn_catcher_visual.json", 6, 5, 500000, "DQN Kenny"),
    #
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/model_free_v2/catcher_visual/plot.png", interval=10000)

    # settings = [
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite_b.json", 0, 5, 600000, "DQN 0.0003"),
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite_b.json", 1, 5, 600000, "DQN 0.0001"),
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite_b.json", 2, 5, 600000, "DQN 3e-05"),
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite_b.json", 3, 5, 600000, "DQN 1e-05"),
    #
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite_b.json", 4, 5, 600000, "DQN 0.0003"),
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite_b.json", 5, 5, 600000, "DQN 0.0001"),
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite_b.json", 6, 5, 600000, "DQN 3e-05"),
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite_b.json", 7, 5, 600000, "DQN 1e-05"),
    #
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite_b.json", 8, 5, 600000, "DQN 0.0003"),
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite_b.json", 9, 5, 600000, "DQN 0.0001"),
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite_b.json", 10, 5, 600000, "DQN 3e-05"),
    #             # ("experiment/archive/model_free_v2/dqn_catcher_lite_b.json", 11, 5, 600000, "DQN 1e-05"),
    #
    #             ("experiment/archive/model_free_v2/dqn_catcher_lite_b.json", 12, 5, 600000, "DQN 0.0003"),
    #             ("experiment/archive/model_free_v2/dqn_catcher_lite_b.json", 13, 5, 600000, "DQN 0.0001"),
    #             ("experiment/archive/model_free_v2/dqn_catcher_lite_b.json", 14, 5, 600000, "DQN 3e-05"),
    #             ("experiment/archive/model_free_v2/dqn_catcher_lite_b.json", 15, 5, 600000, "DQN 1e-05"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/model_free_v2/catcher_lite/600k/catcher_lite_c.png", interval=10000)


    # print("dqn_catacher_raw")
    # _eval_lines(config_files='experiment/archive/model_free_v2/dqn_catcher_raw.json', last_points=20, start_idx=0, end_idx=40, max_steps=200000, interval=5000)
    # print("mve_perfect_catcher_raw")
    # _eval_lines(config_files='experiment/archive/mve_perfect/mve_catcher_raw.json', last_points=20, start_idx=0, end_idx=160, max_steps=200000, interval=5000)
    # settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
    #             ("experiment/archive/mve_perfect/mve_catcher_raw.json", 2, 10, 200000, "MVE k=2"),
    #             ("experiment/archive/mve_perfect/mve_catcher_raw.json", 6, 10, 200000, "MVE k=3"),
    #             ("experiment/archive/mve_perfect/mve_catcher_raw.json", 10, 10, 200000, "MVE k=4"),
    #             ("experiment/archive/mve_perfect/mve_catcher_raw.json", 14, 10, 200000, "MVE k=5"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/mve_perfect/catcher_raw/plot_a.png", interval=5000, title="steps_size 0.0001")

    # settings = [("experiment/archive/model_free_v2/dqn_test.json", 56, 5, 200000, "DQN-Acrobot discount=1.0"),
    #             ("experiment/archive/model_free_v2/dqn_test.json", 32, 5, 200000, "DQN-Acrobot discount=0.999"),
    #             ("experiment/archive/model_free_v2/dqn_test.json", 8, 5, 200000, "DQN-Acrobot discount=0.99"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/model_free_v2/acrobot/plot_a.png", interval=10000,
    #                       title="Discount rates | h: 512 x 512, LR 0.001, TNUF 256")
    # settings = [("experiment/archive/model_free_v2/dqn_test.json", 32, 5, 200000, "DQN-Acrobot h:512 x 512"),
    #             ("experiment/archive/model_free_v2/dqn_test.json", 28, 5, 200000, "DQN-Acrobot  h: 512"),
    #             ("experiment/archive/model_free_v2/dqn_test.json", 26, 5, 200000, "DQN-Acrobot  h: 128"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/model_free_v2/acrobot/plot_b.png", interval=10000,
    #                       title="Architectures | TNUF 256")

    # settings = [("experiment/archive/model_free_v2/dqn_test.json", 68, 5, 200000, "DQN-Acrobot discount=1.0"),
    #             ("experiment/archive/model_free_v2/dqn_test.json", 44, 5, 200000, "DQN-Acrobot discount=0.999"),
    #             ("experiment/archive/model_free_v2/dqn_test.json", 20, 5, 200000, "DQN-Acrobot discount=0.99"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/model_free_v2/acrobot/plot_c.png", interval=10000,
    #                       title="Discount rates | h: 512 x 512, LR 0.001, TNUF 1024")

    # settings = [("experiment/archive/model_free_v2/dqn_test.json", 58, 5, 200000, "DQN-Acrobot discount=1.0"),
    #             ("experiment/archive/model_free_v2/dqn_test.json", 34, 5, 200000, "DQN-Acrobot discount=0.999"),
    #             ("experiment/archive/model_free_v2/dqn_test.json", 10, 5, 200000, "DQN-Acrobot discount=0.99"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/model_free_v2/acrobot/plot_d.png", interval=10000,
    #                       title="Discount rates | h: 512 x 512, LR 0.0001, TNUF 256")

    # settings = [("experiment/archive/model_free_v2/dqn_test.json", 48, 5, 200000, "DQN-Acrobot LR=0.001"),
    #             ("experiment/archive/model_free_v2/dqn_test.json", 49, 5, 200000, "DQN-Acrobot LR=0.0003"),
    #             ("experiment/archive/model_free_v2/dqn_test.json", 50, 5, 200000, "DQN-Acrobot LR=0.0001"),
    #             ("experiment/archive/model_free_v2/dqn_test.json", 51, 5, 200000, "DQN-Acrobot LR=0.00003"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/model_free_v2/acrobot/plot_g.png", interval=10000,
    #                       title="Discount rates | h: 128,  TNUF 256, discount=1.0")

    # settings = [("experiment/archive/model_free_v3/dqn_test.json", 0, 10, 100000, "DQN-Acrobot"),
    #             ("experiment/archive/mve_perfect/mve_acrobot.json", 0, 10, 100000, "MVE-Acrobot LR=0.001 K=2"),
    #             ("experiment/archive/mve_perfect/mve_acrobot.json", 4, 10, 100000, "MVE-Acrobot LR=0.001 K=3"),
    #             ("experiment/archive/mve_perfect/mve_acrobot.json", 8, 10, 100000, "MVE-Acrobot LR=0.001 K=4"),
    #             ("experiment/archive/mve_perfect/mve_acrobot.json", 12, 10, 100000, "MVE-Acrobot LR=0.001 K=5"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/mve_perfect/acrobot/plot_a.png", interval=2000,
    #                       title="MF vs MVE LR=0.001")

    # settings = [("experiment/archive/model_free_v3/dqn_navigation_c.json", 2, 10, 300000, "DQN"),
    #             ("experiment/archive/mve_perfect/mve_navigation_c.json", 1, 10, 300000, "MVE K=2"),
    #             ("experiment/archive/mve_perfect/mve_navigation_c.json", 6, 10, 300000, "MVE K=3"),
    #             ("experiment/archive/mve_perfect/mve_navigation_c.json", 10, 10, 300000, "MVE K=4"),
    #             ("experiment/archive/mve_perfect/mve_navigation_c.json", 13, 10, 300000, "MVE K=5"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/mve_perfect/navigation/plot.png", interval=10000,
    #                       title="MF vs MVE")


    # settings = [("experiment/archive/model_free_v3/dqn_cartpole.json", 1, 10, 50000, "DQN"),
    #             ("experiment/archive/mve_perfect/mve_cartpole.json", 1, 10, 50000, "MVE K=2"),
    #             ("experiment/archive/mve_perfect/mve_cartpole.json", 5, 10, 50000, "MVE K=3"),
    #             ("experiment/archive/mve_perfect/mve_cartpole.json", 9, 10, 50000, "MVE K=4"),
    #             ("experiment/archive/mve_perfect/mve_cartpole.json", 13, 10, 50000, "MVE K=5"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/mve_perfect/cartpole/plot_a.png", interval=2000,
    #                       title="MF vs MVE")

    # settings = [("experiment/archive/model_free_v3/dqn_navigation_b.json", 2, 10, 200000, "DQN"),
    #             ("experiment/archive/zach_perfect/zach_navigation.json", 2, 10, 200000, "Zach K=2"),
    #             ("experiment/archive/zach_perfect/zach_navigation.json", 6, 10, 200000, "Zach K=3"),
    #             ("experiment/archive/zach_perfect/zach_navigation.json", 9, 10, 200000, "Zach K=4"),
    #             ("experiment/archive/zach_perfect/zach_navigation.json", 13, 10, 200000, "Zach K=5"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/zach_perfect/navigation/plot_a.png", interval=10000,
    #                       title="MF vs MVE")

    # settings = [("experiment/archive/model_free_v3/dqn_navigation_b.json", 2, 10, 200000, "DQN"),
    #             ("experiment/archive/mve_perfect/mve_navigation_b.json", 2, 10, 200000, "MVE K=2"),
    #             ("experiment/archive/mve_perfect/mve_navigation_b.json", 6, 10, 200000, "MVE K=3"),
    #             ("experiment/archive/mve_perfect/mve_navigation_b.json", 9, 10, 200000, "MVE K=4"),
    #             ("experiment/archive/mve_perfect/mve_navigation_b.json", 13, 10, 200000, "MVE K=5"),
    #
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/mve_perfect/navigation/plot_nav_c.png", interval=10000,
    #                       title="MF vs MVE")

    # settings = [("experiment/archive/model_free_v3/dqn_navigation_b.json", 2, 10, 200000, "DQN"),
    #             ("experiment/archive/zach_perfect/zach_navigation.json", 2, 10, 200000, "Zach DQN K=1"),
    #             ("experiment/archive/zach_perfect/zach_navigation.json", 5, 10, 200000, "Zach DQN K=2"),
    #             ("experiment/archive/zach_perfect/zach_navigation.json", 9, 10, 200000, "Zach DQN K=3"),
    #             ("experiment/archive/zach_perfect/zach_navigation.json", 13, 10, 200000, "Zach DQN K=4"),
    #             ("experiment/archive/zach_perfect/zach_navigation.json", 18, 10, 200000, "Zach DQN K=5"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/zach_perfect/navigation/plot.png", interval=10000,
    #                       title="MF vs MVE")


    # settings = [
    #             ("experiment/archive/model_free_v2/dqn_catcher_visual.json", 6, 5, 500000, "DQN"),
    #             ("experiment/archive/mve_perfect/mve_catcher_visual.json", 0, 5, 500000, "MVE k=2"),
    #             ("experiment/archive/mve_perfect/mve_catcher_visual.json", 3, 5, 500000, "MVE k=3"),
    #             ("experiment/archive/mve_perfect/mve_catcher_visual.json", 6, 5, 500000, "MVE k=4"),
    #             ("experiment/archive/mve_perfect/mve_catcher_visual.json", 9, 5, 500000, "MVE k=5"),
    #             ]
    # draw_lunar_lander_dqn(settings, save_path="plots/mve_perfect/catcher_visual/plot.png", interval=10000)

    print("DONE")


    settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN", "-", "blue"),
                # ("experiment/archive/mve_avg_perfect/mve_catcher_raw.json", 2, 10, 200000, "MVE k=2", "-", "orange"),
                ("experiment/archive/mve_avg_perfect/mve_catcher_raw.json", 10, 10, 200000, "MVE k=4", "--", "orange"),
                ]
    # draw_lunar_lander_dqn_linestyle(settings, save_path="plots/sprint_1/mve_error_awarev2/catcher_raw/update/plot_k4_perfect.png", interval=5000,
    #                       ylim=(-5, 60))

    settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN", "-", "blue"),
                # ("experiment/archive/mve_avg_learnedv2/catcher_raw.json", 1, 10, 200000, "MVE LM k=2", "--", "orange"),
                ("experiment/archive/mve_avg_learnedv2/catcher_raw.json", 5, 10, 200000, "MVE LM k=4", "--", "orange")
                ]
    # draw_lunar_lander_dqn_linestyle(settings, save_path="plots/sprint_1/mve_error_awarev2/catcher_raw/update/plot_k4_learned.png", interval=5000,
    #                       ylim=(-5, 60))


    settings = [("experiment/archive/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN", "-", "blue"),

                # ("experiment/archive/mve_avg_perfect/mve_catcher_raw.json", 2, 10, 200000, "MVE k=2"),
                # ("experiment/archive/mve_avg_learnedv2/catcher_raw.json", 1, 10, 200000, "MVE LM k=2"),
                # ("experiment/archive/mve_error_awarev2/catcher_raw.json", 0, 10, 200000, "MVE LM EAw k=2 tau 100.0", "--", "red"),
                # ("experiment/archive/mve_error_awarev2/catcher_raw.json", 2, 10, 200000, "MVE LM EAw k=2 tau 10.0", "--", "maroon"),
                # ("experiment/archive/mve_error_awarev2/catcher_raw.json", 4, 10, 200000, "MVE LM EAw k=2 tau 1.0", "--", "magenta"),
                # ("experiment/archive/mve_error_awarev2/catcher_raw.json", 6, 10, 200000, "MVE LM EAw k=2 tau 0.1", "--", "purple"),

                # ("experiment/archive/mve_error_awarev2/catcher_raw.json", 1, 10, 200000, "MVE LM EAw k=4 tau 100.0", "--", "red"),
                # ("experiment/archive/mve_error_awarev2/catcher_raw.json", 3, 10, 200000, "MVE LM EAw k=4 tau 10.0", "--", "maroon"),
                # ("experiment/archive/mve_error_awarev2/catcher_raw.json", 5, 10, 200000, "MVE LM EAw k=4 tau 1.0", "--", "magenta"),
                ("experiment/archive/mve_error_awarev2/catcher_raw.json", 7, 10, 200000, "MVE LM EAw k=4 tau 0.1", "--", "purple"),
                ]
    # draw_lunar_lander_dqn_linestyle(settings, save_path="plots/sprint_1/mve_error_awarev2/catcher_raw/update/plot_k4_learned_tau_4.png", interval=5000,
    #                                 ylim=(-5, 60))


    settings = [# ("experiment/config_fil,es/model_free_v2/dqn_catcher_raw.json", 2, 10, 200000, "DQN"),
                # ("experiment/archive/mve_error_awarev2/catcher_raw.json", 0, 10, 200000, "MVE LM EW k=2 tau 100.0", "--", "red"),
                # ("experiment/archive/mve_error_awarev2/catcher_raw.json", 2, 10, 200000, "MVE LM EW k=2 tau 10.0", "--", "maroon"),
                # ("experiment/archive/mve_error_awarev2/catcher_raw.json", 4, 10, 200000, "MVE LM EW k=2 tau 1.0", "--", "magenta"),
                # ("experiment/archive/mve_error_awarev2/catcher_raw.json", 6, 10, 200000, "MVE LM EW k=2 tau 0.1", "--", "purple"),

                ("experiment/archive/mve_error_awarev2/catcher_raw.json", 1, 10, 200000, "MVE LM EW k=4 tau 100.0", "--", "red"),
                ("experiment/archive/mve_error_awarev2/catcher_raw.json", 3, 10, 200000, "MVE LM EW k=4 tau 10.0", "--", "maroon"),
                ("experiment/archive/mve_error_awarev2/catcher_raw.json", 5, 10, 200000, "MVE LM EW k=4 tau 1.0",  "--", "magenta"),
                ("experiment/archive/mve_error_awarev2/catcher_raw.json", 7, 10, 200000, "MVE LM EW k=4 tau 0.1",  "--", "purple"),


                ]
    # draw_expected_rollout_linestyle(settings, save_path="plots/sprint_1/mve_error_awarev2/catcher_raw/update/rollout_length_plot_k4.png", interval=5000,
    #                                 ylabel="Expected Rollout Length", ylim=(0.95, 2.55))




