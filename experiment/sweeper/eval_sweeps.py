import os
import numpy as np
import sys
import argparse

sys.path.append(os.path.abspath('../../'))

from sweeper import Sweeper


def extract_line(lines, max_steps, interval=0):
    """
    Retrieves num_steps at which episodes were completed
    and returns an array where the value at index i represents
    the number of episodes completed until step i
    """
    step = 0
    # rewards_over_time = np.zeros(max_steps//interval+1) # for GANModel training
    rewards_over_time = np.zeros(max_steps // interval + 1) # for DQN training
    try:
        for line in lines:
            if 'total steps' not in line:
                continue
            num_steps = float(line.split("|")[1].split(",")[0].split(" ")[-1])
            if num_steps > max_steps:
                break
            # reward = float(line.split("|")[1].split(",")[1].split(" ")[2]) # for GANModel training
            reward = float(line.split("|")[1].split(",")[2].split("/")[0].split(" ")[-1]) # for DQN training
            rewards_over_time[step] = reward
            step += 1
        return rewards_over_time
    except:
        print(line)
        print('step:{}'.format(step))
        raise


def get_max_steps(lines):
    for line in lines[::-1]:
        if 'total steps' in line:
            max_steps = float(line.split("|")[1].split(",")[0].split(" ")[-1])
            return max_steps
    return -1


def _eval_lines(config_file, last_points, start_idx, end_idx, max_steps, interval=10000):
    print('config_files: {}'.format(config_file))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sweeper = Sweeper(os.path.join(project_root, config_file))
    eval = []
    eval_lines = []
    for k in range(sweeper.total_combinations):
        eval.append([])
        eval_lines.append([])

    for idx in range(start_idx, end_idx):
        cfg = sweeper.parse(idx)
        cfg.data_root = os.path.join(project_root, 'data', 'output')
        log_dir = cfg.get_log_dir()
        log_path = os.path.join(log_dir, 'log')
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            continue
        if len(lines) == 0:
            continue
        # ugly parse based on the log_file format
        try:
            num_steps = get_max_steps(lines)
            if num_steps >= max_steps:
                assert idx % sweeper.total_combinations == cfg.param_setting
                avg_eval_steps = extract_line(lines, max_steps, interval=interval)
                # eval[id % sweeper.total_combinations].append(np.mean(avg_eval_steps[-int(len(avg_eval_steps)):]))
                eval[idx % sweeper.total_combinations].append(np.mean(avg_eval_steps[-last_points:]))
                # eval[id % sweeper.total_combinations].append(np.mean(avg_eval_steps))


        except IndexError:
            print(idx)
            raise
    summary = list(map(lambda x: (x[0], np.mean(x[1]), np.std(x[1]), len(x[1])), enumerate(eval)))
    summary = [x for x in summary if np.isnan(x[1]) == False]
    # new_summary = []
    # for s in summary:
    #     if np.isnan(s[1]) == False:
    #         new_summary.append(s)
    # print(summary[0])
    # print(new_summary[0])
    # quit()

    summary = sorted(summary, key=lambda s: s[1], reverse=True)

    for idx, mean, std, num_runs in summary:
        print("Param Setting # {:>3d} | Rewards: {:>10.10f} +/- {:>5.2f} ({:>2d} runs) {} | ".format(
            idx, mean, std, num_runs, sweeper.param_setting_from_id(idx)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--cfg-file', default='default.json')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=72, type=int)
    parser.add_argument('--max-steps', default=50000, type=int)
    parser.add_argument('--interval', default=10, type=int)

    args = parser.parse_args()



    #############
    # Navigation

    # print("dqn_navigation")
    # _eval_lines(config_files='experiment/archive/model_free_v3/dqn_navigation_c.json', last_points=15, start_idx=0, end_idx=40, max_steps=300000, interval=10000)
    # # print("zach_navigation")
    # _eval_lines(config_files='experiment/archive/zach_perfect_2/zach_navigation.json', last_points=15, start_idx=0, end_idx=200, max_steps=300000, interval=10000)

    # print("mve_avg_navigation")
    # _eval_lines(config_files='experiment/archive/mve_avg_perfect/mve_navigation.json', last_points=15, start_idx=0, end_idx=160, max_steps=300000, interval=10000)

    # print("mve_avg_navigation_learnedv2")
    # _eval_lines(config_files='experiment/archive/mve_avg_learnedv2/navigation.json', last_points=15, start_idx=0, end_idx=60, max_steps=300000, interval=10000)

    # print("search_dt_navigation")
    # _eval_lines(config_files='experiment/archive/search_dt_perfect/navigation.json', last_points=15, start_idx=0, end_idx=80, max_steps=300000, interval=10000)
    #
    # print("search_dt_navigation_learnedv2")
    # _eval_lines(config_files='experiment/archive/search_dt_learnedv2/navigation.json', last_points=15, start_idx=0, end_idx=80, max_steps=200000, interval=10000)

    # print("mve_sw_model")
    # _eval_lines(config_files='experiment/archive/mve_sw_model/navigation.json', last_points=15, start_idx=0, end_idx=240, max_steps=300000, interval=10000)

    # print("mve_learned/navigation")
    # _eval_lines(config_files='experiment/config_files_2/mve_learned/navigation.json', last_points=15, start_idx=0, end_idx=160, max_steps=300000, interval=10000)
    # #############


    #############
    # Cartpole
    #
    # print("dqn_cartpole")
    # _eval_lines(config_files='experiment/archive/model_free_v3/dqn_cartpole.json', last_points=12, start_idx=0, end_idx=40, max_steps=50000, interval=2000)

    # print("mve_cartpole")
    # _eval_lines(config_files='experiment/archive/mve_perfect/mve_cartpole.json', last_points=12, start_idx=0, end_idx=160, max_steps=50000, interval=2000)

    # print("mve_avg_cartpole")
    # _eval_lines(config_files='experiment/archive/mve_avg_perfect/mve_cartpole.json', last_points=12, start_idx=0, end_idx=160, max_steps=50000, interval=2000)
    #
    # print("zach_cartpole")
    # _eval_lines(config_files='experiment/archive/zach_perfect_2/zach_cartpole.json', last_points=12, start_idx=0, end_idx=200, max_steps=50000, interval=2000)

    # print("mve_cartpole_learned")
    # _eval_lines(config_files='experiment/archive/mve_avg_learned/cartpole.json', last_points=12, start_idx=0, end_idx=80, max_steps=50000, interval=2000)
    #
    # print("mve_sw_model")
    # _eval_lines(config_files='experiment/archive/mve_sw_model/cartpole.json', last_points=12, start_idx=0, end_idx=240, max_steps=50000, interval=2000)


    # print("dqn_cartpole")
    # _eval_lines(config_files='experiment/archive/model_free_v3/dqn_cartpole_b.json', last_points=12, start_idx=0, end_idx=40, max_steps=50000, interval=2000)
    #
    # print("zach_cartpole_b")
    # _eval_lines(config_files='experiment/archive/zach_perfect_2/zach_cartpole_b.json', last_points=12, start_idx=0, end_idx=120, max_steps=50000, interval=2000)
    #
    # print("zach_cartpole_c")
    # _eval_lines(config_files='experiment/archive/zach_perfect_2/zach_cartpole_c.json', last_points=12, start_idx=0, end_idx=200, max_steps=50000, interval=2000)

    # print("zach_cartpole_d")
    # _eval_lines(config_files='experiment/archive/zach_perfect_2/zach_cartpole_d.json', last_points=12, start_idx=0,
    #             end_idx=200, max_steps=50000, interval=2000)
    # #
    # print("search_dt_perfect")
    # _eval_lines(config_files='experiment/archive/search_dt_perfect/cartpole.json', last_points=12, start_idx=0, end_idx=120, max_steps=50000, interval=2000)

    # print("mve_avg_learnedv2")
    # _eval_lines(config_files='experiment/archive/mve_avg_learnedv2/cartpole.json', last_points=12, start_idx=0, end_idx=90, max_steps=50000, interval=2000)

    # print("search_dt_learnedv2")
    # _eval_lines(config_files='experiment/archive/search_dt_learnedv2/cartpole.json', last_points=12, start_idx=0, end_idx=120, max_steps=50000, interval=2000)

    # print("mve_learned/cartpole")
    # _eval_lines(config_files='experiment/config_files_2/mve_learned/cartpole.json', last_points=12, start_idx=0, end_idx=160, max_steps=50000, interval=2000)


    # ############


    #############
    # Catcher Raw
    # print("dqn_catcher_raw")
    # _eval_lines(config_files='experiment/archive/model_free_v2/dqn_catcher_raw.json', last_points=20, start_idx=0, end_idx=40, max_steps=200000, interval=5000)
    #
    # print("dqn_catcher_raw_VF_sw")
    # _eval_lines(config_files='experiment/config_files_2/model_free/catcher_raw.json', last_points=20, start_idx=0, end_idx=270, max_steps=200000, interval=5000)

    # print("dqn_catcher_raw_VF_sw_huber")
    # _eval_lines(config_files='experiment/config_files_2/model_free/catcher_raw_huber.json', last_points=5, start_idx=0, end_idx=270, max_steps=200000, interval=5000)

    # print("dqn_catcher_raw_architecture_sweep")
    # _eval_lines(config_files='experiment/archive/model_free_v4/catcher_raw.json', last_points=20, start_idx=0, end_idx=160, max_steps=200000, interval=5000)
    #
    # print("dqn_catcher_raw_architecture_sweep_smaller_lr")
    # _eval_lines(config_files='experiment/archive/model_free_v4/catcher_raw_smaller_lr.json', last_points=20, start_idx=0, end_idx=150, max_steps=200000, interval=5000)

    # print("zach_catcher_raw")
    # _eval_lines(config_files='experiment/archive/zach_perfect_2/zach_catcher_raw.json', last_points=20, start_idx=0, end_idx=200, max_steps=200000, interval=5000)

    # print("zach_catcher_raw_b")
    # _eval_lines(config_files='experiment/archive/zach_perfect_2/zach_catcher_raw_b.json', last_points=20, start_idx=0, end_idx=160, max_steps=200000, interval=5000)
    #
    # print("mve_avg_catcher_raw")
    # _eval_lines(config_files='experiment/archive/mve_avg_perfect/mve_catcher_raw.json', last_points=20, start_idx=0, end_idx=160, max_steps=200000, interval=5000)

    # print("mve_avg_catcher_learned")
    # _eval_lines(config_files='experiment/archive/mve_avg_learned/catcher_raw.json', last_points=20, start_idx=0, end_idx=90, max_steps=200000, interval=5000)
    #
    # print("mve_avg_catcher_learned_model_sw")
    # _eval_lines(config_files='experiment/archive/mve_sw_model/catcher_raw.json', last_points=20, start_idx=0, end_idx=90, max_steps=200000, interval=5000)
    # print("mve_avg_catcher_learned_error_aware")
    # _eval_lines(config_files='experiment/archive/mve_error_aware/catcher_raw.json', last_points=20, start_idx=0, end_idx=120, max_steps=200000, interval=5000)



    # print("mve_avg_catcher_learned_model_sweeps")
    # _eval_lines(config_files='experiment/archive/mve_sw_model/catcher_raw.json', last_points=20, start_idx=0, end_idx=240, max_steps=200000, interval=5000)

    # print("search_dt_perfect")
    # _eval_lines(config_files='experiment/archive/search_dt_perfect/catcher_raw.json', last_points=20, start_idx=0, end_idx=80, max_steps=200000, interval=5000)

    # print("search_dt_learned")
    # _eval_lines(config_files='experiment/archive/search_dt_learned/catcher_raw.json', last_points=20, start_idx=0, end_idx=80, max_steps=200000, interval=5000)
    #
    # print("search_dt_learnedv2")
    # _eval_lines(config_files='experiment/archive/search_dt_learnedv2/catcher_raw.json', last_points=20, start_idx=0, end_idx=80, max_steps=200000, interval=5000)


    # print("mve_avg_catcher_learnedv2")
    # _eval_lines(config_files='experiment/archive/mve_avg_learnedv2/catcher_raw.json', last_points=20, start_idx=0, end_idx=80, max_steps=200000, interval=5000)
    #
    # print("mve_avg_catcher_learned_error_aware")
    # _eval_lines(config_files='experiment/archive/mve_error_awarev2/catcher_raw.json', last_points=20, start_idx=0, end_idx=120, max_steps=200000, interval=5000)

    # print("mve_learned")
    # _eval_lines(config_files='experiment/config_files_2/mve_learned/catcher_raw.json', last_points=20, start_idx=0, end_idx=200, max_steps=200000, interval=5000)
    # # print("mse_learned_mse")
    # _eval_lines(config_files='experiment/config_files_2/mve_learned/catcher_raw_mse.json', last_points=20, start_idx=0,
    #             end_idx=200, max_steps=200000, interval=5000)

    # print("search_learned")
    # _eval_lines(config_files='experiment/config_files_2/search_learned/catcher_raw.json', last_points=20, start_idx=0, end_idx=200, max_steps=200000, interval=5000)
    # # print("search_learned_mse")
    # _eval_lines(config_files='experiment/config_files_2/search_learned/catcher_raw_mse.json', last_points=20, start_idx=0, end_idx=200, max_steps=200000, interval=5000)

    # print("mve_learned_v2_mse")
    # _eval_lines(config_files='experiment/config_files_2/mve_learned/catcher_raw_v2.json', last_points=20, start_idx=0, end_idx=180, max_steps=200000, interval=5000)
    #
    # print("mve_learned_v3_huber")
    # _eval_lines(config_files='experiment/config_files_2/mve_learned/catcher_raw_v3.json', last_points=20, start_idx=0, end_idx=180, max_steps=200000, interval=5000)
    #
    # print("mve_learned_v4_mse_highlr")
    # _eval_lines(config_files='experiment/config_files_2/mve_learned/catcher_raw_v4.json', last_points=20, start_idx=0, end_idx=90, max_steps=200000, interval=5000)
    #
    # print("search_learned_v2")
    # _eval_lines(config_files='experiment/config_files_2/search_learned/catcher_raw_v2.json', last_points=20, start_idx=0, end_idx=180, max_steps=200000, interval=5000)
    #
    # print("search_learned_v3_huber")
    # _eval_lines(config_files='experiment/config_files_2/search_learned/catcher_raw_v3.json', last_points=20, start_idx=0, end_idx=180, max_steps=200000, interval=5000)
    #
    # print("search_learned_v4_mse_highlr")
    # _eval_lines(config_files='experiment/config_files_2/search_learned/catcher_raw_v4.json', last_points=20, start_idx=0, end_idx=180, max_steps=200000, interval=5000)

    # print("search_learned_v4_mse_huber")
    # _eval_lines(config_files='experiment/config_files_2/search_learned/catcher_raw_v5.json', last_points=20, start_idx=0, end_idx=180, max_steps=200000, interval=5000)

    # print("capacity_catcher_raw")
    # _eval_lines(config_files='experiment/config_files_2/search_learned/capacity_catcher_raw.json', last_points=20, start_idx=0, end_idx=1540, max_steps=200000, interval=5000)


    # print("dqn_catcher_raw_VF_sw_huber")
    # _eval_lines(config_files='experiment/config_files_2/model_free/catcher_raw_huber.json', last_points=20, start_idx=0, end_idx=270, max_steps=200000, interval=5000)

    # print("capacity_catcher_raw_8")
    # _eval_lines(config_files='experiment/config_files_2/search_learned/capacity/catcher_raw_8.json', last_points=20, start_idx=0, end_idx=20, max_steps=200000, interval=5000)
    # print("capacity_catcher_raw_16")
    # _eval_lines(config_files='experiment/config_files_2/search_learned/capacity/catcher_raw_16.json', last_points=20, start_idx=0, end_idx=30, max_steps=200000, interval=5000)
    # print("capacity_catcher_raw_32")
    # _eval_lines(config_files='experiment/config_files_2/search_learned/capacity/catcher_raw_32.json', last_points=20, start_idx=0, end_idx=40, max_steps=200000, interval=5000)
    # print("capacity_catcher_raw_64")
    # _eval_lines(config_files='experiment/config_files_2/search_learned/capacity/catcher_raw_64.json', last_points=20, start_idx=0, end_idx=50, max_steps=200000, interval=5000)
    # print("capacity_catcher_raw_128")
    # _eval_lines(config_files='experiment/config_files_2/search_learned/capacity/catcher_raw_128.json', last_points=20, start_idx=0, end_idx=60, max_steps=200000, interval=5000)
    # print("capacity_catcher_raw_256")
    # _eval_lines(config_files='experiment/config_files_2/search_learned/capacity/catcher_raw_256.json', last_points=20, start_idx=0, end_idx=70, max_steps=200000, interval=5000)

    # print("search_learned")
    # _eval_lines(config_files='experiment/config_files_2/search_learned/catcher_raw_huber.json', last_points=20, start_idx=0, end_idx=300, max_steps=200000, interval=5000)


    # ###########


    ############
    # Acrobot
    # print("dqn_acrobot")
    # _eval_lines(config_files='experiment/archive/model_free_v3/dqn_test.json', last_points=25, start_idx=0, end_idx=40, max_steps=100000, interval=2000)

    # print("mve_acrobot")
    # _eval_lines(config_files='experiment/archive/mve_perfect/mve_acrobot.json', last_points=25, start_idx=0, end_idx=160, max_steps=100000, interval=2000)
    #
    # print("mve_avg_acrobot")
    # _eval_lines(config_files='experiment/archive/mve_avg_perfect/mve_acrobot.json', last_points=25, start_idx=0, end_idx=160, max_steps=100000, interval=2000)
    # #
    # print("mve_avg_acrobot_learnedv2")
    # _eval_lines(config_files='experiment/archive/mve_avg_learnedv2/acrobot.json', last_points=25, start_idx=0, end_idx=60, max_steps=100000, interval=2000)
    # #
    # print("mve_sw_model")
    # _eval_lines(config_files='experiment/archive/mve_sw_model/acrobot.json', last_points=25, start_idx=0, end_idx=240, max_steps=100000, interval=2000)
    #
    # print("zach_acrobot")
    # _eval_lines(config_files='experiment/archive/zach_perfect_2/zach_acrobot.json', last_points=25, start_idx=0, end_idx=200, max_steps=100000, interval=2000)
    #
    # print("search_dt_acrobot")
    # _eval_lines(config_files='experiment/archive/search_dt_perfect/acrobot.json', last_points=25, start_idx=0, end_idx=80, max_steps=100000, interval=2000)
    #
    # print("search_dt_acrobot_learnedv2")
    # _eval_lines(config_files='experiment/archive/search_dt_learnedv2/acrobot.json', last_points=25, start_idx=0, end_idx=80, max_steps=100000, interval=2000)
    # print("mve_learned/acrobot")
    # _eval_lines(config_files='experiment/config_files_2/mve_learned/acrobot.json', last_points=25, start_idx=0, end_idx=160, max_steps=100000, interval=2000)



    print("search_learned_fixed/acrobot")
    _eval_lines(config_file='experiment/config_files_2/search_learned_fixed/acrobot.json', last_points=25, start_idx=0, end_idx=500, max_steps=100000, interval=2000)

    print("search_learned_fixed/acrobot exponential")
    _eval_lines(config_file='experiment/config_files_2/search_learned_fixed/acrobot_exponential.json', last_points=25, start_idx=0, end_idx=500, max_steps=100000, interval=2000)


    # # ############


    ############
    # Catcher visual

    # print("dqn_catacher_visual")
    # _eval_lines(config_files='experiment/archive/model_free_v2/dqn_catcher_visual.json', last_points=25, start_idx=0, end_idx=45, max_steps=500000, interval=10000)
    # print("mve_catcher_visual")
    # _eval_lines(config_files='experiment/archive/mve_avg_perfect/mve_catcher_visual.json', last_points=25, start_idx=0, end_idx=120, max_steps=500000, interval=10000)
    # print("zach_catcher_visual")
    # _eval_lines(config_files='experiment/archive/zach_perfect_2/zach_catcher_visual_b.json', last_points=25, start_idx=0, end_idx=150, max_steps=500000, interval=10000)
    # ############

    ## model_free
    # print("dqn_catacher_raw")
    # _eval_lines(config_files='experiment/archive/model_free/dqn_catcher_raw.json', start_idx=0, end_idx=120, max_steps=400000, interval=10000)

    # print("dqn_catacher_raw_2h")
    # _eval_lines(config_files='experiment/archive/model_free/dqn_catcher_raw_2h.json', start_idx=0, end_idx=60, max_steps=400000, interval=10000)

    # print("dqn_catacher_visual_gray")
    # _eval_lines(config_files='experiment/archive/model_free/dqn_catcher_visual_gray.json', start_idx=0, end_idx=24, max_steps=200000, interval=10000)

    ## model_free_v2
    # print("dqn_catacher_raw")
    # _eval_lines(config_files='experiment/archive/model_free_v2/dqn_catcher_raw.json', last_points=20, start_idx=0, end_idx=40, max_steps=200000, interval=5000)

    # print("dqn_acrobot")
    # _eval_lines(config_files='experiment/archive/model_free_v2/dqn_test.json', last_points=10, start_idx=0, end_idx=359, max_steps=200000, interval=10000)

    # print("dqn_catacher_lite")
    # _eval_lines(config_files='experiment/archive/model_free_v2/dqn_catcher_lite.json', start_idx=0, end_idx=60, max_steps=400000, interval=10000)

    # print("dqn_catacher_lite_b")
    # _eval_lines(config_files='experiment/archive/model_free_v2/dqn_catcher_lite_b.json', start_idx=0, end_idx=80, max_steps=600000, interval=10000)

    # print("dqn_catacher_visual")
    # _eval_lines(config_files='experiment/archive/model_free_v2/dqn_catcher_visual.json', start_idx=0, end_idx=45, max_steps=500000, interval=10000)

    # print("dqn_acrobot")
    # _eval_lines(config_files='experiment/archive/model_free_v2/dqn_test.json', start_idx=0, end_idx=360, max_steps=200000, interval=10000)

    # print("mve_perfect_catcher_raw")
    # _eval_lines(config_files='experiment/archive/mve_perfect/mve_catcher_raw.json', last_points=20, start_idx=0, end_idx=160, max_steps=200000, interval=5000)

    # print("dqn_acrobot")
    # _eval_lines(config_files='experiment/archive/model_free_v3/dqn_test.json', last_points=25, start_idx=0, end_idx=40, max_steps=100000, interval=2000)
    #
    # print("mve_acrobot")
    # _eval_lines(config_files='experiment/archive/mve_perfect/mve_acrobot.json', last_points=25, start_idx=0, end_idx=160, max_steps=100000, interval=2000)

    # print("dqn_navigation")
    # _eval_lines(config_files='experiment/archive/model_free_v3/dqn_navigation_b.json', last_points=10, start_idx=0, end_idx=40, max_steps=200000, interval=10000)
    #
    # print("mve_navigation")
    # _eval_lines(config_files='experiment/archive/mve_perfect/mve_navigation.json', last_points=10, start_idx=0, end_idx=160, max_steps=200000, interval=10000)

    # print("dqn_cartpole")
    # _eval_lines(config_files='experiment/archive/model_free_v3/dqn_cartpole.json', last_points=12, start_idx=0, end_idx=40, max_steps=50000, interval=2000)
    # #
    # print("mve_cartpole")
    # _eval_lines(config_files='experiment/archive/mve_perfect/mve_cartpole.json', last_points=12, start_idx=0, end_idx=160, max_steps=50000, interval=2000)

    # print("dqn_navigation_b")
    # _eval_lines(config_files='experiment/archive/model_free_v3/dqn_navigation_c.json', last_points=15, start_idx=0, end_idx=40, max_steps=300000, interval=10000)
    #
    # print("mve_navigation_b")
    # _eval_lines(config_files='experiment/archive/mve_perfect/mve_navigation_c.json', last_points=15, start_idx=0, end_idx=160, max_steps=300000, interval=10000)

    # print("zach_navigation")
    # _eval_lines(config_files='experiment/archive/zach_perfect/zach_navigation.json', last_points=10, start_idx=0, end_idx=160, max_steps=200000, interval=10000)

    # print("dqn_catacher_visual")
    # _eval_lines(config_files='experiment/archive/model_free_v2/dqn_catcher_visual.json', last_points=25, start_idx=0, end_idx=45, max_steps=500000, interval=10000)
    #
    # print("mve_catacher_visual")
    # _eval_lines(config_files='experiment/archive/mve_perfect/mve_catcher_visual.json', last_points=25,
    #             start_idx=0, end_idx=60, max_steps=500000, interval=10000)





















    # TRACKING
    # Catcher Raw

    # print("sprint_3/catcher_raw/horizon_avg")
    # _eval_lines(config_files='experiment/config_files_3/search_learned/catcher_raw/horizon_avg.json', last_points=20, start_idx=0, end_idx=50, max_steps=200000, interval=5000)
    #
    #
    # print("sprint_3/catcher_raw/horizon_leaf")
    # _eval_lines(config_files='experiment/config_files_3/search_learned/catcher_raw/horizon_leaf.json', last_points=20, start_idx=0, end_idx=50, max_steps=200000, interval=5000)
    #
    #
    # print("sprint_3/catcher_raw/horizon_avg_tune_vf_lr")
    # _eval_lines(config_files='experiment/config_files_3/search_learned/catcher_raw/horizon_avg_tune_vf_lr.json', last_points=20, start_idx=0, end_idx=360, max_steps=200000, interval=5000)
    #
    #
    #
    #
    #
    #
