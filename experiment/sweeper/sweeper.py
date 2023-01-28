import os
import json

import core.utils.config as config


class Sweeper(object):
    """
    The purpose of this class is to take an index, identify a configuration
    of hyper-parameters and create a Config object
    Important: parameters part of the sweep are provided in a list
    """
    def __init__(self, project_root, config_file, extra_name=None):
        config_path = os.path.join(project_root, config_file)
        with open(config_path) as f:
            self.config_dict = json.load(f)
        if extra_name:
            self.config_dict['fixed_parameters']['exp_name'] = self.config_dict['fixed_parameters']['exp_name'].format(extra_name)
        self.total_combinations = 1
        self.seed = None
        self.set_total_combinations()
        self.project_root = project_root

    def set_total_combinations(self):
        if 'sweep_parameters' in self.config_dict:
            sweep_params = self.config_dict['sweep_parameters']
            # calculating total_combinations
            tc = 1
            for params, values in sweep_params.items():
                tc = tc * len(values)
            self.total_combinations = tc

    def parse(self, id):
        config_class = getattr(config, self.config_dict['config_class'])
        cfg = config_class()

        # Populating fixed parameters
        fixed_params = self.config_dict['fixed_parameters']
        for param, value in fixed_params.items():
            setattr(cfg, param, value)

        cumulative = 1

        # Populating sweep parameters
        if 'sweep_parameters' in self.config_dict:
            sweep_params = self.config_dict['sweep_parameters']
            for param, values in sweep_params.items():
                num_values = len(values)
                setattr(cfg, param, values[int(id / cumulative) % num_values])
                cumulative *= num_values
        cfg.cumulative = cumulative
        cfg.run = int(id / cumulative)
        cfg.param_setting = id % cumulative
        cfg.data_root = os.path.join(self.project_root, 'data/output')
        cfg.id = id
        self.total_combinations = cumulative
        self.seed = cfg.run
        cfg.seed = cfg.run
        return cfg

    def param_setting_from_id(self, idx):
        sweep_params = self.config_dict['sweep_parameters']
        param_setting = {}
        cumulative = 1
        for param, values in sweep_params.items():
            num_values = len(values)
            param_setting[param] = values[int(idx/cumulative) % num_values]
            cumulative *= num_values
        return param_setting


if __name__ == '__main__':
    sweeper = Sweeper("archive/lunar_lander/sweepv0_6a.json")
    cfg = sweeper.parse(28)
    print(cfg)
