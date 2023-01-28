import os

from core.utils import torch_utils

class EmptyConfig:
    def __init__(self):
        self.exp_name = 'test'
        self.data_root = None
        self.run = 0
        self.param_setting = 0
        self.logger = None
        self.log_observations = False
        self.tensorboard_logs = False
        self.batch_size = 0
        self.replay_with_len = False
        self.memory_size = 1
        self.evaluation_criteria = "return"
        self.checkpoints = False
        self.warm_up_step = 0
        self.debug = False
        return

    def get_log_dir(self):
        d = os.path.join(self.data_root, self.exp_name, "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting))
        torch_utils.ensure_dir(d)
        return d

    def log_config(self):
        attrs = self.get_print_attrs()
        for param, value in attrs.items():
            self.logger.info('{}: {}'.format(param, value))

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        return attrs

    @property
    def env_fn(self):
        return self.__env_fn

    @env_fn.setter
    def env_fn(self, env_fn):
        self.__env_fn = env_fn
        self.state_dim = env_fn().state_dim
        self.action_dim = env_fn().action_dim

    def get_parameters_dir(self):
        d = os.path.join(self.data_root, self.exp_name,
                         "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting),
                         "parameters")
        torch_utils.ensure_dir(d)
        return d

class Config(EmptyConfig):
    def __init__(self):
        super().__init__()
        self.exp_name = 'test'
        self.data_root = None
        self.device = None
        self.run = 0
        self.param_setting = 0

        self.env_name = None
        self.state_dim = None
        self.action_dim = None
        self.max_steps = 0

        self.log_interval = int(1e3)
        self.save_interval = 0
        self.eval_interval = 0
        self.num_eval_episodes = 5
        self.timeout = None
        self.stats_queue_size = 10
        self.early_save = False

        self.__env_fn = None
        self.logger = None

        self.tensorboard_logs = False
        self.tensorboard_interval = 100

        self.state_normalizer = None
        self.state_norm_coef = 0
        self.reward_normalizer = None
        self.reward_norm_coef = 1.0

        self.early_cut_off = False

        self.testset_paths = None
        self.tester_fn_config = {}
        self.evalset_path = {}
        self.true_value_paths = None
        self.visualize = False
        self.evaluate_overestimation = False
        self.evaluate_feature_mdp = False
        self.evaluate_action_value = False
        
        self.polyak = 0
        self.discrete_control = True
        self.offline_data_path = None

    def get_log_dir(self):
        d = os.path.join(self.data_root, self.exp_name, "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting))
        torch_utils.ensure_dir(d)
        return d

    def log_config(self):
        attrs = self.get_print_attrs()
        for param, value in attrs.items():
            self.logger.info('{}: {}'.format(param, value))

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        return attrs

    @property
    def env_fn(self):
        return self.__env_fn

    @env_fn.setter
    def env_fn(self, env_fn):
        self.__env_fn = env_fn
        self.state_dim = env_fn().state_dim
        self.action_dim = env_fn().action_dim

    def get_visualization_dir(self):
        d = os.path.join(self.data_root, self.exp_name, "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting), "visualizations")
        torch_utils.ensure_dir(d)
        return d

    def get_parameters_dir(self):
        d = os.path.join(self.data_root, self.exp_name,
                         "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting),
                         "parameters")
        torch_utils.ensure_dir(d)
        return d

    def get_data_dir(self):
        d = os.path.join(self.data_root, self.exp_name,
                         "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting),
                         "data")
        torch_utils.ensure_dir(d)
        return d

    def get_warmstart_property_dir(self):
        d = os.path.join(self.data_root, self.exp_name,
                         "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting),
                         "warmstart_property")
        torch_utils.ensure_dir(d)
        return d

    def get_logdir_format(self):
        return os.path.join(self.data_root, self.exp_name,
                            "{}_run",
                            "{}_param_setting".format(self.param_setting))


class VIConfig(Config):
    def __init__(self):
        super(VIConfig).__init__()
        self.agent = 'ValueIteration'
        self.activation_config = {'name': 'None'}
        self.constraint = None
        self.optimizer_type = 'SGD'
        self.vf_loss = 'mse'
        self.replay_with_len = None
        self.replay = None
        self.epsilon = 0
        self.load_offline_data = False
        self.tester_fn_config = []
        self.batch_size = 0
        self.polyak = 0
        self.learning_rate = 1
        self.warm_up_step = 0
        self.early_cut_off = False
        self.log_observations = False
        self.evaluate_overestimation = False
        self.evaluate_feature_mdp = False
        self.evaluate_action_value = False
        
    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'state_normalizer',
                  'reward_normalizer', 'rep_fn', 'val_fn', 'rep_activation_fn',
                  'eval_data']:
            del attrs[k]
        return attrs


class DQNAgentConfig(Config):
    def __init__(self):
        super().__init__()
        self.agent = 'DQNAgent'
        self.learning_rate = 0

        self.decay_epsilon = False
        self.epsilon = 0.1
        self.epsilon_start = None
        self.epsilon_end = None
        self.eps_schedule = None
        self.epsilon_schedule_steps = None
        self.random_action_prob = None

        self.discount = None

        self.network_type = 'fc'
        self.batch_size = None
        self.use_target_network = True
        self.memory_size = None
        self.optimizer_type = 'Adam'
        self.optimizer_fn = None
        self.val_net = None
        self.target_network_update_freq = None
        self.update_network = True

        self.replay = True
        self.replay_fn = None
        self.replay_with_len = False
        self.memory_size = 10000

        self.evaluation_criteria = "return"
        self.vf_loss = "mse"
        self.vf_loss_fn = None
        self.constraint = None
        self.constr_fn = None
        self.constr_weight = 0
        self.rep_type = "default"

        self.save_params = False
        self.save_early = None

        self.activation_config = {'name': 'None'}
        self.offline_data = None
        self.eval_data = None
        self.load_offline_data = False

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'eps_schedule', 'optimizer_fn', 'constr_fn',
                  'replay_fn', 'vf_loss_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'rep_fn', 'val_fn', 'rep_activation_fn',
                  'offline_data', 'eval_data']:
            del attrs[k]
        return attrs


class SarsaAgentConfig(Config):
    def __init__(self):
        super().__init__()
        self.agent = 'SarsaAgent'
        self.learning_rate = 0
        self.discount = None
        
        self.network_type = 'fc'
        self.batch_size = 1
        self.use_target_network = False
        self.target_network_update_freq = None
        self.memory_size = None
        self.optimizer_type = 'Adam'
        self.optimizer_fn = None
        self.val_net = None
        self.update_network = True

        self.replay = False

        self.evaluation_criteria = "return"
        self.vf_loss = "mse"
        self.vf_loss_fn = None
        self.constraint = None
        self.constr_fn = None
        self.constr_weight = 0
        self.rep_type = "default"

        self.save_params = False
        self.save_early = None

        self.activation_config = {'name': 'None'}

        self.decay_epsilon = False
        self.epsilon = 0.1
        self.epsilon_start = None
        self.epsilon_end = None
        self.eps_schedule = None
        self.epsilon_schedule_steps = None
        self.random_action_prob = None

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'eps_schedule', 'optimizer_fn', 'constr_fn',
                  'replay_fn', 'vf_loss_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'rep_fn', 'val_fn', 'rep_activation_fn']:
            del attrs[k]
        return attrs


class SarsaOfflineConfig(SarsaAgentConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'SarsaOffline'
        self.learning_rate = 0
        self.discount = None
        self.early_cut_threshold = 3
        self.tester_fn_config = []

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'eps_schedule', 'optimizer_fn', 'constr_fn',
                  'replay_fn', 'vf_loss_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'rep_fn', 'val_fn', 'rep_activation_fn',
                  'offline_data', 'tester_fn', 'eval_data']:
            del attrs[k]
        return attrs
    
    
class SarsaOfflineBatchConfig(SarsaOfflineConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'SarsaOfflineBatch'
        
        
class MonteCarloOfflineConfig(Config):
    def __init__(self):
        super().__init__()
        self.agent = 'MonteCarloOffline'
        self.activation_config = {'name': 'None'}
        self.constraint = None
        self.vf_loss = "mse"
        self.vf_loss_fn = None
        self.batch_size = 32
        self.early_cut_threshold = 3
        self.tester_fn_config = []
        self.epsilon = 0

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'eps_schedule', 'optimizer_fn', 'constr_fn',
                  'replay_fn', 'vf_loss_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'rep_fn', 'val_fn', 'rep_activation_fn',
                  'offline_data', 'tester_fn', 'eval_data']:
            del attrs[k]
        return attrs


class MonteCarloConfig(Config):
    def __init__(self):
        super().__init__()
        self.agent = 'MonteCarloAgent'
        self.activation_config = {'name': 'None'}
        self.constraint = None
        self.vf_loss = "mse"
        self.epsilon = 0.1

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'eps_schedule', 'optimizer_fn', 'constr_fn',
                  'replay_fn', 'vf_loss_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'rep_fn', 'val_fn', 'rep_activation_fn']:
            del attrs[k]
        return attrs

class CQLOfflineConfig(MonteCarloOfflineConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'CQLAgentOffline'
        self.cql_alpha = 1.0


class FQIConfig(DQNAgentConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'FQI'
        self.early_cut_threshold = 3
        self.tester_fn_config = []
        self.memory_size = 0

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'eps_schedule', 'optimizer_fn', 'constr_fn',
                  'replay_fn', 'vf_loss_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'rep_fn', 'val_fn', 'rep_activation_fn',
                  'offline_data', 'tester_fn', 'eval_data']:
            del attrs[k]
        return attrs

class QmaxCloneConfig(Config):
    def __init__(self):
        super().__init__()
        self.agent = 'QmaxCloneAgent'
        self.learning_rate = 0

        self.decay_epsilon = False
        self.epsilon = 0
        self.epsilon_start = None
        self.epsilon_end = None
        self.eps_schedule = None
        self.epsilon_schedule_steps = None
        self.random_action_prob = None

        self.discount = None

        self.network_type = 'fc'
        self.batch_size = None
        self.use_target_network = False
        self.memory_size = None
        self.optimizer_type = 'Adam'
        self.optimizer_fn = None
        self.val_net = None

        self.replay = False
        # self.replay_fn = None
        # self.replay_with_len = False
        # self.memory_size = 10000

        self.evaluation_criteria = "return"
        self.vf_loss = "mse"
        self.vf_loss_fn = None
        self.constraint = None
        self.constr_fn = None
        self.constr_weight = 0
        self.rep_type = "default"

        self.save_params = False
        self.save_early = None

        self.activation_config = {'name': 'None'}

        self.early_cut_threshold = 3
        self.tester_fn_config = []
        self.memory_size = 0

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'eps_schedule', 'optimizer_fn', 'constr_fn',
                  'replay_fn', 'vf_loss_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'rep_fn', 'val_fn', 'rep_activation_fn',
                  'offline_data', 'tester_fn', 'eval_data']:
            del attrs[k]
        return attrs


class QmaxConstrConfig(QmaxCloneConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'QmaxConstrAgent'
        self.constr_weight = 1.0


class QRCOnlineConfig(DQNAgentConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'QRCOnline'
        self.load_offline_data = False
        self.target_network_update_freq = 32
        # From the paper, for Ant Maze task
        self.qrc_beta = 1.0 # from paper
        # self.qrc_beta = 0.01
        self.qrc_beta_1 = 0.99
        self.qrc_beta_2 = 0.999

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'eps_schedule', 'optimizer_fn', 'constr_fn',
                  'replay_fn', 'vf_loss_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'rep_fn', 'val_fn', 'rep_activation_fn',
                  'offline_data', 'tester_fn', 'eval_data']:
            del attrs[k]
        return attrs


class QRCOfflineConfig(QRCOnlineConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'QRCOfflineConfig'


class AWACOnlineConfig(Config):
    def __init__(self):
        super().__init__()
        self.agent = 'AWACOnline'
        # self.update_after = 0
        # self.update_every = 1
        self.target_network_update_freq = 1
        self.polyak = 0.995
        self.awac_lambda = 1 # From the MuJoCo benchmark in AWAC paper
        self.awac_remove_const = False
        self.load_offline_data = False

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'replay_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'policy_fn', 'critic_fn',
                  'policy_optimizer_fn', 'critic_optimizer_fn', 'alpha_optimizer_fn',
                  'offline_data', 'tester_fn', 'eval_data']:
            del attrs[k]
        return attrs

class AWACOfflineConfig(AWACOnlineConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'AWACOffline'

class SACConfig(AWACOnlineConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'SACConfig'
        self.target_network_update_freq = 1
        self.polyak = 0.995
        # self.discrete_sac_loss = False
        self.load_offline_data = False

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'replay_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'policy_fn', 'critic_fn',
                  'policy_optimizer_fn', 'critic_optimizer_fn', 'alpha_optimizer_fn',
                  'offline_data', 'tester_fn', 'eval_data']:
            del attrs[k]
        return attrs

class SACOfflineConfig(SACConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'SACOffline'

class IQLOnlineConfig(AWACOnlineConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'IQLOnline'
        self.load_offline_data = False
        # From the paper
        self.target_network_update_freq = 1
        self.polyak = 0.995
        self.clip_grad_param = 100
        # From the paper, Appendix C
        self.expectile = 0.8
        self.temperature = 10 #For smaller hyperparameter values, the objective behaves similarly to behavioral cloning, while for larger values, it attempts to recover the maximum of the Q-function

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'replay_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'policy_fn', 'critic_fn', 'state_value_fn',
                  'policy_optimizer_fn', 'critic_optimizer_fn', 'alpha_optimizer_fn', 'vs_optimizer_fn',
                  'offline_data', 'tester_fn', 'eval_data']:
            del attrs[k]
        return attrs

class IQLOfflineConfig(IQLOnlineConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'IQLOffline'
        # # From the paper, Appendix B Ant Maze task
        # self.expectile = 0.9
        # self.temperature = 10 #For smaller hyperparameter values, the objective behaves similarly to behavioral cloning, while for larger values, it attempts to recover the maximum of the Q-function
        # From the paper, Appendix B MuJoCo task
        self.expectile = 0.7
        self.temperature = 3
        self.actor_cosin_schedule = False


class InSampleConfig(AWACOfflineConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'InSample'
        self.load_offline_data = False
        self.target_network_update_freq = 1
        self.polyak = 0.995
        self.tau = 0.01
        self.constant = 1
        self.clip_grad_param = 100
        self.exp_threshold = 10000
        self.eq18_v_calculation = False
        # self.ac_onPolicy = True
        self.automatic_tmp_tuning = False
        self.pretrain_beta = False

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'replay_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'policy_fn', 'critic_fn', 'state_value_fn', 'rep_fn',
                  'policy_optimizer_fn', 'critic_optimizer_fn', 'alpha_optimizer_fn', 'vs_optimizer_fn',
                  'offline_data', 'tester_fn', 'eval_data']:
            del attrs[k]
        return attrs


class CQLSACOfflineConfig(AWACOfflineConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'CQLSAC'
        self.load_offline_data = False
        self.cql_wight = 1.0,
        self.temperature = 1.0,
        self.target_action_gap = 10


    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'replay_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'policy_fn', 'critic_fn', 'state_value_fn', 'rep_fn',
                  'policy_optimizer_fn', 'critic_optimizer_fn', 'alpha_optimizer_fn', 'vs_optimizer_fn',
                  'offline_data', 'tester_fn', 'eval_data']:
            del attrs[k]
        return attrs


class TD3BCConfig(AWACOfflineConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'TD3BC'
        self.policy_freq = 2
        self.alpha = 2.5
        
    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'replay_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'policy_fn', 'critic_fn', 'state_value_fn', 'rep_fn',
                  'policy_optimizer_fn', 'critic_optimizer_fn', 'alpha_optimizer_fn', 'vs_optimizer_fn',
                  'offline_data', 'tester_fn', 'eval_data']:
            del attrs[k]
        return attrs


class OneStepConfig(AWACOfflineConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'OneStep'
        self.alpha = 0.1
        
    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'replay_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'policy_fn', 'critic_fn', 'state_value_fn', 'rep_fn',
                  'policy_optimizer_fn', 'critic_optimizer_fn', 'alpha_optimizer_fn', 'vs_optimizer_fn',
                  'offline_data', 'tester_fn', 'eval_data']:
            del attrs[k]
        return attrs
