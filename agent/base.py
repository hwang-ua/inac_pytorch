import torch
import copy

from utils import torch_utils

import os
import numpy as np
import pickle


class Replay:
    def __init__(self, memory_size, batch_size, seed=0):
        self.rng = np.random.RandomState(seed)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.pos = 0
    
    def feed(self, experience):
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:
            self.data[self.pos] = experience
        self.pos = (self.pos + 1) % self.memory_size
    
    def feed_batch(self, experience):
        for exp in experience:
            self.feed(exp)
    
    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        sampled_indices = [self.rng.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]
        batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))
        return batch_data
    
    def sample_array(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        
        sampled_indices = [self.rng.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]
        
        return sampled_data
    
    def size(self):
        return len(self.data)
    
    def persist_memory(self, dir):
        for k in range(len(self.data)):
            transition = self.data[k]
            with open(os.path.join(dir, str(k)), "wb") as f:
                pickle.dump(transition, f)
    
    def clear(self):
        self.data = []
        self.pos = 0
    
    def get_buffer(self):
        return self.data


class Agent:
    def __init__(self,
                 project_root,
                 log_dir,
                 logger,
                 env,
                 eval_env,
                 id_,
                 device,
                 offline_data=None,
                 load_offline_data=True,
                 offline_setting=True
                 ):
        if offline_setting: assert load_offline_data
        self.offline_setting = offline_setting
        self.device = device
        self.offline_data = offline_data
        self.project_root = project_root
        self.log_dir = log_dir
        self.logger = logger
        self.gamma = 0.99
        self.timeout = 1000
        self.env = env
        self.memory_size = 2000000
        self.batch_size = 256
        self.replay = Replay(memory_size=self.memory_size, batch_size=self.batch_size, seed=id_)
        self.eval_env = eval_env #copy.deepcopy(env)
        self.state_normalizer = lambda x: x
        
        self.episode_reward = 0
        self.episode_rewards = []
        self.total_steps = 0
        self.reset = True
        self.ep_steps = 0
        self.num_episodes = 0
        self.stats_queue_size = 5
        self.batch_indices = torch.arange(self.batch_size).long().to(device)
        self.ep_returns_queue_train = np.zeros(self.stats_queue_size)
        self.ep_returns_queue_test = np.zeros(self.stats_queue_size)
        self.train_stats_counter = 0
        self.test_stats_counter = 0
        self.agent_rng = np.random.RandomState(id_)
        self.test_rng = np.random.RandomState(id_)
        
        self.populate_latest = False
        self.populate_states, self.populate_actions, self.populate_true_qs = None, None, None
        self.automatic_tmp_tuning = False
        
        self.state = None
        self.action = None
        self.next_state = None
        
        self.eps = 1e-8
        self.use_target_network = True
        self.target_network_update_freq = 1
        self.polyak = 0.995
        
    
    def offline_param_init(self):
        # self.trainset, self.testset = self.training_set_construction(self.offline_data)
        # self.training_size = len(self.trainset[0])
        # self.training_indexs = np.arange(self.training_size)
        self.training_loss = []
        self.test_loss = []
        self.tloss_increase = 0
        self.tloss_rec = np.inf
        # assert self.training_size >= self.memory_size
        # self.fill_offline_data_to_buffer()
        
    def feed_data(self):
        return
    
    def get_data(self):
        states, actions, rewards, next_states, terminals = self.replay.sample()
        in_ = torch_utils.tensor(self.state_normalizer(states), self.device)
        r = torch_utils.tensor(rewards, self.device)
        ns = torch_utils.tensor(self.state_normalizer(next_states), self.device)
        t = torch_utils.tensor(terminals, self.device)
        data = {
            'obs': in_,
            'act': actions,
            'reward': r,
            'obs2': ns,
            'done': t
        }
        return data
    
    def fill_offline_data_to_buffer(self):
        self.trainset, self.testset = self.training_set_construction(self.offline_data)
        self.training_size = len(self.trainset[0])
        self.training_indexs = np.arange(self.training_size)
        train_s, train_a, train_r, train_ns, train_t, _, _, _, _ = self.trainset
        for idx in range(len(train_s)):
            self.replay.feed([train_s[idx], train_a[idx], train_r[idx], train_ns[idx], train_t[idx]])
        if self.memory_size < self.training_size:
            self.logger.info("\nWARNING: Buffer size is less than training set size, {} vs {}\n".format(self.memory_size, self.training_size))

    def step(self):
        trans = self.feed_data()
        data = self.get_data()
        losses = self.update(data)
        return trans, losses
    
    def update(self, data):
        raise NotImplementedError
    
    def reset_population_flag(self):
        # Done evaluation, regenerate data at next checkpoint
        self.populate_latest = False
        self.populate_states, self.populate_actions, self.populate_true_qs = None, None, None
    
    def update_stats(self, reward, done):
        self.episode_reward += reward
        self.total_steps += 1
        self.ep_steps += 1
        # print(self.ep_steps, self.total_steps, done)
        if done or self.ep_steps == self.timeout:
            self.episode_rewards.append(self.episode_reward)
            self.num_episodes += 1
            self.add_train_log(self.episode_reward)
            self.episode_reward = 0
            self.ep_steps = 0
            self.reset = True
    
    def add_train_log(self, ep_return):
        self.ep_returns_queue_train[self.train_stats_counter] = ep_return
        self.train_stats_counter += 1
        self.train_stats_counter = self.train_stats_counter % self.stats_queue_size
    
    def add_test_log(self, ep_return):
        self.ep_returns_queue_test[self.test_stats_counter] = ep_return
        self.test_stats_counter += 1
        self.test_stats_counter = self.test_stats_counter % self.stats_queue_size
    
    def populate_returns(self, log_traj=False, total_ep=None, initialize=False):
        total_ep = self.stats_queue_size if total_ep is None else total_ep
        total_steps = 0
        total_states = []
        total_actions = []
        total_returns = []
        for ep in range(total_ep):
            ep_return, steps, traj = self.eval_episode(log_traj=log_traj)
            total_steps += steps
            total_states += traj[0]
            total_actions += traj[1]
            total_returns += traj[2]
            self.add_test_log(ep_return)
            if initialize:
                self.add_train_log(ep_return)
        return [total_states, total_actions, total_returns]
    
    def random_fill_buffer(self, total_steps):
        state = self.eval_env.reset()
        for _ in range(total_steps):
            action = self.agent_rng.randint(0, self.env.action_dim)
            last_state = state
            state, reward, done, _ = self.eval_env.step([action])
            self.replay.feed([last_state, action, reward, state, int(done)])
            if done:
                state = self.eval_env.reset()
    
    def eval_episode(self, log_traj=False):
        ep_traj = []
        state = self.eval_env.reset()
        total_rewards = 0
        ep_steps = 0
        done = False
        while True:
            action = self.eval_step(state)
            last_state = state
            state, reward, done, _ = self.eval_env.step([action])
            # print(np.abs(state-last_state).sum(), "\n",action)
            if log_traj:
                ep_traj.append([last_state, action, reward])
            total_rewards += reward
            ep_steps += 1
            if done or ep_steps == self.timeout:
                break
        
        states = []
        actions = []
        rets = []
        if log_traj:
            ret = 0
            for i in range(len(ep_traj) - 1, -1, -1):
                s, a, r = ep_traj[i]
                ret = r + self.gamma * ret
                rets.insert(0, ret)
                actions.insert(0, a)
                states.insert(0, s)
        return total_rewards, ep_steps, [states, actions, rets]
    
    def eval_episodes(self):
        return
    
    def log_return(self, log_ary, name, elapsed_time):
        total_episodes = len(self.episode_rewards)
        mean, median, min_, max_ = np.mean(log_ary), np.median(log_ary), np.min(log_ary), np.max(log_ary)
        log_str = '%s LOG: steps %d, episodes %3d, ' \
                  'returns %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), %.2f steps/s'
        self.logger.info(log_str % (name, self.total_steps, total_episodes, mean, median,
                                        min_, max_, len(log_ary),
                                        elapsed_time))
        return mean, median, min_, max_
    
    def log_file(self, elapsed_time=-1, test=True):
        if not self.offline_setting:
            mean, median, min_, max_ = self.log_return(self.ep_returns_queue_train, "TRAIN", elapsed_time)
        # self.populate_returns()
        if test:
            self.populate_states, self.populate_actions, self.populate_true_qs = self.populate_returns(log_traj=True)
            self.populate_latest = True
            mean, median, min_, max_ = self.log_return(self.ep_returns_queue_test, "TEST", elapsed_time)
            try:
                normalized = np.array([self.eval_env.env.unwrapped.get_normalized_score(ret_) for ret_ in self.ep_returns_queue_test])
                mean, median, min_, max_ = self.log_return(normalized, "Normalized", elapsed_time)
            except:
                pass
    
    def policy(self, state, eps):
        raise NotImplementedError
    
    def eval_step(self, state):
        # action = self.policy(state, 0)
        # return action
        raise NotImplementedError
    
    def save(self):
        raise NotImplementedError
    
    def load(self, filename):
        raise NotImplementedError
        
    def training_set_construction(self, data_dict):
        assert len(list(data_dict.keys())) == 1
        data_dict = data_dict[list(data_dict.keys())[0]]
        states = data_dict['states']
        actions = data_dict['actions']
        rewards = data_dict['rewards']
        next_states = data_dict['next_states']
        terminations = data_dict['terminations']
        next_actions = np.concatenate([data_dict['actions'][1:], data_dict['actions'][-1:]])  # Should not be used when using the current estimation in target construction

        thrshd = int(len(states))
        training_s = states[: thrshd]
        training_a = actions[: thrshd]
        training_r = rewards[: thrshd]
        training_ns = next_states[: thrshd]
        training_t = terminations[: thrshd]
        training_na = next_actions[: thrshd]
        
        testing_s = states[thrshd:]
        testing_a = actions[thrshd:]
        testing_r = rewards[thrshd:]
        testing_ns = next_states[thrshd:]
        testing_t = terminations[thrshd:]
        testing_na = next_actions[thrshd:]
        
        return [training_s, training_a, training_r, training_ns, training_t, training_na, None, None, None], \
               [testing_s, testing_a, testing_r, testing_ns, testing_t, testing_na, None, None, None]


