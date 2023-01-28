import os

import numpy as np
import torch
import copy

import matplotlib.pyplot as plt
import matplotlib as mpl

from core.utils import torch_utils
from core.utils import helpers


class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.gamma = cfg.discount
        self.env = cfg.env_fn()
        self.replay = cfg.replay_fn()
        self.eval_env = copy.deepcopy(cfg.env_fn)()

        self.episode_reward = 0
        self.episode_rewards = []
        self.total_steps = 0
        self.reset = True
        self.ep_steps = 0
        self.num_episodes = 0
        self.timeout = cfg.timeout
        self.batch_indices = torch.arange(self.cfg.batch_size).long().to(cfg.device)
        self.ep_returns_queue_train = np.zeros(cfg.stats_queue_size)
        self.ep_returns_queue_test = np.zeros(cfg.stats_queue_size)
        self.train_stats_counter = 0
        self.test_stats_counter = 0
        self.agent_rng = np.random.RandomState(self.cfg.seed)
        self.test_rng = np.random.RandomState(self.cfg.seed)
        self.true_q_predictor = self.cfg.tester_fn.get('true_value_estimator', lambda x:None)
        self.eval_set = self.property_evaluation_dataset(getattr(cfg, 'eval_data', None), getattr(cfg, 'qmax_table', None))

        self.populate_latest = False
        self.populate_states, self.populate_actions, self.populate_true_qs = None, None, None
        self.automatic_tmp_tuning = False
        
        self.state = None
        self.action = None
        self.next_state = None
        self.eps = 1e-8
        
        self.temp = 0

    def offline_param_init(self):
        self.trainset, self.testset = self.training_set_construction(self.cfg.offline_data)
        self.training_size = len(self.trainset[0])
        self.training_indexs = np.arange(self.training_size)

        self.training_loss = []
        self.test_loss = []
        self.tloss_increase = 0
        self.tloss_rec = np.inf

    def feed_data(self):
        if self.reset is True:
            self.state = self.env.reset()
            self.reset = False
        action = self.policy(self.state, eval=False)
        next_state, reward, done, _ = self.env.step([action])
        self.replay.feed([self.state, action, reward, next_state, int(done)])
        prev_state = self.state
        self.state = next_state
        self.update_stats(reward, done)
        return prev_state, action, reward, next_state, int(done)

    def get_data(self):
        states, actions, rewards, next_states, terminals = self.replay.sample()
        in_ = torch_utils.tensor(self.cfg.state_normalizer(states), self.cfg.device)
        r = torch_utils.tensor(rewards, self.cfg.device)
        ns = torch_utils.tensor(self.cfg.state_normalizer(next_states), self.cfg.device)
        t = torch_utils.tensor(terminals, self.cfg.device)
        data = {
            'obs': in_,
            'act': actions,
            'reward': r,
            'obs2': ns,
            'done': t
        }
        return data

    def get_offline_data(self):
        train_s, train_a, train_r, train_ns, train_t, train_na, _, timeouts, _ = self.trainset
        idxs = self.agent_rng.randint(0, len(train_s), size=self.cfg.batch_size) \
            if self.cfg.batch_size < len(train_s) else np.arange(len(train_s))

        in_ = torch_utils.tensor(self.cfg.state_normalizer(train_s[idxs]), self.cfg.device)
        act = train_a[idxs]
        r = torch_utils.tensor(train_r[idxs], self.cfg.device)
        ns = torch_utils.tensor(self.cfg.state_normalizer(train_ns[idxs]), self.cfg.device)
        t = torch_utils.tensor(train_t[idxs], self.cfg.device)
        na = train_na[idxs]
        to = timeouts[idxs]

        data = {
            'obs': in_,
            'act': act,
            'reward': r,
            'obs2': ns,
            'done': t,
            'act2': na,
            'timeout': to
        }
        return data

    def get_offline_traj(self, traj_len=3):
        train_s, train_a, train_r, train_ns, train_t, train_na, _, timeouts, _ = self.trainset
        idxs = self.agent_rng.randint(0, len(train_s), size=self.cfg.batch_size) \
            if self.cfg.batch_size < len(train_s) else np.arange(len(train_s))

        in_ = []
        act = []
        r = []
        ns = []
        t = []
        na = []
        to = []
        for k in range(traj_len):
            in_.append(torch_utils.tensor(self.cfg.state_normalizer(train_s[idxs-k]), self.cfg.device))
            act.append(train_a[idxs-k])
            r.append(torch_utils.tensor(train_r[idxs-k], self.cfg.device))
            ns.append(torch_utils.tensor(self.cfg.state_normalizer(train_ns[idxs-k]), self.cfg.device))
            t.append(torch_utils.tensor(train_t[idxs-k], self.cfg.device))
            na.append(train_na[idxs-k])
            to.append(timeouts[idxs-k])
        
        starts = np.where(idxs < traj_len)[0]
        for st in starts:
            t[idxs[st]][st] = 1
        
        data = {
            'obs': in_,
            'act': act,
            'reward': r,
            'obs2': ns,
            'done': t,
            'act2': na,
            'timeout': to
        }
        return data

    def get_weighted_offline_data(self, higher_priority_index, higher_priority_prob):
        train_s, train_a, train_r, train_ns, train_t, train_na, _, timeouts, _ = self.trainset
        
        eps = self.agent_rng.rand(self.cfg.batch_size)
        idxs = np.zeros(self.cfg.batch_size, dtype=int)
        highpris = np.where(eps < higher_priority_prob)[0]
        idxs[highpris] = self.agent_rng.randint(0, higher_priority_index, size=len(highpris))
        lowpris = np.where(eps >= higher_priority_prob)[0]
        idxs[lowpris] = self.agent_rng.randint(higher_priority_index, len(train_s), size=len(lowpris))
        # eps = self.agent_rng.rand()
        # if eps < higher_priority_prob:
        #     idxs = self.agent_rng.randint(0, higher_priority_index, size=self.cfg.batch_size) \
        #         if self.cfg.batch_size < higher_priority_index else np.arange(higher_priority_index)
        # else:
        #     idxs = self.agent_rng.randint(higher_priority_index, len(train_s), size=self.cfg.batch_size) \
        #         if self.cfg.batch_size < len(train_s)-higher_priority_index else np.arange(len(train_s)-higher_priority_index)+higher_priority_index

        in_ = torch_utils.tensor(self.cfg.state_normalizer(train_s[idxs]), self.cfg.device)
        act = train_a[idxs]
        r = torch_utils.tensor(train_r[idxs], self.cfg.device)
        ns = torch_utils.tensor(self.cfg.state_normalizer(train_ns[idxs]), self.cfg.device)
        t = torch_utils.tensor(train_t[idxs], self.cfg.device)
        na = train_na[idxs]
        to = timeouts[idxs]

        data = {
            'obs': in_,
            'act': act,
            'reward': r,
            'obs2': ns,
            'done': t,
            'act2': na,
            'timeout': to
        }
        return data

    def get_uniform_offline_data(self, probs):
        train_s, train_a, train_r, train_ns, train_t, train_na, _, timeouts, _ = self.trainset
        idxs = self.agent_rng.choice(np.arange(len(train_s)), size=self.cfg.batch_size, replace=True, p=probs)

        in_ = torch_utils.tensor(self.cfg.state_normalizer(train_s[idxs]), self.cfg.device)
        act = train_a[idxs]
        r = torch_utils.tensor(train_r[idxs], self.cfg.device)
        ns = torch_utils.tensor(self.cfg.state_normalizer(train_ns[idxs]), self.cfg.device)
        t = torch_utils.tensor(train_t[idxs], self.cfg.device)
        na = train_na[idxs]
        to = timeouts[idxs]

        data = {
            'obs': in_,
            'act': act,
            'reward': r,
            'obs2': ns,
            'done': t,
            'act2': na,
            'timeout': to
        }
        return data

    def fill_offline_data_to_buffer(self):
        self.trainset, self.testset = self.training_set_construction(self.cfg.offline_data)
        train_s, train_a, train_r, train_ns, train_t, _, _, _, _ = self.trainset
        for idx in range(len(train_s)):
            self.replay.feed([train_s[idx], train_a[idx], train_r[idx], train_ns[idx], train_t[idx]])

    def step(self):
        trans = self.feed_data()
        data = self.get_data()
        if self.check_update():#self.cfg.policy_fn_config["train_params"] and self.cfg.critic_fn_config["train_params"]:
            losses = self.update(data)
        else:
            losses = None
        return trans, losses
    
    def check_update(self):
        return self.cfg.policy_fn_config["train_params"] or self.cfg.critic_fn_config["train_params"]
    
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
            if self.cfg.evaluation_criteria == "return":
                self.add_train_log(self.episode_reward)
            elif self.cfg.evaluation_criteria == "steps":
                self.add_train_log(self.ep_steps)
            else:
                raise NotImplementedError
            self.episode_reward = 0
            self.ep_steps = 0
            self.reset = True

    def add_train_log(self, ep_return):
        self.ep_returns_queue_train[self.train_stats_counter] = ep_return
        self.train_stats_counter += 1
        self.train_stats_counter = self.train_stats_counter % self.cfg.stats_queue_size

    def add_test_log(self, ep_return):
        self.ep_returns_queue_test[self.test_stats_counter] = ep_return
        self.test_stats_counter += 1
        self.test_stats_counter = self.test_stats_counter % self.cfg.stats_queue_size

    def populate_returns(self, log_traj=False, total_ep=None, initialize=False):
        total_ep = self.cfg.stats_queue_size if total_ep is None else total_ep
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
            if self.cfg.evaluation_criteria == "return":
                self.add_test_log(ep_return)
                if initialize:
                    self.add_train_log(ep_return)
            elif self.cfg.evaluation_criteria == "steps":
                self.add_test_log(steps)
                if initialize:
                    self.add_train_log(steps)
            else:
                raise NotImplementedError
        return [total_states, total_actions, total_returns]
    
    # hacking environment and policy
    def populate_returns_random_start(self, start_pos=None, start_policy=None, total_ep=None):
        total_ep = self.cfg.stats_queue_size if total_ep is None else total_ep
        total_states = []
        total_actions = []
        total_returns = []
        for ep in range(total_ep):
            s, a, ret = self.eval_episode_random_start(start_pos=start_pos, random_insert_policy=start_policy)
            total_states.append(s)
            total_actions.append(a)
            total_returns.append(ret)
        return [total_states, total_actions, total_returns]

    def random_fill_buffer(self, total_steps):
        state = self.eval_env.reset()
        for _ in range(total_steps):
            action = self.agent_rng.randint(0, self.cfg.action_dim)
            last_state = state
            state, reward, done, _ = self.eval_env.step([action])
            self.replay.feed([last_state, action, reward, state, int(done)])
            if done:
                state = self.eval_env.reset()
                # print("Done")

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
            if done or ep_steps == self.cfg.timeout:
                break

        states = []
        actions = []
        rets = []
        if log_traj:
            # s, a, r = ep_traj[len(ep_traj)-1]
            # ret = r if done else self.true_q_predictor(self.cfg.state_normalizer(s))[a]
            # states = [s]
            # actions = [a]
            # rets = [ret]
            # for i in range(len(ep_traj)-2, -1, -1):
            ret = 0
            for i in range(len(ep_traj)-1, -1, -1):
                s, a, r = ep_traj[i]
                ret = r + self.cfg.discount * ret
                rets.insert(0, ret)
                actions.insert(0, a)
                states.insert(0, s)
        return total_rewards, ep_steps, [states, actions, rets]

    # hacking environment and policy
    def eval_episode_random_start(self, start_pos, random_insert_policy):
        if start_pos is None:
            ep_traj = []
            state = self.eval_env.reset()
            total_rewards = 0
            ep_steps = 0
            while True:
                action = self.eval_step(state)
                last_state = state
                state, reward, done, _ = self.eval_env.step([action])
                ep_traj.append([last_state, action, reward])
                total_rewards += reward
                ep_steps += 1
                if done or ep_steps == self.cfg.timeout:
                    break
            
            random_idx = self.test_rng.randint(len(ep_traj))
            random_start = ep_traj[random_idx][0]
        else:
            random_start = start_pos

        action = random_insert_policy(random_start)
        state, reward, done, _ = self.eval_env.hack_step(random_start, action)
        if done:
            return random_start, action, reward
        ep_traj = []
        total_rewards = 0
        ep_steps = 0
        while True:
            action = self.eval_step(state)
            last_state = state
            state, reward, done, _ = self.eval_env.hack_step(last_state, action)
            ep_traj.append([last_state, action, reward])
            total_rewards += reward
            ep_steps += 1
            if done or ep_steps == self.cfg.timeout:
                break

        ret = 0
        for i in range(len(ep_traj)-1, -1, -1):
            s, a, r = ep_traj[i]
            ret = r + self.cfg.discount * ret
        return s, a, ret

    def eval_episodes(self):
        return

    def log_return(self, log_ary, name, elapsed_time):
        rewards = log_ary
        total_episodes = len(self.episode_rewards)
        mean, median, min_, max_ = np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards)

        log_str = '%s LOG: steps %d, episodes %3d, ' \
                  'returns %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), %.2f steps/s'

        self.cfg.logger.info(log_str % (name, self.total_steps, total_episodes, mean, median,
                                        min_, max_, len(rewards),
                                        elapsed_time))
        return mean, median, min_, max_

    def log_file(self, elapsed_time=-1, test=True):
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
        return mean, median, min_, max_

    def policy(self, o, eval=False):
        o = torch_utils.tensor(self.cfg.state_normalizer(o), self.cfg.device)
        with torch.no_grad():
            a, _ = self.ac.pi(o, deterministic=eval)
            # a, _ = self.ac.pi(o)
        a = torch_utils.to_np(a)
        return a

    def eval_step(self, state):
        a = self.policy(state, eval=True)
        return a

    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError

    def one_hot_action(self, actions):
        one_hot = np.zeros((len(actions), self.cfg.action_dim))
        np.put_along_axis(one_hot, actions.reshape((-1, 1)), 1, axis=1)
        return one_hot
    
    def default_value_predictor(self):
        def vp(x):
            with torch.no_grad():
                q1, q2 = self.ac.q1q2(x)
            return torch.minimum(q1, q2)
        return vp

    def default_rep_predictor(self):
        def rp(x):
            with torch.no_grad():
                rep = self.ac.pi.body(self.ac.pi.rep(x))
            return rep
        return rp
    
    def training_set_construction(self, data_dict, value_predictor=None):
        if value_predictor is None:
            value_predictor = self.default_value_predictor()
       
        assert len(list(data_dict.keys())) == 1
        data_dict = data_dict[list(data_dict.keys())[0]]
        states = data_dict['states']
        actions = data_dict['actions']
        rewards = data_dict['rewards']
        next_states = data_dict['next_states']
        terminations = data_dict['terminations']
        next_actions = np.concatenate([data_dict['actions'][1:], data_dict['actions'][-1:]])  # Should not be used when using the current estimation in target construction
        if 'timeouts' in data_dict:
            timeout = data_dict['timeouts']
        else:
            timeout = np.zeros(len(states))
        # thrshd = int(len(states) * 0.8)
        thrshd = int(len(states))
        training_s = states[: thrshd]
        training_a = actions[: thrshd]
        training_r = rewards[: thrshd]
        training_ns = next_states[: thrshd]
        training_t = terminations[: thrshd]
        training_na = next_actions[: thrshd]
        training_timeout = timeout[: thrshd]

        testing_s = states[thrshd:]
        testing_a = actions[thrshd:]
        testing_r = rewards[thrshd:]
        testing_ns = next_states[thrshd:]
        testing_t = terminations[thrshd:]
        testing_na = next_actions[thrshd:]
        testing_timeout = timeout[thrshd:]
        return [training_s, training_a, training_r, training_ns, training_t, training_na, None, training_timeout, None], \
               [testing_s, testing_a, testing_r, testing_ns, testing_t, testing_na, None, testing_timeout, None]

    def property_evaluation_dataset(self, data_dict, qmax_table):
        if data_dict is None:
            return
        states = []
        actions = []
        returns = []
        qtable = None
        if self.cfg.evaluate_overestimation:
            assert qmax_table is not None
            qtable = torch.load(qmax_table)

        for name in data_dict:
            states.append(data_dict[name]['states'])
            actions.append(data_dict[name]['actions'])
            # if 'returns' in data_dict[name]:
            #     returns.append(data_dict[name]['returns'])
            if 'qmax' in data_dict[name]:
                returns.append(data_dict[name]['qmax'])
            else:
                true_returns = np.zeros(len(data_dict[name]['states']))
                rewards = data_dict[name]['rewards']
                next_states = data_dict[name]['next_states']
                terminations = data_dict[name]['terminations']
                for i in range(len(states) - 1, -1, -1):
                    if i == len(states) - 1 or (not np.array_equal(next_states[i], states[i + 1])):
                        true_pred = self.true_q_predictor(self.cfg.state_normalizer(states[i]))
                        true_returns[i] = 0 if true_pred is None else true_pred[actions[i]]
                    else:
                        end = 1.0 if terminations[i] else 0.0
                        true_returns[i] = rewards[i] + (1 - end) * self.cfg.discount * true_returns[i + 1]
                returns.append(true_returns)

        if len(states) > 0:
            states = np.concatenate(states)
            actions = np.concatenate(actions)
            returns = np.concatenate(returns)
        return states, actions, returns, qtable

    def log_overestimation(self):
        test_s, _, _, qmax_table = self.eval_set
        # _, _, _, qmax_table = self.eval_set
        # test_s, _, _, _, _, _, _, _, _ = self.trainset
        test_s = [tuple(row) for row in test_s]
        test_s = np.unique(test_s, axis=0)
 
        with torch.no_grad():
            q_values = self.default_value_predictor()(torch_utils.tensor(self.cfg.state_normalizer(test_s), self.cfg.device))
            q_values = torch_utils.to_np(q_values)
        onpolicy_a = np.argmax(q_values, axis=1)
        onpolicy_q = q_values[np.arange(len(q_values)), onpolicy_a]
        qmax = qmax_table[tuple(test_s.T)][np.arange(len(test_s)), onpolicy_a]
        all_diff = onpolicy_q - qmax
        # abs_diff = np.abs(onpolicy_q - qmax)
        log_str = 'TRAIN LOG: steps %d, ' \
                  'Overestimation: %.8f/%.8f/%.8f (mean/min/max)'
        self.cfg.logger.info(log_str % (self.total_steps, all_diff.mean(), all_diff.min(), all_diff.max()))
        # log_str = 'TRAIN LOG: steps %d, ' \
        #           'EstimationAbsError: %.8f/%.8f/%.8f (mean/min/max)'
        # self.cfg.logger.info(log_str % (self.total_steps, abs_diff.mean(), abs_diff.min(), abs_diff.max()))
        
    def log_overestimation_current_pi(self):
        if not self.populate_latest:
            self.populate_states, self.populate_actions, self.populate_true_qs = self.populate_returns(log_traj=True)
        states = np.array(self.populate_states)
        true_qs = np.array(self.populate_true_qs)
        with torch.no_grad():
            # phis = self.rep_net(self.cfg.state_normalizer(states))
            # q_values = self.val_net(phis)
            q_values = self.default_value_predictor()(self.cfg.state_normalizer(states))
            q_values = torch_utils.to_np(q_values)
        onpolicy_q = q_values[np.arange(len(q_values)), self.populate_actions]
        all_diff = onpolicy_q - true_qs
        log_str = 'TRAIN LOG: steps %d, ' \
                  'OverestimationCurrentPi: %.8f/%.8f/%.8f (mean/min/max)'
        self.cfg.logger.info(log_str % (self.total_steps, all_diff.mean(), all_diff.min(), all_diff.max()))

    def log_rep_rank(self):
        """ From https://arxiv.org/pdf/2207.02099.pdf Appendix A.11"""
        test_s, _, _, _ = self.eval_set

        def compute_rank_from_features(feature_matrix, rank_delta=0.01):
            sing_values = np.linalg.svd(feature_matrix, compute_uv=False)
            cumsum = np.cumsum(sing_values)
            nuclear_norm = np.sum(sing_values)
            approximate_rank_threshold = 1.0 - rank_delta
            threshold_crossed = (
                cumsum >= approximate_rank_threshold * nuclear_norm)
            effective_rank = sing_values.shape[0] - np.sum(threshold_crossed) + 1
            return effective_rank
    
        states = torch_utils.tensor(self.cfg.state_normalizer(test_s), self.cfg.device)
        with torch.no_grad():
            phi_s = torch_utils.to_np(self.default_rep_predictor()(states))
        erank = compute_rank_from_features(phi_s)
        log_str = 'TRAIN LOG: steps %d, ' \
                  'RepRank: %.8f'
        self.cfg.logger.info(log_str % (self.total_steps, erank))

    def draw_action_value(self):
        # test_s, _, _, _ = self.eval_set
        states, obstacles = self.eval_env.get_state_space()
        goal = self.eval_env.get_goal_coord()
        with torch.no_grad():
            qs = self.default_value_predictor()(torch_utils.tensor(self.cfg.state_normalizer(states), self.cfg.device))
            torch_utils.to_np(qs)

        fig, axs = plt.subplots(1, self.cfg.action_dim+1, figsize=(13, 5))
        for a in range(self.cfg.action_dim):
            template = np.zeros((self.eval_env.num_cols, self.eval_env.num_rows))
            for idx, s in enumerate(states):
                template[s[0], s[1]] = qs[idx, a]
            img = axs[a].imshow(template, cmap="Blues", vmin=qs[:, a].min(), vmax=qs[:, a].max())
            axs[a].set_title("Action{}".format(self.eval_env.actions[a]))
            for obs in obstacles:
                axs[a].text(obs[1]-0.3, obs[0]+0.3, "X", color="orange", fontsize=7)
            axs[a].text(goal[1]-0.3, goal[0]+0.3, "G", color="orange", fontsize=7)
            plt.colorbar(img, ax=axs[a], shrink=0.5)

        template = np.zeros((self.eval_env.num_cols, self.eval_env.num_rows))
        policy = np.zeros((self.eval_env.num_cols, self.eval_env.num_rows), dtype=str)
        action_list = self.eval_env.directions
        for idx, s in enumerate(states):
            template[s[0], s[1]] = qs[idx].max()
            policy[s[0], s[1]] = action_list[self.policy(s, 0)]
        img = axs[-1].imshow(template, cmap="Blues", vmin=qs.min(), vmax=qs.max())
        axs[-1].set_title("{}".format(self.eval_env.actions))
        for obs in obstacles:
            axs[-1].text(obs[1]-0.3, obs[0]+0.3, "X", color="orange", fontsize=7)
        for idx, s in enumerate(states):
            axs[-1].text(s[1]-0.3, s[0]+0.3, policy[s[0], s[1]], color="black", fontsize=7)
        axs[-1].text(goal[1]-0.3, goal[0]+0.3, "G", color="orange", fontsize=7)
        plt.colorbar(img, ax=axs[-1], shrink=0.5)
        plt.tight_layout()
        pth = self.cfg.get_visualization_dir()
        plt.savefig("{}/action_values.png".format(pth), dpi=300, bbox_inches='tight')
        # plt.savefig("{}/action_values_{}.png".format(pth, self.total_steps), dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()

