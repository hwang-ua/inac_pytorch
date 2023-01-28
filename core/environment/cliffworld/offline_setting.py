import sys
# import os
# sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from collections import deque
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random
import itertools
import time

import torch
import torch.nn.functional as F
from agents import DQN_Agent
from networks import small_QNetwork as cnn
from networks import mlp

from q_learning import QLearningAgent
from cliffworld_env import Environment
from optimal_cliffworld import count_correct,optimal_value

plt.style.use('seaborn-darkgrid') # seaborn-darkgrid, seaborn-whitegrid
plt.rc('font', size=20)

class Offline_data_buffer():
    def __init__(self,buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size) # (s_t,a_t,r_t,t_t,s_t1) s and a are scalar number
        self.state_list = [] # state in buffer, only state here

    def clear_buffer(self):
        self.buffer = deque(maxlen=self.buffer_size)  # (s_t,a_t,r_t,t_t,s_t1) s and a are scalar number
        self.state_list = []  # state in buffer, only state here

    def add_data(self,transition):
        self.buffer.append(transition)

    def sample_batch(self,n):
        index = np.random.randint(len(self.buffer),size = n)
        batch = [self.buffer[i] for i in index]

        return batch

class Offline_training():
    def __init__(self):
        self.discount = 0.99
        self.data_num = 10000 # 17*4 # 10000

        self.update_steps = 100 # update how many times
        self.plot_num = 100 # only draw 100 points to plot
        self.step_index = range(0,self.update_steps,self.update_steps//self.plot_num)

        self.env = Environment(scalar_obs=True)
        self.offline_data = Offline_data_buffer(self.data_num)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_num = 1 # run multiple seeds
        self.target_update_interval = 500
        self.rand_base = self.random_baseline()

    def reset_update_steps(self,update_steps):
        self.update_steps = update_steps
        self.step_index = range(0,self.update_steps,self.update_steps//self.plot_num)
        self.rand_base = self.random_baseline()

    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def data_collecting(self):
        self.offline_data.clear_buffer()
        # assert len(self.offline_data.buffer) == 0

        if self.data_num//4 < 37:
            # states = np.random.choice(self.env.state_list,self.data_num//4,replace=False)
            # for s in states:
            #     self.offline_data.state_list.append(s)
            #     for a in range(self.env.action_space):
            #         self.env.set_state(s)
            #         s_prime, reward, done, info = self.env.step(a)
            #
            #         self.offline_data.add_data([s, a, reward, done, s_prime])

            for i in range(self.data_num):
                s = self.env.reset(random_choice=True)
                a = np.random.randint(self.env.action_space)
                s_prime, reward, done, info = self.env.step(a)

                self.offline_data.add_data([s,a,reward,done,s_prime])
        else:
            self.offline_data.state_list = self.env.state_list
            for i in range(self.data_num):
                s = self.env.reset(random_choice=True)
                a = np.random.randint(self.env.action_space)
                s_prime, reward, done, info = self.env.step(a)

                self.offline_data.add_data([s,a,reward,done,s_prime])

        # assert len(self.offline_data.buffer) == self.data_num

    def data_collecting_all_transition(self):
        self.offline_data.clear_buffer()
        for s in self.env.state_list:
            for a in range(4):
                self.env.current_state = self.env.obs2state(s)
                s_prime, reward, done, info = self.env.step(a)
                self.offline_data.add_data([s,a,reward,done,s_prime])

    def q_learning(self,batch_size,step_size):
        correct_count_data = []
        loss_data = []
        for seed in range(self.run_num):
            print('seed :',seed)
            self.setup_seed(seed)

            # self.data_collecting()
            self.data_collecting_all_transition()
            self.q_agent = QLearningAgent()

            correct_count_list = []
            loss_list = []

            for i in range(self.update_steps):
                sample_data = self.offline_data.sample_batch(batch_size)
                for s,a,reward,done,s_prime in sample_data:
                    target = reward + (1-done) * self.discount * np.max(self.q_agent.q[s_prime,:])
                    self.q_agent.q[s,a] = self.q_agent.q[s,a] + step_size * (target - self.q_agent.q[s,a])

                if i%(self.update_steps//self.plot_num) == 0:
                    num,wrong_action_list = count_correct(self.q_agent.q)
                    loss = np.mean(np.abs(self.q_agent.q - optimal_value))

                    correct_count_list.append(num)
                    loss_list.append(loss)

            tmp_correct_count = {'steps': self.step_index, 'correct_count': correct_count_list}
            tmp_correct_count = pd.DataFrame(tmp_correct_count)
            tmp_loss = {'steps': self.step_index, 'loss': loss_list}
            tmp_loss = pd.DataFrame(tmp_loss)

            correct_count_data.append(tmp_correct_count)
            loss_data.append(tmp_loss)

        correct_count_data = pd.concat(correct_count_data)
        loss_data = pd.concat(loss_data)

        self.plot(correct_count_data,'correct_count','q learning: correct action')
        self.plot(loss_data,'loss','q learning: loss')

    def plot(self,data,ylabel,tittle,hue = None):
        png = plt.figure(figsize=(10, 10))
        if hue:
            sns.lineplot(x='steps', y=ylabel, data=data,hue = hue,palette = sns.color_palette()[:len(set(data[hue]))])
        else:
            sns.lineplot(x='steps', y=ylabel, data=data)

        if ylabel == 'correct_count':
            # sns.lineplot(x='steps', y=ylabel, data=random_baseline,ci='sd')
            sns.lineplot(x='steps', y=ylabel, data=self.rand_base,ci=None)

        plt.ylim(ymin=0)
        plt.title(tittle)
        plt.show()

    def random_baseline(self):
        random_baseline = []
        for seed in range(self.run_num):
            self.setup_seed(seed)

            random_q = np.random.random((48, 4))
            num, wrong_action_list = count_correct(random_q)

            tmp_correct_count = {'steps': self.step_index, 'correct_count': [num] * self.plot_num}
            tmp_correct_count = pd.DataFrame(tmp_correct_count)

            random_baseline.append(tmp_correct_count)

        random_baseline = pd.concat(random_baseline)
        return random_baseline

    def supervised_learning(self):
        # cross_entropy_weight_list = [0]
        # use all state action pair in a batch
        batch_size = 37*4
        correct_count_data = []
        loss_data = []

        all_states = self.all_array_states(vector=True) # all state
        all_states_tensor = torch.from_numpy(np.array(all_states)).to(self.device).float()

        states_tensor = torch.from_numpy(np.delete(np.array(all_states), range(1, 12), 0)).to(self.device).float() # remove cliff states and goal state
        target_values = torch.from_numpy(np.delete(optimal_value, range(1, 12), 0)).to(self.device).float()

        for seed in range(self.run_num):
            # print('seed:',seed)
            print('seed:',seed)
            self.setup_seed(seed)

            self.data_collecting()
            if len(self.offline_data.state_list)<37:
                print(self.offline_data.state_list)
                states_tensor = torch.from_numpy(np.array(all_states)[self.offline_data.state_list]).to(self.device).float()
                target_values = torch.from_numpy(optimal_value[self.offline_data.state_list]).to(self.device).float()

            # self.net = cnn(self.env.action_space, self.env.observation_space).to(self.device)
            self.net = mlp(self.env.action_space, 2).to(self.device)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)

            correct_count_list = []
            loss_list = []

            for i in range(self.update_steps):
                q_values = self.net(states_tensor)

                loss = F.smooth_l1_loss(q_values, target_values)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.)
                self.optimizer.step()

                if i % (self.update_steps // self.plot_num) == 0:

                    with torch.no_grad():
                        q_values = self.net(all_states_tensor)
                    q_values = q_values.detach().cpu().numpy()

                    num,wrong_action_list = count_correct(q_values)
                    loss = np.mean(np.abs(np.delete(q_values, range(1, 12), 0) - np.delete(optimal_value, range(1, 12), 0)))
                    correct_count_list.append(num)
                    loss_list.append(loss)

            tmp_correct_count = {'steps': self.step_index, 'correct_count': correct_count_list,'batch_size':[batch_size]*self.plot_num}
            tmp_correct_count = pd.DataFrame(tmp_correct_count)
            tmp_loss = {'steps': self.step_index, 'loss': loss_list,'batch_size':[batch_size]*self.plot_num}
            tmp_loss = pd.DataFrame(tmp_loss)

            correct_count_data.append(tmp_correct_count)
            loss_data.append(tmp_loss)

        correct_count_data = pd.concat(correct_count_data)
        loss_data = pd.concat(loss_data)

        self.plot(correct_count_data,'correct_count','supervised learning all states: correct action')
        self.plot(loss_data,'loss','supervised learning all states: loss')

    def sampled_supervised_learning(self):
        # batch_size_list = [4,16,64,256]
        batch_size_list = [64]
        correct_count_data = []
        loss_data = []

        all_states = self.all_array_states(vector=True)
        all_states_tensor = torch.from_numpy(np.array(all_states)).to(self.device).float()
        states_tensor = torch.from_numpy(np.delete(np.array(all_states), range(1, 12), 0)).to(self.device).float()
        target_values = torch.from_numpy(np.delete(optimal_value, range(1, 12), 0)).to(self.device).float()

        for seed, batch_size in itertools.product(range(self.run_num),batch_size_list):
            print('seed:',seed,'batch_size:',batch_size)
            self.setup_seed(seed)

            self.data_collecting()
            if len(self.offline_data.state_list)<37:
                print(self.offline_data.state_list)
                states_tensor = torch.from_numpy(np.array(all_states)[self.offline_data.state_list]).to(self.device).float()
                target_values = torch.from_numpy(optimal_value[self.offline_data.state_list]).to(self.device).float()

            # self.net = cnn(self.env.action_space, self.env.observation_space).to(self.device)
            self.net = mlp(self.env.action_space, 2).to(self.device)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)

            correct_count_list = []
            loss_list = []

            for i in range(self.update_steps):
                index = np.random.randint(len(states_tensor), size=batch_size)
                q_values = self.net(states_tensor[index])

                loss = F.smooth_l1_loss(q_values, target_values[index])

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.)
                self.optimizer.step()

                if i % (self.update_steps // self.plot_num) == 0:
                    with torch.no_grad():
                        q_values = self.net(all_states_tensor)
                    q_values = q_values.detach().cpu().numpy()

                    num,wrong_action_list = count_correct(q_values)
                    loss = np.mean(np.abs(np.delete(q_values, range(1, 12), 0) - np.delete(optimal_value, range(1, 12), 0)))
                    correct_count_list.append(num)
                    loss_list.append(loss)
                    print('---')
                    print(num,wrong_action_list)
                    print(loss)

            tmp_correct_count = {'steps': self.step_index, 'correct_count': correct_count_list, 'batch_size':[batch_size]*self.plot_num}
            tmp_correct_count = pd.DataFrame(tmp_correct_count)
            tmp_loss = {'steps': self.step_index, 'loss': loss_list, 'batch_size':[batch_size]*self.plot_num}
            tmp_loss = pd.DataFrame(tmp_loss)

            correct_count_data.append(tmp_correct_count)
            loss_data.append(tmp_loss)

        correct_count_data = pd.concat(correct_count_data)
        loss_data = pd.concat(loss_data)

        self.plot(correct_count_data, 'correct_count', 'sampled supervised learning: correct action', hue='batch_size')
        self.plot(loss_data, 'loss', 'sampled supervised learning: loss', hue='batch_size')

    def dqn_all_state(self):
        # use all state action pair in a batch
        batch_size = 37*4
        correct_count_data = []
        loss_data = []
        bellman_loss_data = []

        all_states = self.all_array_states(vector=True)
        all_states_tensor = torch.from_numpy(np.array(all_states)).to(self.device).float()

        if self.data_num//4 >= 37:
            s_t = []
            a_t = []
            r_t = []
            t_t = []
            s_t1 = []
            # self.env.array_obs = True  # return array state
            for state in range(48):
                if state in range(1,12):
                    pass
                else:
                    for action in range(4):
                        # s_t.append(self.env.array_observation(self.env.obs2state(state))) #array state
                        s_t.append(self.env.obs2state(state)) # vector state
                        self.env.set_state(state)
                        s_prime, reward, done, info = self.env.step(action)
                        a_t.append(action)
                        r_t.append(np.float32(reward))
                        t_t.append(np.float32(done))
                        s_t1.append(self.env.obs2state(s_prime))
            # assert len(s_t) == len(a_t) == len(r_t) == len(t_t) == len(s_t1) == 37*4
            # self.env.array_obs = False # return scalar state

        for seed in range(self.run_num):
            # print('seed:',seed)
            print('seed:',seed)
            self.setup_seed(seed)

            if self.data_num // 4 < 37:
                s_t = []
                a_t = []
                r_t = []
                t_t = []
                s_t1 = []
                self.data_collecting()
                print(self.offline_data.state_list)
                batch_size = self.data_num

                for s, a, reward, done, s_prime in self.offline_data.buffer:
                    s_t.append(self.env.obs2state(s))
                    a_t.append(a)
                    r_t.append(np.float32(reward))
                    t_t.append(np.float32(done))
                    s_t1.append(self.env.obs2state(s_prime))

                # print(len(s_t))
                # for state in self.offline_data.state_list:
                #     for action in range(4):
                #         # s_t.append(self.env.array_observation(self.env.obs2state(state))) #array state
                #         s_t.append(self.env.obs2state(state))  # vector state
                #         self.env.set_state(state)
                #         s_prime, reward, done, info = self.env.step(action)
                #         a_t.append(action)
                #         r_t.append(np.float32(reward))
                #         t_t.append(np.float32(done))
                #         s_t1.append(self.env.obs2state(s_prime))

            self.dqn_agent = DQN_Agent(action_space=self.env.action_space,
                                       state_space=2,
                                       net=mlp, atari_name='cliffworld',train=True,
                                       exploration_final_eps=0.1, batch_size=batch_size, batch_num=1,
                                       double_dqn=False, matrix_update=False, cross_entropy_loss=False, cross_entropy_weight=0, update_time=1)
            # most of these parameters are not used, to just instantiate the dqn_agent
            correct_count_list = []
            loss_list = [] # oracl loss, different between  current q with optimal q
            bellman_loss_list = []

            for i in range(self.update_steps):
                # print(i)
                if i % self.target_update_interval == 0:
                    self.dqn_agent.target_net.load_state_dict(self.dqn_agent.net.node_dict())

                bellman_loss, other_q_s_a, other_q_s_a_max, bad_update_rate, q_max_diff, q_max_magnitude = self.dqn_agent.update.learn(0, batch_size, self.env.action_space, s_t, a_t, r_t, t_t, s_t1)

                if i %(self.update_steps//self.plot_num) == 0:
                    # print(states_tensor.shape)
                    with torch.no_grad():
                        q_values = self.dqn_agent.net(all_states_tensor)
                    q_values = q_values.detach().cpu().numpy()

                    # print(bellman_loss,q_values.shape)
                    num,wrong_action_list = count_correct(q_values)
                    loss = np.mean(np.abs(np.delete(q_values, range(1, 12), 0) - np.delete(optimal_value, range(1, 12), 0)))
                    correct_count_list.append(num)
                    loss_list.append(loss)
                    bellman_loss_list.append(bellman_loss)

            # tmp_correct_count = {'steps': self.step_index, 'correct_count': correct_count_list,'batch_size':[batch_size]*self.plot_num}
            # tmp_correct_count = pd.DataFrame(tmp_correct_count)
            # tmp_loss = {'steps': self.step_index, 'loss': loss_list,'batch_size':[batch_size]*self.plot_num}
            # tmp_loss = pd.DataFrame(tmp_loss)

            tmp_correct_count = {'steps': self.step_index, 'correct_count': correct_count_list,'batch_size':[batch_size]*self.plot_num}
            tmp_correct_count = pd.DataFrame(tmp_correct_count)
            tmp_loss = {'steps': self.step_index, 'loss': loss_list,'batch_size':[batch_size]*self.plot_num}
            tmp_loss = pd.DataFrame(tmp_loss)
            tmp_bellman_loss = {'steps': self.step_index, 'bellman_loss': bellman_loss_list,'batch_size':[batch_size]*self.plot_num}
            tmp_bellman_loss = pd.DataFrame(tmp_bellman_loss)

            correct_count_data.append(tmp_correct_count)
            loss_data.append(tmp_loss)
            bellman_loss_data.append(tmp_bellman_loss)

        correct_count_data = pd.concat(correct_count_data)
        loss_data = pd.concat(loss_data)
        bellman_loss_data = pd.concat(bellman_loss_data)

        self.plot(correct_count_data, 'correct_count', 'dqn all (s,a) pairs: correct action')
        self.plot(loss_data, 'loss', 'dqn all (s,a) pairs: loss')
        self.plot(bellman_loss_data, 'bellman_loss', 'dqn all (s,a) pairs: bellman loss')

    def dqn(self):
        # sample a batch to update
        # batch_size_list = [4,16,64,256]
        batch_size_list = [256,1024]
        correct_count_data = []
        loss_data = []
        bellman_loss_data = []


        all_states = self.all_array_states(vector=True)
        all_states_tensor = torch.from_numpy(np.array(all_states)).to(self.device).float()

        for seed, batch_size in itertools.product(range(self.run_num),batch_size_list):
            print('seed:',seed,'batch_size:',batch_size)
            self.setup_seed(seed)

            self.data_collecting()
            print(self.offline_data.state_list)
            self.dqn_agent = DQN_Agent(action_space=self.env.action_space,
                                       state_space=2,
                                       net=mlp, atari_name='cliffworld',train=True,
                                       exploration_final_eps=0.1, batch_size=batch_size, batch_num=1,
                                       double_dqn=False, matrix_update=False, cross_entropy_loss=False, cross_entropy_weight=0, update_time=1)
            # most of these parameters are not used, to just instantiate the dqn_agent
            correct_count_list = []
            loss_list = []
            bellman_loss_list = []

            for i in range(self.update_steps):
                if i % self.target_update_interval == 0:
                    self.dqn_agent.target_net.load_state_dict(self.dqn_agent.net.node_dict())

                sample_data = self.offline_data.sample_batch(batch_size)
                s_t = []
                a_t = []
                r_t = []
                t_t = []
                s_t1 = []

                for s, a, reward, done, s_prime in sample_data:
                    s_t.append(self.env.obs2state(s))
                    a_t.append(a)
                    r_t.append(np.float32(reward))
                    t_t.append(np.float32(done))
                    s_t1.append(self.env.obs2state(s_prime))

                bellman_loss, other_q_s_a, other_q_s_a_max, bad_update_rate, q_max_diff, q_max_magnitude = self.dqn_agent.update.learn(0, batch_size, self.env.action_space, s_t, a_t, r_t, t_t, s_t1)

                if i % (self.update_steps // self.plot_num) == 0:
                    # print(states_tensor.shape)
                    with torch.no_grad():
                        q_values = self.dqn_agent.net(all_states_tensor)
                    q_values = q_values.detach().cpu().numpy()

                    # print(bellman_loss,q_values.shape)
                    num,wrong_action_list = count_correct(q_values)
                    loss = np.mean(np.abs(np.delete(q_values, range(1, 12), 0) - np.delete(optimal_value, range(1, 12), 0)))
                    correct_count_list.append(num)
                    loss_list.append(loss)
                    bellman_loss_list.append(bellman_loss)

            tmp_correct_count = {'steps': self.step_index, 'correct_count': correct_count_list,'batch_size':[batch_size]*self.plot_num}
            tmp_correct_count = pd.DataFrame(tmp_correct_count)
            tmp_loss = {'steps': self.step_index, 'loss': loss_list,'batch_size':[batch_size]*self.plot_num}
            tmp_loss = pd.DataFrame(tmp_loss)
            tmp_bellman_loss = {'steps': self.step_index, 'bellman_loss': bellman_loss_list,'batch_size':[batch_size]*self.plot_num}
            tmp_bellman_loss = pd.DataFrame(tmp_bellman_loss)

            correct_count_data.append(tmp_correct_count)
            loss_data.append(tmp_loss)
            bellman_loss_data.append(tmp_bellman_loss)

        correct_count_data = pd.concat(correct_count_data)
        loss_data = pd.concat(loss_data)
        bellman_loss_data = pd.concat(bellman_loss_data)

        self.plot(correct_count_data,'correct_count','dqn: correct action',hue='batch_size')
        self.plot(loss_data,'loss','dqn: loss',hue = 'batch_size')
        self.plot(bellman_loss_data, 'bellman_loss', 'dqn: bellman loss',hue = 'batch_size')

    def iql(self):
        from implicitQ.iql import iql_agent
        # sample a batch to update
        # batch_size_list = [4,16,64,256,1024]
        batch_size_list = [64]
        q1_correct_count_data = []
        q2_correct_count_data = []
        q1_loss_data = []
        q2_loss_data = []


        all_states = self.all_array_states(vector=True)
        all_states_tensor = torch.from_numpy(np.array(all_states)).to(self.device).float()

        for seed, batch_size in itertools.product(range(self.run_num),batch_size_list):
            print('seed:',seed,'batch_size:',batch_size)
            self.setup_seed(seed)

            self.data_collecting()
            print(self.offline_data.state_list)
            agent = iql_agent
            q1_correct_count_list = []
            q2_correct_count_list = []
            q1_loss_list = []
            q2_loss_list = []

            for i in range(self.update_steps):
                sample_data = self.offline_data.sample_batch(batch_size)
                s_t = []
                a_t = []
                r_t = []
                t_t = []
                s_t1 = []

                for s, a, reward, done, s_prime in sample_data:
                    s_t.append(self.env.obs2state(s))
                    a_t.append(a)
                    r_t.append(np.float32(reward))
                    t_t.append(np.float32(done))
                    s_t1.append(self.env.obs2state(s_prime))

                s_t = torch.from_numpy(np.array(s_t)).to(self.device).float()
                a_t = torch.LongTensor(a_t).view(-1, 1).to(self.device)
                r_t = torch.from_numpy(np.array(r_t)).to(self.device)
                t_t = torch.from_numpy(np.array(t_t)).to(self.device)
                s_t1 = torch.from_numpy(np.array(s_t1)).to(self.device).float()
                # print(s_t.shape)
                data = {'obs': s_t, 'act': a_t, 'reward': r_t, 'obs2': s_t1, 'done': t_t}
                agent.update(data)


                if i % (self.update_steps // self.plot_num) == 0:
                    with torch.no_grad():
                        q1, q2 = agent.ac.q1q2(all_states_tensor)
                        print('q1:',q1.shape)
                    q1 = q1.detach().cpu().numpy()
                    q2 = q2.detach().cpu().numpy()
                    q1_num, q1_wrong_action_list = count_correct(q1)
                    q2_num, q2_wrong_action_list = count_correct(q2)
                    q1_loss = np.mean(np.square(np.delete(q1, range(1, 12), 0) - np.delete(optimal_value, range(1, 12), 0)))
                    q2_loss = np.mean(np.square(np.delete(q2, range(1, 12), 0) - np.delete(optimal_value, range(1, 12), 0)))
                    q1_correct_count_list.append(q1_num)
                    q2_correct_count_list.append(q2_num)
                    q1_loss_list.append(q1_loss)
                    q2_loss_list.append(q2_loss)
                    print('---',i)
                    print('iql q1 correct num:', q1_num, q1_wrong_action_list, '|loss:', q1_loss)
                    print('iql q2 correct num:', q2_num, q2_wrong_action_list, '|loss:', q2_loss)

            tmp_correct_count = {'steps': self.step_index, 'correct_count': q1_correct_count_list,'batch_size':[batch_size]*self.plot_num}
            tmp_correct_count = pd.DataFrame(tmp_correct_count)
            tmp_loss = {'steps': self.step_index, 'loss': q1_loss_list,'batch_size':[batch_size]*self.plot_num}
            tmp_loss = pd.DataFrame(tmp_loss)
            q1_correct_count_data.append(tmp_correct_count)
            q1_loss_data.append(tmp_loss)

            tmp_correct_count = {'steps': self.step_index, 'correct_count': q2_correct_count_list,'batch_size':[batch_size]*self.plot_num}
            tmp_correct_count = pd.DataFrame(tmp_correct_count)
            tmp_loss = {'steps': self.step_index, 'loss': q2_loss_list,'batch_size':[batch_size]*self.plot_num}
            tmp_loss = pd.DataFrame(tmp_loss)
            q2_correct_count_data.append(tmp_correct_count)
            q2_loss_data.append(tmp_loss)

        q1_correct_count_data = pd.concat(q1_correct_count_data)
        q2_correct_count_data = pd.concat(q2_correct_count_data)
        q1_loss_data = pd.concat(q1_loss_data)
        q2_loss_data = pd.concat(q2_loss_data)

        self.plot(q1_correct_count_data,'correct_count','q1: correct action',hue='batch_size')
        self.plot(q2_correct_count_data,'correct_count','q2: correct action',hue='batch_size')
        self.plot(q1_loss_data,'loss','q1: loss',hue = 'batch_size')
        self.plot(q2_loss_data,'loss','q2: loss',hue = 'batch_size')

    def iql_v2(self):
        from discrete_iql.agent import IQL
        # sample a batch to update
        # batch_size_list = [4,16,64,256,1024]
        batch_size_list = [64]
        q1_correct_count_data = []
        q2_correct_count_data = []
        q1_loss_data = []
        q2_loss_data = []


        all_states = self.all_array_states(vector=True)
        all_states_tensor = torch.from_numpy(np.array(all_states)).to(self.device).float()

        for seed, batch_size in itertools.product(range(self.run_num),batch_size_list):
            print('seed:',seed,'batch_size:',batch_size)
            self.setup_seed(seed)

            # self.data_collecting()
            self.data_collecting_all_transition()
            print(len(self.offline_data.buffer),self.offline_data.state_list)
            agent = IQL(state_size=self.env.observation_space,
                        action_size=self.env.action_space,
                        device=self.device)
            q1_correct_count_list = []
            q2_correct_count_list = []
            q1_loss_list = []
            q2_loss_list = []

            for i in range(self.update_steps):
                sample_data = self.offline_data.sample_batch(batch_size)
                s_t = []
                a_t = []
                r_t = []
                t_t = []
                s_t1 = []

                for s, a, reward, done, s_prime in sample_data:
                    s_t.append(self.env.obs2state(s))
                    a_t.append(a)
                    r_t.append(np.float32(reward))
                    t_t.append(np.float32(done))
                    s_t1.append(self.env.obs2state(s_prime))

                s_t = torch.from_numpy(np.array(s_t)).to(self.device).float()
                a_t = torch.LongTensor(a_t).view(-1, 1).to(self.device)
                r_t = torch.from_numpy(np.array(r_t)).to(self.device)
                s_t1 = torch.from_numpy(np.array(s_t1)).to(self.device).float()
                t_t = torch.from_numpy(np.array(t_t)).to(self.device)
                # print(s_t.shape,a_t.shape,r_t.shape,s_t1.shape,t_t.shape)

                policy_loss, critic1_loss, critic2_loss, value_loss = agent.learn((s_t,a_t,r_t,s_t1,t_t))

                if i % (self.update_steps // self.plot_num) == 0:
                    with torch.no_grad():
                        q1 = agent.critic1(all_states_tensor)
                        q2 = agent.critic2(all_states_tensor)
                        print('q1:',q1.shape)
                    q1 = q1.detach().cpu().numpy()
                    q2 = q2.detach().cpu().numpy()
                    q1_num, q1_wrong_action_list = count_correct(q1)
                    q2_num, q2_wrong_action_list = count_correct(q2)
                    q1_loss = np.mean(np.square(np.delete(q1, range(1, 12), 0) - np.delete(optimal_value, range(1, 12), 0)))
                    q2_loss = np.mean(np.square(np.delete(q2, range(1, 12), 0) - np.delete(optimal_value, range(1, 12), 0)))
                    q1_correct_count_list.append(q1_num)
                    q2_correct_count_list.append(q2_num)
                    q1_loss_list.append(q1_loss)
                    q2_loss_list.append(q2_loss)
                    print('---',i)
                    print('iql q1 correct num:', q1_num, q1_wrong_action_list, '|loss:', q1_loss)
                    print('iql q2 correct num:', q2_num, q2_wrong_action_list, '|loss:', q2_loss)

            tmp_correct_count = {'steps': self.step_index, 'correct_count': q1_correct_count_list,'batch_size':[batch_size]*self.plot_num}
            tmp_correct_count = pd.DataFrame(tmp_correct_count)
            tmp_loss = {'steps': self.step_index, 'loss': q1_loss_list,'batch_size':[batch_size]*self.plot_num}
            tmp_loss = pd.DataFrame(tmp_loss)
            q1_correct_count_data.append(tmp_correct_count)
            q1_loss_data.append(tmp_loss)

            tmp_correct_count = {'steps': self.step_index, 'correct_count': q2_correct_count_list,'batch_size':[batch_size]*self.plot_num}
            tmp_correct_count = pd.DataFrame(tmp_correct_count)
            tmp_loss = {'steps': self.step_index, 'loss': q2_loss_list,'batch_size':[batch_size]*self.plot_num}
            tmp_loss = pd.DataFrame(tmp_loss)
            q2_correct_count_data.append(tmp_correct_count)
            q2_loss_data.append(tmp_loss)

        q1_correct_count_data = pd.concat(q1_correct_count_data)
        q2_correct_count_data = pd.concat(q2_correct_count_data)
        q1_loss_data = pd.concat(q1_loss_data)
        q2_loss_data = pd.concat(q2_loss_data)

        self.plot(q1_correct_count_data,'correct_count','q1: correct action',hue='batch_size')
        self.plot(q2_correct_count_data,'correct_count','q2: correct action',hue='batch_size')
        self.plot(q1_loss_data,'loss','q1: loss',hue = 'batch_size')
        self.plot(q2_loss_data,'loss','q2: loss',hue = 'batch_size')


    def all_array_states(self,vector=False):
        # inlcude cliff states and goal state.
        # return array state or vector state
        all_states = []
        if vector:
            for i in range(48):
                all_states.append(self.env.obs2state(i))
        else:
            for i in range(48):
                all_states.append(self.env.array_observation(self.env.obs2state(i)))
        # print(all_states)
        return all_states


offline_q = Offline_training()

offline_q.reset_update_steps(100000) # 50000
# offline_q.iql()
offline_q.iql_v2()

# offline_q.q_learning(batch_size=1, step_size=0.5)
# offline_q.supervised_learning()
# offline_q.sampled_supervised_learning()

# offline_q.reset_update_steps(100000)
# offline_q.dqn_all_state()
# offline_q.dqn()
