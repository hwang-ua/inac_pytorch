from cliffworld_wrappers import Baselines_DummyVecEnv
import numpy as np
from optimal_cliffworld import count_correct,optimal_value
from collections import deque
import os

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value_head
        final_p: float
            final output value_head
        from baselines
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value_head"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

class buffer():
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
        index = np.random.randint(min(self.buffer_size,len(self.buffer)),size = n)
        batch = [self.buffer[i] for i in index]

        return batch

class QLearningAgent():
    def __init__(self,final_step=int(1e5),step_size=0.5):
        """Setup for the agent called when the experiment first starts.
        
        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            step_size (float): The step-size,
            discount (float): The discount factor,
        }

        """
        # Store the parameters provided in agent_init_info.
        self.num_actions = 4
        self.num_states = 48
        self.exploration_decay = LinearSchedule(schedule_timesteps=final_step/2,final_p=0.01,initial_p=1.)

        self.step_size = step_size # 0.5
        self.final_step = final_step
        self.buffer_size = int(1e5)
        self.buffer = buffer(self.buffer_size) # (s_t,a_t,r_t,t_t,s_t1) s and a are scalar number
        self.learning_starts = final_step * 0.005
        self.discount = 0.99
        self.batch_size = 32

        # Create an array for action-value estimates and initialize it to zero.
        # self.q = np.random.random((self.num_states, self.num_actions)) # The array of action-value estimates.
        self.q = np.zeros((self.num_states, self.num_actions)) # The array of action-value estimates.

        self.prev_state = None
        self.prev_action = None

    def act(self,state,reward, done, info,train,current_step=0):
        # print(state)
        current_q = self.q[state, :]
        if train:
            epsilon = self.exploration_decay.value(current_step)
            if np.random.random() < epsilon:
                action = np.random.randint(self.num_actions)
            else:
                action = self.argmax(current_q)

            if self.prev_state != None:
                self.buffer.add_data([self.prev_state,self.prev_action,reward,done,state])

            if current_step > self.learning_starts:
                batch = self.buffer.sample_batch(self.batch_size)
                for transition in batch: # (s_t,a_t,r_t,t_t,s_t1)
                    target = transition[2] + (1 - transition[3]) * self.discount * np.max(self.q[transition[4], :])
                    self.q[transition[0], transition[1]] = self.q[transition[0], transition[1]] + self.step_size * (target - self.q[transition[0], transition[1]])

            # Perform an update (1 line)
            # if done != None:
            #     alpha = self.lr_decay.value(current_step)
            #     lr = self.step_size * alpha
            #     target = reward + (1-done) * self.discount * np.max(self.q[state, :])
            #     self.q[self.prev_state, self.prev_action] = self.q[self.prev_state, self.prev_action] + lr * (target - self.q[self.prev_state, self.prev_action])
        else:
            action = self.argmax(current_q)

        self.prev_state = state
        self.prev_action = action
        return action
        
    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []
        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return np.random.choice(ties)

if __name__ == '__main__':
    seed_list = [1,2,3]
    step_size = 0.5
    for seed in seed_list:
        log_path = 'log/' + str(seed) + '_' + str(step_size)

        correct_num_list = []
        wrong_action_list_list = []
        loss_list = []
        mean_score_list = []
        steps_list = []

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        env = Baselines_DummyVecEnv('cliffworld',1,array_obs=False,scalar_obs=True)
        env.envs[0].env.env.array_obs = False
        np.random.seed(seed)

        rewards, dones, info = [None],[None],[None]
        current_step = 0
        final_step = int(1e5)
        states = env.reset()
        agent = QLearningAgent(final_step,step_size)

        while current_step <= final_step:
            action = agent.act(states[0],rewards[0],dones[0],info,True,current_step)
            next_states, rewards, dones, info = env.step([action])
            states = next_states
            # if dones[0]:
            #     print(env.get_episode_rewmean())

            if current_step % (final_step//20) == 0:
                test_env = Baselines_DummyVecEnv('cliffworld',1,array_obs=False,scalar_obs=True)
                test_env.envs[0].env.env.array_obs = False
                test_rewards, test_dones, test_info = [None],[None],[None]
                test_current_step = 0
                test_final_step = 10000
                test_states = test_env.reset()

                while test_current_step < test_final_step:
                    test_action = agent.act(test_states[0],test_rewards[0],test_dones[0],test_info,False)
                    test_next_states, test_rewards, test_dones, test_info = test_env.step([test_action])
                    test_states = test_next_states
                    test_current_step += 1

                print('---',current_step)
                mean_score = test_env.get_episode_rewmean()
                num, wrong_action_list = count_correct(agent.q)
                loss = np.mean(np.square(np.delete(agent.q, range(1, 12), 0) - np.delete(optimal_value, range(1, 12), 0)))
                print('mean_score :', mean_score)
                print('mean len :',test_env.get_episode_lenmean())
                print('correct state :',num, wrong_action_list)
                print('loss :',loss)
                mean_score_list.append(mean_score)
                correct_num_list.append(num)
                wrong_action_list_list.append(wrong_action_list)
                loss_list.append(loss)
                steps_list.append(current_step)

            current_step += 1

        np.savez_compressed(log_path + '/statistics',
                            mean_score = mean_score_list,
                            steps = steps_list,
                            correct_num = correct_num_list,
                            wrong_action_list = wrong_action_list_list,
                            loss = loss_list)

    # print(agent.q)
    # print(count_correct(agent.q))
    # print(np.mean(np.abs(agent.q-optimal_value)))