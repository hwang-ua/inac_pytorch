#!/usr/bin/env python

import numpy as np
from copy import deepcopy

class Environment():
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    def __init__(self, array_obs=False,scalar_obs=False):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """
        self.rows = 4
        self.cols = 12
        self.start = [0,0]
        self.goal = [0,11]
        self.current_state = None
        # array observation, used for conv net
        # three channels, first one shows the goal's position, second one shows cliff's position, the third one shows the agent's position
        self.array_obs = array_obs
        self.scalar_obs = scalar_obs
        if array_obs:
            self.observation_space = [self.rows,self.cols,3]
        else:
            self.observation_space = 2
        self.action_space = 4

        self.goal_array = np.zeros((self.rows,self.cols)).astype('int8')
        self.goal_array[0,self.cols-1] = 1
        self.goal_array = np.flipud(self.goal_array)

        self.cliff_array = np.zeros((self.rows,self.cols)).astype('int8')
        for i in range(1,self.cols-1):
            self.cliff_array[0,i] = 1
        self.cliff_array = np.flipud(self.cliff_array)

        self.state_list = [0]
        self.state_list.extend(list(range(12,48)))
        self.state_dim = self.observation_space
        self.action_dim = self.action_space

    @property
    def num_rows(self):
        return self.rows

    @property
    def num_cols(self):
        return self.cols

    def get_state_space(self):
        obs = self.get_obstacles_map()
        empty = np.argwhere(obs == 0)
        obs = np.argwhere(obs != 0)
        return empty, obs

    def get_goal_coord(self):
        return [self.goal_x, self.goal_y]

    def reset(self, random_choice=True):
        """The first method called when the episode starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        if not random_choice:
            self.current_state = self.start  # An empty NumPy array
        else:
            self.set_state(np.random.choice(self.state_list))
        # print(np.random.choice(self.state_list))
        # print(self.current_state)

        if self.array_obs:
            self.reward_obs_term = (self.array_observation(self.current_state),0.0, False,{})
        else:
            if self.scalar_obs:
                self.reward_obs_term = (self.observation(self.current_state),0.0, False,{})
            else:
                self.reward_obs_term = (self.current_state,0.0, False,{})
        return self.reward_obs_term[0]

    def step(self, action):
        action = action[0]
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        new_state = deepcopy(self.current_state)
        # print(new_state)

        if action == 0: #right
            new_state[1] = min(new_state[1]+1, self.cols-1)
        elif action == 1: #down
            new_state[0] = max(new_state[0]-1, 0)
        elif action == 2: #left
            new_state[1] = max(new_state[1]-1, 0)
        elif action == 3: #up
            new_state[0] = min(new_state[0]+1, self.rows-1)
        else:
            raise Exception("Invalid action.")
        self.current_state = new_state
        # if action == 3 and new_state%12 == 4:
        #     reward
        # reward = -1
        reward = 0
        is_terminal = False
        if self.current_state[0] == 0 and self.current_state[1] > 0:
            if self.current_state[1] < self.cols - 1:
                # reward = -100.0
                # reward = -10.0
                reward = -1.0
                # self.current_state = deepcopy(self.start)
                is_terminal = True
            else:
                is_terminal = True
                reward = 1.
        if self.array_obs:
            self.reward_obs_term = self.array_observation(self.current_state), reward, is_terminal, {}
        else:
            if self.scalar_obs:
                self.reward_obs_term = self.observation(self.current_state),reward, is_terminal,{}
            else:
                self.reward_obs_term = self.current_state,reward, is_terminal,{}

        return self.reward_obs_term

    def observation(self, state):
        # 2d to 1d
        return state[0] * self.cols + state[1]

    def obs2state(self,state):
        # 1d to 2d
        return [state // self.cols, state % self.cols]
    
    def set_state(self,state):
        self.current_state = [state//self.cols,state%self.cols]

    def array_observation(self,state):
        # three channels, first one shows the goal's position, second one shows cliff's position, the third one shows the agent's position

        self.agent_array = np.zeros((self.rows, self.cols)).astype('int8')
        self.agent_array[state[0], state[1]] = 1
        self.agent_array = np.flipud(self.agent_array)

        obs = np.concatenate((self.goal_array[:, :, np.newaxis], self.cliff_array[:, :, np.newaxis], self.agent_array[:, :, np.newaxis]), axis=-1)

        return obs

    def array2state(self,obs):
        x,y = np.where(np.flipud(obs[:,:,-1]==1))
        return [x[0],y[0]]


if __name__ == '__main__':
    env = Environment()
    s = env.reset()
    obs = env.array_observation([2,0])
    print(len(obs))
    print(obs[:,:,-1])
    x,y = np.where(np.flipud(obs[:,:,-1]==1))
    print(x,y)
    # print(obs.transpose(2, 0, 1)[-1])
