"""
4rooms gridworld environment.
"""

import numpy as np
import random

class GridWorld:
  _num_rows = -1
  _num_cols = -1
  _num_states = -1
  _reward_function = None
  _num_actions = 4

  _start_x = 0
  _start_y = 0
  _goal_x = 0
  _goal_y = 0

  _r = None
  _P = None

  def __init__(self,
              random_start=False,
              use_negative_rewards=False,
              episode_len=50) -> None:
    """
    Initialize by reading from a gridworld definitation string.
    """
    self._parse_string('core/environment/env/4rooms.mdp')
    self._curr_x = self._start_x
    self._curr_y = self._start_y
    self._num_states = self._num_rows * self._num_cols
    self._use_negative_rewards = use_negative_rewards
    self._episode_len = episode_len
    self._random_start = random_start
    if self._use_negative_rewards:
        self._rewards = (-1., 0.)
    else:
        self._rewards = (0., 1.)
    self._build()

    self.state_dim = (2,)
    self.action_dim = 4

  @property
  def episode_len(self):
    return self._episode_len

  @property
  def actions(self):
    self.directions = ["A", ">", "V", "<"]
    # return ['up', 'right', 'down', 'left', 'stay']
    return ['up', 'right', 'down', 'left']
  
  
  @property
  def P(self):
    return self._P

  @property
  def r(self):
    return self._r

  @property
  def num_states(self):
    return self._num_states
    
  @property
  def num_actions(self):
    return self._num_actions

  @property
  def matrix_mdp(self):
    return self._matrix_mdp

  @property
  def num_rows(self):
    return self._num_rows
    
  @property
  def num_cols(self):
    return self._num_cols

  def get_curr_state(self):
    return self.pos_to_state(self._curr_x, self._curr_y)

  def random_action(self):
    """
    Randomly sample an action.
    """
    return random.randrange(self._num_actions)

  def _parse_string(self, path):
    """
    Parsing the definition string.
    """
    # read string
    file_name = open(path, 'r')
    gw_str = ''
    for line in file_name:
        gw_str += line

    # read file
    data = gw_str.split('\n')
    self._num_rows = int(data[0].split(',')[0])
    self._num_cols = int(data[0].split(',')[1])
    self._matrix_mdp = np.zeros((self._num_rows, self._num_cols))
    mdp_data = data[1:]

    for i in range(len(mdp_data)):
      for j in range(len(mdp_data[1])):
        if mdp_data[i][j] == 'X':
          self._matrix_mdp[i][j] = -1 # wall
        elif mdp_data[i][j] == '.':
          self._matrix_mdp[i][j] = 0 # ground
        elif mdp_data[i][j] == 'S':
          self._matrix_mdp[i][j] = 0 # start
          self._start_x = i
          self._start_y = j
        elif mdp_data[i][j] == 'G':
          self._matrix_mdp[i][j] = 0 # goal
          self._goal_x = i
          self._goal_y = j
          
  def get_state_space(self):
    obs = np.where(self._matrix_mdp == -1, 1, 0)
    empty = np.argwhere(obs == 0)
    obs = np.argwhere(obs != 0)
    return empty, obs

  def get_goal_coord(self):
      return [self._goal_x, self._goal_y]

  def state_to_pos(self, idx):
    """
    Compute pos of the given state.
    """
    y = int(idx % self._num_cols)
    x = int(idx / self._num_cols)
    return x, y

  def pos_to_state(self, x, y):
    """
    Return the index of a state given a coordinate
    """
    idx = x * self._num_cols + y
    return idx
    
  def _get_next_state(self, x, y, a):
    a = a[0] if type(a) == list else a
    """
    Compute the next state by taking an action at (x,y).

    Note that the boarder of the mdp must be walls.
    """
    assert self._matrix_mdp[x][y] != -1
    next_x = x
    next_y = y
    action = self.actions[a]

    if action == 'up' and x > 0:
      next_x = x - 1
      next_y = y
    elif action == 'right' and y < self._num_cols - 1:
      next_x = x
      next_y = y + 1
    elif action == 'down' and x < self._num_rows - 1:
      next_x = x + 1
      next_y = y
    elif action == 'left' and y > 0:
      next_x = x
      next_y = y - 1

    if self._matrix_mdp[next_x][next_y] != -1:
      return next_x, next_y
    else:
      return x, y

  def _get_next_reward(self, next_x, next_y):
    """
    Get reward by taking an action.
    """
    if next_x == self._goal_x and next_y == self._goal_y:
      return self._rewards[1]
    else:
      return self._rewards[0]
      
  def step(self, action):
    """
    One environment step.
    """
    next_x, next_y = self._get_next_state(
                self._curr_x,
                self._curr_y,
                action
            )
    reward = self._get_next_reward(next_x, next_y)
    self._curr_x = next_x
    self._curr_y = next_y
    obs = np.array((self._curr_x, self._curr_y))
    info = {'state_idx': self.pos_to_state(self._curr_x, self._curr_y)}
    return self._normalize_pos(obs), reward, False, info

  def reset(self):
    """
    Reset the agent to the start position.
    """
    if self._random_start:
      pos = self._random_empty_grids(1)[0]
      self._curr_x = pos[0]
      self._curr_y = pos[1]
    else:
      self._curr_x = self._start_x
      self._curr_y = self._start_y
    obs = np.array((self._curr_x, self._curr_y))
    return self._normalize_pos(obs)

  def _normalize_pos(self, pos):
    return pos
    # x = pos[0] / self._num_rows - 0.5
    # y = pos[1] / self._num_cols - 0.5
    # return np.array([x, y])

  def _random_empty_grids(self, k):
    """
    Return k random empty positions.
    """
    ground = np.argwhere(self._matrix_mdp==0)
    selected = np.random.choice(
            np.arange(ground.shape[0]),
            size=k,
            replace=False
            )
    return ground[selected]

  def _build(self):
    """
    Build reward and transition matrix.
    """
    def one_hot(i, n):
        vec = np.zeros(n)
        vec[i] = 1
        return vec

    self._r = np.zeros((self._num_states, self._num_actions))
    self._P = np.zeros((self._num_states, self._num_actions, self._num_states))

    for x in range(self._num_rows):
      for y in range(self._num_cols):
        if self._matrix_mdp[x][y] == -1:
          continue
        s_idx = self.pos_to_state(x, y)
        for a in range(self._num_actions):
          nx, ny = self._get_next_state(x, y, a)
          ns_idx = self.pos_to_state(nx, ny)
          self._P[s_idx][a] = one_hot(ns_idx, self._num_states)
          self._r[s_idx][a] = self._get_next_reward(nx, ny)

  # def model_predict(self, x, y, action):
  #   """
  #   Given a position x,y and action, output a transition
  #   """
  #   next_x, next_y = self._get_next_state(x, y, action)
  #   reward = self._get_next_reward(next_x, next_y)
  #   obs = np.array((next_x, next_y))
  #   return self._normalize_pos(obs), reward, False