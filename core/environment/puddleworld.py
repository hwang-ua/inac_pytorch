import numpy as np

class Puddle():
    def __init__(self, headX, headY, tailX, tailY, radius, length, axis):
        self.headX = headX
        self.headY = headY
        self.tailX = tailX
        self.tailY = tailY
        self.radius = radius
        self.length = length
        self.axis = axis

    def getDistance(self, xCoor, yCoor):

        if self.axis == 0:
            u = (xCoor - self.tailX)/self.length
        else:
            u = (yCoor - self.tailY)/self.length

        dist = 0.0

        if u < 0.0 or u > 1.0:
            if u < 0.0:
                dist = np.sqrt(np.power((self.tailX - xCoor),2) + np.power((self.tailY - yCoor),2))
            else:
                dist = np.sqrt(np.power((self.headX - xCoor),2) + np.power((self.headY - yCoor),2))
        else:
            x = self.tailX + u * (self.headX - self.tailX)
            y = self.tailY + u * (self.headY - self.tailY)

            dist = np.sqrt(np.power((x - xCoor),2) + np.power((y - yCoor),2))

        if dist < self.radius:
            return (self.radius - dist)
        else:
            return 0

class PuddleWorld():
    def __init__(self, seed=1e-5, env_randomstart=True, normalized=False):
        self.env_rng = np.random.RandomState(seed)
        self.random_start = env_randomstart

        self.state_dim = (2,)
        self.action_dim = 4
        self.state = None
        self.puddle1 = Puddle(0.45,0.75,0.1,0.75,0.1,0.35,0)
        self.puddle2 = Puddle(0.45,0.8,0.45,0.4,0.1,0.4,1)

        self.pworld_min_x = 0.0
        self.pworld_max_x = 1.0
        self.pworld_min_y = 0.0
        self.pworld_max_y = 1.0
        self.pworld_mid_x = (self.pworld_max_x - self.pworld_min_x)/2.0
        self.pworld_mid_y = (self.pworld_max_y - self.pworld_min_y)/2.0

        self.goalDimension = 0.1
        self.defDisplacement = 0.05

        self.sigma = 0.01

        self.goalXCoor = self.pworld_max_x - self.goalDimension #1000#
        self.goalYCoor = self.pworld_max_y - self.goalDimension #1000#
        self.normalized = normalized

        self.wasReset = False

    def internal_reset(self):
        if not self.wasReset:
            self.state = self.env_rng.uniform(low=0.0, high=0.1, size=(2,))

            reset = False
            if self.random_start:
                while not reset:
                    self.state[0] = self.env_rng.uniform(low=0, high=1)
                    self.state[1] = self.env_rng.uniform(low=0, high=1)
                    if not self._terminal():
                        reset = True
            else:
                raise NotImplementedError
            print("\nStart state:", self.state)
            self.wasReset = True
        return self._get_ob()

    def reset(self):
        self.wasReset = False
        return self.internal_reset()

    def _get_ob(self):
        if self.normalized:
            s = self.state
            s0 = (s[0] - self.pworld_mid_x) * 2.0
            s1 = (s[1] - self.pworld_mid_y) * 2.0
            return np.array([s0, s1])
        else:
            s = self.state
            return np.array([s[0], s[1]])

    def _terminal(self):
        s = self.state
        return bool((s[0] >= self.goalXCoor) and (s[1] >= self.goalYCoor))

    def _reward(self,x,y,terminal):
        if terminal:
            return -1
        reward = -1
        dist = self.puddle1.getDistance(x, y)
        reward += (-400. * dist)
        dist = self.puddle2.getDistance(x, y)
        reward += (-400. * dist)
        reward = reward
        return reward

    def step(self,a):
        s = self.state

        xpos = s[0]
        ypos = s[1]

        n = self.env_rng.normal(scale=self.sigma)

        if a == 0: #left
            xpos -= self.defDisplacement+n
        elif a == 1: #right
            xpos += self.defDisplacement+n
        elif a == 2: #down
            ypos -= self.defDisplacement+n
        else: #up
            ypos += self.defDisplacement+n

        if xpos > self.pworld_max_x:
            xpos = self.pworld_max_x
        elif xpos < self.pworld_min_x:
            xpos = self.pworld_min_x

        if ypos > self.pworld_max_y:
            ypos = self.pworld_max_y
        elif ypos < self.pworld_min_y:
            ypos = self.pworld_min_y

        s[0] = xpos
        s[1] = ypos
        self.state = s

        terminal = self._terminal()
        reward = self._reward(xpos,ypos,terminal)

        return (self._get_ob(), reward, terminal, {})
