import gym
from gym import spaces
import numpy as np

class EthOptimize(gym.Env):
    def __init__(self):
        self.upper_bsize = 300000
        self.lower_bsize = 150000
        self.upper_btime = 25
        self.lower_btime = 5
        self.action_space = spaces.Discrete(5) # 0, 1, 2, 3, 4
        self.observation_space = spaces.Box(np.array([self.lower_bsize, self.upper_bsize]), np.array([self.lower_btime, self.upper_btime]))
        self.state = None
        self.counts = 0

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        x, y = self.state
        if action == 0:
            x=x
            y=y
        elif action == 1:
            x = x + 1500
            y = y
        elif action == 2:
            x = x - 1500
            y = y
        elif action == 3:
            x = x
            y = y + 1
        elif action == 4:
            x = x
            y = y - 1
        self.state = np.array([x, y])
        self.counts += 1

        done = None # justify border 
        done = bool(done)

        reward = None
        '''
        if not done:
            reward -= 0.1
        else:
            if ...
        '''
        
        return self.state, reward, done, {}

    def reset(self):
        # state初始化，回到一个初始状态，为下一个周期准备
        self.state = np.array([151500, 7])
        return self.state 


if __name__ == '__main__':
    env = EthOptimize()
    env.reset()
    env.step(2)
    print(env.state)
    env.step(4)
    print(env.state)
