import gym
import numpy as np

class ethOptimize(gym.Env):
    def __init__(self):
        self.action_space = None
        self.observation_space = None
        pass

    def step(self, action):
        return self.state, reward, done, {}

    def reset(self):
        return self.state 

