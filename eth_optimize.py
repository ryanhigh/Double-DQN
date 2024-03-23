import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EthOptimize(gym.Env):
    def __init__(self):
        self.upper_bsize = 300000
        self.lower_bsize = 150000
        self.upper_btime = 25
        self.lower_btime = 5
        self.action_space = spaces.Discrete(5) # 0, 1, 2, 3, 4
        self.observation_space = spaces.Box(
            low=np.array([self.lower_bsize, self.lower_btime], dtype=np.float32),
            high=np.array([self.upper_bsize, self.upper_btime], dtype=np.float32),
            dtype=np.float32
        )
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

    def reset(self, seed=None):
        # state初始化，回到一个初始状态，为下一个周期准备
        self.state = np.array([151500, 7])
        return self.state, {} 
    
    def render(self, mode='human'):
        # Implement rendering logic
        pass


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env 
    # 如果你安装了pytorch，则使用上面的，如果你安装了tensorflow，则使用from stable_baselines.common.env_checker import check_env
    env = EthOptimize()
    check_env(env)
