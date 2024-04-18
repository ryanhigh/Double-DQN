import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random
import math
from clean_data import Final_Dataframe

df = Final_Dataframe()


def getPerformance(x, y):
    row_indices = df.index[(df['gaslimit'] == x) & (df['period(s)'] == y)].tolist()
    if not bool(row_indices):
        print(row_indices, x, y)
    tps = df['tps(tx/s)'].iloc[row_indices].item()
    latency = df['latency(ms)'].iloc[row_indices].item()
    perform = np.array([tps, latency])
    return perform


def getReward(perform_, perform, perform0):
    # get delta value
    gamma = 1
    T_0, L_0 = perform0
    T_t, L_t = perform_
    T_t_1, L_t_1 = perform

    delta_T_orig = (T_t - T_0) / T_0
    delta_T_step = (T_t - T_t_1) / T_t_1
    delta_L_orig = (- L_t + L_0) / L_0
    delta_L_step = (- L_t + L_t_1) / L_t_1

    # compute reward value rT and rL
    if delta_T_step > 0:
        rT = math.exp(gamma  * delta_T_orig * delta_T_step)
    else:
        rT = - math.exp(- gamma * delta_T_orig * delta_T_step)
    
    if delta_L_step > 0:
        rL = - math.exp(gamma * delta_L_orig * delta_L_step)
    else:
        rL = math.exp(- gamma * delta_L_orig * delta_L_step)
    reward = 0.7 * rT + 0.3 * rL
    return reward


def getReward2(perform_):
    tps, delay = perform_
    reward = tps / delay
    return reward


class EthOptimize(gym.Env):
    def __init__(self):
        self.upper_bsize = 30000000
        self.lower_bsize = 15000000
        self.upper_btime = 15
        self.lower_btime = 5
        self.action_space = spaces.Discrete(5)# 0, 1, 2, 3, 4
        self.observation_space = spaces.Box(
            low=np.array([self.lower_bsize, self.lower_btime], dtype=np.float32),
            high=np.array([self.upper_bsize, self.upper_btime], dtype=np.float32),
            dtype=np.float32
        )
        self.state = None
        self.counts = 0
        self.performance0 = None

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        x, y = self.state
        # 记录当前性能
        perform = getPerformance(x, y)

        if action == 0:
            x=x
            y=y
        elif action == 1:
            x = x + 150000
            y = y
        elif action == 2:
            x = x - 150000
            y = y
        elif action == 3:
            x = x
            y = y + 1
        elif action == 4:
            x = x
            y = y - 1
        self.state = np.array([x, y])
        self.counts += 1

        max_tps = df['tps(tx/s)'].max()
        border = (x <= self.lower_bsize or x >= self.upper_bsize) or (y < self.lower_btime or y > self.upper_btime)  # justify border
        if border:
            done = True
            perform_ = perform
        else:
            perform_ = getPerformance(x, y)
            tps_current = perform_[0]
            if tps_current == max_tps:
                done = True
            else:
                done = False

        if not done:
            reward = 0
        else:
            if border:
                reward  = -0.5
            else:    
                reward = 10 * getReward2(perform_)
        # reward = getReward(perform_, perform, self.performance0)
        return self.state, reward, done, {}, perform_

    def reset(self, seed=None):
        # state初始化，回到一个初始状态，为下一个周期准备
        iloc_num = random.randint(10, 1000)
        # print(iloc_num)
        blocksize = df['gaslimit'].iloc[iloc_num].item()
        period = df['period(s)'].iloc[iloc_num].item()
        # print(type(blocksize), type(period))
        tps = df['tps(tx/s)'].iloc[iloc_num].item()
        latency = df['latency(ms)'].iloc[iloc_num].item()
        self.performance0 = np.array([tps, latency])
        self.state = np.array([blocksize, period])
        self.counts = 0
        return self.state, {}
    
    def SaveState(self, blocksize, internal):
        self.state = np.array([blocksize, internal])
        self.performance0 = getPerformance(blocksize, internal)
        return self.state


# if __name__ == "__main__":
#     env = EthOptimize()
#     state = env.reset()
#     state = state[0]
#     x, y  = state
#     tps, latency = getPerformance(x, y)
#     print(tps/latency)
