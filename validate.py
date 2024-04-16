"""
validate optimization of reward function
in this file, we load the last checkpoint we save, and 
"""
import os
import gym
import numpy as np
import argparse
import pandas as pd
from DDQN import DDQN
from eth_optimize import EthOptimize, getPerformance
from utils import plot_learning_curve, create_directory, plot_validate_curve

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_length', type=int, default=1000)
parser.add_argument('--max_episodes', type=int, default=20000)
parser.add_argument('--ckpt_dir', type=str, default='./test2/DDQN/')
parser.add_argument('--vali_dir', type=str, default='./validate/')
parser.add_argument('--tpsctpt', type=str, default='./validate/EP20000tps_ct.png')
parser.add_argument('--latctpt', type=str, default='./EP20000lat_ct.png')
parser.add_argument('--reward_path', type=str, default='./EP20000avg_reward.png')
parser.add_argument('--dataset_path', type=str, default='./result2.csv')

args = parser.parse_args()

def main():
    env = EthOptimize()
    agent = DDQN(alpha=0.0003, state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
                 fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.9, tau=0.005, epsilon=1.0,
                 eps_end=0.05, eps_dec=5e-4, max_size=1000000, batch_size=256)
    agent.load_models(20000) # load the number 20k model
    total_rewards, avg_rewards, tps_l, tps_l2 = [], [], [], []
    delay, delay_ = [], []
    data = pd.read_csv(args.dataset_path)

    for i in range(args.dataset_length):
        total_reward = 0
        blocksize = data['gaslimit'].iloc[i].item()
        period = data['period(s)'].iloc[i].item()
        tps = data['tps(tx/s)'].iloc[i].item()
        latency = data['latency(ms)'].iloc[i].item()

        tps_l.append(tps)
        delay.append(latency)

        observation = env.SaveState(blocksize, period)
        action = agent.choose_action(observation, isTrain=False)
        observation_, reward, done, info, border = env.step(action)
        blocksize_, period_ = observation_
        if border:
            perform = np.array([tps, latency])
        else:
            perform = getPerformance(blocksize_, period_)
        tps_, latency_ = perform

        tps_l2.append(tps_)
        delay_.append(latency_)

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
    
    episodes = [i for i in range(args.dataset_length)]
    plot_validate_curve(episodes, tps_l, tps_l2, 'TPS', 'tps(tx/s)', args.tpsctpt)
    plot_validate_curve(episodes, delay, delay_, 'Latency', 'latency', args.latctpt)
    plot_learning_curve(episodes, avg_rewards, 'Reward', 'reward', args.reward_path)

if __name__ == "__main__":
    create_directory('./validate/', sub_dirs=[])
    main()




    
