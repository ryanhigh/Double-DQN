import os
import gym
import numpy as np
import argparse
from PPO import PPO
from eth_optimize import EthOptimize, getPerformance, getReward, getReward2
from trainPPO import ppo_train
from train import ddqn_train
from utils import plot_learning_curve, plot_validate_curve, create_directory

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=5000)
parser.add_argument('--ckpt_dir', type=str, default='./Val_n_base/')
parser.add_argument('--reward_path', type=str, default='./Val_n_base/avg_reward_r2v2_test1.png')

args = parser.parse_args()


def ComputeBaseline(ind):
    env = EthOptimize()
    observation = env.reset()
    observation = observation[0]
    blk, internal = observation
    perform0 = getPerformance(blk, internal)

    # which reward func is used
    # ind = 1:  complex version
    # ind = 2:  tps/delay easy version
    if ind == 1:
        baseline = getReward(perform0, perform0, perform0)
    elif ind == 2:
        baseline = getReward2(perform0)
    
    return baseline
    

def main():
    #################  train environment name: Eth_Optimize  #################
    create_directory(args.ckpt_dir, sub_dirs=[])
    # ppo train
    max_episodes, ppo_rewards = ppo_train()
    # ddqn train
    i_episodes, ddqn_rewards = ddqn_train(max_episodes)
    episodes = [i for i in range(max_episodes)]

    # w/o DRL
    baseline = ComputeBaseline(2)
    base_return = [baseline for _ in range(max_episodes)]
    plot_validate_curve(episodes, base_return, ppo_rewards, ddqn_rewards, 'Reward', 'rewards', args.reward_path)
    


if __name__ == "__main__":
    main()
    
