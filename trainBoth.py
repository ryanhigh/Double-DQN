import os
import gym
import numpy as np
import argparse
from PPO import PPO
from eth_optimize import EthOptimize
from trainPPO import ppo_train
from train import ddqn_train
from utils import plot_learning_curve, plot_validate_curve, create_directory

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=10000)
parser.add_argument('--ckpt_dir', type=str, default='./validate/')
parser.add_argument('--reward_path', type=str, default='./validate/avg_reward.png')

args = parser.parse_args()

def main():
    #################  train environment name: Eth_Optimize  #################
    create_directory(args.ckpt_dir, sub_dirs=[])
    # ppo train
    max_episodes, ppo_rewards = ppo_train()
    # ddqn train
    i_episodes, ddqn_rewards = ddqn_train(max_episodes)
    episodes = [i for i in range(max_episodes)]
    plot_validate_curve(episodes, ppo_rewards, ddqn_rewards, 'Reward', 'rewards', args.reward_path)


if __name__ == "__main__":
    main()
    
