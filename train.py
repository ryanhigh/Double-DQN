import os
import gym
import numpy as np
import argparse
from DDQN import DDQN
from eth_optimize import EthOptimize
from utils import plot_learning_curve, create_directory, plot_validate_curve
os.environ['KMP_DUPLICATE_LIB_OK']='True'
envpath = '/home/xgq/conda/envs/pytorch1.6/lib/python3.6/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=1000)
parser.add_argument('--ckpt_dir', type=str, default='./Val_n_base/')
parser.add_argument('--reward_path', type=str, default='./Val_n_base/ddqn_avg_reward.png')
parser.add_argument('--epsilon_path', type=str, default='./Val_n_base/ddqn_epsilon.png')

args = parser.parse_args()


def ddqn_train(max_episodes):
    # env = gym.make('LunarLander-v2')
    env = EthOptimize()
    agent = DDQN(alpha=0.0003, state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
                 fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.9, tau=0.005, epsilon=1.0,
                 eps_end=0.05, eps_dec=5e-4, max_size=1000000, batch_size=256)
    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    total_rewards, avg_rewards, eps_history = [], [], []

    for episode in range(max_episodes):
        total_reward = 0
        done = False
        observation = env.reset()
        observation = observation[0]
        while not done:
            action = agent.choose_action(observation, isTrain=True)
            observation_, reward, done, info, perform_ = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            total_reward += reward
            observation = observation_

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        eps_history.append(agent.epsilon)
        print('EP:{} reward:{} avg_reward:{} epsilon:{}'.
              format(episode + 1, total_reward, avg_reward, agent.epsilon))

        if (episode + 1) % 50 == 0:
            agent.save_models(episode + 1)

    episodes = [i for i in range(args.max_episodes)]
    return episodes, avg_rewards
    # plot_learning_curve(episodes, avg_rewards, 'Reward', 'reward', args.reward_path)
    # plot_learning_curve(episodes, eps_history, 'Epsilon', 'epsilon', args.epsilon_path)


if __name__ == '__main__':
    ddqn_train()
    # env = Car2DEnv()
    # print("action_dim", env.action_space.n)
    # print("state_dim", env.observation_space.shape[0])
