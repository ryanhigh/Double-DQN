import os
import gym
import numpy as np
import argparse
from DDQN import DDQN
from PPO import PPO
from eth_optimize import EthOptimize, getPerformance
import matplotlib.pyplot as plt
from utils import plot_learning_curve, create_directory, plot_validate_performance_curve
os.environ['KMP_DUPLICATE_LIB_OK']='True'
envpath = '/home/xgq/conda/envs/pytorch1.6/lib/python3.6/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=1000)
parser.add_argument('--ckpt_dir', type=str, default='./Val_n_base/')
parser.add_argument('--tps_path', type=str, default='./Val_n_base/ppo_10group_tps.png')
parser.add_argument('--delay_path', type=str, default='./Val_n_base/ppo_10group_delay.png')

args = parser.parse_args()
env = EthOptimize()
agent = DDQN(alpha=0.0003, state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
            fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.9, tau=0.005, epsilon=1.0,
            eps_end=0.05, eps_dec=5e-4, max_size=1000000, batch_size=256)
agent.load_models(9650)
agent_ppo = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2)
agent_ppo.load(args.ckpt_dir + 'PPO/ppo_9782.pth')

def contrast():
    ddqn_tps_l, ddqn_delay_l = [], []
    ppo_tps_l, ppo_delay_l = [], []


    done = False
    ppo_done = False
    observation = env.reset()
    observation = observation[0]
    x0, y0 = observation
    perform0 = getPerformance(x0, y0)
    tps0, delay0 = perform0
    ddqn_tps_l.append(tps0)
    ppo_tps_l.append(tps0)
    ddqn_delay_l.append(delay0)
    ppo_delay_l.append(delay0)
    timestep = 0
    ddqn_observation = observation
    ppo_observation = observation
    # while (not done) and (not ppo_done):
    #     # DDQN model run
    #     ddqn_action = agent.choose_action(ddqn_observation, isTrain=False)
    #     ddqn_observation_, reward, done, info, perform_ = env.step(ddqn_action)
    #     x, y = ddqn_observation_
    #     perform_i = getPerformance(x, y)
    #     tps_i, delay_i = perform_i
    #     ddqn_tps_l.append(tps_i)
    #     ddqn_delay_l.append(delay_i)
    #     ddqn_observation = ddqn_observation_

    #     # PPO model run
    #     ppo_action = agent_ppo.choose_action(ppo_observation)
    #     ppo_observation_, ppo_reward, ppo_done, info, perform_ = env.step(ppo_action)
    #     ppo_x, ppo_y = ppo_observation_
    #     ppo_perform_i = getPerformance(ppo_x, ppo_y)
    #     ppo_tps_i, ppo_delay_i = ppo_perform_i
    #     ppo_tps_l.append(ppo_tps_i)
    #     ppo_delay_l.append(ppo_delay_i)
    #     ppo_observation = ppo_observation_

    #     timestep += 1

    while not done:
        # DDQN run
        ddqn_action = agent.choose_action(ddqn_observation, isTrain=False)
        ddqn_observation_, reward, done, info, perform_ = env.step(ddqn_action)
        x, y = ddqn_observation_
        perform_i = getPerformance(x, y)
        tps_i, delay_i = perform_i
        ddqn_tps_l.append(tps_i)
        ddqn_delay_l.append(delay_i)
        ddqn_observation = ddqn_observation_
        timestep += 1

    for i in range(timestep):
        if ppo_done:
            ppo_observation_ = ppo_observation
        else:
            ppo_action = agent_ppo.choose_action(ppo_observation)
            ppo_observation_, ppo_reward, ppo_done, info, perform_ = env.step(ppo_action)
        
        ppo_x, ppo_y = ppo_observation_
        ppo_perform_i = getPerformance(ppo_x, ppo_y)
        ppo_tps_i, ppo_delay_i = ppo_perform_i
        ppo_tps_l.append(ppo_tps_i)
        ppo_delay_l.append(ppo_delay_i)
        ppo_observation = ppo_observation_

    time = [i for i in range(timestep+1)]
    plot_validate_performance_curve(time, ddqn_tps_l, ppo_tps_l, 'TPS', 'tps', args.tps_path)
    plot_validate_performance_curve(time, ddqn_delay_l, ppo_delay_l, 'Latency', 'latency', args.delay_path)


def group10():
    time = [i for i in range(21)]
    color = ['r', 'k', 'g', 'skyblue', 'b', 'y', 'c', 'teal', 'm', 'pink']
    plt.figure()
    avg_tps, avg_delay = [], []
    for t in range(10):
        tps_l, delay_l = [], []

        done = False
        observation = env.reset()
        observation = observation[0]
        x0, y0 = observation
        perform0 = getPerformance(x0, y0)
        tps0, delay0 = perform0
        tps_l.append(tps0)
        delay_l.append(delay0)
        for i in range(20):
            if done:
                observation_ = observation
                ppo_x, ppo_y = observation_
                ppo_perform_i = getPerformance(ppo_x, ppo_y)
                ppo_tps_i, ppo_delay_i = ppo_perform_i
                tps_l.append(ppo_tps_i)
                delay_l.append(ppo_delay_i)
                observation = observation_
            else:
                action = agent_ppo.choose_action(observation)
                observation_, ppo_reward, done, info, perform_ = env.step(action)
                if done:
                    observation_ = observation
                               
                ppo_x, ppo_y = observation_
                ppo_perform_i = getPerformance(ppo_x, ppo_y)
                ppo_tps_i, ppo_delay_i = ppo_perform_i
                tps_l.append(ppo_tps_i)
                delay_l.append(ppo_delay_i)
                observation = observation_
            print("finish_{}".format(i))
        
        plt.plot(time, delay_l, linestyle='-', linewidth = 2, color=color[t], label='ppo_group_{}'.format(t))
        Optim_tps = (tps_l[20] - tps0) / tps0
        Optim_delay = (delay0 - delay_l[20]) / delay0
        avg_tps.append(Optim_tps)
        avg_delay.append(Optim_delay)
    
    plt.legend()
    plt.title('10 Group Latency')
    plt.xlabel('time')
    plt.ylabel('latency')

    plt.show()
    plt.savefig(args.delay_path)
    avgt = sum(avg_tps) / len(avg_tps)
    avgd = sum(avg_delay) / len(avg_delay)
    print("average tps optimization is {}% and average latency optimization is {}%".format(avgt*100, avgd*100))


if __name__ == "__main__":
    group10()
    # ppo_x, ppo_y = 16650000, 5
    # ppo_perform_i = getPerformance(ppo_x, ppo_y)
    # print(ppo_perform_i)


