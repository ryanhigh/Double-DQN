import numpy as np
import pandas as pd
from utils import LabberRing
figure_path = './Val_n_base/avg_reward_r2_vboth_wid500_test2.png'
evalc = pd.read_csv('./Val_n_base/rewards.csv')
episodes = evalc['episodes'].tolist()
baseline = evalc['baseline'].tolist()
ddqn = evalc['ddqn'].tolist()
ppo = evalc['ppo'].tolist()
ddqn_convergence = np.mean(ddqn) / 10
ppo_convergence = np.mean(ppo) / 10
ppo_ratio = (ppo_convergence - baseline[0]) / baseline[0]
ddqn_ratio = (ddqn_convergence - baseline[0]) / baseline[0]
#print(ppo_ratio, ddqn_ratio)
LabberRing(500, episodes, baseline, ddqn, ppo, 'Evaluate Criteria', 'evalc(x10)', figure_path)








    
