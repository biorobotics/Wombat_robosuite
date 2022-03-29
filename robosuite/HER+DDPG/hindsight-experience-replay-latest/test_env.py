import numpy as np
from arguments import get_args
import numpy as np
import robosuite as suite
import matplotlib.pyplot as plt
import transforms3d as t3d
from collections import OrderedDict
from robosuite import load_controller_config

import math
import os
from datetime import datetime
from gym import spaces
import random
import time

from train_ur3e import UR3e_env


def dyn_rand():
    # phone_x = 0.578#np.random.uniform(0.428, 0.728)
    phone_x = 0.4

    phone_speed = -0.20#np.random.uniform(-0.14, -0.18)
    phone_orient = 0.0
    # phone_orient = np.random.uniform(-0.05, 0.05)
    return phone_x, phone_speed, phone_orient

if __name__ == '__main__':
    args = get_args()
    env = UR3e_env(args,is_render = True)
    phone_x, phone_speed, phone_orient = dyn_rand()

    obs = env.set_env(phone_x, phone_speed, phone_orient)
    observ = obs['observation']
    env_params = {'obs': obs['observation'].shape[0], 
                'goal': obs['desired_goal'].shape[0], 
                'action': env.action_network_dim, 
                'action_max': env.action_high[0],
                }

    observation_init = obs['observation']
    # env.model.opt.gravity[-1] = 0

    max_time_steps = 11000
    action_zero = np.array([1.31e-1, 3.915e-1, 2.05e-1, -3.14, 0,0,-0.4, 0.4])
    desired =  np.array([-np.pi/2, -2.0, -np.pi/2, -1.01,  1.57, np.pi *0/180.0])

    # action_zero = np.zeros(8)
    # action_zero[6:8] = np.array([-0.4, 0.4])
    # action_zero[0] = np.array([1.31e-1])
    for t in range(max_time_steps):
        obs, reward, done = env.step(desired)

        obs_current = obs['observation']
        joint_0 = env.sim.data.get_joint_qpos('robot0_joint_1')
        # if(t>1000):
            # action_zero[0:3]-=0.0005
        # env.sim.step()
        
