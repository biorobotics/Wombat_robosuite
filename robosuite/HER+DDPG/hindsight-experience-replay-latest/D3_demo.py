import torch
from rl_modules.models import actor
from arguments import get_args
import gym
import numpy as np
from mpi_utils.normalizer import normalizer

import numpy as np
import robosuite as suite
import matplotlib.pyplot as plt
import transforms3d as t3d
from collections import OrderedDict
import ipdb
import cv2
import h5py
from robosuite import load_controller_config
import json
import argparse

import math
import os
from datetime import datetime
from gym import spaces
import random

from train import D3_pick_place_env

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

# pre_process the inputs
def _preproc_inputs(obs, g):
    obs_norm = o_norm.normalize(obs)
    goal_norm = g_norm.normalize(g)
    # concatenate the stuffs
    inputs = np.concatenate([obs_norm, goal_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
    if args.cuda:
        inputs = inputs.cuda()
    return inputs

if __name__ == '__main__':
    args = get_args()
    # load the model param
    model_path = args.save_dir + args.env_name + '/model.pt'
    o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    # env = PickPlace_env(args)
    env = D3_pick_place_env(args,is_render=True)
    # env = gym.make(args.env_name)
    # get the env param
    # observation = env.reset()
    # obs = env.env.reset()
    obs = env.reset()
    # obs,_ = env.robot_obs2obs(obs)
    # get the environment params
    env_params = {'obs': obs['observation'].shape[0], 
                  'goal': obs['desired_goal'].shape[0], 
                  'action': env.action_network_dim, 
                  'action_max': env.action_high[0],
                  }
    
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    Success = 0
    for i in range(args.demo_length):
        observation = env.reset()
        # observation_,observation = env.robot_obs2obs(observation)
        # observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']#+ np.array([0.0,0.,0.03])
        obs_current = obs.copy()#obs['observation']
        ag = observation['achieved_goal']
        # create the normalizer
        o_norm = normalizer(size=env_params['obs'], default_clip_range=args.clip_range)
        g_norm = normalizer(size=env_params['goal'], default_clip_range=args.clip_range)
        action_zero = np.array([0,0,0.6,0,0,0,-0.2,-0.2])
        obs_current = np.zeros(28)
        obs_last = obs_current.copy()
        pp = 0
        pos_y = 0
        pos_x = 0
        for t in range(env._max_episode_steps):
            # env.render()
            if t%1000==0:
                print(t)
            # print("action_zero", action_zero)
            obs,reward,done,_ = env.step(action_zero)
            if t>=0 and t<1400 and pp==0:
                action_network = np.zeros(3)
                action_zero[1] -= 0.0002 #motion happening here
                if np.linalg.norm(obs_current[19]-obs_current[12])>0.002:
                    pos_x_dir = (obs_current[19]-obs_current[12])/np.abs(obs_current[19]-obs_current[12])
                    pos_x += 0.0002*pos_x_dir  #motion happening here
                action_zero[0] = pos_x
                # ipdb.set_trace()  
                obs,reward,done,_ = env.step(action_zero)
                obs_current = obs['observation'] 
                print(obs_current[19])
                print(obs_current[12])
                # print("Stage 1 ",pp)
            ### Calculate variables to pick ###
            elif t>=1400 and t<1500 and pp==0:
                action_network = np.zeros(3)
                obs,reward,done,_ = env.step(action_zero) 
                obs_current = obs['observation']
                vel_iPhone = (obs_current[13] - obs_last[13])
                steps_to_reach = ((0.0 - obs_current[13])/vel_iPhone)
                vel_iPhone_rt = (obs_current[13] - obs_last[13])/(0.002) #rt ==> real_time
                # print("last and current iPhone : ",obs_current[13],obs_last[13])
                # print("steps to reach: ",steps_to_reach)
                # print("Stage 2 ",pp)
            ### calculate target velocity ###
            elif t == 1500 and pp==0:
                action_network = np.zeros(3)
                vel_z = (obs_current[21] - 0.81)/int(steps_to_reach)
                vel_y = (obs_current[20] - 0.0)/int(steps_to_reach)
                # print("Stage 3 ",pp)
            ### Start snatch motion ###
            elif np.linalg.norm(obs_current[20]-obs_current[13])<0.001 or pp == 1:
                # print("Stage 4 ",pp)
                # print("len of ep_obs: ",len(mb_obs))
                obs,reward,done,_ = env.step(action_zero)
                obs_current = obs['observation']
                with torch.no_grad():
                    # ipdb.set_trace()
                    input_tensor = _preproc_inputs(obs_current, g)
                    pi = actor_network(input_tensor)
                    action_network = pi.cpu().numpy().squeeze()
                    # print("input_tensor", input_tensor)
                    # print("pi", pi)
                    # print("action_network", action_network)

                #residual trajectory added here
                # action_network = np.zeros(3)
                # print("action_zero just before Stage 4", action_zero)
                action_zero[0]+= action_network[0] #adding del_x to current_motion
                action_zero[1]+= (vel_iPhone_rt)*0.002 + action_network[1] #adding del_y to motion
                action_zero[2]+= vel_z*10 + action_network[2]  #adding del_z to motion
                # print("action_zero just after Stage 4", action_zero)
                # print("obs_current[26]: ",obs_current[26],np.linalg.norm(obs_current[21]-0.83))
                if np.linalg.norm(obs_current[21]-0.83)<0.001 and pp == 1 and obs_current[26]< 0.1:
                    # print("stay!!")
                    action_zero[2] = pos_z
                    action_zero[6] = 0.4
                    action_zero[7] = 0.4
                                
                elif obs_current[26]> 0.1 and action_zero[2]>0.55:
                    # print("go up!! ship is sinking")
                    pos_z -= 0.0005 #motion happening here
                    action_zero[2] = pos_z
                    action_zero[6] = 0.4
                    action_zero[7] = 0.4
                                
                elif action_zero[2]<0.55:
                    # print("breaking up of the loop")
                    break
                pos_z = action_zero[2]
                pp =1
            # re-assign the observation
            else:
                obs,reward,done,_ = env.step(action_zero)
                obs_current = obs['observation']
                # print("stage 5 {}, {}".format(pp,np.linalg.norm(obs_current[20]-obs_current[13])))
                action_network = np.zeros(3)
            obs_last = obs_current.copy()
            # inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            # with torch.no_grad():
            #     pi = actor_network(inputs)
            # action = pi.detach().numpy().squeeze()
            # # action_robot = env.action2robot_cmd(action,observation)
            # if t <= 10:
            #     action_robot[0:3] = g[0:3]
            #     # action_robot[2] += 0.05
            # # put actions into the environment
            # observation_new, reward, _, info = env.env.step(action_robot)
            # observation_new_,observation_new = env.robot_obs2obs(observation_new)
            # obs = observation_new_['observation']
            # observation = observation_new
        print("Expected goal: ",g)
        print("Actual position: ",obs['achieved_goal'])
        # print("Gripper location: ",observation_new[12:15])
        # print("Reward is: ",env.compute_reward(obs['achieved_goal'].reshape(1,-1),g.reshape(1,-1),None))renv.compute_reward(obs['observation'],obs['observation'],None)
        reward = env.compute_reward(obs['observation'],obs['observation'],None)
        print("Reward is: ",reward)
        if reward[1]>=5:
            Success = Success + 1
        print("Run no.: {}, total Successes: {}".format(i+1,Success))
            # obs = observation_new['observation']
        # print('the episode is: {}, is success: {}'.format(i, info['is_success']))
