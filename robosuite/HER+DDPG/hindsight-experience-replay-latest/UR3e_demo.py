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
import time

from train_ur3e import UR3e_env


from wombat_dmp.srv import *
import rospy
from geometry_msgs.msg import Pose, Vector3 
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R



#TODO: Things to do next: Verify once with the x  and y action frames 
# TODO: DMP tolerance 
# TODO Get a baseline 
# TODO: Test clip joint functions 



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
def dyn_rand():
    # phone_x = 0.578#np.random.uniform(0.428, 0.728)
    phone_x = 0.3

    phone_speed = -0.20#np.random.uniform(-0.14, -0.18)
    phone_orient = 0.0
    # phone_orient = np.random.uniform(-0.05, 0.05)
    return phone_x, phone_speed, phone_orient


def dmp_client( start, goal, mode, phone_velocity):
    rospy.wait_for_service('dmp_calc_path')
    try:
        dmp_calc_path = rospy.ServiceProxy('dmp_calc_path', dmpPath)
        resp1 = dmp_calc_path(start,goal, mode, phone_velocity)
        return resp1.traj
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


def quat_to_euler(  quat):
    r_quat = R.from_quat([quat.x,quat.y,quat.z,quat.w])
    e_angles = r_quat.as_euler('xyz', degrees=False)
    return e_angles

def euler_to_quat(euler):
    rot = R.from_euler('xyz',[euler[0], euler[1], euler[2]], degrees=False)
    quat = rot.as_quat()
    return quat
        

if __name__ == '__main__':
    args = get_args()
    # load the model param
    model_path = args.save_dir + args.env_name + '/Modelnew_dynamics_epoch_36.pt'
    o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    # env = PickPlace_env(args)
    # env = gym.make(args.env_name)
    # get the env param
    # observation = env.reset()
    # obs = env.env.reset()

    success = 0
    for i in range(args.demo_length):
        print(f"Restarting the episode")
        # obs = env.reset()
        phone_x, phone_speed, phone_orient = dyn_rand()
        env = UR3e_env(args,is_render=True)

        env.set_env(phone_x,phone_speed,phone_orient)
        obs = env.observation_current
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
        # reset the environment
        # observation = env.reset(phone_x, phone_speed, phone_orient)
        # observation_,observation = env.robot_obs2obs(observation)
        # observation = env.reset()
        # start to do the demo
        observation_init = obs['observation']
        # g = obs['desired_goal']#+ np.array([0.0,0.,0.03])
        obs_current = observation_init.copy()#obs['observation']
        # ag = obs['achieved_goal']
        # create the normalizer
        o_norm = normalizer(size=env_params['obs'], default_clip_range=args.clip_range)
        g_norm = normalizer(size=env_params['goal'], default_clip_range=args.clip_range)
        action_zero=np.array([1.31e-1, 3.915e-1, 2.05e-1, -3.14, 0,0,-0.4, 0.4])
        obs_current = np.zeros(34)
        obs_last = obs_current.copy()
        pp_snatch = 0
        pp_grip = 0
        pos_y = 0
        pos_x = 0
        time_reset = 50
        time_stay = 100
        ##to plot
        # ! how do these numbers are calculated?
        observation_x = np.zeros(11000)
        max_time_steps = 11000
        t_arr = np.linspace(0,max_time_steps,max_time_steps+1)
        observation_x_commanded = np.zeros(max_time_steps+1)
        observation_y_commanded  = np.zeros(max_time_steps+1)
        observation_z_commanded  = np.zeros(max_time_steps+1)

        observation_x_real = np.zeros(max_time_steps+1)
        observation_y_real = np.zeros(max_time_steps+1)
        observation_z_real = np.zeros(max_time_steps+1)
        

        T_sim_real = np.eye(4)
        T_real_actual = np.eye(4)
        T_sim_actual = np.eye(4)
        t_real = [0]*6


        plan_flag= True
        wait_flag = False
        path_executed = False
        time.sleep(0.1)
        print(f"Printing before first timestep")
        for t in range(env._max_episode_steps):

            # try:
            #     obs,reward,done = env.step(action_zero)
            # except Exception as e:
            #     print(f"{e}")
            print(f"demo: action_zero: {action_zero}")
            obs,reward,done = env.step(action_zero)

            # print(f"obs {obs}")
            obs_current = obs['observation']
            # print(obs_current[21])
            vel_iPhone_rt = (obs_current[13] - obs_last[13])/(0.002) #rt ==> real_time
            # When phone crosses this point, planner will get started
            # TODO: Recalib for UR3e
            pre_grasp_pos = 0.6 #0.9
            proximal_tol = 0.1
            
            gripper_pos = obs_current[19:26] 
            phone_pos = obs_current[12:15] 
            print(f"phone_pose: {phone_pos[0:3]}, gripper_pose: {gripper_pos[0:3]}")
            
            # print(f"delta: {delta} ")
            if phone_pos[0]>pre_grasp_pos :
                print(f"demo 2 action_zero {action_zero}")
                obs,reward,done= env.step(action_zero)
                obs_current = obs['observation'] 
                

            
            if (phone_pos[0]<pre_grasp_pos and path_executed==False and t>50):


                # Init DMP traj 
                gripper_pos = obs_current[19:26] 
                phone_pos = obs_current[12:15] 

                delta = action_zero[0:3] - gripper_pos[0:3]
                action_goal = phone_pos + delta
                
                print(f"phone_pose: {phone_pos[0:3]}, gripper_pose: {gripper_pos[0:3]}")
                if(plan_flag):          
                    start= action_zero[0:3]
                    start_pose = Pose()

                    start_pose.position.x = start[0]
                    start_pose.position.y = start[1]
                    start_pose.position.z = start[2]
                    start_pose.orientation.x = 0
                    start_pose.orientation.y = 0
                    start_pose.orientation.z = 0
                    start_pose.orientation.w = 1


                    goal = np.zeros(3)
                    goal[0]  = action_zero[0]
                    goal[1] = action_goal[1]
                    goal[2] = action_goal[2]        
                    end_pose = Pose()
                    # TODO: Recalib for UR
                    end_pose.position.x = goal[0] 
                    end_pose.position.y = goal[1] 
                    end_pose.position.z = 0.09
                    end_pose.orientation.x = 0
                    end_pose.orientation.y = 0
                    end_pose.orientation.z = 0
                    end_pose.orientation.w =1
                    mode = String()
                    mode.data = "pick"

                    phone_velo = Vector3()
                    phone_velo.x = 0
                    phone_velo.y =obs_current[29]
                    phone_velo.z = 0
                    traj =dmp_client(start_pose, end_pose, mode, phone_velo)
                    desired_traj = np.zeros((len(traj), 6))
                    for i in range(len(traj)):
                        desired_traj[i][0] =traj[i].position.x 
                        desired_traj[i][1] =traj[i].position.y 
                        desired_traj[i][2] =traj[i].position.z
                        desired_traj[i][5] =quat_to_euler(traj[i].orientation)[0]
                        desired_traj[i][4] =quat_to_euler(traj[i].orientation)[1]
                        desired_traj[i][3] =quat_to_euler(traj[i].orientation)[2]
                    traj_index = 0
                    plan_flag = False

                # ! -> Execute this traj

                # Calculate current position in the trajectory : i^th index 
                
                
                path_executed = traj_index==desired_traj.shape[0]
                if(path_executed==True):
                    print(f"Path Executed {path_executed}")
                if(path_executed==False):
                    action_zero[0:3] = desired_traj[traj_index, 0:3]
                    # print(f"action_zero {action_zero}")

                    # TODO: Recalib for UR
                    # if action_zero[2]<=0.84:
                    #     action_zero[2]=0.84
                    print(f"action_zero from dmp {action_zero}")
                    obs,reward,done= env.step(action_zero)
                    obs_current = obs['observation'] 
                    traj_index+=1



            # env.render()
            # if t%1000==0:
            #     print(t)
            # print("action_zero", action_zero)
            # obs,reward,done,_ = env.step(action_zero)
            # if t>=0 and t<1700 and pp_snatch==0:
            #     action_network = np.zeros(3)
            #     action_zero[1] += 0.0002 #motion happening here
            #     if action_zero[1]>0.05:
            #         action_zero[1]=0.05
            #     if np.linalg.norm(obs_current[19]-obs_current[12])>0.002:
            #         pos_x_dir = (obs_current[19]-obs_current[12])/np.abs(obs_current[19]-obs_current[12])
            #         pos_x += 0.0001*pos_x_dir  #motion happening here
            #         # print("x",obs_current[26])
            #         # print(obs_current[27])
            #     action_zero[0] = pos_x
            #     # ipdb.set_trace()  
            #     print(action_zero)
            #     obs,reward,done,_ = env.step(action_zero)
            #     obs_current = obs['observation'] 
            #     # print("Stage 1 ",pp)
            # ### Calculate variables to pick ###
            # elif t>=1700 and t<1800 and pp_snatch==0:
            #     action_network = np.zeros(3)
            #     obs,reward,done,_ = env.step(action_zero) 
            #     obs_current = obs['observation']
            #     vel_iPhone = (obs_current[13] - obs_last[13])
            #     steps_to_reach = ((0.0 - obs_current[13])/vel_iPhone)
            #     vel_iPhone_rt = (obs_current[13] - obs_last[13])/(0.002) #rt ==> real_time
            #     # print("Stage 2 ",pp)
            # ### calculate target velocity ###
            # elif t == 1800 and pp_snatch==0:
            #     action_network = np.zeros(3)
            #     vel_z = (obs_current[21] - 0.885)/int(steps_to_reach)
            #     vel_y = (obs_current[20] - 0.0)/int(steps_to_reach)
            #     # print("Stage 3 ",pp)
            ### Start snatch motion ###
            elif path_executed or np.linalg.norm(obs_current[20]-obs_current[13])<0.001 or pp_snatch == 1:
                if(wait_flag==False):
                    completion_time = t
                    wait_flag =True
                if action_zero[2]>=0.763:
                    action_zero[2]=0.763
                obs,reward,done = env.step(action_zero)
                obs_current = obs['observation']
                
                
                
                # with torch.no_grad():
                #     # ipdb.set_trace()
                #     input_tensor = _preproc_inputs(obs_current, g)
                #     pi = actor_network(input_tensor)
                #     action_network = pi.detach().numpy().squeeze()
                #     # print("input_tensor", input_tensor)
                #     # print("pi", pi)
                #     # print("action_network", action_network)

                # #residual trajectory added here
                # action_network = np.zeros(3)
                # # print("action_zero just before Stage 4", action_zero)
                # action_zero[0]+= action_network[0] #adding del_x to current_motion
                # action_zero[1]+= action_network[1] #adding del_y to motion
                # action_zero[2]+= action_network[2]  #adding del_z to motion
                # # action_zero[3]+= action_network[3] #adding orient_x to current_motion
                # # action_zero[4]+= action_network[4] #adding orient_y to motion
                # # action_zero[5]+= action_network[5]  #adding orient_z to motion


                # print(f"phone_y {phone_pos[1]}")
                # print(f"gripper_y {gripper_pos[1]}")
                resume_flag=True
                # if(wait_flag==True):
                #     if(t-completion_time)>1:
                #         resume_flag = True

                if path_executed and  pp_snatch == 1 and pp_grip == 0 and resume_flag:
                    action_zero[2] = pos_z
                    # action_zero[6] = 0.4
                    # action_zero[7] = 0.4
                    action_zero[6] = 0.4
                    action_zero[7] = -0.4
                    time_stay-=1
                    if time_stay<=0:
                        pp_grip=1
                                
                elif pp_grip==1 and action_zero[2]>0.55:
                    # print("go up!! ship is sinking")
                    last_time = t
                    time_reset-=1
                    if time_reset<=0:
                        pos_z -= 0.0005 #motion happening here
                        action_zero[2] = pos_z
                        # action_zero[6] = 0.4
                        # action_zero[7] = 0.4
                        action_zero[6] = 0.4
                        action_zero[7] = -0.4

                    if time_reset<=-100:
                        env.clip_grip_vel()
                                
                elif action_zero[2]<0.55:
                    # print("breaking up of the loop")
                    break
                pos_z = action_zero[2]
                pp_snatch =1
            # re-assign the observation
            # else:
            #     print(f"if else last stage")
            #     # print("no snatching")
            #     if action_zero[2]>=0.763:
            #         action_zero[2]=0.763
            #     # obs,reward,done,_ = env.step(action_zero)

            #     obs_current = obs['observation']
                # print("stage 5 {}, {}".format(pp,np.linalg.norm(obs_current[20]-obs_current[13])))
                # action_network = np.zeros(3)
            obs_last = obs_current.copy()
            T_sim_real[0:3,0:3] = np.array([[-1,0,0],
                            [0,1,0],
                            [0,0,-1]])
            T_sim_real[0:3,3] = [0.5983,0.1196,0.3987]
            # T_sim_target[0:3,0:3] = t3d.euler.euler2mat(target_sim[3],target_sim[4],target_sim[5],'szyx')
            T_sim_actual[0:3,3] = [obs_current[19],obs_current[20],obs_current[21]]
            T_real_actual = np.matmul(np.linalg.inv(T_sim_real),T_sim_actual)
            t_real[0],t_real[1],t_real[2] = T_real_actual[0:3,3]
            t_real[2] = - t_real[2]
            # observation_x[t] = t_real[0]
            # observation_y[t] = t_real[1]
            # observation_z[t] = t_real[2]
            # t_arr[t] = t*0.002
            # observation_x_commanded[t] = action_zero[0]
            # observation_y_commanded[t] = action_zero[1]
            # observation_z_commanded[t] = action_zero[2]


            # observation_x_real[t] = gripper_pos[0] - delta[0]
            # observation_y_real[t] = gripper_pos[1] - delta[1]
            # observation_z_real[t] = gripper_pos[2] - delta[2]
            





        # plt.plot(t_arr[0:last_time]*0.002, observation_x_commanded[0:last_time])
        # plt.plot(t_arr[0:last_time]*0.002, observation_x_real[0:last_time])
        # plt.legend(["commanded", "real"])
        # plt.title('End-effector x-coordinate vs. time')
        # plt.xlabel('time(in seconds)')
        # plt.ylabel('End-effector x-coordinate(in meter)')
        # plt.show()
        # plt.plot(t_arr[0:last_time]*0.002, observation_y_commanded[0:last_time])
        # plt.plot(t_arr[0:last_time]*0.002, observation_y_real[0:last_time])
        # plt.legend(["commanded", "real"])
        # plt.title('End-effector y-coordinate vs. time')
        # plt.xlabel('time(in seconds)')
        # plt.ylabel('End-effector y-coordinate(in meter)')
        # plt.show()
        # plt.plot(t_arr[0:last_time]*0.002, observation_z_commanded[0:last_time])
        # plt.plot(t_arr[0:last_time]*0.002, observation_z_real[0:last_time])
        # plt.legend(["commanded", "real"])
        # plt.title('End-effector z-coordinate vs. time')
        # plt.xlabel('time(in seconds)')
        # plt.ylabel('End-effector z-coordinate(in meter)')
        # plt.show()
        
                        
        # print("Expected goal: ",g)
        # print("Actual position: ",obs['achieved_goal'])
        reward = env.compute_reward(obs['observation'])
        print("Reward is: ",reward)
        
        if reward[1]==50:
            success = success + 1
        print("Run no.: {}, Total_successes: {}".format(i+1,success))
        # plt.plot(t_arr, observation_x)
        # plt.show()
        env.close_window()