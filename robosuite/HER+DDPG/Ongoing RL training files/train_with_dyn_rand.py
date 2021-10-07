import numpy as np
import gym
import os, sys
from arguments import get_args
from mpi4py import MPI
from rl_modules.ddpg_agent import ddpg_agent
import random
import torch

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
"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)
"""


import D3_controller.trajGenerator as trg
from D3_controller.interpolateTraj import interpolateTraj
import D3_controller.invK as invK

import robosuite
from robosuite.models.objects import BoxObject
from robosuite.models.robots import Wombat_arm
from robosuite.models.arenas import EmptyArena
from robosuite.models.grippers import gripper_factory
import math
from mujoco_py import MjSim, MjViewer
import numpy as np
import time
import matplotlib as mpl
import ipdb
mpl.use('TkAgg')
import matplotlib.pyplot as plt 
from robosuite.models import MujocoWorldBase



class D3_pick_place_env(object):

	def __init__ (self,args=None,is_render=False):

		self.is_render = is_render
		self.action_dim = 6 #actual robot
		self.action_network_dim = 3 # Gripper pose
		self.obs_dim = 34#28#26
		self.q_pos_last = np.zeros(self.action_dim)
		self.observation_current = None
		self.observation_last = None
		self.observation_last2last = None
		self.joint_sim_last = None

		self.done = False

		self.action_high = np.array([0.00005]*self.action_dim)
		self.action_low = np.array([-0.00005]*self.action_dim)

		self.action_network_high = np.array([0.00005]*self.action_network_dim)
		self.action_network_low = np.array([-0.00005]*self.action_network_dim)	 

		self._max_episode_steps = 11000#9000#20000

		pass

	def angDiff(self,a1,a2):
		d1=(a1-a2)%(2*np.pi)
		d2=(a2-a1)%(2*np.pi)
		if d1<d2:
			return -d1
		return d2
		#given two pairs of joint values, finds the rotary values for j2 that
		#are closest to j1. Assumes values are in rad
	def nextClosestJointRad(self,j1,j2):
		j2c=np.copy(j2)
		for i in range(0,3):
			aDiff1=self.angDiff(j1[i],j2[i])
			aDiff2=self.angDiff(j1[i],j2[i]+np.pi)
			if abs(aDiff1)<abs(aDiff2):
				j2c[i]=j1[i]+aDiff1
			else:
				j2c[i]=j1[i]+aDiff2
		return j2c

	def set_env(self,phone_x,phone_speed,phone_orient):

		self.phone_x = phone_x
		self.phone_speed = phone_speed
		self.phone_orient = phone_orient
		self.world = MujocoWorldBase()
		self.mujoco_robot = Wombat_arm()

		self.mujoco_robot.set_base_xpos([0, 0.0, 0])
		self.world.merge(self.mujoco_robot)

		self.mujoco_arena =EmptyArena()
		# mujoco_arena.set_origin([0.8, 0, 0])
		self.world.merge(self.mujoco_arena)

		self.iphonebox = BoxObject(name="iphonebox",size=[0.035,0.07,0.01],rgba=[0,0,0,1],friction=[1,1,5]).get_obj()
		self.iphonebox.set('pos', '{} 1.35 1'.format(self.phone_x)) #0.75
		self.iphonebox.set('quat', '{} 0 0 1'.format(self.phone_orient)) #0
		self.world.worldbody.append(self.iphonebox)

		self.box = BoxObject(name="box",size=[0.35,9.7,0.37],rgba=[0.5,0.5,0.5,1],friction=[1,1,1]).get_obj()
		self.box.set('pos', '0.6 -2 0')
		self.world.worldbody.append(self.box)

		self.model = self.world.get_model(mode="mujoco_py")

		self.sim = MjSim(self.model)
		
		if self.is_render:
			self.viewer = MjViewer(self.sim)
			self.viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh
			self.viewer.render()

		self.timestep= 0.0005
		self.sim_state = self.sim.get_state()
		self.joint_names = ['robot0_branch1_linear_joint','robot0_branch2_linear_joint','robot0_branch3_linear_joint',
							'robot0_branch1_joint','robot0_branch2_joint','robot0_branch3_joint']

		# ipdb.set_trace()
		self.observation_current = self.get_observation()

		return self.observation_current

	def get_signal(self,action,obs_last,obs_last2last):
		action = self.clip_action(action)

		action = action.reshape(1,-1)[:,:6]
		# ipdb.set_trace()
		joint_target_action=trg.jointTraj(action)
		# q_pos_last[0:6] = obs_last2last[0:6]
		try :
			joint_real=joint_target_action
		except TypeError:
			print("found an issue")
			joint_real=joint_real_last

		ee_pose=invK.real2sim_wrapper(action[0])
		# print("action[0] in get_signal", action[0])
		# print("joint_real[0] in get_signal", joint_real[0])
		self.joint_sim=invK.ik_wrapper(joint_real[0])
		j1 = np.array([self.sim.data.get_joint_qpos("robot0_branch1_joint"),self.sim.data.get_joint_qpos("robot0_branch2_joint"),self.sim.data.get_joint_qpos("robot0_branch3_joint")])
		j2 = np.array([self.joint_sim[3],self.joint_sim[4],self.joint_sim[5]])
		self.joint_sim[3:6] = self.nextClosestJointRad(j1,j2)
		PD_scale= self.PD_signal_scale(action[0],joint_real[0])
		# ipdb.set_trace()
		PD_signal=[self.PD_controller_rot(self.joint_sim[3],obs_last[0],obs_last2last[0],PD_scale[0]),
			   self.PD_controller_rot(self.joint_sim[4],obs_last[1],obs_last2last[1],PD_scale[1]),
			   self.PD_controller_rot(self.joint_sim[5],obs_last[2],obs_last2last[2],PD_scale[2]),
			   self.PD_controller_lin(self.joint_sim[0],obs_last[3],obs_last2last[3],PD_scale[3]),
			   self.PD_controller_lin(self.joint_sim[1],obs_last[4],obs_last2last[4],PD_scale[4]),
			   self.PD_controller_lin(self.joint_sim[2],obs_last[5],obs_last2last[5],PD_scale[5])]

		self.joint_real_last = joint_real

		return PD_signal


	def reset(self,phone_x=0.75,phone_speed=-0.2,phone_orient=0):
		# ipdb.set_trace()
		obs = self.set_env(phone_x,phone_speed,phone_orient)
		return obs

	def grip_signal(self,des_state,obs_last,obs_last2last):
		if des_state=='open':
			left_finger_open = -0.287884
			right_finger_open = -0.295456

			grip_signal=[self.Gripper_PD_controller(left_finger_open,obs_last[26],obs_last2last[26]),
						 self.Gripper_PD_controller(right_finger_open,obs_last[27],obs_last2last[27])]
		if des_state=='close':
			left_finger_close = 0.246598
			right_finger_close = 0.241764

			grip_signal=[self.Gripper_PD_controller(left_finger_close,obs_last[26],obs_last2last[26]),
						 self.Gripper_PD_controller(right_finger_close,obs_last[27],obs_last2last[27])]
		return grip_signal

	def clip_action(self,action):
		if action[0]>0.45:
			action[0]=0.45
		if action[0]<-0.45:
			action[0]=-0.45
		if action[1]>0.45:
			action[1]=0.45
		if action[1]<-0.45:
			action[1]=-0.45
		if action[2]>0.78:
			action[2]=0.78

		return action
		

	def action2robot(action_RL,action_robot):

		action = np.zeros(self.action_dim)
		action = action_robot[0:3]+action_RL[0:3]

		return action

	def step(self,action):
		# action = action.reshape(1,-1)
		if self.observation_last is not None:
			# ipdb.set_trace()
			self.observation_last2last = self.observation_last
		else:
			self.observation_last2last = np.zeros(self.obs_dim)


		self.observation_last = self.observation_current['observation']
		# print("action", action)
		# action = self.action2robot(action_RL,action_robot)
		# print("action,obs_last,obs_last2last in train.py step()", action,self.observation_last,self.observation_last2last)
		PD_signal = self.get_signal(action,self.observation_last,self.observation_last2last)

		self.sim.data.ctrl[0:6] = PD_signal[0:6]
		if action[6] > 0:
			des_state='close'
		else:
			des_state='open'
		self.sim.data.ctrl[6:8] = self.grip_signal(des_state,self.observation_last,self.observation_last2last)
		self.clip_grip_action()
		# print("torque reading: ",self.sim.data.ctrl[0:6])
		# print("action", action)
		self.sim.step()
		if self.is_render:
			# pass
			self.viewer.render()
		# print("sending steps")
		self.observation_current = self.get_observation()
		self.reward = self.compute_reward(self.observation_current['observation'],self.observation_current['observation'],None)
		self.sim.data.set_joint_qvel('box_joint0', [0, self.phone_speed, 0, 0, 0, 0])
		self.done = self.is_done
		# ipdb.set_trace()
		# print("sensor data: ",self.sim.data.sensordata)
		return self.observation_current,self.reward,self.done,None


	def get_observation(self):

		observation = np.zeros(self.obs_dim)

		observation[0] = self.sim.data.get_joint_qpos("robot0_branch1_joint")
		observation[1] = self.sim.data.get_joint_qpos("robot0_branch2_joint")
		observation[2] = self.sim.data.get_joint_qpos("robot0_branch3_joint")
		#linear
		observation[3] = self.sim.data.get_joint_qpos("robot0_branch1_linear_joint")
		observation[4] = self.sim.data.get_joint_qpos("robot0_branch2_linear_joint")
		observation[5] = self.sim.data.get_joint_qpos("robot0_branch3_linear_joint")

		observation[6] = self.sim.data.get_joint_qvel("robot0_branch1_joint")
		observation[7] = self.sim.data.get_joint_qvel("robot0_branch2_joint")
		observation[8] = self.sim.data.get_joint_qvel("robot0_branch3_joint")
		#linear
		observation[9] = self.sim.data.get_joint_qvel("robot0_branch1_linear_joint")
		observation[10] = self.sim.data.get_joint_qvel("robot0_branch2_linear_joint")
		observation[11] = self.sim.data.get_joint_qvel("robot0_branch3_linear_joint")


		observation[12:19] = self.sim.data.get_joint_qpos('iphonebox_joint0')
		observation[19:26] = self.sim.data.sensordata[0:7]	#gripper base link pose
		observation[26] = self.sim.data.get_joint_qpos('robot0_gripper_left_finger_joint')
		observation[27] = self.sim.data.get_joint_qpos('robot0_gripper_right_finger_joint')
		observation[28:34] = self.sim.data.get_joint_qvel('iphonebox_joint0')
		# observation[28:31] = self.sim.render(width=200, height=200, camera_name='frontview', mode='offscreen')[::-1,:]

		goal = 0.88#0.83		#z value of the phone should be at 0.83 m from floor
		achieved_goal = observation[14] #Current z value of the iPhone
		observation = {'observation':observation,'desired_goal':np.array([goal]),'achieved_goal': np.array([achieved_goal])}

		# goal = np.array([True, True, True])		#fingers closed, dist(gripper-phone)<0.11, gripper away from conveyor
		# achieved_goal = np.array([observation[26]>0, observation[21]-observation[14]<= 0.11, observation[21]>0.99])
		## raises error
		# goal = np.array([1, 1, 1])		#fingers closed, dist(gripper-phone)<0.11, gripper away from conveyor
		# achieved_goal = np.array([1 if observation[26]>0 else 0, 1 if observation[21]-observation[14]<= 0.11 else 0, 1 if observation[21]>0.99 else 0])
		# observation = {'observation':observation,'desired_goal':goal,'achieved_goal': achieved_goal}
		
		# ipdb.set_trace()
		# print("gripper pose: ",self.sim.data.get_joint_qpos('robot0_gripper_base_link_joint'))
		return observation

		pass

	def compute_reward(self,obs,obs2,obs3):
		# reward_grasp = []
		# for i in range(obs.shape[0]):
		# 	if np.linalg.norm(obs[i]-obs[i]) < 0.005:
		# 		reward_grasp.append(10)
		# 	else:
		# 		reward_grasp.append(10)

		# reward_grasp = np.array(reward_grasp)
		##modified reward function
		reward_grasp = []
		# print("obs.shape[0]", obs.shape[0])
		# print("obs[21]-obs[14]", obs[21]-obs[14])
		# print("obs[21]", obs[21])
		# ipdb.set_trace()
		# print(obs[21]-obs[14] <= 0.11)
		# print(obs[26] > 0)
		# print(obs[21] > 0.99)
		# for i in range(obs.shape[0]):
		# 	if (obs[21]-obs[14] <= 0.11) and (obs[26] > 0) and (obs[21] > 0.99):
		# 		reward_grasp.append(20)
		# 		# print("Reward: +20")
		# 	elif (obs[21]-obs[14] <= 0.16) and (obs[21]-obs[14] > 0.11) and (obs[26] > 0) and (obs[21] > 0.94):
		# 		reward_grasp.append(5)
		# 		# print("Reward: +5")
		# 	else:
		# 		reward_grasp.append(0)
		for i in range(obs.shape[0]):
			if obs[14]>=0.88 and (obs[21]-obs[14] <= 0.113) and (abs(obs[19]-obs[12]) <= 0.04) and obs[26] > -0.29:
				reward_grasp.append(50)
				# print("Reward: +50")
			elif obs[14] >= 0.80 and obs[26] > -0.29:
				reward_grasp.append(5)
				# print("Reward: +5")
			else:
				reward_grasp.append(0)
		reward_grasp = np.array(reward_grasp)
		return reward_grasp

	def is_done(self,obs):
		# if ((obs[21]-obs[14]) < 0.14) and (obs[21] > 0.99):
		# 	return True
		# else:
		return False

	def PD_controller_rot(self,des,current,q_pos_last,scale):
	
		kp=20*scale #10 #20 
		kd=8 #4 #1
		qpos = des+kp*(des-current)-kd*(current-q_pos_last)
		# print(kp*(des-current))
		return qpos

	def PD_controller_lin(self,des,current,q_pos_last,scale):
		
		kp=1000 #1000 #200
		kd=1500 #1500
		qpos = des+kp*(des-current)-kd*(current-q_pos_last)
		# print(kp*(des-current))
		return qpos

	def PD_signal_scale(self,ee_pos,joint_vals):

		#scales the PD signal based on the ee pos or joint values; wombat_arm needs
		#different PD values depending on where it is, position-wise

		ee_xy_disp=np.array([math.sqrt(ee_pos[0]**2+ee_pos[1]**2)]*6)+1.0
		lin_vals=np.array([joint_vals[2],joint_vals[0],joint_vals[1]]*2)+1.0
		scale=7
		PD_scale_factor=((np.multiply(ee_xy_disp,lin_vals)**2)-1)*scale
		return PD_scale_factor

	def Gripper_PD_controller(self,des,current,last):
		kp=10
		kd=2
		pos = des+kp*(des-current)-kd*(current-last)
		return pos

	def clip_grip_action(self):
		if self.sim.data.get_joint_qpos('robot0_gripper_left_finger_joint')>0.25:
			self.sim.data.set_joint_qpos('robot0_gripper_left_finger_joint', 0.246598)
		if self.sim.data.get_joint_qpos('robot0_gripper_right_finger_joint')>0.25:
			self.sim.data.set_joint_qpos('robot0_gripper_right_finger_joint', 0.241764)
		if self.sim.data.get_joint_qpos('robot0_gripper_left_finger_joint')<-0.29:
			self.sim.data.set_joint_qpos('robot0_gripper_left_finger_joint', -0.287884)
		if self.sim.data.get_joint_qpos('robot0_gripper_right_finger_joint')<-0.30:
			self.sim.data.set_joint_qpos('robot0_gripper_right_finger_joint', -0.295456)

	def seed(self, seed=None):
		"""
		Utility function to set numpy seed
		Args:
			seed (None or int): If specified, numpy seed to set
		Raises:
			TypeError: [Seed must be integer]
		"""
		# Seed the generator
		if seed is not None:
			try:
				np.random.seed(seed)
			except:
				TypeError("Seed must be an integer type!")


def get_env_params(env):
	# ipdb.set_trace()
	obs_dict = env.reset()	#not yet written
	# ipdb.set_trace()
	# close the environment
	# ipdb.set_trace()
	params = {'obs': obs_dict['observation'].shape[0],
			'goal': obs_dict['desired_goal'].shape[0],
			'action': env.action_network_dim,
			'action_max': env.action_high[0],
			}
	params['max_timesteps'] = env._max_episode_steps
	return params

def launch(args,env):
	# create the ddpg_agent
	
	# set random seeds for reproduce
	env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
	random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
	np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
	torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
	if args.cuda:
		torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
	# get the environment parameters
	env_params = get_env_params(env)
	ddpg_trainer = ddpg_agent(args, env, env_params)
	ddpg_trainer.learn()
	# create the ddpg agent to interact with the environment 
	#####################################################works till this point############################################
	####### not yet there ##########

if __name__ == '__main__':
	# take the configuration for the HER
	os.environ['OMP_NUM_THREADS'] = '1'
	os.environ['MKL_NUM_THREADS'] = '1'
	os.environ['IN_MPI'] = '1'
	# get the params
	args = get_args()
	# print(args)


	from time import localtime, strftime

	current_time = strftime("%Y-%m-%d-%H-%M-%S", localtime())
	print("Current Time =", current_time)
	# args = parser.parse_args()
	# args.log = os.path.join(args.log)
	pick_place_env = D3_pick_place_env(args,is_render=False)
	# pick_place_env = PickPlace_env(args)
	# pick_place_env.run()
	launch(args,pick_place_env)