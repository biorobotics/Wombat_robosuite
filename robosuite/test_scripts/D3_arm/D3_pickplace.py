import trajGenerator as trg
from interpolateTraj import interpolateTraj
import robosuite
from robosuite.models.objects import BoxObject
from robosuite.models.robots import Wombat_arm
from robosuite.models.arenas import EmptyArena
from robosuite.models.grippers import gripper_factory
import invK
import math
import copy
from mujoco_py import MjSim, MjViewer
import numpy as np
import time
import matplotlib as mpl
import ipdb
mpl.use('TkAgg')
import matplotlib.pyplot as plt 
from robosuite.models import MujocoWorldBase
import pyautogui
import cv2

class D3_pick_place_env(object):

	def __init__ (self,args=None,is_render=False):

		self.is_render = True#is_render
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

	##PD controller below
	def PD_controller_rot(self,des,current,q_pos_last,scale):
		
		#kp = 10
		#kp=10
		#kp = 1
		#kd = 0.3
		#kp = 5
		#kd = 0.6
		kp=10*scale#20*scale ####10
		kd=4#0.6 ####4
		qpos = des+kp*(des-current)-kd*(current-q_pos_last)
		# print(kp*(des-current))
		return qpos

	# return np.array(points)
	def PD_controller_lin(self,des,current,q_pos_last,scale):
		
		#kp = 10
		#kd = 0.8
		#kp=10
		#kd=0.1
		kp=800#150 ####1000
		kd=1500 ####1500
		qpos = des+kp*(des-current)-kd*(current-q_pos_last)
		# print(kp*(des-current))
		return qpos

	##Gripper PD controller below
	def Gripper_PD_controller(self,des,current,last):
		kp=10
		kd=2
		pos = des+kp*(des-current)-kd*(current-last)
		return pos


	#scales the PD signal based on the ee pos or joint values; wombat_arm needs
	#different PD values depending on where it is, position-wise
	def PD_signal_scale(self,ee_pos,joint_vals):
		ee_xy_disp=np.array([math.sqrt(ee_pos[0]**2+ee_pos[1]**2)]*6)+1.0
		lin_vals=np.array([joint_vals[2],joint_vals[0],joint_vals[1]]*2)+1.0
		scale=7
		PD_scale_factor=((np.multiply(ee_xy_disp,lin_vals)**2)-1)*scale
		
		return PD_scale_factor

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
		# self.iphonebox = BoxObject(name="iphonebox",size=[0.035,0.07,0.015],rgba=[0,0,0,1],friction=[1,1,5]).get_obj()
		##iphone 12 pro max dimensions
		self.iphonebox = BoxObject(name="iphonebox",size=[0.039,0.08,0.0037],rgba=[0,0,0,1],friction=[1,1,5]).get_obj()
		##iphone xr dimensions
		# self.iphonebox = BoxObject(name="iphonebox",size=[0.03785,0.07545,0.00415],rgba=[0,0,0,1],friction=[1,1,5]).get_obj()
		##iphone se/5/5s dimensions
		# self.iphonebox = BoxObject(name="iphonebox",size=[0.0293,0.0619,0.0038],rgba=[0,0,0,1],friction=[1,1,5]).get_obj()
		self.iphonebox.set('pos', '{} 1.35 1'.format(self.phone_x)) #0.75
		self.iphonebox.set('quat', '{} 0 0 1'.format(self.phone_orient)) #0
		self.world.worldbody.append(self.iphonebox)


		##Apple watch dimensions
		# self.watch_x = self.phone_x
		# self.watch_orient = self.phone_orient
		# self.watchbox = BoxObject(name="iphonebox",size=[0.01665,0.0193,0.00525], rgba=[0,0,0,1],friction=[1,1,5]).get_obj()
		# self.watchbox.set('pos', '{} 1.35 1'.format(self.phone_x)) #0.75
		# self.watchbox.set('quat', '{} 0 0 1'.format(self.phone_orient)) #0
		# self.world.worldbody.append(self.watchbox)

		##Apple iPad 10.2 9th gen. dimensions
		# self.ipad_x = self.phone_x
		# self.ipad_orient = self.phone_orient
		# self.ipadbox = BoxObject(name="iphonebox",size=[0.08705,0.1253,0.00375], rgba=[0,0,0,1],friction=[1,1,5]).get_obj()
		# self.ipadbox.set('pos', '{} 1.35 1'.format(self.ipad_x)) #0.75
		# self.ipadbox.set('quat', '{} 0 0 1'.format(self.ipad_orient)) #0
		# self.world.worldbody.append(self.ipadbox)

		self.box = BoxObject(name="box",size=[0.35,9.7,0.37],rgba=[0.5,0.5,0.5,1],friction=[1,1,1]).get_obj()
		self.box.set('pos', '0.6 -2 0')
		self.world.worldbody.append(self.box)

		# self.iPhone_collision = MujocoXMLObject("/home/biorobotics-ms/catkin_ws/src/Wombat_robosuite/robosuite/models/assets/objects/iphone12promax.xml",name="iPhone12ProMaxObject")
		# self.iPhone_visual = MujocoXMLObject("/home/yashraghav/robosuite/robosuite/models/assets/objects/can-visual.xml",name="CanVisualObject")

		# self.iPhone_collision = iPhone12ProMaxObject(name="iPhone12ProMax")
		# self.iPhone_visual = iPhone12ProMaxVisualObject(name="iPhone12ProMaxVisual")

		# self.iPhone_collision = CanObject(name="Can")
		# self.iPhone_visual = CanVisualObject(name="CanVisual")

		# self.world.worldbody.append(self.iPhone_collision)
		# self.world.worldbody.append(self.iPhone_visual)
		# self.world.merge(self.iPhone_collision)
		# self.world.merge(self.iPhone_visual)

		# self.objects = [self.iPhone_collision,self.iPhone_visual]

		# for mujoco_obj in self.objects:
		# 	self.world.merge_assets(mujoco_obj)
		# 	self.world.worldbody.append(mujoco_obj.get_obj())
		
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


	def grip_signal(self,des_state,obs_last,obs_last2last):
		if des_state=='open':
			# left_finger_open = -0.287884
			# right_finger_open = -0.295456
			left_finger_open = -0.5#-0.6
			right_finger_open = 0.5#0.6

			grip_signal=[self.Gripper_PD_controller(left_finger_open,obs_last[26],obs_last2last[26]),
						 self.Gripper_PD_controller(right_finger_open,obs_last[27],obs_last2last[27])]
		if des_state=='close':
			# left_finger_close = 0.246598
			# right_finger_close = 0.241764
			left_finger_close = 0.5#0.5
			right_finger_close = -0.5#-0.5

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

	def step(self,action):
		# action = action.reshape(1,-1)
		if self.observation_last is not None:
			# ipdb.set_trace()
			self.observation_last2last = self.observation_last
		else:
			self.observation_last2last = np.zeros(self.obs_dim)


		self.observation_last = self.observation_current['observation']
		
		PD_signal = self.get_signal(action,self.observation_last,self.observation_last2last)

		self.sim.data.ctrl[0:6] = PD_signal[0:6]
		if action[6] > 0:
			des_state='close'
		else:
			des_state='open'
		self.sim.data.ctrl[6:8] = self.grip_signal(des_state,self.observation_last,self.observation_last2last)
		self.clip_grip_action()
		
		self.sim.step()
		if self.is_render:
			# pass
			self.viewer.render()
		if self.is_render:
			self.viewer1 = MjViewer(self.sim)
			self.viewer1.vopt.geomgroup[0] = 0 # disable visualization of collision mesh
			self.viewer1.render()
		# print("sending steps")
		self.observation_current = self.get_observation()
		self.reward = self.compute_reward(self.observation_current['observation'],self.observation_current['observation'],None)
		self.sim.data.set_joint_qvel('box_joint0', [0, self.phone_speed, 0, 0, 0, 0])
		# self.sim.data.set_joint_qpos('robot0_base_left_torque_joint',0)
		# self.sim.data.set_joint_qpos('robot0_base_right_torque_joint',0)
		self.done = self.is_done
		
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
		observation[19] = observation[19] + 0.02
		
		observation[26] = self.sim.data.get_joint_qpos('robot0_base_left_short_joint')
		observation[27] = self.sim.data.get_joint_qpos('robot0_base_right_short_joint')
		observation[28:34] = self.sim.data.get_joint_qvel('iphonebox_joint0')
		# print(self.sim.data.get_joint_qpos('robot0_base_left_torque_joint'))
		# print(self.sim.data.get_joint_qpos('robot0_base_right_torque_joint'))
		goal = 0.88#0.83		#z value of the phone should be at 0.83 m from floor
		achieved_goal = observation[14] #Current z value of the iPhone
		observation = {'observation':observation,'desired_goal':np.array([goal]),'achieved_goal': np.array([achieved_goal])}
		
		return observation

	def reset(self,phone_x=0.78,phone_speed=-0.2,phone_orient=0):
		# ipdb.set_trace()
		obs = self.set_env(phone_x,phone_speed,phone_orient)
		return obs

	def clip_grip_action(self):
		self.j_pos = 0.5
		if self.sim.data.get_joint_qpos('robot0_base_left_short_joint')>self.j_pos:
			self.sim.data.set_joint_qpos('robot0_base_left_short_joint', self.j_pos)
			# print(1)
		if self.sim.data.get_joint_qpos('robot0_base_right_short_joint')>self.j_pos:
			self.sim.data.set_joint_qpos('robot0_base_right_short_joint', self.j_pos)
			# print(2)
		if self.sim.data.get_joint_qpos('robot0_base_left_short_joint')<-self.j_pos:
			self.sim.data.set_joint_qpos('robot0_base_left_short_joint', -self.j_pos)
			# print(3)
		if self.sim.data.get_joint_qpos('robot0_base_right_short_joint')<-self.j_pos:
			self.sim.data.set_joint_qpos('robot0_base_right_short_joint', -self.j_pos)
			# print(4)

		if self.sim.data.get_joint_qpos('robot0_base_left_torque_joint')>0.1:
			self.sim.data.set_joint_qpos('robot0_base_left_torque_joint', 0.1)
		if self.sim.data.get_joint_qpos('robot0_base_right_torque_joint')>0.1:
			self.sim.data.set_joint_qpos('robot0_base_right_torque_joint', 0.1)
		if self.sim.data.get_joint_qpos('robot0_base_left_torque_joint')<-0.1:
			self.sim.data.set_joint_qpos('robot0_base_left_torque_joint', -0.1)
		if self.sim.data.get_joint_qpos('robot0_base_right_torque_joint')<-0.1:
			self.sim.data.set_joint_qpos('robot0_base_right_torque_joint', -0.1)

	def clip_grip_vel(self):
		
		self.sim.data.set_joint_qvel('robot0_base_left_short_joint', 0.005)
		self.sim.data.set_joint_qvel('robot0_base_right_short_joint', -0.005)
		# print("left",self.sim.data.get_joint_qvel('robot0_base_left_short_joint'))
		# print("right",self.sim.data.get_joint_qvel('robot0_base_right_short_joint'))
		

	def compute_reward(self,obs,obs2,obs3):
		##modified reward function
		reward_grasp = []
		for i in range(obs.shape[0]):
			if obs[14]>=0.88 and (obs[21]-obs[14] <= 0.12) and (abs(obs[19]-obs[12]) <= 0.04) and obs[26] > -0.29:
				reward_grasp.append(50)
				# print("Reward: +50")
			elif obs[14] >= 0.83 and obs[26] > -0.29 and (obs[21]-obs[14] <= 0.18):
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

def dyn_rand():
	phone_x = 0.578#0.578#np.random.uniform(0.428, 0.728)
	phone_speed = -0.22#np.random.uniform(-0.14, -0.18)
	phone_orient = 0.0
	# phone_orient = np.random.uniform(-0.05, 0.05)
	return phone_x, phone_speed, phone_orient

env = D3_pick_place_env(is_render=False)
obs = env.reset()

Partial_success = 0
Full_success = 0
for i in range(1):
	phone_x, phone_speed, phone_orient = dyn_rand()
	# reset the environment
	observation = env.reset(phone_x, phone_speed, phone_orient)
	obs = observation['observation']
	g = observation['desired_goal']#+ np.array([0.0,0.,0.03])
	obs_current = obs.copy()#obs['observation']
	ag = observation['achieved_goal']
	action_zero = np.array([0,0,0.6,0,0,0,-0.5,0.5])#-0.6,0.6])
	obs_current = np.zeros(34)
	obs_last = obs_current.copy()
	pp_snatch = 0
	pp_grip = 0
	pos_y = 0
	pos_x = 0
	time_reset = 50
	time_stay = 100
	##to plot
	observation_x = np.zeros(11000)
	t_arr = np.linspace(0,11000,11000)
	for t in range(env._max_episode_steps):
		obs,reward,done,_ = env.step(action_zero)
		if t>=0 and t<1700 and pp_snatch==0:
			action_network = np.zeros(3)
			action_zero[1] += 0.0002 #motion happening here
			if action_zero[1]>0.05:
				action_zero[1]=0.05
			if np.linalg.norm(obs_current[19]-obs_current[12])>0.001:
				pos_x_dir = (obs_current[19]-obs_current[12])/np.abs(obs_current[19]-obs_current[12])
				pos_x += 0.0001*pos_x_dir  #motion happening here
				# print("x",obs_current[26])
				# print(obs_current[27])
			action_zero[0] = pos_x
			obs,reward,done,_ = env.step(action_zero)
			obs_current = obs['observation'] 
			# print("Stage 1 ",pp)
		### Calculate variables to pick ###
		elif t>=1700 and t<1800 and pp_snatch==0:
			obs,reward,done,_ = env.step(action_zero) 
			obs_current = obs['observation']
			vel_iPhone = (obs_current[13] - obs_last[13])
			steps_to_reach = ((0.0 - obs_current[13])/vel_iPhone)
			vel_iPhone_rt = (obs_current[13] - obs_last[13])/(0.002) #rt ==> real_time
			# print("Stage 2 ",pp)
		### calculate target velocity ###
		elif t == 1800 and pp_snatch==0:
			vel_z = (obs_current[21] - 0.885)/int(steps_to_reach)
			vel_y = (obs_current[20] - 0.0)/int(steps_to_reach)
			# print("Stage 3 ",pp)
		### Start snatch motion ###
		elif np.linalg.norm(obs_current[20]-obs_current[13])<0.001 or pp_snatch == 1:
			# print("Stage 4 ",pp)
			# print("obs_26",obs_current[26])
			# print("obs_27",obs_current[27])
			if action_zero[2]>=0.76:
				action_zero[2]=0.76
			obs,reward,done,_ = env.step(action_zero)
			obs_current = obs['observation']
			#residual trajectory added here
			# print("action_zero just before Stage 4", action_zero)
			action_zero[1]+= (vel_iPhone_rt)*0.002 #adding del_y to motion
			action_zero[2]+= vel_z*10#adding del_z to motion
			# if np.linalg.norm(obs_current[21]-0.838)<0.001 and pp_snatch == 1 and obs_current[26]< -0.29: #855####885
			# 	print("stay!!")
			# 	# print("obs",obs_current[26])
			# 	action_zero[2] = pos_z
			# 	action_zero[6] = 0.5
			# 	action_zero[7] = -0.5
								
			# elif obs_current[26]> -0.29 and action_zero[2]>0.55:
			# 	print("go up!! ship is sinking")
			# 	time_reset-=1
			# 	if time_reset<=0:
			# 		pos_z -= 0.0005 #motion happening here
			# 		action_zero[2] = pos_z
			# 		action_zero[6] = 0.5
			# 		action_zero[7] = -0.5
								
			# elif action_zero[2]<0.55:
			# 	# print("breaking up of the loop")
			# 	break
			# pos_z = action_zero[2]
			# pp_snatch =1

			##modified snatching motion
			if np.linalg.norm(obs_current[21]-0.838)<0.001 and pp_snatch == 1 and pp_grip == 0: #855####885
				print("stay!!")
				# print("obs",obs_current[26])
				action_zero[2] = pos_z
				action_zero[6] = 0.5
				action_zero[7] = -0.5
				time_stay-=1
				if time_stay<=0:
					pp_grip=1
								
			elif pp_grip==1 and action_zero[2]>0.55:
				print("go up!! ship is sinking")
				time_reset-=1
				if time_reset<=0:
					pos_z -= 0.0005 #motion happening here
					action_zero[2] = pos_z
					action_zero[6] = 0.5
					action_zero[7] = -0.5
				if time_reset<=-100:
					env.clip_grip_vel()
				print(time_reset)				
			elif action_zero[2]<0.55:
				# print("breaking up of the loop")
				break
			pos_z = action_zero[2]
			pp_snatch =1
		# re-assign the observation
		else:
			# print("no snatching")
			if action_zero[2]>=0.76:
				action_zero[2]=0.76
			obs,reward,done,_ = env.step(action_zero)
			obs_current = obs['observation']
			
		obs_last = obs_current.copy()
			
						
print("Expected goal: ",g)
print("Actual position: ",obs['achieved_goal'])
reward = env.compute_reward(obs['observation'],obs['observation'],None)
print("Reward is: ",reward)
if reward[1]==5:
	Partial_success = Partial_success + 1
if reward[1]==50:
	Full_success = Full_success + 1
print("Run no.: {}, Partial_successes: {}, Full_successes: {}".format(i+1,Partial_success,Full_success))
