import trajGenerator as trg
from interpolateTraj import interpolateTraj
import robosuite
from robosuite.models.objects import BoxObject
from robosuite.models.robots import Wombat_arm
from robosuite.models.arenas import EmptyArena
from robosuite.models.grippers import gripper_factory
import invK
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

	def __init__ (self,is_render,phone_speed,phone_x):

		self.action_dim = 6
		self.obs_dim = 28#26
		self.q_pos_last = np.zeros(self.action_dim)
		self.observation_current = np.zeros(self.obs_dim)
		self.observation_last = np.zeros(self.obs_dim)
		self.observation_last2last = np.zeros(self.obs_dim)
		self.is_render = is_render
		self.done = False
		self.phone_speed = phone_speed
		self.phone_x = phone_x
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

	def set_env(self):

		self.world = MujocoWorldBase()
		self.mujoco_robot = Wombat_arm()

		self.mujoco_robot.set_base_xpos([0, 0.0, 0])
		self.world.merge(self.mujoco_robot)

		self.mujoco_arena =EmptyArena()
		# mujoco_arena.set_origin([0.8, 0, 0])
		self.world.merge(self.mujoco_arena)

		self.iphonebox = BoxObject(name="iphonebox",size=[0.035,0.07,0.02],rgba=[0,0,0,1],friction=[1,1,5]).get_obj()
		self.iphonebox.set('pos', '{} -2 1'.format(self.phone_x)) #0.75
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

	def get_signal(self,action,obs_last,obs_last2last):


		# obs_last = obs_last_dict['observation']
		# obs_last2last = obs_last2last_dict['observation']

		action = action.reshape(1,-1)[:,:6]
		joint_target_action=trg.jointTraj(action)
		# q_pos_last[0:6] = obs_last2last[0:6] 
		joint_real=joint_target_action
		ee_pose=invK.real2sim_wrapper(action[0])
		self.joint_sim=invK.ik_wrapper(joint_real[0])
		j1 = np.array([self.sim.data.get_joint_qpos("robot0_branch1_joint"),self.sim.data.get_joint_qpos("robot0_branch2_joint"),self.sim.data.get_joint_qpos("robot0_branch3_joint")])
		j2 = np.array([self.joint_sim[3],self.joint_sim[4],self.joint_sim[5]])
		self.joint_sim[3:6] = self.nextClosestJointRad(j1,j2)
		PD_scale= self.PD_signal_scale(action[0],joint_real[0])
		ipdb.set_trace()

		PD_signal=[self.PD_controller_rot(self.joint_sim[3],obs_last[0],obs_last2last[0],PD_scale[0]),
			   self.PD_controller_rot(self.joint_sim[4],obs_last[1],obs_last2last[1],PD_scale[1]),
			   self.PD_controller_rot(self.joint_sim[5],obs_last[2],obs_last2last[2],PD_scale[2]),
			   self.PD_controller_lin(self.joint_sim[0],obs_last[3],obs_last2last[3],PD_scale[3]),
			   self.PD_controller_lin(self.joint_sim[1],obs_last[4],obs_last2last[4],PD_scale[4]),
			   self.PD_controller_lin(self.joint_sim[2],obs_last[5],obs_last2last[5],PD_scale[5])]


		return PD_signal

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

	def step(self,action):
		# action = action.reshape(1,-1)
		self.observation_last2last = self.observation_last
		self.observation_last = self.observation_current
		PD_signal = self.get_signal(action,self.observation_last['observation'],self.observation_last2last['observation'])
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
			self.viewer.render()
		# print("sending steps")
		self.observation_current = self.get_observation()
		self.reward = self.calculate_reward(self.observation_current)
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
		# print("gripper pose: ",self.sim.data.get_joint_qpos('robot0_gripper_base_link_joint'))
		return observation

		pass


	def calculate_reward(self,obs):

		return 10


	def is_done(self,obs):

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


	def run(self):



if __name__ == "__main__":

	D3_pp = D3_pick_place_env(True,conveyor_speed[0],pxl)
	D3_pp.set_env()
	obs_current = np.zeros(28)
	obs_last = np.zeros(28)
	gripper_to_finger = 0.09
