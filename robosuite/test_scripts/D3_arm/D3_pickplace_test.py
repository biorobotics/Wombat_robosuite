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
from mujoco_py import MjSim, MjViewer, const
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

	def __init__ (self,args=None,is_render=True):

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
		kp=20*scale#20*scale ####10
		kd=0.6#0.6 ####4
		qpos = des+kp*(des-current)-kd*(current-q_pos_last)
		# print(kp*(des-current))
		return qpos

	# return np.array(points)
	def PD_controller_lin(self,des,current,q_pos_last,scale):
		
		#kp = 10
		#kd = 0.8
		#kp=10
		#kd=0.1
		kp=150#800#150 ####1000
		kd=1500 ####1500
		qpos = des+kp*(des-current)-kd*(current-q_pos_last)
		# print(kp*(des-current))
		return qpos

	##Gripper PD controller below
	def Gripper_PD_controller(self,des,current,last):
		kp=100
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
		# self.iphonebox = BoxObject(name="iphonebox",size=[0.039,0.08,0.0037],rgba=[0,0,0,1],friction=[1,1,5]).get_obj()
		##iphone xr dimensions
		# self.iphonebox = BoxObject(name="iphonebox",size=[0.03785,0.07545,0.00415],rgba=[0,0,0,1],friction=[1,1,5]).get_obj()
		##iphone se/5/5s dimensions
		self.iphonebox = BoxObject(name="iphonebox",size=[0.0293,0.0619,0.0038],rgba=[0,0,0,1],friction=[1,1,5]).get_obj()
		self.iphonebox.set('pos', '{} 0 0.8'.format(self.phone_x)) #0.75
		self.iphonebox.set('quat', '{} 0 0 1'.format(self.phone_orient)) #0
		self.world.worldbody.append(self.iphonebox)


		##Apple watch dimensions
		# self.watch_x = self.phone_x
		# self.watch_orient = self.phone_orient
		# self.watchbox = BoxObject(name="iphonebox",size=[0.01665,0.0193,0.00525],rgba=[0,0,0,1],friction=[1,1,5]).get_obj()
		# self.watchbox.set('pos', '{} 1.35 1'.format(self.phone_x)) #0.75
		# self.watchbox.set('quat', '{} 0 0 1'.format(self.phone_orient)) #0
		# self.world.worldbody.append(self.watchbox)

		# self.box = BoxObject(name="box",size=[0.35,9.7,0.37],rgba=[0.5,0.5,0.5,1],friction=[1,1,1]).get_obj()
		# self.box.set('pos', '0.6 -2 0')
		# self.world.worldbody.append(self.box)

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
			left_finger_close = 1#0.5
			right_finger_close = -1#-0.5

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
		self.x = 1.0
		if self.sim.data.get_joint_qpos('robot0_base_left_short_joint')>self.x:
			self.sim.data.set_joint_qpos('robot0_base_left_short_joint', self.x)
		if self.sim.data.get_joint_qpos('robot0_base_right_short_joint')>self.x:
			self.sim.data.set_joint_qpos('robot0_base_right_short_joint', self.x)
		if self.sim.data.get_joint_qpos('robot0_base_left_short_joint')<-self.x:
			self.sim.data.set_joint_qpos('robot0_base_left_short_joint', -self.x)
		if self.sim.data.get_joint_qpos('robot0_base_right_short_joint')<-self.x:
			self.sim.data.set_joint_qpos('robot0_base_right_short_joint', -self.x)

		if self.sim.data.get_joint_qpos('robot0_base_left_torque_joint')>0.1:
			self.sim.data.set_joint_qpos('robot0_base_left_torque_joint', 0.1)
		if self.sim.data.get_joint_qpos('robot0_base_right_torque_joint')>0.1:
			self.sim.data.set_joint_qpos('robot0_base_right_torque_joint', 0.1)
		if self.sim.data.get_joint_qpos('robot0_base_left_torque_joint')<-0.1:
			self.sim.data.set_joint_qpos('robot0_base_left_torque_joint', -0.1)
		if self.sim.data.get_joint_qpos('robot0_base_right_torque_joint')<-0.1:
			self.sim.data.set_joint_qpos('robot0_base_right_torque_joint', -0.1)

	



env = D3_pick_place_env(is_render=False)
obs = env.reset()

t = 0
timestep=0.0005
t_final = 10000#468
sim_state = env.sim.get_state()
joint_angle_fdbk = np.zeros([t_final,6])
distance_ee = np.zeros([t_final,9])
extent = env.sim.model.stat.extent


q_pos_last = [0]*6
ee_pose_des = []
ee_pose_current = []
ee_pose_goal=[]
ee_pose_goal_maxlen=100
ee_pose_des_past=[]
ee_pose_des_past_maxlen=100
ee_pose_timeskip=90
dt=1.0/(t_final-t)
#construct target trajectory here. path must have t_final-t waypoints
# target_wp=np.array([[0.0,0.0,0.65,0,0,0],
# 					[0.15,0.25,0.50,0,0,0],
# 					[0.15,0.25,0.50,0,0,0]])
# #					[0.15,0.3,0.65,-0.4,0.1,0],
# #					[0.15,0.3,0.65,-0.4,0.1,0]])
# target_traj=interpolateTraj(target_wp,5001)

#target_traj=np.array([[0.0,0.0,0.65,0.0,0.0,0.0] for i in range(t,t_final)])

#z movement
# target_traj=np.array([[0,0.0,0.6+0.1-0.1*np.cos(i*2*np.pi*dt),0,0,0] for i in range(t,t_final)])

#y movement
# r=0.2
# target_traj=np.array([[0.0,(r/3)-(r/2)*np.cos(i*2*np.pi*dt),0.6,0,0,0] for i in range(t,t_final)])#has singularities

#x movement
# r = 0.1
# target_traj=np.array([[r*np.sin(i*2*np.pi*dt),0.0,0.6,0,0,0] for i in range(t,t_final)])

#in-between axisx+y movement
# r=0.1
# target_traj=np.array([[(-(r/2)+(r/2)*np.cos(i*2*np.pi*dt))*-1*np.cos(np.pi/6),
# 						(-(r/2)+(r/2)*np.cos(i*2*np.pi*dt))*-1*np.sin(np.pi/6),
# 						0.6,0,0,0] for i in range(t,t_final)])

#in-between axisz+y movement
# r=0.05
# target_traj=np.array([[(-r+r*np.cos(i*2*np.pi*dt))*-1*np.cos(5*np.pi/6),
# 						(-r+r*np.cos(i*2*np.pi*dt))*-1*np.sin(5*np.pi/6),
# 						0.6+0.05-0.05*np.cos(i*2*np.pi*dt),0,0,0] for i in range(t,t_final)])

#horizontal circle
r=0.3
tLin=200
dt=1.0/(t_final-t-tLin)
target_traj1=np.array([[0,r*np.sin(np.pi/2*(float(i)/tLin)),0.6,0,0,0] for i in range(0,tLin)])
target_traj2=np.array([[r*np.sin((i)*dt*np.pi*2),r*np.cos(i*dt*np.pi*2),0.6,0,0,0] for i in range(t,t_final-tLin)])
target_traj=np.block([[target_traj1],[target_traj2]])


#another trajectory(to run this change t_final to 468 since 93+104+74+104+93=468)
# target_wp_beg=np.array([[0.0,0.0,0.6,0,0,0],
# 					[0.0,0,0.6,0,0,0]])
# target_wp_asc=np.array([[0.0,0.0,0.6,0,0,0],
# 					[0.0,-0.2,0.6,0,0,0]])
# #					[0.15,0.3,0.65,-0.4,0.1,0],
# #					[0.15,0.3,0.65,-0.4,0.1,0]])
# target_wp_mid=np.array([[0.0,-0.2,0.6,0,0,0],
# 					[0.0,-0.2,0.6,0,0,0]])
# target_wp_dec=np.array([[0.0,-0.2,0.6,0,0,0],
# 					[0.0,0.0,0.6,0,0,0]])
# #					[0.15,0.3,0.65,-0.4,0.1,0],
# #					[0.15,0.3,0.65,-0.4,0.1,0]])
# target_wp_end=np.array([[0.0,0.0,0.6,0,0,0],
# 					[0.0,0.0,0.6,0,0,0]])
# target_traj1=interpolateTraj(target_wp_beg,93)
# target_traj2=interpolateTraj(target_wp_asc,104)
# target_traj3=interpolateTraj(target_wp_mid,74)
# target_traj4=interpolateTraj(target_wp_dec,104)
# target_traj5=interpolateTraj(target_wp_end,93)
# target_traj=np.block([[target_traj1],[target_traj2],[target_traj3],[target_traj4],[target_traj5]])

#vertical circle
# r=0.1
# dt=1.0/(t_final-t)
# target_traj=np.array([[r*np.sin(i*dt*np.pi*2),0.0,0.7-r*np.cos(i*dt*np.pi*2),0,0,0] for i in range(t,t_final)])


#trajectory for checking orientation
# r=0.5
# dt=1.0/(t_final-t)
# target_traj=np.array([[0,0.0,0.7,r*np.sin(i*dt*np.pi*2),-r*np.sin(i*dt*np.pi*2),r*np.cos(i*dt*np.pi*2)] for i in range(t,t_final)])
# target_traj=np.array([[0,0.0,0.7,0,0,r*np.sin(i*dt*np.pi*2)] for i in range(t,t_final)])

#moving in the shape of "8"
# r=0.05
# d=0.08255
# tLin=1000
# dt=1.0/(t_final-t-tLin)
# dz=0.1
# target_traj1=np.array([[0,-d*np.sin(np.pi/2*(float(i)/tLin)),0.6+dz*np.sin(np.pi/2*float(i)/tLin),0,0,0] for i in range(0,tLin)])
# target_traj2=np.array([[-r*np.sin(2*np.pi*dt*i),-r*np.sin(2*np.pi*dt*i)*np.cos(2*np.pi*dt*i)-d,0.6+dz,0,0,0]
# 					 for i in range(t,t_final-tLin)])
# # target_traj2=np.array([[-r*np.sin(2*np.pi*dt*i),-r*np.sin(2*np.pi*dt*i)*np.cos(2*np.pi*dt*i)-d,0.6+dz,0,0,0]
# # 					 for i in range(t,t_final-tLin)])

# target_traj=np.block([[target_traj1],[target_traj2]])

#convert to joint trajectory
joint_target_traj=trg.jointTraj(target_traj) #qu is the actual quaternion based on ik code



while True:
	
	env.sim.set_state(sim_state)
	ee_pose_des = []
	ee_pose_des_past = []
	ee_pose_current = []
	t = 0
	#joint values in the simulation
	j_actual=np.zeros((t_final-t,6))
	j_actual_real=np.zeros((t_final-t,6))
	#goal joint values for simulation to follow
	j_goal=np.zeros((t_final-t,6))
	#motor force/torque values
	motor_effort=np.zeros((t_final-t,6))
	t_arr=np.linspace(timestep*t,timestep*t_final,t_final-t)
	
	
	
	
	#ipdb.set_trace()
	
	while t<t_final:
		
		#rotary
		joint_angle_fdbk[t,0] = env.sim.data.get_joint_qpos("robot0_branch1_joint")
		joint_angle_fdbk[t,0] = env.sim.data.get_joint_qpos("robot0_branch2_joint")
		joint_angle_fdbk[t,0] = env.sim.data.get_joint_qpos("robot0_branch3_joint")
		#linear
		joint_angle_fdbk[t,0] = env.sim.data.get_joint_qpos("robot0_branch1_linear_joint")
		joint_angle_fdbk[t,0] = env.sim.data.get_joint_qpos("robot0_branch2_linear_joint")
		joint_angle_fdbk[t,0] = env.sim.data.get_joint_qpos("robot0_branch3_linear_joint")

		#rotary
		q_pos_last[0] = env.sim.data.get_joint_qpos("robot0_branch1_joint")
		q_pos_last[1] = env.sim.data.get_joint_qpos("robot0_branch2_joint")
		q_pos_last[2] = env.sim.data.get_joint_qpos("robot0_branch3_joint")
		#linear
		q_pos_last[3] = env.sim.data.get_joint_qpos("robot0_branch1_linear_joint")
		q_pos_last[4] = env.sim.data.get_joint_qpos("robot0_branch2_linear_joint")
		q_pos_last[5] = env.sim.data.get_joint_qpos("robot0_branch3_linear_joint")
		
		env.sim.step()
		
		if True:
			env.viewer.render()

		#current target joint values, in IK frame
		joint_real=joint_target_traj[t]
		# file = open("joints.txt" ,"a")
		# file.write(str(joint_real[0]) + " " + str(joint_real[1]) + " " + str(joint_real[2]) + " " + str(joint_real[3]) + " " + str(joint_real[4]) + " " + str(joint_real[5]) + '\n')
		# file.close()
		#print(joint_real)
		
		#target ee pos, converted to simulation frame
		#target_traj[t] = [0, 0, 0.6, 0, 0, 0]
		print("traj",target_traj[t])
		ee_pose=invK.real2sim_wrapper(target_traj[t])
		
		# (ji,vi)=invK.invK(target_traj[t])
		# ji=ji.flatten()
		# if vi==0:
		# 	print("IK resulted in joints that are invalid!\n")
			
		
		env.viewer.add_marker(pos=np.array(ee_pose[0:3]),
                      size=np.array([0.01,0.01,0.01]),
                      type=const.GEOM_SPHERE,
					  label="",
					  rgba=[0,0,1,1])
		
		ee_pose_des.append(ee_pose)
		#convert current target joint values, in sim frame
		joint_sim=invK.ik_wrapper(joint_real)
		j_goal[t,:]=np.array(joint_sim)
		
		#calculate/send PD control signal to the motors
		PD_scale=env.PD_signal_scale(target_traj[t],joint_target_traj[t])
		PD_signal=[env.PD_controller_rot(joint_sim[3],env.sim.data.get_joint_qpos("robot0_branch1_joint"),q_pos_last[0],PD_scale[0]),
				env.PD_controller_rot(joint_sim[4],env.sim.data.get_joint_qpos("robot0_branch2_joint"),q_pos_last[1],PD_scale[1]),
				env.PD_controller_rot(joint_sim[5],env.sim.data.get_joint_qpos("robot0_branch3_joint"),q_pos_last[2],PD_scale[2]),
				env.PD_controller_lin(joint_sim[0],env.sim.data.get_joint_qpos("robot0_branch1_linear_joint"),q_pos_last[3],PD_scale[3]),
				env.PD_controller_lin(joint_sim[1],env.sim.data.get_joint_qpos("robot0_branch2_linear_joint"),q_pos_last[4],PD_scale[4]),
				env.PD_controller_lin(joint_sim[2],env.sim.data.get_joint_qpos("robot0_branch3_linear_joint"),q_pos_last[5],PD_scale[5])]
		
		
		env.sim.data.ctrl[0]=PD_signal[0]
		env.sim.data.ctrl[1]=PD_signal[1]
		env.sim.data.ctrl[2]=PD_signal[2]
		env.sim.data.ctrl[3]=PD_signal[3]
		env.sim.data.ctrl[4]=PD_signal[4]
		env.sim.data.ctrl[5]=PD_signal[5]
		
		
		
		j_actual[t,:]=np.array([env.sim.data.get_joint_qpos("robot0_branch1_linear_joint"),
						 env.sim.data.get_joint_qpos("robot0_branch2_linear_joint"),
						 env.sim.data.get_joint_qpos("robot0_branch3_linear_joint"),
						 env.sim.data.get_joint_qpos("robot0_branch1_joint"),
						 env.sim.data.get_joint_qpos("robot0_branch2_joint"),
						 env.sim.data.get_joint_qpos("robot0_branch3_joint")])
		
		
		#ee position and orientation in simulation
		ee_current_xyz=copy.copy(env.sim.data.sensordata[9:12])
		ee_current=np.append(ee_current_xyz,
					copy.copy(env.sim.data.sensordata[15:18]))
		ee_pose_current.append(ee_current)
		
		#adding marker for visualization
		env.viewer.add_marker(pos=np.array(env.sim.data.sensordata[9:12]),
                      size=np.array([0.01,0.01,0.01]),
                      type=const.GEOM_SPHERE,
					  label="",
					  rgba=[0,1,0,1])
		
		#mark past positions with a marker
		if t%ee_pose_timeskip==0:
			if len(ee_pose_des_past)>=ee_pose_des_past_maxlen:
				ee_pose_des_past.pop(0)
			ee_pose_des_past.append(copy.copy(env.sim.data.sensordata[9:12]))
		for p in ee_pose_des_past:			
			env.viewer.add_marker(pos=np.array(p[0:3]),
                      size=np.array([0.01,0.01,0.01]),
                      type=const.GEOM_SPHERE,
					  label="",
					  rgba=[0,1,0,1])
	
		#mark past goal positions with a marker
		#mark past positions with a marker
		if t%ee_pose_timeskip==0:
			if len(ee_pose_goal)>=ee_pose_goal_maxlen:
				ee_pose_goal.pop(0)
			ee_pose_goal.append(copy.copy(ee_pose[0:3]))
		for p in ee_pose_goal:			
			env.viewer.add_marker(pos=np.array(p[0:3]),
                      size=np.array([0.01,0.01,0.01]),
                      type=const.GEOM_SPHERE,
					  label="",
					  rgba=[0,0,1,1])
		t += 1