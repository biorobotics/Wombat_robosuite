import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, critic
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
from torch.utils.tensorboard import SummaryWriter
import cv2
import time
from time import localtime, strftime
import ipdb
import imageio
import transforms3d as t3d
import matplotlib.pyplot as plt


from wombat_dmp.srv import *
import rospy
from geometry_msgs.msg import Pose, Vector3 
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R
from ur_ikfast import ur_kinematics 

ur3e_arm = ur_kinematics.URKinematics('ur3e')

"""
ddpg with HER (MPI-version)
"""
class grasping_agent:
	def __init__(self, args, env, env_params):
		self.args = args
		self.env = env
		print("self env,: ",self.env)
		self.env_params = env_params
		# create the network
		self.actor_network = actor(env_params)
		self.critic_network = critic(env_params)

		# sync the networks across the cpus
		sync_networks(self.actor_network)
		sync_networks(self.critic_network)
		
		# build up the target network
		self.actor_target_network = actor(env_params)
		self.critic_target_network = critic(env_params)
		
		# load the weights into the target networks
		self.actor_target_network.load_state_dict(self.actor_network.state_dict())
		self.critic_target_network.load_state_dict(self.critic_network.state_dict())
		
		# if use gpu
		if self.args.cuda:
			self.actor_network.cuda()
			self.critic_network.cuda()
			self.actor_target_network.cuda()
			self.critic_target_network.cuda()
		
		# create the optimizer
		self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
		self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
		
		# her sampler
		self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
		
		# create the replay buffer
		self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
		
		# create the normalizer
		self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
		self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
		
		# create the dict for store the model
		if MPI.COMM_WORLD.Get_rank() == 0:
			if not os.path.exists(self.args.save_dir):
				os.mkdir(self.args.save_dir)
			# path to save the model
			self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
			if not os.path.exists(self.model_path):
				os.mkdir(self.model_path)

		self.writer = SummaryWriter()
		self.write_counter = 0
		self.record_video = True
		#! timestep incosistent from 11000
		self.timesteps = 5500#4000

	def add_padding(self,transition,keys):

		# [mb_obs, mb_ag, mb_g, mb_actions]
		transition_new = []
		for idx,k in enumerate(transition):

			# ipdb.set_trace()
			if keys[idx] == 'obs' or keys[idx] == 'ag':
				t = np.zeros([k.shape[0],self.timesteps+1,k.shape[2]])
				# print("1st if", t)
			else:
				# print(f"k.shape {k.shape}, idx = {idx}")
				t = np.zeros([k.shape[0],self.timesteps,k.shape[2]])
				# print("2nd if", t)
			# print("k", k)
			t[:,:k.shape[1],:] = k
			transition_new.append(t)


		return transition_new
		# pass

	def dmp_client(self, start, goal, mode, phone_velocity):
		rospy.wait_for_service('dmp_calc_path')
		try:
			dmp_calc_path = rospy.ServiceProxy('dmp_calc_path', dmpPath)
			resp1 = dmp_calc_path(start,goal, mode, phone_velocity)
			return resp1.traj
		except rospy.ServiceException as e:
			print("Service call failed: %s"%e)
	def quat_to_euler(self, quat):
		r_quat = R.from_quat([quat.x,quat.y,quat.z,quat.w])
		e_angles = r_quat.as_euler('zyx', degrees=False)
		return e_angles

	def learn(self):
		"""
		train the network
		"""
		# start to collect samples
		observation_new_ = None
		Partial_success = 0
		Full_success = 0
		Total_episodes = 0
		for epoch in range(self.args.n_epochs):
			for n_cycles in range(self.args.n_cycles):
				mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
				break_flag = False
				for r_mpi in range(self.args.num_rollouts_per_mpi):
					
					#TODO: Set up the flags for baseline loop
					#Pre Reach 
					# This stage will move the robot so that the orientation and pos_x 
					# of the robot matches the phone


					#Pre Grasp 
					#If the phone has entered -> init the pick dmp motion
					# till DMP convergence 

					#RL for Grasping


					#Pick up on success 
					# reset the rollouts
					ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
					phone_x, phone_speed, phone_orient = self.dyn_rand()
					# reset the environment
					obs = self.env.reset(phone_x,phone_speed,phone_orient)
					obs = self.env.observation_current
					obs_current = obs.copy()
					ag = obs['achieved_goal']
					g = obs['desired_goal']
					# start to collect samples
					if MPI.COMM_WORLD.Get_rank() == 0:
						image_new = []
					# TODO: New action zero: calc from Fk tho...
					action_zero=np.array([1.31e-1, 3.915e-1, 2.05e-1, -3.14, 0,0,-0.4, 0.4])
					desired_joints =  np.array([-np.pi/2, -2.0, -np.pi/2, -1.01,  1.57, np.pi *0/180.0])
					prev_joint = desired_joints
					action_zero[:6] = desired_joints
					ee_pose_init = ur3e_arm.forward(desired_joints)
					ee_pose = ee_pose_init
					pick_ht = 0.12
					obs_current = np.zeros(34)
					obs_last = obs_current.copy()
					pp_snatch = 0 # pp =gripping flag
					pp_grip = 0
					pos_y = 0
					pos_x = 0
					time_reset = 50
					time_stay = 100
					T_sim_real = np.eye(4)
					T_real_actual = np.eye(4)
					T_sim_actual = np.eye(4)
					t_real = [0]*6
					max_time_steps = self.env_params['max_timesteps']
					observation_x = np.zeros(max_time_steps+1)
					observation_y = np.zeros(max_time_steps+1)
					observation_z = np.zeros(max_time_steps+1)
					t_arr = np.linspace(0,max_time_steps,max_time_steps+1)
					# R_ie = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
					if n_cycles == self.args.n_cycles - 1 and MPI.COMM_WORLD.Get_rank() == 0 and r_mpi == 1 and self.record_video:
						video_writer = imageio.get_writer('saved_models/New_Gripping_policy_dyn_rand/epoch_{}.mp4'.format(epoch+1), fps=60)
					
					plan_flag= True
					wait_flag = False
					path_executed = False
					done = False
					time.sleep(0.1)

					for t in range(self.env_params['max_timesteps']):

						# try:
						# 	obs,reward,done = self.env.step(action_zero)
						# 	# print(f"action_zero {action_zero}")
						# except:
						# 	break_flag = True
						# 	break
						obs,reward,done = self.env.step(action_zero)
						obs_current = obs['observation']
						# print(f"Observation1 from self.env {obs_current[1]}, timestep {t}")
						#Pre Reach 
						# This stage will move the robot so that the orientation and pos_x 
						# of the robot matches the phone


						#Pre Grasp 
						#If the phone has entered -> init the pick dmp motion
						# till DMP convergence 
						vel_iPhone_rt = (obs_current[12] - obs_last[12])/(0.002) #rt ==> real_time
						# flags
						pre_grasp_pos = 0.5
						proximal_tol = 0.1


						# gripper_pos = obs_current[19:26] 
						# phone_pos = obs_current[12:15] 
						gripper_pos = ee_pose_init
						phone_pos = obs_current[12:15] 
						# delta = gripper_pos[0:3] - action_zero[0:3]

						# if phone_pos[1]>pre_grasp_pos:
						# 	try:
						# 		obs,reward,done= self.env.step(action_zero)
						# 		# print(f"action_zero {action_zero}")

						# 	except:
						# 		break
						# 	obs_current = obs['observation'] 
						# print(phone_pos[0])
						if phone_pos[0]<pre_grasp_pos:
							if(done):
								print("episode done")
							# print(f"stage0")
							# print(f"demo 2 action_zero {action_zero}")
							obs,reward,done= self.env.step(action_zero)
							obs_current = obs['observation'] 
							# print("pp_snatch",pp_snatch)
							# print(obs_current[20])
							# print(obs_current[13])


						if (phone_pos[0]>=pre_grasp_pos and path_executed==False):
							# print(f"stage1: DMP")

							# Init DMP traj 
							phone_pos = obs_current[12:15]


							# Init DMP traj 
							if(plan_flag):          
								start= ee_pose[0:3]
								start_pose = Pose()

								start_pose.position.x = start[0]
								start_pose.position.y = start[1]
								start_pose.position.z = start[2]
								start_pose.orientation.x = 0
								start_pose.orientation.y = 0
								start_pose.orientation.z = 0
								start_pose.orientation.w = 1


								goal = np.zeros(3)
								goal[0] = phone_pos[0] - 0.5
								goal[1] = phone_pos[1] 
								goal[2] = 0.09#phone_pos[2]        
								end_pose = Pose()
								end_pose.position.x = goal[0] 
								end_pose.position.y = goal[1] 
								end_pose.position.z = goal[2]
								end_pose.orientation.x = 0
								end_pose.orientation.y = 0
								end_pose.orientation.z = 0
								end_pose.orientation.w =1
								mode = String()
								mode.data = "pick"
								vel_iPhone_rt = (obs_current[12] - obs_last[12])/(0.002) #rt ==> real_time

								phone_velo = Vector3()
								phone_velo.x = vel_iPhone_rt
								phone_velo.y = 0
								phone_velo.z = 0

								traj =self.dmp_client(start_pose, end_pose, mode, phone_velo)
								desired_traj = np.zeros((len(traj), 6))
								for i in range(len(traj)):
									desired_traj[i][0] =traj[i].position.x 
									desired_traj[i][1] =traj[i].position.y 
									desired_traj[i][2] =traj[i].position.z
									desired_traj[i][5] =self.quat_to_euler(traj[i].orientation)[0]
									desired_traj[i][4] =self.quat_to_euler(traj[i].orientation)[1]
									desired_traj[i][3] =self.quat_to_euler(traj[i].orientation)[2]
								traj_index = 0
								plan_flag = False

							# ! -> Execute this traj

							# Calculate current position in the trajectory : i^th index 
							path_executed = traj_index==desired_traj.shape[0]
							action_network=np.zeros(6)
							
							if(path_executed==True):
								print(f"Path Executed {path_executed}")
							if(path_executed==False):
								# action_zero[0:3] = desired_traj[traj_index, 0:3]
								ee_pose[0:3] = desired_traj[traj_index, 0:3]
								ee_pose[2] = np.clip(ee_pose[2], a_min=pick_ht, a_max=0.3)#desired_traj[traj_index, 0:3]
								desired_joint = ur3e_arm.inverse(ee_pose, q_guess = prev_joint)
								if(desired_joint is None):
									desired_joint = prev_joint
								prev_joint = desired_joint
								action_zero[:6] = desired_joint
								

								
								# obs,reward,done= env.step(action_zero)
								obs_current = obs['observation'] 
								traj_index+=1
							
							if(True):

								# print(f"executing agent actions")
								##RL for Grasping
								with torch.no_grad():
									input_tensor = self._preproc_inputs(obs_current, g)
									pi = self.actor_network(input_tensor)
									action_network = self._select_actions(pi)
								action_zero[0]+= action_network[0] #adding del to joint1
								action_zero[1]+=  action_network[1] #adding del to joint2
								action_zero[2]+=  action_network[2]  #adding del to joint3
								action_zero[3]+= action_network[3] #adding del to joint4
								action_zero[4]+=  action_network[4] #adding del to joint5
								action_zero[5]+=  action_network[5]  #adding del to joint6
								# Instead these should be a boolean value for the signal to grasp  
								# action_zero[6] += action_network[3]
								# action_zero[7] += action_network[4]
							
							
							# # clip the action if its touching the conveyor
							# if action_zero[2]>=0.763:
							# 	action_zero[2]=0.763								
								
							obs,reward,done = self.env.step(action_zero)
							# try:
							# 	obs,reward,done = self.env.step(action_zero)
							# 	# print(f"action_zero {action_zero}")

							# except:
							# 	print(f"Passing !!!!")
							# 	_ = self.env.reset()
							# 	break
							# traj_index+=1

						## Start snatch motion ###
						elif path_executed or np.linalg.norm(obs_current[19]-obs_current[12])<0.001 or pp_snatch == 1:
							if(wait_flag==False):
								completion_time = t
								wait_flag =True
							
							obs,reward,done = self.env.step(action_zero)
							obs_current = obs['observation']
							

							resume_flag=True#False
							
							if path_executed and pp_snatch == 1 and pp_grip == 0 and resume_flag:
								# print(f"stage 3 Grip")
								# print("stay!!")
								# action_zero[2] = pos_z
								action_zero[6] = 0.5
								action_zero[7] = -0.5
								desired_joint = ur3e_arm.inverse(ee_pose, q_guess = prev_joint)
								if(desired_joint is None):
									desired_joint = prev_joint
								prev_joint = desired_joint
								action_zero[:6] = desired_joint

								time_stay-=1
								if time_stay<=0:
									pp_grip=1
								

							elif pp_grip==1 and ee_pose[2]<=0.3:
								# print("go up!! ship is sinking")
								last_time = t
								time_reset-=1
								if time_reset<=0:
									ee_pose[2] += 0.0005 #motion happening here

									desired_joint = ur3e_arm.inverse(ee_pose, q_guess = prev_joint)
									if(desired_joint is None):
										desired_joint = prev_joint
									prev_joint = desired_joint
									action_zero[:6] = desired_joint
					
									action_zero[6] = 0.5
									action_zero[7] = -0.5

								# if time_reset<=-100:
								# 	self.env.clip_grip_vel()
								

							elif done or phone_pos[0]>0.9:
								print("breaking up of the loop")
								break
							# print("phone_pos[0]",phone_pos[0])
							# print("phone_pos[2]",phone_pos[2])
							pp_snatch =1

							ep_obs.append(obs['observation'].copy())
							ep_ag.append(obs['achieved_goal'].copy())
							ep_g.append(obs['desired_goal'].copy())
							ep_actions.append(action_network.copy())

						# print(f"ep_ag {ep_ag}")
						obs_last = obs_current.copy()
						ag = obs['achieved_goal']
						
						#write a video
						if n_cycles == self.args.n_cycles - 1 and MPI.COMM_WORLD.Get_rank() == 0 and r_mpi == 1 and self.record_video:
							video_img = self.env.sim.render(height=512, width=512, camera_name='robot0_camera_1', mode='offscreen')[::-1]
							video_writer.append_data(video_img)
							if t%1000==0:
								print("Video_making & t:",t)
						
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
					# plt.plot(t_arr, observation_x)
					# plt.title('End-effector x-coordinate vs. time')
					# plt.xlabel('time(in seconds)')
					# plt.ylabel('End-effector x-coordinate(in meter)')
					# plt.show()
					# plt.plot(t_arr, observation_y)
					# plt.title('End-effector y-coordinate vs. time')
					# plt.xlabel('time(in seconds)')
					# plt.ylabel('End-effector y-coordinate(in meter)')
					# plt.show()
					# plt.plot(t_arr, observation_z)
					# plt.title('End-effector z-coordinate vs. time')
					# plt.xlabel('time(in seconds)')
					# plt.ylabel('End-effector z-coordinate(in meter)')
					# plt.show()
					# self.env.close_window()
					if(self.env.is_render):
						self.env.close_window()

					
					Total_episodes = Total_episodes +1
					if reward[1] == 5:
						Partial_success = Partial_success + 1
					if reward[1] == 50:
						Full_success = Full_success + 1
					if n_cycles == self.args.n_cycles - 1 and MPI.COMM_WORLD.Get_rank() == 0 and r_mpi == 1 and self.record_video:
						video_writer.close()
						print("video done!")
						# time.sleep(100)
					ep_obs.append(obs_last.copy())
					ep_ag.append(ag.copy())
					mb_obs.append(ep_obs)
					mb_ag.append(ep_ag)
					mb_g.append(ep_g)
					mb_actions.append(ep_actions)
					print(f"final timestep {t}")
					print("epoch: {}, cycles: {}, r_mpi: {}, Partial_success: {}, Full_success: {}, Episodes: {}".format(epoch,n_cycles,r_mpi,Partial_success,Full_success,Total_episodes))
					current_time = strftime("%Y-%m-%d-%H-%M-%S", localtime())
					

				# convert them into arrays
				if(break_flag):
					break
				array_l = [len(mb_obs[0]),len(mb_obs[1])]
				min_l = min(array_l)
				for r_mpi in range(self.args.num_rollouts_per_mpi):
					#done to equal the array out for concatenate
					l = len(mb_obs[r_mpi])
					mb_obs[r_mpi] = mb_obs[r_mpi][l-min_l:]	
					mb_ag[r_mpi] = mb_ag[r_mpi][l-min_l:]	
					mb_g[r_mpi] = mb_g[r_mpi][l-min_l:]	
					mb_actions[r_mpi] = mb_actions[r_mpi][l-min_l:]	
					

				
				mb_obs = np.array(mb_obs)
				mb_ag = np.array(mb_ag)
				mb_g = np.array(mb_g)
				mb_actions = np.array(mb_actions)
				# print(f"mb_actions shape{mb_actions.shape}")
				self.write_counter += 1

				mb_obs, mb_ag, mb_g, mb_actions = self.add_padding([mb_obs, mb_ag, mb_g, mb_actions],['obs','ag','g','action'])
				
				# store the episodes
				self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
				self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
				for _ in range(self.args.n_batches):
					# train the network
					self._update_network()
				# soft update
				self._soft_update_target_network(self.actor_target_network, self.actor_network)
				self._soft_update_target_network(self.critic_target_network, self.critic_network)
			# start to do the evaluation
			# success_rate = self._eval_agent()
			if MPI.COMM_WORLD.Get_rank() == 0:
				# print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
				# self.writer.add_scalar(success_rate,epoch)
				torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
							self.model_path + '/new_grasping_policy_epoch_{}.pt'.format(epoch+1))

	# pre_process the inputs
	def _preproc_inputs(self, obs, g):
		obs_norm = self.o_norm.normalize(obs)
		g_norm = self.g_norm.normalize(g)
		# concatenate the stuffs
		inputs = np.concatenate([obs_norm, g_norm])
		inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
		if self.args.cuda:
			inputs = inputs.cuda()
		return inputs
	
	def _make_video(self,video_buffer,epoch):
		# ipdb.set_trace()
		height, width, layers = video_buffer[0].shape
		print("Size of video buffer: ",len(video_buffer))
		size = (width,height)
		 
		out = cv2.VideoWriter(os.path.join(self.model_path,'trial_{}.avi'.format(epoch)),cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
		 
		for i in range(len(video_buffer)):
			im_rgb = cv2.cvtColor(video_buffer[i], cv2.COLOR_BGR2RGB)
			out.write(im_rgb)
		out.release()

		print("Video done")



	# this function will choose action for the agent and do the exploration
	def _select_actions(self, pi):
		action = pi.cpu().numpy().squeeze()
		# add the gaussian
		action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
		action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
		# random actions...
		random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
											size=self.env_params['action'])
		# choose if use the random actions
		action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
		return action

	# update the normalizer
	def _update_normalizer(self, episode_batch):
		mb_obs, mb_ag, mb_g, mb_actions = episode_batch
		mb_obs_next = mb_obs[:, 1:, :]
		mb_ag_next = mb_ag[:, 1:, :]
		# get the number of normalization transitions
		num_transitions = mb_actions.shape[1]
		# create the new buffer to store them
		# print(f"mb_ag: {mb_ag}" )
		buffer_temp = {'obs': mb_obs, 
					   'ag': mb_ag,
					   'g': mb_g, 
					   'actions': mb_actions, 
					   'obs_next': mb_obs_next,
					   'ag_next': mb_ag_next,
					   }
		transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
		obs, g = transitions['obs'], transitions['g']
		# pre process the obs and g
		transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
		# update
		self.o_norm.update(transitions['obs'])
		self.g_norm.update(transitions['g'])
		# recompute the stats
		self.o_norm.recompute_stats()
		self.g_norm.recompute_stats()

	def _preproc_og(self, o, g):
		o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
		g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
		return o, g

	# soft update
	def _soft_update_target_network(self, target, source):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

	# update the network
	def _update_network(self):
		# print("Updating the network")
		# sample the episodes
		transitions = self.buffer.sample(self.args.batch_size)
		# pre-process the observation and goal
		o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
		transitions['obs'], transitions['g'] = self._preproc_og(o, g)
		transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
		# start to do the update
		obs_norm = self.o_norm.normalize(transitions['obs'])
		g_norm = self.g_norm.normalize(transitions['g'])
		inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
		obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
		g_next_norm = self.g_norm.normalize(transitions['g_next'])
		inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
		# transfer them into the tensor
		inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
		inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
		actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
		r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
		if self.args.cuda:
			inputs_norm_tensor = inputs_norm_tensor.cuda()
			inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
			actions_tensor = actions_tensor.cuda()
			r_tensor = r_tensor.cuda()
		# calculate the target Q value function
		with torch.no_grad():
			# do the normalization
			# concatenate the stuffs
			actions_next = self.actor_target_network(inputs_next_norm_tensor)
			q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
			q_next_value = q_next_value.detach()
			target_q_value = r_tensor + self.args.gamma * q_next_value
			target_q_value = target_q_value.detach()
			# clip the q value
			clip_return = 1 / (1 - self.args.gamma)
			target_q_value = torch.clamp(target_q_value, -clip_return, 0)
		# the q loss
		real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
		critic_loss = (target_q_value - real_q_value).pow(2).mean()
		# the actor loss
		# print("inputs_norm_tensor", torch.max(inputs_norm_tensor), torch.min(inputs_norm_tensor))
		actions_real = self.actor_network(inputs_norm_tensor)
		
		actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
		
		# actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
		# start to update the network
		self.actor_optim.zero_grad()
		actor_loss.backward()
		
		if MPI.COMM_WORLD.Get_rank() == 0:
			# print("Actor_loss: ", actor_loss)
			self.writer.add_scalar('actor_loss/train',actor_loss,self.write_counter)
			self.writer.add_scalar('critic_loss/train',critic_loss,self.write_counter)
		sync_grads(self.actor_network)
		self.actor_optim.step()
		# update the critic_network
		self.critic_optim.zero_grad()
		critic_loss.backward()
		sync_grads(self.critic_network)
		self.critic_optim.step()

	# do the evaluation
	def _eval_agent(self):
		total_success_rate = []
		for _ in range(self.args.n_test_rollouts):
			per_success_rate = []
			observation = self.env.reset()
			observation_,observation = self.env.robot_obs2obs(observation)
			obs = observation_['observation']
			g = observation_['desired_goal']
			for _ in range(self.env_params['max_timesteps']):
				with torch.no_grad():
					input_tensor = self._preproc_inputs(obs, g)
					pi = self.actor_network(input_tensor)
					# convert the actions
					actions = pi.detach().cpu().numpy().squeeze()
					action_robot = self.env.action2robot_cmd(actions,observation)

				observation_new, _, _, info = self.env.env.step(action_robot)
				observation_new_,observation_new = self.env.robot_obs2obs(observation_new)


				obs = observation_new_['observation']
				g = observation_new_['desired_goal']
		#         per_success_rate.append(info['is_success'])
		#     total_success_rate.append(per_success_rate)
		# total_success_rate = np.array(total_success_rate)
		# local_success_rate = np.mean(total_success_rate[:, -1])
		# global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
		global_success_rate = 0.5
		return True

	def dyn_rand(self):
		# phone_x = 0.578#np.random.uniform(0.428, 0.728)
		# phone_speed = -0.20#np.random.uniform(-0.14, -0.25)
		# phone_orient = 0.0
		phone_x = np.random.uniform(0.232, 0.532)#0.382 is taken as the middle point
		phone_speed = np.random.uniform(0.20, 0.35)
		phone_orient = 0.0



		# phone_orient = np.random.uniform(-0.05, 0.05)

		return phone_x, phone_speed, phone_orient