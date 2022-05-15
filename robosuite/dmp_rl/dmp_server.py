#!/usr/bin/env python3
"""
Ginesi et al. 2019, fig 3a
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import seaborn
import rospy
from dmp import dmp_cartesian
from dmp import obstacle_superquadric
import copy
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import plotly.express as px
from gen_traj import generate_traj

class DMP:
	def __init__(self,primitive=None,qprimitive=None,num_dmps=3,num_bfs=40,K_val=150,dtime=0.006,a_s=4.0,tolerance=8e-02,rescale='rotodilation', T= 1.5):
		#self.can_update=True
		self.num_line_points=50
		self.moving_goal = False
		if self.moving_goal:
			self.target_velocity = 0.2

		else:
			self.target_velocity = 0

		self.primitive=primitive
		self.qprimitive=qprimitive
		self.dt = dtime

		#cartesian DMP for xyz position
		self.dmp=dmp_cartesian.DMPs_cartesian(n_dmps=num_dmps,n_bfs=num_bfs,K=K_val,dt=dtime,alpha_s=a_s,tol=tolerance, rescale=rescale, T=T)	
		
		#quaternion DMP 
		#TODO: Plot this and see if the no of basis functions here are fine
		# self.qdmp.imitate_path(q_des=self.qprimitive)
		
		#list to hold all obstacles
		self.obstacles=[]
		self.dmp.reset_state()


	#calculates the dmp rollout, given a start and end. requires that
	#the dmp be trained beforehand
	def handle_dmp_path(self,start_pose, goal_pose, mode="supereclipse", phone_velocity = 0):
		#store obstacles. TODO: implement
		
		# Init the planner
		self.dmp.reset_state()
		# self.qdmp.reset_state()
		print("num obstacles is: ",len(self.obstacles))
		print("handling dmp path\n")
		start = start_pose
		goal = goal_pose
		# Recieve the start and goal position

		self.dmp.reset_state()
		# self.qdmp.reset_state()
		print("num obstacles is: ",len(self.obstacles))
		print("handling dmp path\n")
		# Recieve the start and goal position
		start=np.array([start_pose.position.x,
						start_pose.position.y,
						start_pose.position.z,
						start_pose.orientation.w,
						start_pose.orientation.x,
						start_pose.orientation.y,
						start_pose.orientation.z])
		goal=np.array([goal_pose.position.x,
						goal_pose.position.y,
						goal_pose.position.z,
						goal_pose.orientation.w,
						goal_pose.orientation.x,
						goal_pose.orientation.y,
						goal_pose.orientation.z])



		self.target_velocity = phone_velocity
		print(f"phone_velocity {self.target_velocity}")
		if(self.target_velocity ==0):
			self.moving_goal = False







		print("got start and goal")
		print("start=",start)
		print("goal=",goal)
		print("mode =",mode)

		self.scaling_factor = 10
		start[0:3] = start[0:3]*self.scaling_factor
		goal[0:3] = goal[0:3]*self.scaling_factor
		self.target_velocity *=self.scaling_factor

		traj_desired = generate_traj(mode)

		traj_len = len(traj_desired.poses)
		self.primitive  = np.zeros((traj_len, 3))
		self.Qprimitive = np.zeros((traj_len, 4))
		print("Got the Primitive Trajectory")
		for i in range(len(traj_desired.poses)):
			self.primitive[i,:] = np.array([traj_desired.poses[i].position.x, traj_desired.poses[i].position.y,
							traj_desired.poses[i].position.z])
			self.Qprimitive[i,:] = np.array([traj_desired.poses[i].orientation.w, 
										traj_desired.poses[i].orientation.x, 
										traj_desired.poses[i].orientation.y, 
										traj_desired.poses[i].orientation.z])
		
		primitive_start = self.primitive[0,:]
		new_start = copy.deepcopy(start[0:3])
		shift = np.array(new_start - primitive_start)

		self.dmp.imitate_path(x_des=(self.primitive + shift))


		self.dmp.x_0=copy.deepcopy(start[0:3])
		self.dmp.x_goal=copy.deepcopy(goal[0:3])


		#calculate the path
		#Convergence Flags for cartesian and quaternion DMP

		flag=False
		qflag=False



		x_track_s=copy.deepcopy(start[0:3])
		dx_track_s=np.zeros(self.dmp.n_dmps)
		ddx_track_s=np.zeros(self.dmp.n_dmps)
		

		path=np.array([start[0:3]])
		dx_track=np.zeros((1,self.dmp.n_dmps))
		ddx_track=np.zeros((1,self.dmp.n_dmps))

		self.dmp.t=0
		# self.qdmp.t=0
		print("beginning to calculate path")
		
		
		while(not (flag)):
			
			x_track_s,dx_track_s,ddx_track_s=self.dmp.step(
				external_force=None,adapt=False)
			
			path=np.append(path,[x_track_s],axis=0)
			dx_track=np.append(dx_track,[dx_track_s],axis=0)
			ddx_track=np.append(ddx_track,[ddx_track_s],axis=0)
			
			if self.moving_goal:
				self.dmp.x_goal[0] = self.dmp.x_goal[0] + self.target_velocity*self.dt
				pass
			self.dmp.t+=1
			flag=(np.linalg.norm(x_track_s-self.dmp.x_goal)/np.linalg.norm(self.dmp.x_goal-self.dmp.x_0)<=self.dmp.tol)

		#convert the path to geometry_msgs/Pose array
		print("found path with length: ", path.shape[0])#. path is:\n")
		path /=self.scaling_factor
		geomPath=[Pose() for i in range(path.shape[0])]
		# traj = Path()
		x_traj = list(path[:,0])
		y_traj = list(path[:,1])
		z_traj = list(path[:,2])
		len_path_plot = list(range(len(x_traj)))


		for i in range(path.shape[0]):
			geomPath[i].position.x=path[i,0]
			geomPath[i].position.y=path[i,1]
			geomPath[i].position.z=path[i,2]
			# print("Z values, ", path[i,2])
			geomPath[i].orientation.w=1
			geomPath[i].orientation.x=0
			geomPath[i].orientation.y=0
			geomPath[i].orientation.z=0
		#remove restriction on updating objects
		#self.can_update=True
		
		return geomPath
