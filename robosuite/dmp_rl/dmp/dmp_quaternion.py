'''
Copyright (C) 2020 Michele Ginesi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.linalg
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import copy
import pdb

from dmp.cs import CanonicalSystem
from dmp.exponential_integration import exp_eul_step
from dmp.exponential_integration import phi1
from dmp.derivative_matrices import compute_D1, compute_D2
from wombat_kinematics import invK
from wombat_kinematics import forK
from wombat_kinematics import condition_no

# ---------------------------------------------------------------------------- #
# quaternion DMPs in Cartesian Space
# ---------------------------------------------------------------------------- #
#multiply two quaternions together
def quat_mul(q0,q1):
	q=np.zeros(4)
	q[0]=q0[0]*q1[0]-q0[1]*q1[1]-q0[2]*q1[2]-q0[3]*q1[3]
	q[1]=q0[0]*q1[1]+q0[1]*q1[0]+q0[2]*q1[3]-q0[3]*q1[2]
	q[2]=q0[0]*q1[2]-q0[1]*q1[3]+q0[2]*q1[0]+q0[3]*q1[1]
	q[3]=q0[0]*q1[3]+q0[1]*q1[2]-q0[2]*q1[1]+q0[3]*q1[0]
	return q
#take log of a quaternion
def quat_log(q):
	norm=np.linalg.norm(q[1:4])
	if norm<0.000001:
		return np.zeros(3)
	return np.arccos(q[0])*q[1:4]/norm

def quat_exp(q):
	if np.linalg.norm(q)<0.000001:
		return np.array([1,0,0,0])
	q_exp=np.zeros(4)
	imag_part=q[0:3]
	if len(q)>3:
		imag_part=q[1:4]
	inorm=np.linalg.norm(imag_part)
	q_exp[0]=np.cos(inorm)
	q_exp[1:4]=np.sin(inorm)*imag_part/inorm
	return q_exp

def log_R(R):
	acos_term=(np.trace(R)-1.0)/2.0
	theta=np.arccos(np.clip(acos_term,-1.0,1.0))
	# print(theta)
	if theta<0.000001:
		return np.zeros(3)
	nc=1.0/(2*np.sin(theta))
	nv=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
	n=nc*nv
	return theta*n

class DMPs_quaternion(object):
	'''
	Implementation of discrete Dynamic Movement Primitives in cartesian space,
	as described in
	[1] Park, D. H., Hoffmann, H., Pastor, P., & Schaal, S. (2008, December).
		Movement reproduction and obstacle avoidance with Dynamic movement
		primitives and potential fields.
		In Humanoid Robots, 2008. Humanoids 2008. 8th IEEE-RAS International
		Conference on (pp. 91-98). IEEE.
	[2] Hoffmann, H., Pastor, P., Park, D. H., & Schaal, S. (2009, May).
		Biologically-inspired dynamical systems for movement generation:
		automatic real-time goal adaptation and obstacle avoidance.
		In Robotics and Automation, 2009. ICRA'09. IEEE International
		Conference on (pp. 2587-2592). IEEE.

	Note: code modified for quaternion dmps, as described in 
	[3] Kramberger, A., Gams, A., Nemec, B., and Ude, A. (2016). 
	"Generalization of orientational motion in unit quaternion space",
	in IEEE-RAS International Conference on Humanoid Robots (Cancun),
	 808-813.
	'''

	def __init__(self,
		n_dmps = 1, n_bfs = 50, dt = 0.01, q_0 = None, q_goal = None, T = 1.0,
		K = 50, D = None, w = None, tol = 0.01, alpha_s = 4.0, basis = 'gaussian', **kwargs):
		'''
		n_dmps int	 : number of dynamic movement primitives (i.e. dimensions)
		n_bfs int	 : number of basis functions per DMP (actually, they will
					   be one more)
		dt float	 : timestep for simulation
		x_0 array	  : initial state of DMPs
		x_goal array   : x_goal state of DMPs
		T float		 : final time
		K float		 : elastic parameter in the dynamical system
		D float		 : damping parameter in the dynamical system
		w array		 : associated weights
		tol float	 : tolerance
		alpha_s float: constant of the Canonical System
		basis string : type of basis functions
		'''
		#surya singularity stuff
		self.sx=0.2
		self.sy=0.2
		self.sz=0.2
		#az=bz*4, bz=12-> bz*bz*4, bz*4
		# Tolerance for the accuracy of the movement: the trajectory will stop
		# when || x - g || <= tol
		self.tol = copy.deepcopy(tol)
		self.n_dmps = copy.deepcopy(n_dmps)
		self.n_bfs = copy.deepcopy(n_bfs)

		# Default values give as in [2]
		#TODO: find better default values for quaternion DMPs
		self.K = copy.deepcopy(K)
		if D is None:
			D = 2 * np.sqrt(self.K)
		self.D = copy.deepcopy(D)

		# Set up the CS
		self.cs = CanonicalSystem(dt = dt, run_time = T, alpha_s = alpha_s)

		# Create the matrix of the linear component of the problem
		#self.compute_linear_part()

		# Set up the DMP system. if not set, goal and start will be set to id
		if q_0 is None:
			#q_0 = np.zeros(self.n_dmps)
			q_0=np.array([1,0,0,0])
		if q_goal is None:
			#q_goal = np.ones(self.n_dmps)
			q_goal=np.array([1,0,0,0])
		self.q_0 = copy.deepcopy(q_0)
		self.q_goal = copy.deepcopy(q_goal)
		self.basis = copy.deepcopy(basis)
		self.reset_state()
		self.gen_centers()
		self.gen_width()

		# If no weights are give, set them to zero
		if w is None:
			#w = np.zeros([self.n_dmps, self.n_bfs + 1])
			w=np.zeros([3,self.n_bfs+1])
		self.w = copy.deepcopy(w)
	
	def compute_linear_part(self):
		'''
		Compute the linear component of the problem.
		'''
		#Note: not used. TODO: remove?
		self.linear_part = np.zeros([2 * self.n_dmps, 2 * self.n_dmps])
		self.linear_part\
			[range(0, 2 * self.n_dmps, 2), range(0, 2 * self.n_dmps, 2)] = \
				- self.D
		self.linear_part\
			[range(0, 2 * self.n_dmps, 2), range(1, 2 * self.n_dmps, 2)] = \
				- self.K
		self.linear_part\
			[range(1, 2 * self.n_dmps, 2), range(0, 2 * self.n_dmps, 2)] = 1.
	#no need to mod
	def gen_centers(self):
		'''
		Set the centres of the basis functions to be spaced evenly throughout
		run time
		'''
		# Desired activations throughout time
		self.c = np.exp(- self.cs.alpha_s * self.cs.run_time *
			((np.cumsum(np.ones([1, self.n_bfs + 1])) - 1) / self.n_bfs))
	#no need to mod
	def gen_psi(self, s):
		'''
		Generates the activity of the basis functions for a given canonical
		system rollout.
		 s : array containing the rollout of the canonical system
		'''
		c = np.reshape(self.c, [self.n_bfs + 1, 1])
		w = np.reshape(self.width, [self.n_bfs + 1,1 ])
		if (self.basis == 'gaussian'):
			xi = w * (s - c) * (s - c)
			psi_set = np.exp(- xi)
		else:
			xi = np.abs(w * (s - c))
			if (self.basis == 'mollifier'):
				psi_set = (np.exp(- 1.0 / (1.0 - xi * xi))) * (xi < 1.0)
			elif (self.basis == 'wendland2'):
				psi_set = ((1.0 - xi) ** 2.0) * (xi < 1.0)
			elif (self.basis == 'wendland3'):
				psi_set = ((1.0 - xi) ** 3.0) * (xi < 1.0)
			elif (self.basis == 'wendland4'):
				psi_set = ((1.0 - xi) ** 4.0 * (4.0 * xi + 1.0)) * (xi < 1.0)
			elif (self.basis == 'wendland5'):
				psi_set = ((1.0 - xi) ** 5.0 * (5.0 * xi + 1)) * (xi < 1.0)
			elif (self.basis == 'wendland6'):
				psi_set = ((1.0 - xi) ** 6.0 * 
					(35.0 * xi ** 2.0 + 18.0 * xi + 3.0)) * (xi < 1.0)
			elif (self.basis == 'wendland7'):
				psi_set = ((1.0 - xi) ** 7.0 *
					(16.0 * xi ** 2.0 + 7.0 * xi + 1.0)) * (xi < 1.0)
			elif (self.basis == 'wendland8'):
				psi_set = (((1.0 - xi) ** 8.0 *
					(32.0 * xi ** 3.0 + 25.0 * xi ** 2.0 + 8.0 * xi + 1.0)) *
					(xi < 1.0))
		psi_set = np.nan_to_num(psi_set)
		return psi_set

	def gen_weights(self, f_target):
		'''
		Generate a set of weights over the basis functions such that the
		target forcing term trajectory is matched.
		 f_target shaped 3 x n_time_steps
		'''
		# Generate the basis functions
		s_track = self.cs.rollout()
		psi_track = self.gen_psi(s_track)
		# Compute useful quantities
		sum_psi = np.sum(psi_track, 0)
		sum_psi_2 = sum_psi * sum_psi
		s_track_2 = s_track * s_track
		# Set up the minimization problem
		A = np.zeros([self.n_bfs + 1, self.n_bfs + 1])
		b = np.zeros([self.n_bfs + 1])
		# The matrix does not depend on f
		for k in range(self.n_bfs + 1):
			A[k, k] = scipy.integrate.simps(
				psi_track[k] * psi_track[k] * s_track_2 / sum_psi_2, s_track)
			for h in range(k + 1, self.n_bfs + 1):
				A[k, h] = scipy.integrate.simps(
					psi_track[k] * psi_track[h] * s_track_2 / sum_psi_2,
					s_track)
				A[h, k] = A[k, h].copy()
		LU = scipy.linalg.lu_factor(A)
		# The problem is decoupled for each dimension
		#for d in range(self.n_dmps):
		for d in range(3):
			# Create the vector of the regression problem
			for k in range(self.n_bfs + 1):
				b[k] = scipy.integrate.simps(
					f_target[d] * psi_track[k] * s_track / sum_psi, s_track)
			# Solve the minimization problem
			self.w[d] = scipy.linalg.lu_solve(LU, b)
		self.w = np.nan_to_num(self.w)

	def gen_width(self):
		'''
		Set the "widths" for the basis functions.
		'''
		if (self.basis == 'gaussian'):
			self.width = 1.0 / np.diff(self.c) / np.diff(self.c)
			self.width = np.append(self.width, self.width[-1])
		else:
			self.width = 1.0 / np.diff(self.c)
			self.width = np.append(self.width[0], self.width)
	def calc_log_arr(self,goal,path):
		log_arr=np.zeros((path.shape[0],3))
		
		for i in range(path.shape[0]):
			qi=path[i]
			qi_conj=copy.deepcopy(qi)
			qi_conj[1:4]=-qi_conj[1:4]
			q_mult=quat_mul(goal,qi_conj)
			log_arr[i,:]=quat_log(q_mult)
		return log_arr
	def imitate_path(self, q_des, eta_des = None, deta_des = None, t_des = None,
		g_w = True, **kwargs):
		'''
		Takes in a desired trajectory and generates the set of system
		parameters that best realize this path.
		  q_des array shaped num_timesteps x 4
		  t_des 1D array of num_timesteps component
		  g_w boolean, used to separate the one-shot learning from the
					   regression over multiple demonstrations
		'''

		## Set initial state and x_goal
		self.q_0 = q_des[0].copy()
		self.q_goal = q_des[-1].copy()

		## Set t_span
		if t_des is None:
			# Default value for t_des
			t_des = np.linspace(0, self.cs.run_time, q_des.shape[0])
		else:
			# Warp time to start from zero and end up to T
			t_des -= t_des[0]
			t_des /= t_des[-1]
			t_des *= self.cs.run_time
		time = np.linspace(0., self.cs.run_time, self.cs.timesteps)

		## Piecewise linear interpolation
		# Interpolation function
		#TODO: replace with quaternion interpolation

		#path_gen = scipy.interpolate.interp1d(t_des, x_des.transpose())
		#convert q_des to Rotations
		q_des_wfirst=np.block([q_des[:,1:4],q_des[:,0:1]])
		rots=R.from_quat(q_des_wfirst)
		#rots=R.random(q_des.shape[0],random_state=1243)
		#for i in range(q_des.shape[0]):
		#	rots[i]=R.from_quat(q_des[i])
		
		path_gen = Slerp(t_des,rots)
		# Evaluation of the interpolant
		rpath = path_gen(time)
		qpath=rpath.as_quat()
		#this returns in scalar-last format; we want scalar-first
		#to be consistent
		path=np.block([qpath[:,3:4],qpath[:,0:3]])
		q_des = path#.transpose()

		## Second order estimates of the derivatives
		## (the last non centered, all the others centered)
		#TODO: replace with quaternion derivatives
		if eta_des is None:
			D1 = compute_D1(self.cs.timesteps, self.cs.dt)
			dq_des = np.dot(D1, q_des)
			#calculate eta_des, the scaled angular velocity,
			#from
			eta_des=np.zeros((q_des.shape[0],3)) 
			for i in range(q_des.shape[0]):
				q_des_i_conj=copy.deepcopy(q_des[i,:])
				q_des_i_conj[1:4]=-q_des_i_conj[1:4]
				ang_vel_i=2*quat_mul(dq_des[i],q_des_i_conj)
				eta_des[i,:]=ang_vel_i[1:4]
		else:
			dpath = np.zeros([self.cs.timesteps, self.n_dmps])
			#dpath_gen = scipy.interpolate.interp1d(t_des, dx_des)
			dpath_gen=Slerp(t_des,eta_des)
			dpath = dpath_gen(time)
			eta_des = dpath.transpose()
		if deta_des is None:
			#D2 = compute_D2(self.cs.timesteps, self.cs.dt)
			D1=compute_D1(self.cs.timesteps,self.cs.dt)
			deta_des = np.dot(D1, eta_des)
		else:
			ddpath = np.zeros([self.cs.timesteps, self.n_dmps])
			#ddpath_gen = scipy.interpolate.interp1d(t_des, ddq_des)
			ddpath_gen=Slerp(t_des,deta_des)
			ddpath = ddpath_gen(time)
			deta_des = ddpath.transpose()

		## Find the force required to move along this trajectory
		s_track = self.cs.rollout()
		#TODO: replace with forcing fxn for quaternions. make sure it adjusts for the 3 eqns
		#current:
		#(K/ddx-(x_goal-x_des)+D*dx/K+arr(x_goal-x_0)
		#print(self.q_goal)
		#print(self.q_0)
		Rg=R.from_quat(self.q_goal)
		R0=R.from_quat(self.q_0)
		R0_inv=R0.inv()
		#TODO: replace with DMP that can handle the same start/goal
		#constraints
		Do=log_R(( np.matmul((Rg.as_matrix()),(R0_inv.as_matrix())) ))
		#Do_diag_inv=1/Do
		#array of log(Rg*R.T)
		
		Dx_arr=np.zeros((3,self.cs.timesteps))
		for i in range(self.cs.timesteps):
			R_inv=(R.from_quat(qpath[i])).inv()
			Dx_arr[:,i]=log_R(np.matmul(Rg.as_matrix(),R_inv.as_matrix()))
		#f_target = ((deta_des / self.K - 2*self.calc_log_arr(self.q_goal,q_des) + 
		#	self.D / self.K * eta_des).transpose() +
		#	np.reshape((Do_diag_inv), [3, 1]) * s_track)
		f_target= deta_des.transpose()/self.K -Dx_arr+(self.D/self.K)*eta_des.transpose()+np.reshape(Do,[3,1])*s_track
		#f_target = ((ddx_des / self.K - (self.x_goal - x_des) + 
		#	 self.D / self.K * dx_des).transpose() +
		#	 np.reshape((self.x_goal - self.x_0), [self.n_dmps, 1]) * s_track)
		if g_w:
			# Efficiently generate weights to realize f_target
			# (only if not called by paths_regression)
			self.gen_weights(f_target)
			self.reset_state()
			#TODO: re-add this. seems not to be used though.
			#self.learned_position = self.x_goal - self.x_0
		return f_target

	def reset_state(self, v0 = None, **kwargs):
		'''
		Reset the system state
		'''
		self.q = self.q_0.copy()
		print("resetting state; self.q is:",self.q)
		if v0 is None:
			v0 = np.zeros(3)#0.0 * self.q_0
		self.eta = v0
		self.deta = np.zeros(3)
		self.cs.reset_state()
		self.sx=0.2
		self.sy=0.2
		self.sz=0.2

	def rollout(self, tau = 1.0, v0 = None, **kwargs):
		'''
		Generate a system trial, no feedback is incorporated.
		  tau scalar, time rescaling constant
		  v0 scalar, initial velocity of the system
		'''

		# Reset the state of the DMP
		if v0 is None:
			v0 = np.zeros(3)#0.0 * self.q_0
		self.reset_state(v0 = v0)
		q_track = np.array([self.q_0])
		eta_track = np.array([v0])
		t_track = np.array([0])
		psi = self.gen_psi(self.cs.s)
		#forcing function.
		f0 = (np.dot(self.w, psi[:, 0])) / (np.sum(psi[:, 0])) * self.cs.s
		f0 = np.nan_to_num(f0)

		Rg=R.from_quat(self.q_goal)
		R0=R.from_quat(self.q_0)
		R0_inv=R0.inv()
		#TODO: replace with DMP that can handle the same start/goal
		#constraints
		Do_diag=log_R((np.matmul((Rg.as_matrix()),(R0_inv.as_matrix())) ))
	
		deta_track = np.array([-self.D * v0 + self.K*f0])
		#TODO: replace with quaternion difference instead of this
		err = np.linalg.norm(q_track[-1] - self.q_goal)
		#P = phi1(self.cs.dt * self.linear_part / tau)
		while err > self.tol:
			psi = self.gen_psi(self.cs.s)
			f = (np.dot(self.w, psi[:, 0])) / (np.sum(psi[:, 0])) * self.cs.s
			f = np.nan_to_num(f)
			
			#calculate dmp acceleration
			#calculate the 2 log term
			q_inv=copy.deepcopy(q_track[-1])
			q_inv[1:4]=-q_inv[1:4]
			logterm=2*quat_log(quat_mul(self.q_goal,q_inv))
			#TODO: check the Do_diag multiplication term
			#deta=self.K*2*logterm - self.D*eta_track[-1]+f*Do_diag
			deta=self.K*2*logterm-self.D*eta_track[-1]-self.K*Do_diag*self.cs.s+self.K*f
			eta=eta_track[-1]+deta*tau*self.cs.dt
			quat_exp_term=self.cs.dt*eta/(2*tau)
			print("quat exp term inner",quat_exp_term)
			print("quat exp term:",quat_exp(quat_exp_term))
			print("q_track[-1]",q_track[-1])
			q=quat_mul(quat_exp(quat_exp_term),q_track[-1])
			print("resulting q:",q)
			#record pos, vel, acc
			deta_track=np.append(deta_track,[deta],axis=0)
			eta_track=np.append(eta_track,[eta],axis=0)
			q_track=np.append(q_track,[q],axis=0)
			#calculate dmp velocity from acceleration.
			#TODO: replace this forward euler with something else

			t_track = np.append(t_track, t_track[-1] + self.cs.dt)
			#TODO: replace with quaternion difference instead of this
			err = np.linalg.norm(q_track[-1] - self.q_goal)
			self.cs.step(tau=tau)
		return q_track, eta_track, deta_track, t_track

	def step(self, tau = 1.0, error = 0.0, external_force = None,
		adapt=False, tols=None, pos=None,**kwargs):
		'''
		Run the DMP system for a single timestep.
		  tau float: time rescaling constant
		  error float: optional system feedback
		  external_force 1D array: external force to add to the system
		  adapt bool: says if using adaptive step
		  tols float list: [rel_tol, abs_tol]
		'''

		## Initialize
		if tols is None:
			tols = [1e-03, 1e-06]
		## Setup
		# Scaling matrix
		# Coupling term in canonical system
		error_coupling = 1.0 + error
		self.cs.step(tau=tau, error_coupling=error_coupling)
		#force scaling term, Do_diag
		Rg=R.from_quat(self.q_goal)
		R0=R.from_quat(self.q_0)
		R0_inv=R0.inv()
		#TODO: replace with DMP that can handle the same start/goal
		#constraints
		Do_diag=log_R((np.matmul((Rg.as_matrix()),(R0_inv.as_matrix())) ))
		#forcing function
		psi = self.gen_psi(self.cs.s)
		f = (np.dot(self.w, psi[:, 0])) / (np.sum(psi[:, 0])) * self.cs.s
		f = np.nan_to_num(f)
		#calculate dmp acceleration
		#calculate the 2 log term
		q_inv=copy.deepcopy(self.q)
		q_inv[1:4]=-q_inv[1:4]
		logterm=2*quat_log(quat_mul(self.q_goal,q_inv))
		#TODO: check the Do_diag multiplication term
		#deta=self.K*2*logterm - self.D*self.eta+f*Do_diag
		#print("printing terms of deta.")
		
		deta=self.K*2*logterm-self.D*self.eta-self.K*Do_diag*self.cs.s+self.K*f
		#print("eta:",self.eta)
		#print("f:",f)
		#print("logterm:",logterm)
		#print("deta:",deta)
		#add external forces
		if external_force is not None:
			deta=deta+external_force(self.q,self.eta)/tau
		# if pos is not None and len(pos)==3:
		# # #add singularity avoidance
		# # 	homing_pos=np.array([1,0,0,0])
		# # 	sing_logterm=2*quat_log(quat_mul(homing_pos,q_inv))
		# # 	sing_deta_term=self.K*2*sing_logterm
		# # 	#check condition number to scale this

		# # 	#TODO: integrate with cartesian DMP so that this doesn't need to be run
		# # 	current_R=R.from_quat(self.q)
		# # 	eul_angs=current_R.as_euler('zyx',degrees=False)
			
		# # 	curr_j=np.array([pos[0],pos[1],pos[2],
		# # 					eul_angs[2],eul_angs[1],eul_angs[0]])
		# # 	#solve for joint angles using IK
		# # 	j_actual_real,valid, _=invK.invK(curr_j)
		# # 	_, _,solX,solY,solZ,all_joints=forK.fk_plot(
		# # 		j_actual_real.flatten(),self.sx,self.sy,self.sz,self.cs.dt)
		# # 	self.sx=solX
		# # 	self.sy=solY
		# # 	self.sz=solZ
		# # 	cond_num=condition_no.cond_no(np.array(all_joints))
		# # 	cond_mod=max(np.log(cond_num)-6,0.0)
		# # 	print("condition mod=",cond_mod)
		# # 	# add the singularity avoidance term to the system acceleration
		# # 	deta=deta+2.2*cond_mod*sing_deta_term	
	
		eta=self.eta+deta*tau*self.cs.dt
		quat_exp_term=self.cs.dt*eta/(2*tau)
		#print("quat exp term inner",quat_exp_term)
		#print("quat exp term:",quat_exp(quat_exp_term))
		#print("self.q:",self.q)
		q=quat_mul(quat_exp(quat_exp_term),self.q)
		#print("resulting self.q:",q)
		self.q=q
		self.eta=eta
		self.deta=deta
		#print("q before reutrning is,",self.q," and q is",q)
		#print(self.eta)
		#print(self.deta)
		return self.q,self.eta,self.deta


