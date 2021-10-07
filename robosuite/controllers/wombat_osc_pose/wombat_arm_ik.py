"""
***********************************************************************************

NOTE: IK is only supported for the following robots:

:Wombat_arm:

Attempting to run IK with any other robot will raise an error!

***********************************************************************************
"""
import os
from os.path import join as pjoin
import robosuite
import robosuite.controllers.invK
import robosuite.controllers.trajGenerator# as trg

from robosuite.controllers.base_controller import Controller
from robosuite.utils.control_utils import *
import robosuite.utils.transform_utils as T
import numpy as np
import math


# Dict of supported ik robots
SUPPORTED_IK_ROBOTS = {"Wombat_arm"}





class WombatController(Controller):
	"""
	Controller for controlling robot arm via inverse kinematics. Allows position and orientation control of the
	robot's end effector.

	Inverse kinematics solving is handled by mujoco only.

	NOTE: Control input actions are assumed to be the absolute position / orientation of the end effector
	and are taken as the array (x_dpos, y_dpos, z_dpos, x_rot, y_rot, z_rot).

	Args:
		sim (MjSim): Simulator instance this controller will pull robot state updates from

		eef_name (str): Name of controlled robot arm's end effector (from robot XML)

		joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

			:`'joints'`: list of indexes to relevant robot joints
			:`'qpos'`: list of indexes to relevant robot joint positions
			:`'qvel'`: list of indexes to relevant robot joint velocities

		robot_name (str): Name of robot being controlled. Can be {"Sawyer", "Panda", or "Baxter"}

		actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

		policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

		qpos_limits (2-list of float or 2-list of Iterable of floats): Limits (rad) below and above which the magnitude
			of a calculated goal joint position will be clipped. Can be either be a 2-list (same min/max value for all
			joint dims), or a 2-list of list (specific min/max values for each dim)

		interpolator (Interpolator): Interpolator object to be used for interpolating from the current state to
			the goal state during each timestep between inputted actions

		**kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
			via an argument dict that has additional extraneous arguments won't raise an error

	Raises:
		AssertionError: [Unsupported robot]
	"""

	def __init__(self,
				 sim,
				 eef_name,
				 joint_indexes,
				 robot_name,
				 actuator_range,
				 policy_freq=20,
				 position_limits=None,
				 orientation_limits=None,
				 qpos_limits=None,
				 interpolator_pos=None,
				 interpolator_ori=None,
				 control_ori=True,
				 control_delta=False,
				 uncouple_pos_ori=True,
				 **kwargs # does nothing; used so no error raised when dict is passed with extra terms used previously
				 ):

		self.qpos_limits = np.array([[-2,-4,-0.2226,-4,-4,-4,-2,-4,-0.2226,-4,-4,-4,-2,-4,-0.2226,-4,-4,-4],[2,4,0.2954,4,4,4,2,4,0.2954,4,4,4,2,4,0.2954,4,4,4]])
		print("wombat_controller_file")
		# Run superclass inits
		super().__init__(
			sim=sim,
			eef_name=eef_name,
			joint_indexes=joint_indexes,
			actuator_range=actuator_range#,
			#**kwargs
		)

		# Verify robot is supported by IK
		assert robot_name in SUPPORTED_IK_ROBOTS, "Error: Tried to instantiate IK controller for unsupported robot! " \
												  "Inputted robot: {}, Supported robots: {}".format(
			robot_name, SUPPORTED_IK_ROBOTS)

		# Initialize ik-specific attributes
		self.robot_name = robot_name        # Name of robot (e.g.: "Panda", "Sawyer", etc.)

		# Determine whether this is pos ori or just pos
		self.use_ori = control_ori

		# Determine whether we want to use delta or absolute values as inputs
		self.use_delta = control_delta

		# Control dimension
		self.control_dim = 6 if self.use_ori else 3
		#self.name_suffix = "POSE" if self.use_ori else "POSITION"

		# limits
		self.position_limits = np.array(position_limits) if position_limits is not None else position_limits
		self.orientation_limits = np.array(orientation_limits) if orientation_limits is not None else orientation_limits

		# control frequency
		self.control_freq = policy_freq

		# interpolator
		self.interpolator_pos = interpolator_pos
		self.interpolator_ori = interpolator_ori

		# whether or not pos and ori want to be uncoupled
		self.uncoupling = uncouple_pos_ori

		# initialize goals based on initial pos / ori
		self.goal_ori = np.array(self.initial_ee_ori_mat)
		self.goal_pos = np.array(self.initial_ee_pos)

		self.relative_ori = np.zeros(3)
		self.ori_ref = None

		# initialize
		#self.goal_qpos = None
		print("1")

		
		# Set the reference robot target pos / orientation (to prevent drift / weird ik numerical behavior over time)
		#self.reference_target_pos = self.ee_pos
		#self.reference_target_orn = T.mat2quat(self.ee_ori_mat)

 
		# Interpolator
		# self.interpolator_pos = interpolator_pos
		# self.interpolator_ori = interpolator_ori

		# Interpolator-related attributes
		# self.ori_ref = None
		# self.relative_ori = None

 
		# Set ik limits and override internal min / max
		#self.ik_pos_limit = ik_pos_limit
		#self.ik_ori_limit = ik_ori_limit

		# Target pos and ori
		#self.ik_robot_target_pos = None
		#self.ik_robot_target_orn = None                 # note: this currently isn't being used at all

		# Commanded pos and resulting commanded vel
		# self.commanded_joint_positions = None
		# self.commanded_joint_velocities = None

		# Should be in (0, 1], smaller values mean less sensitivity.
		# self.user_sensitivity = .3


	# def get_control(self, dpos=None, rotation=None, update_targets=False):
	#     """
	#     Returns joint velocities to control the robot after the target end effector
	#     position and orientation are updated from arguments @dpos and @rotation.
	#     If no arguments are provided, joint velocities will be computed based
	#     on the previously recorded target.

	#     Args:
	#         dpos (np.array): a 3 dimensional array corresponding to the desired
	#             change in x, y, and z end effector position.
	#         rotation (np.array): a rotation matrix of shape (3, 3) corresponding
	#             to the desired rotation from the current orientation of the end effector.
	#         update_targets (bool): whether to update ik target pos / ori attributes or not

	#     Returns:
	#         np.array: a flat array of joint velocity commands to apply to try and achieve the desired input control.
	#     """
	#     # Sync joint positions for IK.
	#     #self.sync_ik_robot()

	#     # Compute new target joint positions if arguments are provided
	#     if (dpos is not None) and (rotation is not None):
	#         self.commanded_joint_positions = np.array(self.joint_positions_for_eef_command(
	#             dpos, rotation, update_targets
	#         ))

	#     # P controller from joint positions (from IK) to velocities
	#     velocities = np.zeros(self.joint_dim)
	#     deltas = self._get_current_error(
	#         self.joint_pos, self.commanded_joint_positions
	#     )
	#     for i, delta in enumerate(deltas):
	#         velocities[i] = -10. * delta

	#     self.commanded_joint_velocities = velocities
		# return velocities

	def PD_signal_scale(self,ee_pos,joint_vals):
		ee_xy_disp=np.array([math.sqrt(ee_pos[0]**2+ee_pos[1]**2)]*6)+1.0
		lin_vals=np.array([joint_vals[2],joint_vals[0],joint_vals[1]]*2)+1.0
		scale=7
		PD_scale_factor=((np.multiply(ee_xy_disp,lin_vals)**2)-1)*scale
		#print("PD_scale_factor:",PD_scale_factor)
		#PD_scale_factor=np.array([1,1,1,1,1,1])
		return PD_scale_factor

	def inverse_kinematics(self, target):
		"""
		Helper function to do inverse kinematics for a given target position and
		orientation in the Mujoco world frame.

		Args:
			target_position (3-tuple): desired position
			target_orientation (4-tuple): desired orientation quaternion

		Returns:
			list: list of size @num_joints corresponding to the joint angle solution.
		"""
		
		#print(target[0:6])
		target[0:6]=[0.0, 0.03175,-0.6, 3.141592653589793, -0.0, 3.141592653589793]
		print("target",target[0:6])
		ee_pose_real = robosuite.controllers.invK.sim2real_wrapper(target[0:6])
		print("ee_pose_real",ee_pose_real)
		#joint_real = robosuite.controllers.trajGenerator.jointTraj(ee_pose_real)
		(joint_real,vi) = robosuite.controllers.invK.invK(ee_pose_real)
		joint_real=joint_real.flatten()
		if vi==0:
			print("IK resulted in joints that are invalid!\n")
			
		print("joint_real",joint_real)
		joint_sim = robosuite.controllers.invK.ik_wrapper(joint_real)
		print("joint_sim",joint_sim)
		global PD_scale
		PD_scale = self.PD_signal_scale(ee_pose_real,joint_real)
		# ik_solution = list(
		#     p.calculateInverseKinematics(
		#         bodyUniqueId=self.ik_robot,
		#         endEffectorLinkIndex=self.bullet_ee_idx,
		#         targetPosition=target_position,
		#         targetOrientation=target_orientation,
		#         lowerLimits=list(self.sim.model.jnt_range[self.joint_index, 0]),
		#         upperLimits=list(self.sim.model.jnt_range[self.joint_index, 1]),
		#         jointRanges=list(self.sim.model.jnt_range[self.joint_index, 1] -
		#                          self.sim.model.jnt_range[self.joint_index, 0]),
		#         restPoses=self.rest_poses,
		#         jointDamping=[0.1] * self.num_bullet_joints,
		#         physicsClientId=self.bullet_server_id
		#     )
		# )
		print("2")
		return joint_sim


	def set_goal(self, action, set_pos=None, set_ori=None):
		"""
		Sets goal based on input @action. If self.impedance_mode is not "fixed", then the input will be parsed into the
		delta values to update the goal position / pose and the kp and/or damping_ratio values to be immediately updated
		internally before executing the proceeding control loop.

		Note that @action expected to be in the following format, based on impedance mode!

			:Mode `'fixed'`: [joint pos command]
			:Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
			:Mode `'variable_kp'`: [kp values, joint pos command]

		Args:
			action (Iterable): Desired relative joint position goal state
			set_pos (Iterable): If set, overrides @action and sets the desired absolute eef position goal state
			set_ori (Iterable): IF set, overrides @action and sets the desired absolute eef orientation goal state
		"""
		# Update state
		self.update()

		print("action",action)
		delta = action
		print(action)
		if set_pos is None:
			set_pos = delta[:3]
			# Set default control for ori if we're only using position control
			if set_ori is None:
				set_ori = T.quat2mat(T.axisangle2quat(delta[3:6])) if self.use_ori else \
					np.array([[0, -1., 0.], [-1., 0., 0.], [0., 0., 1.]])
			# No scaling of values since these are absolute values
			scaled_delta = delta
		set_qpos = self.inverse_kinematics(action)
		# Get requested delta inputs if we're using interpolators
		#(dpos, dquat) = self._clip_ik_input(set_qpos)

		# Set interpolated goals if necessary
		# if self.interpolator_pos is not None:
		#     # Absolute position goal
		#     self.interpolator_pos.set_goal(set_qpos[:2])

		# if self.interpolator_ori is not None:
		#     # Relative orientation goal
		#     self.interpolator_ori.set_goal(set_qpos[3:5] - self.ee_ori_mat)  # goal is the relative change in orientation
		#     self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start
		#     self.relative_ori = np.zeros(3)  # relative orientation always starts at 0
		#scaled_delta = None

		# We only want to update goal orientation if there is a valid delta ori value OR if we're using absolute ori
		# use math.isclose instead of numpy because numpy is slow
		bools = [0. if math.isclose(elem, 0.) else 1. for elem in scaled_delta[3:]]
		if sum(bools) > 0. or set_ori is not None:
			self.goal_ori = set_goal_orientation(scaled_delta[3:],
												 self.ee_ori_mat,
												 orientation_limit=self.orientation_limits,
												 set_ori=set_ori)
		self.goal_pos = set_goal_position(scaled_delta[:3],
										  self.ee_pos,
										  position_limit=self.position_limits,
										  set_pos=set_pos)

		if self.interpolator_pos is not None:
			self.interpolator_pos.set_goal(self.goal_pos)

		if self.interpolator_ori is not None:
			self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start
			self.interpolator_ori.set_goal(orientation_error(self.goal_ori, self.ori_ref))  # goal is the total orientation error
			self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

		# self.goal_qpos = set_goal_position(scaled_delta,
		# 								   self.joint_pos,
		# 								   position_limit=self.position_limits,
		# 								   set_pos=set_qpos)

		# if self.interpolator is not None:
		# 	self.interpolator.set_goal(self.goal_qpos)

		#action = [150,150,150,20*PD_scale,20*PD_scale,20*PD_scale,set_qpos[0],set_qpos[1],set_qpos[2],set_qpos[3],set_qpos[4],set_qpos[5]]
		# Run ik prepropressing to convert pos, quat ori to desired velocities
		#requested_control = self._make_input(delta, self.reference_target_orn)

		# Compute desired velocities to achieve eef pos / ori
		#velocities = self.get_control(**requested_control, update_targets=True)

		# Set the goal velocities for the underlying velocity controller
		#super().set_goal(action,set_qpos)

	def run_controller(self):
		"""
		Calculates the torques required to reach the desired setpoint.

		Executes Operational Space Control (OSC) -- either position only or position and orientation.

		A detailed overview of derivation of OSC equations can be seen at:
		http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf

		Returns:
			 np.array: Command torques
		"""
		# Update state
		self.update()

		desired_pos = None
		#self.kp = np.array([20*PD_scale[0], 150, 20*PD_scale[0], 150, 20*PD_scale[0], 150])
		#self.kp = np.array([0.2,0.2,0.2,0.2,0.2,0.2])
		self.kp = np.zeros(6)
		print(self.kp)
		#self.kd = np.array([0.6, 1500, 0.6, 1500, 0.6, 1500])
		#self.kd = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
		self.kd = np.zeros(6)
		# Only linear interpolator is currently supported
		# if self.interpolator is not None:
		# 	# Linear case
		# 	if self.interpolator.order == 1:
		# 		desired_qpos = self.interpolator.get_interpolated_goal()
		# 		print("desired_qpos", desired_qpos)
		# 	else:
		# 		# Nonlinear case not currently supported
		# 		pass
		# else:
		# 	desired_qpos = np.array(self.goal_qpos)
		# 	print("desired_qpos", desired_qpos)

		# # torques = pos_err * kp + vel_err * kd
		# print(desired_qpos)
		# print(self.joint_pos)
		# print(self.sim.data.get_joint_qpos("robot0_branch1_joint"))
		# print(self.sim.data.get_joint_qpos("robot0_branch1_linear_joint"))
		# #position_error = desired_qpos - self.joint_pos
		# position_error = np.array([desired_qpos[0],self.sim.data.get_joint_qpos("robot0_branch1_pivot_joint"),desired_qpos[1],self.sim.data.get_joint_qpos("robot0_branch1_linear_revolute_joint"),self.sim.data.get_joint_qpos("robot0_branch1_clevis_joint"),self.sim.data.get_joint_qpos("robot0_branch1_ee_joint"),desired_qpos[2],self.sim.data.get_joint_qpos("robot0_branch2_pivot_joint"),desired_qpos[3],self.sim.data.get_joint_qpos("robot0_branch2_linear_revolute_joint"),self.sim.data.get_joint_qpos("robot0_branch2_clevis_joint"),self.sim.data.get_joint_qpos("robot0_branch2_ee_joint"),desired_qpos[4],self.sim.data.get_joint_qpos("robot0_branch3_pivot_joint"),desired_qpos[5],self.sim.data.get_joint_qpos("robot0_branch3_linear_revolute_joint"),self.sim.data.get_joint_qpos("robot0_branch3_clevis_joint"),self.sim.data.get_joint_qpos("robot0_branch3_ee_joint")]) - np.array([self.sim.data.get_joint_qpos("robot0_branch1_joint"),self.sim.data.get_joint_qpos("robot0_branch1_pivot_joint"),self.sim.data.get_joint_qpos("robot0_branch1_linear_joint"),self.sim.data.get_joint_qpos("robot0_branch1_linear_revolute_joint"),self.sim.data.get_joint_qpos("robot0_branch1_clevis_joint"),self.sim.data.get_joint_qpos("robot0_branch1_ee_joint"),self.sim.data.get_joint_qpos("robot0_branch2_joint"),self.sim.data.get_joint_qpos("robot0_branch2_pivot_joint"),self.sim.data.get_joint_qpos("robot0_branch2_linear_joint"),self.sim.data.get_joint_qpos("robot0_branch2_linear_revolute_joint"),self.sim.data.get_joint_qpos("robot0_branch2_clevis_joint"),self.sim.data.get_joint_qpos("robot0_branch2_ee_joint"),self.sim.data.get_joint_qpos("robot0_branch3_joint"),self.sim.data.get_joint_qpos("robot0_branch3_pivot_joint"),self.sim.data.get_joint_qpos("robot0_branch3_linear_joint"),self.sim.data.get_joint_qpos("robot0_branch3_linear_revolute_joint"),self.sim.data.get_joint_qpos("robot0_branch3_clevis_joint"),self.sim.data.get_joint_qpos("robot0_branch3_ee_joint")])
		# #print(self.sim.data.get_joint_qvel("robot0_branch1_linear_joint"))
		# #vel_pos_error = -self.joint_vel
		# vel_pos_error = - np.array([self.sim.data.get_joint_qvel("robot0_branch1_joint"),0,self.sim.data.get_joint_qvel("robot0_branch1_linear_joint"),0,0,0,self.sim.data.get_joint_qvel("robot0_branch2_joint"),0,self.sim.data.get_joint_qvel("robot0_branch2_linear_joint"),0,0,0,self.sim.data.get_joint_qvel("robot0_branch3_joint"),0,self.sim.data.get_joint_qvel("robot0_branch3_linear_joint"),0,0,0])
		# desired_torque = (np.multiply(position_error, self.kp)
		# 				  + np.multiply(vel_pos_error, self.kd))
		# print(np.array(position_error))
		# print(self.kp)
		# print(desired_torque)
		# # Return desired torques plus gravity compensations
		# print("mass_matrix",self.mass_matrix)
		# print("mass_size",self.mass_matrix.shape)
		# self.torques = np.dot(self.mass_matrix, desired_torque) + self.torque_compensation
		# print("desired torque",desired_torque)
		# print("torques",self.torques)
		# print("torque shape",self.torques.shape)
		# temp = np.array([self.torques[0],self.torques[2],self.torques[6],self.torques[8],self.torques[12],self.torques[14]])
		# print("shape",temp.shape)
		# self.torques = np.zeros(6)
		# self.torques = temp
		# # Always run superclass call for any cleanups at the end
		# super().run_controller()
		# Only linear interpolator is currently supported
		print(np.array([self.sim.data.get_joint_qpos("robot0_branch1_joint"),self.sim.data.get_joint_qpos("robot0_branch1_pivot_joint"),self.sim.data.get_joint_qpos("robot0_branch1_linear_joint"),self.sim.data.get_joint_qpos("robot0_branch1_linear_revolute_joint"),self.sim.data.get_joint_qpos("robot0_branch1_clevis_joint"),self.sim.data.get_joint_qpos("robot0_branch1_ee_joint"),self.sim.data.get_joint_qpos("robot0_branch2_joint"),self.sim.data.get_joint_qpos("robot0_branch2_pivot_joint"),self.sim.data.get_joint_qpos("robot0_branch2_linear_joint"),self.sim.data.get_joint_qpos("robot0_branch2_linear_revolute_joint"),self.sim.data.get_joint_qpos("robot0_branch2_clevis_joint"),self.sim.data.get_joint_qpos("robot0_branch2_ee_joint"),self.sim.data.get_joint_qpos("robot0_branch3_joint"),self.sim.data.get_joint_qpos("robot0_branch3_pivot_joint"),self.sim.data.get_joint_qpos("robot0_branch3_linear_joint"),self.sim.data.get_joint_qpos("robot0_branch3_linear_revolute_joint"),self.sim.data.get_joint_qpos("robot0_branch3_clevis_joint"),self.sim.data.get_joint_qpos("robot0_branch3_ee_joint")]))
		if self.interpolator_pos is not None:
			# Linear case
			if self.interpolator_pos.order == 1:
				desired_pos = self.interpolator_pos.get_interpolated_goal()
			else:
				# Nonlinear case not currently supported
				pass
		else:
			desired_pos = np.array(self.goal_pos)

		if self.interpolator_ori is not None:
			# relative orientation based on difference between current ori and ref
			self.relative_ori = orientation_error(self.ee_ori_mat, self.ori_ref)

			ori_error = self.interpolator_ori.get_interpolated_goal()
		else:
			desired_ori = np.array(self.goal_ori)
			ori_error = orientation_error(desired_ori, self.ee_ori_mat)

		# Compute desired force and torque based on errors
		position_error = desired_pos - self.ee_pos
		print("desired_pos",desired_pos)
		vel_pos_error = -self.ee_pos_vel

		# F_r = kp * pos_err + kd * vel_err
		desired_force = (np.multiply(np.array(position_error), np.array(self.kp[0:3]))
						 + np.multiply(vel_pos_error, self.kd[0:3]))

		vel_ori_error = -self.ee_ori_vel

		# Tau_r = kp * ori_err + kd * vel_err
		desired_torque = (np.multiply(np.array(ori_error), np.array(self.kp[3:6]))
						  + np.multiply(vel_ori_error, self.kd[3:6]))

		# Compute nullspace matrix (I - Jbar * J) and lambda matrices ((J * M^-1 * J^T)^-1)
		lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(self.mass_matrix,
																				 self.J_full,
																				 self.J_pos,
																				 self.J_ori)

		# Decouples desired positional control from orientation control
		if self.uncoupling:
			decoupled_force = np.dot(lambda_pos, desired_force)
			decoupled_torque = np.dot(lambda_ori, desired_torque)
			decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])
		else:
			desired_wrench = np.concatenate([desired_force, desired_torque])
			decoupled_wrench = np.dot(lambda_full, desired_wrench)

		# Gamma (without null torques) = J^T * F + gravity compensations
		self.torques = np.dot(self.J_full.T, decoupled_wrench) + self.torque_compensation

		# Calculate and add nullspace torques (nullspace_matrix^T * Gamma_null) to final torques
		# Note: Gamma_null = desired nullspace pose torques, assumed to be positional joint control relative
		#                     to the initial joint positions
		self.torques += nullspace_torques(self.mass_matrix, nullspace_matrix,
										  self.initial_joint, self.joint_pos, self.joint_vel)

		temp = np.array([self.torques[0],self.torques[2],self.torques[6],self.torques[8],self.torques[12],self.torques[14]])
		# print("shape",temp.shape)
		self.torques = np.zeros(6)
		self.torques = temp
		# Always run superclass call for any cleanups at the end
		super().run_controller()
		print("4")
		return self.torques
		
	def update_initial_joints(self, initial_joints):
		# First, update from the superclass method
		super().update_initial_joints(initial_joints)

		# We also need to reset the goal in case the old goals were set to the initial confguration
		self.reset_goal()

	def reset_goal(self):
		"""
		Resets the goal to the current state of the robot
		"""
		self.goal_ori = np.array(self.ee_ori_mat)
		self.goal_pos = np.array(self.ee_pos)

		# Also reset interpolators if required

		if self.interpolator_pos is not None:
			self.interpolator_pos.set_goal(self.goal_pos)

		if self.interpolator_ori is not None:
			self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start
			self.interpolator_ori.set_goal(orientation_error(self.goal_ori, self.ori_ref))  # goal is the total orientation error
			self.relative_ori = np.zeros(3)  # relative orientation always starts at 0
		print("5")
		# if self.interpolator is not None:
		# 	self.interpolator.set_goal(self.goal_qpos)

	# def _clip_ik_input(self, dpos, rotation):
	#     """
	#     Helper function that clips desired ik input deltas into a valid range.

	#     Args:
	#         dpos (np.array): a 3 dimensional array corresponding to the desired
	#             change in x, y, and z end effector position.
	#         rotation (np.array): relative rotation in scaled axis angle form (ax, ay, az)
	#             corresponding to the (relative) desired orientation of the end effector.

	#     Returns:
	#         2-tuple:

	#             - (np.array) clipped dpos
	#             - (np.array) clipped rotation
	#     """
	#     # scale input range to desired magnitude
	#     if dpos.any():
	#         dpos, _ = T.clip_translation(dpos, self.ik_pos_limit)

	#     # Map input to quaternion
	#     rotation = T.axisangle2quat(rotation)

	#     # Clip orientation to desired magnitude
	#     rotation, _ = T.clip_rotation(rotation, self.ik_ori_limit)

	#     return dpos, rotation

	# def _make_input(self, action, old_quat):
	#     """
	#     Helper function that returns a dictionary with keys dpos, rotation from a raw input
	#     array. The first three elements are taken to be displacement in position, and a
	#     quaternion indicating the change in rotation with respect to @old_quat. Additionally clips @action as well

	#     Args:
	#         action (np.array) should have form: [dx, dy, dz, ax, ay, az] (orientation in
	#             scaled axis-angle form)
	#         old_quat (np.array) the old target quaternion that will be updated with the relative change in @action
	#     """
	#     # Clip action appropriately
	#     dpos, rotation = self._clip_ik_input(action[:3], action[3:])

	#     # Update reference targets
	#     self.reference_target_pos += dpos * self.user_sensitivity
	#     self.reference_target_orn = T.quat_multiply(old_quat, rotation)

	#     return {
	#         "dpos": dpos * self.user_sensitivity,
	#         "rotation": T.quat2mat(rotation)
	#     }

	# @staticmethod
	# def _get_current_error(current, set_point):
	#     """
	#     Returns an array of differences between the desired joint positions and current
	#     joint positions. Useful for PID control.

	#     Args:
	#         current (np.array): the current joint positions
	#         set_point (np.array): the joint positions that are desired as a numpy array

	#     Returns:
	#         np.array: the current error in the joint positions
	#     """
	#     error = current - set_point
	#     return error

	@property
	def control_limits(self):
		"""
		 The limits over this controller's action space, as specified by self.ik_pos_limit and self.ik_ori_limit
		 and overriding the superclass method

		 Returns:
			 2-tuple:

				 - (np.array) minimum control values
				 - (np.array) maximum control values
		 """
		#max_limit = np.concatenate([self.ik_pos_limit * np.ones(3), self.ik_ori_limit * np.ones(3)])
		low = np.array([-2,-0.2226,-2,-0.2226,-2,-0.2226])
		high = np.array([2,0.2954,2,0.2954,2,0.2954])
		return low, high

	@property
	def name(self):
		return 'WOMBAT_ARM_IK'
