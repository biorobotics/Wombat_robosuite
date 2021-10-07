from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Wombat_arm
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.arenas import BinsArena
from robosuite.models.objects import BallObject
from robosuite.models.objects import CanObject
from robosuite.models.objects import iPhoneObject
from robosuite.utils.mjcf_utils import new_joint
from mujoco_py import MjSim, MjViewer
import math
import invK
import trajGenerator as trg
from interpolateTraj import interpolateTraj
import robosuite
import numpy as np
import time

def PD_controller_rot(des,current,q_pos_last,scale):
	
	#kp = 10
	#kp=10
	#kp = 1
	#kd = 0.3
	#kp = 5
	#kd = 0.6
	kp=20*scale
	kd=0.6
	qpos = des+kp*(des-current)-kd*(current-q_pos_last)
	# print(kp*(des-current))
	return qpos

	# return np.array(points)
def PD_controller_lin(des,current,q_pos_last,scale):
	
	#kp = 10
	#kd = 0.8
	#kp=10
	#kd=0.1
	kp=150
	kd=1500
	qpos = des+kp*(des-current)-kd*(current-q_pos_last)
	# print(kp*(des-current))
	return qpos

#scales the PD signal based on the ee pos or joint values; wombat_arm needs
#different PD values depending on where it is, position-wise
def PD_signal_scale(ee_pos,joint_vals):
	ee_xy_disp=np.array([math.sqrt(ee_pos[0]**2+ee_pos[1]**2)]*6)+1.0
	lin_vals=np.array([joint_vals[2],joint_vals[0],joint_vals[1]]*2)+1.0
	scale=7
	PD_scale_factor=((np.multiply(ee_xy_disp,lin_vals)**2)-1)*scale
	#print("PD_scale_factor:",PD_scale_factor)
	#PD_scale_factor=np.array([1,1,1,1,1,1])
	return PD_scale_factor

world = MujocoWorldBase()
mujoco_robot = Wombat_arm()

#gripper = gripper_factory('PandaGripper')
gripper = gripper_factory(None)
#gripper = gripper_factory('RethinkGripper')
#gripper = gripper_factory('Robotiq140Gripper')
#gripper.hide_visualization()
mujoco_robot.add_gripper(gripper)

mujoco_robot.set_base_xpos([0.4, 0.06, 0])
world.merge(mujoco_robot)

#mujoco_arena = TableArena()
mujoco_arena = BinsArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

#sphere = BallObject(
#    name="sphere",
#    size=[0.04],
#    rgba=[0, 0.5, 0.5, 1])#.get_collision()
#sphere.append(new_joint(name='sphere_free_joint', type='free'))
#sphere.set('pos', '1.0 0 1.0')
#world.worldbody.append(sphere)
can = CanObject(name="can")
#can.set('pos', '1.0 0 1.0')
world.merge(can)

model = world.get_model(mode="mujoco_py")

sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh
env = robosuite.make(
	env_name="PickPlaceiPhone", # try with other tasks like "Stack" and "Door"
	robots="Wombat_arm",  # try with other robots like "Sawyer" and "Jaco"
	has_renderer=True,
	has_offscreen_renderer=False,
	use_camera_obs=False,
)
viewer.render()
# reset the environment
env.reset()
#action = np.zeros(7)
# for i in range(1000):
# 	sim.set_state(sim_state)
# 	#action = np.random.randn(env.robots[0].dof) # sample random action
# 	obs, reward, done, info = env.step(action)  # take action in the environment
# 	env.render()  # render on display
t = 0
timestep= 0.0005
t_final = 10000
sim_state = sim.get_state()
q_pos_last = [0]*6
ee_pose_des = []
ee_pose_current = []
#ee_pose_goal=[]
#ee_pose_goal_maxlen=100
#ee_pose_des_past=[]
#ee_pose_des_past_maxlen=100
#ee_pose_timeskip=90
#dt=1.0/(t_final-t)
#vertical circle
#r=0.1
#dt=1.0/(t_final-t)
#target_traj=np.array([[r*np.sin(i*dt*np.pi*2),0.0,0.7-r*np.cos(i*dt*np.pi*2),0,0,0] for i in range(t,t_final)])

#horizontal circle
# r=0.3
# tLin=2000
# dt=1.0/(t_final-t-tLin)
# target_traj1=np.array([[0,-r*np.sin(np.pi/2*(float(i)/tLin)),0.6,0,0,0] for i in range(0,tLin)])
# target_traj2=np.array([[-r*np.sin((i)*dt*np.pi*2),-r*np.cos(i*dt*np.pi*2),0.6,0,0,0] for i in range(t,t_final-tLin)])
# target_traj=np.block([[target_traj1],[target_traj2]])
#vertical circle
r=0.1
dt=1.0/(t_final-t)
target_traj=np.array([[r*np.sin(i*dt*np.pi*2),0.0,0.7-r*np.cos(i*dt*np.pi*2),0,0,0] for i in range(t,t_final)])
#convert to joint trajectory
joint_target_traj=trg.jointTraj(target_traj)
sim.set_state(sim_state)
ee_pose_des = []
#ee_pose_des_past = []
#ee_pose_current = []
#joint values in the simulation
j_actual=np.zeros((t_final-t,6))
j_actual_real=np.zeros((t_final-t,6))
#goal joint values for simulation to follow
j_goal=np.zeros((t_final-t,6))

# while t<t_final:
# 	#rotary
# 	q_pos_last[0] = sim.data.get_joint_qpos("robot0_branch1_joint")
# 	q_pos_last[1] = sim.data.get_joint_qpos("robot0_branch2_joint")
# 	q_pos_last[2] = sim.data.get_joint_qpos("robot0_branch3_joint")
# 	#linear
# 	q_pos_last[3] = sim.data.get_joint_qpos("robot0_branch1_linear_joint")
# 	q_pos_last[4] = sim.data.get_joint_qpos("robot0_branch2_linear_joint")
# 	q_pos_last[5] = sim.data.get_joint_qpos("robot0_branch3_linear_joint")
# 	#print(q_pos_last[0],q_pos_last[1],q_pos_last[2],q_pos_last[3],q_pos_last[4],q_pos_last[5])
# 	sim.step()
# 	#print(sim.data.get_joint_qpos("branch1_joint"),sim.data.get_joint_qpos("branch2_joint"),sim.data.get_joint_qpos("branch3_joint"),sim.data.get_joint_qpos("branch1_linear_joint"),sim.data.get_joint_qpos("branch2_linear_joint"),sim.data.get_joint_qpos("branch3_linear_joint"))
# 	if True:
# 		viewer.render()

# 	#current target joint values, in IK frame
# 	joint_real=joint_target_traj[t]
	# ee_pose_des.append(ee_pose)
	#convert current target joint values, in sim frame
	# joint_sim=invK.ik_wrapper(joint_real)
	# j_goal[t,:]=np.array(joint_sim)
	# #calculate/send PD control signal to the motors
	# PD_scale=PD_signal_scale(target_traj[t],joint_target_traj[t])
	# PD_signal=[PD_controller_rot(joint_sim[3],sim.data.get_joint_qpos("robot0_branch1_joint"),q_pos_last[0],PD_scale[0]),
	# 		   PD_controller_rot(joint_sim[4],sim.data.get_joint_qpos("robot0_branch2_joint"),q_pos_last[1],PD_scale[1]),
	# 		   PD_controller_rot(joint_sim[5],sim.data.get_joint_qpos("robot0_branch3_joint"),q_pos_last[2],PD_scale[2]),
	# 		   PD_controller_lin(joint_sim[0],sim.data.get_joint_qpos("robot0_branch1_linear_joint"),q_pos_last[3],PD_scale[3]),
	# 		   PD_controller_lin(joint_sim[1],sim.data.get_joint_qpos("robot0_branch2_linear_joint"),q_pos_last[4],PD_scale[4]),
	# 		   PD_controller_lin(joint_sim[2],sim.data.get_joint_qpos("robot0_branch3_linear_joint"),q_pos_last[5],PD_scale[5])]
	
	# sim.data.ctrl[0]=PD_signal[0]
	# sim.data.ctrl[1]=PD_signal[1]
	# sim.data.ctrl[2]=PD_signal[2]
	# sim.data.ctrl[3]=PD_signal[3]
	# sim.data.ctrl[4]=PD_signal[4]
	# sim.data.ctrl[5]=PD_signal[5]
	# t=t+1

for i in range(1000):
	ee_pose=invK.real2sim_wrapper(target_traj[t])
	action = ee_pose
	#sim.set_state(sim_state)
	#action = np.random.randn(env.robots[0].dof) # sample random action
	obs, reward, done, info = env.step(action)  # take action in the environment
	env.render()  # render on display
