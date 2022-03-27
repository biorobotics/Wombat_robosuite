import robosuite
from robosuite.models.objects import BoxObject
from robosuite.models.objects import CylinderObject
from robosuite.models.robots import UR3e
from robosuite.models.arenas import EmptyArena
from robosuite.models.grippers import gripper_factory
import math
import copy
from mujoco_py import MjSim, MjViewer
import numpy as np
import time
# import matplotlib as mpl
import ipdb
# mpl.use('TkAgg')
import matplotlib.pyplot as plt 
from robosuite.models import MujocoWorldBase
from robosuite.models.objects.objects import MujocoXMLObject
import pyautogui
from ur_ikfast import ur_kinematics 
from scipy.spatial.transform import Rotation as R



def move_robot(sim, joint_qpos ):
    for i in range(len(joint_qpos)):
        joint_name = "robot0_joint_" + str(i+1)
        sim.data.set_joint_qpos(joint_name, joint_qpos[i])

def quat_to_euler( quat):
	r_quat = R.from_quat([quat[0],quat[1],quat[2],quat[3]])
	e_angles = r_quat.as_euler('zyx', degrees=False)
	return e_angles

ur3e_arm = ur_kinematics.URKinematics('ur3e')

world = MujocoWorldBase()

mujoco_robot = UR3e()

# gripper = gripper_factory('PandaGripper')
# # gripper.hide_visualization()
# mujoco_robot.add_gripper(gripper)
		
mujoco_robot.set_base_xpos([0.5, 0.0, 0.20])
world.merge(mujoco_robot)

mujoco_arena =EmptyArena()
world.merge(mujoco_arena)


box = BoxObject(name="box",size=[9.7,0.35,0.40],rgba=[0.9,0.9,0.9,1],friction=[1,1,1]).get_obj()
box.set('pos', '1 0.4 0.37')
world.worldbody.append(box)

iphonebox = BoxObject(name="iphonebox",size=[0.08,0.039,0.0037],rgba=[0,0,0,1],friction=[1,1,1]).get_obj() 
# iphonebox = BoxObject(name="iphonebox",size=[0.08,0.039,0.003],rgba=[0,0,0,1],friction=[10,10,10]).get_obj()
iphonebox.set('pos', '0.63 0.395 0.8')
iphonebox.set('quat', '1 0.7 0.7 0')
world.worldbody.append(iphonebox)
 
# Gripper touches at z = 0.896

model = world.get_model(mode="mujoco_py")

sim = MjSim(model)

viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

viewer.render()



t_final = 20000

t = 0
timestep= 0.002

sim_state = sim.get_state()

joint_values_init = np.zeros(6)
joint_values_init =np.array([-np.pi/2, -2.0, -np.pi/2, -1.01,  1.57, np.pi *0/180.0])

ee_pose_init = ur3e_arm.forward(joint_values_init)
ee_pose_init[3:] =  [-0.9999997, 0, 0, 0.0007963]
print(f"ee pose shape {ee_pose_init.shape}, {ee_pose_init}")
print(f"euler angles :{quat_to_euler(np.array(ee_pose_init[3:]))}")

phone_pose = sim.data.get_joint_qpos('iphonebox_joint0')
phone_pose[1] = ee_pose_init[1]+ 0.0005
phone_pose[3:] = [1, 0, 0, 0]
# sim.data.set_joint_qpos('iphonebox_joint0',phone_pose )


def ik(pose, q_guess):
    # print(f"pose {pose}")
    return ur3e_arm.inverse(pose, False, q_guess = q_guess)

ee_pose = ee_pose_init
ee_pose[1] = ee_pose[1] + 0.037
last_angles = joint_values_init

reach_flag = False

while t<t_final:
	# sim.data.set_joint_qvel('box_joint0',[-0.2,0,0,0,0,0])
	sim.step()
	viewer.render()
	# print("L1",sim.data.get_joint_qpos('robot0_base_left_short_joint'))
	# print("R1",sim.data.get_joint_qpos('robot0_base_right_short_joint'))
	phone_pose = sim.data.get_joint_qpos('iphonebox_joint0')
	gripper_pose = sim.data.sensordata[0:7]	
	joint_pos = ik(ee_pose, last_angles)
	print(f"gripper_pose_z  {gripper_pose[1]}, phone {phone_pose[1]}")
	if(gripper_pose[2]<=0.895 and t>10 and reach_flag == False):
		reach_flag = True
		reach_time = t
		# print(f"set reach time = {reach_time}")

	if reach_flag==False:
		ee_pose[2] -=0.000025
		sim.data.ctrl[6] = -0.7
		sim.data.ctrl[7] = 0.7
		# sim.data.set_joint_qpos('iphonebox_joint0',[0.63,0.395,0.80337,0,0,0,0])

	if reach_flag:
		if(t-reach_time)>100:
			sim.data.ctrl[6] = -0.25
			sim.data.ctrl[7] = 0.25
			# sim.data.set_joint_qpos('iphonebox_joint0',[0.63,0.395,0.80337,0,0,0,0])

		if((t-reach_time)>1000):
			ee_pose[2]+=0.000025
			# sim.data.set_joint_qvel('iphonebox_joint0',[0,0,0.2,0,0,0])
			# print(f"lifting up")

	# if sim.data.get_joint_qpos('robot0_base_left_torque_joint')>0.1:
	# 	sim.data.set_joint_qpos('robot0_base_left_torque_joint', 0.1)
	# if sim.data.get_joint_qpos('robot0_base_left_torque_joint')<-0.1:
	# 	sim.data.set_joint_qpos('robot0_base_left_torque_joint', -0.1)

	# if sim.data.get_joint_qpos('robot0_base_right_torque_joint')>0.1:
	# 	sim.data.set_joint_qpos('robot0_base_right_torque_joint', 0.1)
	# if sim.data.get_joint_qpos('robot0_base_right_torque_joint')<-0.1:
	# 	sim.data.set_joint_qpos('robot0_base_right_torque_joint', -0.1)

	
	# if sim.data.get_joint_qpos('robot0_base_left_short_joint')>0.7:
	# 	sim.data.set_joint_qpos('robot0_base_left_short_joint', 0.7)
	# if sim.data.get_joint_qpos('robot0_base_left_short_joint')<-0.7:
	# 	sim.data.set_joint_qpos('robot0_base_left_short_joint', -0.7)

	# if sim.data.get_joint_qpos('robot0_base_right_short_joint')>0.7:
	# 	sim.data.set_joint_qpos('robot0_base_right_short_joint', 0.7)
	# if sim.data.get_joint_qpos('robot0_base_right_short_joint')<-0.7:
	# 	sim.data.set_joint_qpos('robot0_base_right_short_joint', -0.7)
	

		
	# print(f"ee_pose {ee_pose}, joint_values {joint_pos}")

	move_robot(sim, joint_pos)

	# phone_pose = sim.data.get_joint_qpos('iphonebox_joint0')
	# gripper_pose = sim.data.sensordata[0:7]	

	
	last_angles = joint_pos


	t=t+1
# while t<t_final:
# 	sim.step()
# 	if True:
# 		viewer.render()
# 	ee_pose[2] -=0.00005
# 	# ee_pose[1] +=0.00005
# 	# ee_pose[0] +=0.00005
# 	# sim.data.set_joint_qvel('box_joint0', [-0.2, 0, 0, 0, 0, 0])
# 	joint_pos = ik(ee_pose, last_angles)
# 	# print(f"joint angles {joint_pos}")
# 	if t>0 and t<2000: ##keeping gripper open
# 		# sim.data.set_joint_qpos('robot0_base_left_short_joint', -0.05)
# 		# sim.data.set_joint_qpos('robot0_base_right_short_joint', 0.05)

# 		# sim.data.set_joint_qpos('robot0_base_left_torque_joint', 0.0)
# 		# sim.data.set_joint_qpos('robot0_base_right_torque_joint', -0.0)
# 		sim.data.ctrl[6] = -0.7
# 		sim.data.ctrl[7] = 0.7
# 	else: ##making gripper close
# 		# sim.data.set_joint_qpos('robot0_base_left_short_joint', 0.0)
# 		# sim.data.set_joint_qpos('robot0_base_right_short_joint', -0.0)
# 		# sim.data.set_joint_qpos('robot0_base_left_torque_joint', 0.0)
# 		# sim.data.set_joint_qpos('robot0_base_right_torque_joint', -0.0)
# 		sim.data.ctrl[6] =-0.1
# 		sim.data.ctrl[7] = 0.1
# 	# print("L1",sim.data.get_joint_qpos('robot0_base_left_torque_joint'))
# 	# print("R1",sim.data.get_joint_qpos('robot0_base_right_torque_joint'))
# 	if sim.data.get_joint_qpos('robot0_base_left_torque_joint')>0.1:
# 		sim.data.set_joint_qpos('robot0_base_left_torque_joint', 0.1)
# 	if sim.data.get_joint_qpos('robot0_base_left_torque_joint')<-0.1:
# 		sim.data.set_joint_qpos('robot0_base_left_torque_joint', -0.1)

# 	if sim.data.get_joint_qpos('robot0_base_right_torque_joint')>0.1:
# 		sim.data.set_joint_qpos('robot0_base_right_torque_joint', 0.1)
# 	if sim.data.get_joint_qpos('robot0_base_right_torque_joint')<-0.1:
# 		sim.data.set_joint_qpos('robot0_base_right_torque_joint', -0.1)

# 	# print("L1",sim.data.get_joint_qpos('robot0_base_left_short_joint'))
# 	# print("R1",sim.data.get_joint_qpos('robot0_base_right_short_joint'))
# 	if sim.data.get_joint_qpos('robot0_base_left_short_joint')>0.7:
# 		sim.data.set_joint_qpos('robot0_base_left_short_joint', 0.7)
# 	if sim.data.get_joint_qpos('robot0_base_left_short_joint')<-0.7:
# 		sim.data.set_joint_qpos('robot0_base_left_short_joint', -0.7)

# 	if sim.data.get_joint_qpos('robot0_base_right_short_joint')>0.7:
# 		sim.data.set_joint_qpos('robot0_base_right_short_joint', 0.7)
# 	if sim.data.get_joint_qpos('robot0_base_right_short_joint')<-0.7:
# 		sim.data.set_joint_qpos('robot0_base_right_short_joint', -0.7)

		
# 	# print(f"ee_pose {ee_pose}, joint_values {joint_pos}")

# 	move_robot(sim, joint_pos)
# 	phone_pose = sim.data.get_joint_qpos('iphonebox_joint0')
# 	gripper_pose = sim.data.sensordata[0:7]	
# 	print(f"gripper_pose_z  {gripper_pose[2]}, phone {phone_pose[2]}")

	
# 	last_angles = joint_pos


# 	t=t+1




