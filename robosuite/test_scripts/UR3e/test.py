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



def move_robot(sim, joint_qpos ):
    for i in range(len(joint_qpos)):
        joint_name = "robot0_joint_" + str(i+1)
        sim.data.set_joint_qpos(joint_name, joint_qpos[i])


ur3e_arm = ur_kinematics.URKinematics('ur3e')

world = MujocoWorldBase()

mujoco_robot = UR3e()

		
mujoco_robot.set_base_xpos([0.5, 0.0, 0.05])
world.merge(mujoco_robot)

mujoco_arena =EmptyArena()
world.merge(mujoco_arena)

iphonebox = BoxObject(name="iphonebox",size=[0.08,0.039,0.0037],rgba=[0,0,0,1],friction=[1,1,5]).get_obj() 
iphonebox.set('pos', '-0.5 0.4 0.9')
world.worldbody.append(iphonebox)

box = BoxObject(name="box",size=[9.7,0.35,0.37],rgba=[0.9,0.9,0.9,1],friction=[1,1,1]).get_obj()
box.set('pos', '1 0.4 0')
world.worldbody.append(box)



model = world.get_model(mode="mujoco_py")

sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

viewer.render()



t_final = 20000

t = 0
timestep= 0.002

sim_state = sim.get_state()

# joint_values_init = np.zeros(6)
joint_values_init = [-0.691, -1.50792, -1.60242, -1.57075, 1.50792, -0.31415]

ee_pose_init = ur3e_arm.forward(joint_values_init)
print(f"ee pose shape {ee_pose_init.shape}, {ee_pose_init}")
def ik(pose, q_guess):
    print(f"pose {pose}")
    return ur3e_arm.inverse(pose, False, q_guess = q_guess)

ee_pose = ee_pose_init
last_angles = joint_values_init

while t<t_final:
	sim.step()
	if True:
		viewer.render()
	# ee_pose[2] +=0.00005
	# ee_pose[1] +=0.00005
	ee_pose[0] +=0.00005
	sim.data.set_joint_qvel('box_joint0', [0.2, 0, 0, 0, 0, 0])
	joint_pos = ik(ee_pose, last_angles)
	print(f"joint angles {joint_pos}")
	# print(f"ee_pose {ee_pose}, joint_values {joint_pos}")

	move_robot(sim, joint_pos)

	last_angles = joint_pos


	t=t+1



