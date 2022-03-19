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
import cv2




##Building a custom world, importing UR5e and adding gripper below
world = MujocoWorldBase()
# mount = CylinderObject(name="mount",size=[0.1,0.3],rgba=[0.5,0.5,0.5,1],friction=[1,1,1]).get_obj()
# mount.set('pos', '0.35 0 0')
# world.worldbody.append(mount)

mujoco_robot = UR3e()

		
# world.merge(can_collision)
# world.merge(can_visual)
# gripper = gripper_factory(None)
# # # gripper = gripper_factory('D3_gripper')
# # #gripper.hide_visualization()
# mujoco_robot.add_gripper(gripper)

# mujoco_robot.set_base_xpos([0.4, 0.06, 0])
mujoco_robot.set_base_xpos([0.5, 0.0, 0])
world.merge(mujoco_robot)

#mujoco_arena = TableArena()
# mujoco_arena = BinsArena()
mujoco_arena =EmptyArena()
# mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

##adding iphone as a box and conveyor belt as a box below
# iphonebox = BoxObject(name="iphonebox",size=[0.11,0.11,0.11],rgba=[0,0,0,1],friction=[1,1,1]).get_obj()
iphonebox = BoxObject(name="iphonebox",size=[0.08,0.039,0.0037],rgba=[0,0,0,1],friction=[1,1,5]).get_obj() ##PIONEER
iphonebox.set('pos', '-0.5 0.6 0.9')
world.worldbody.append(iphonebox)

# box = BoxObject(name="box",size=[0.35,9.7,0.37],rgba=[0.5,0.5,0.5,1],friction=[1,1,1]).get_obj()
# box.set('pos', '1 -0.5 0')
# world.worldbody.append(box)

box = BoxObject(name="box",size=[9.7,0.35,0.37],rgba=[0.75,0.75,0.755,1],friction=[1,1,1]).get_obj()
box.set('pos', '1 0.6 0')
world.worldbody.append(box)

# mount = CylinderObject(name="mount",size=[0.1,0.3],rgba=[0.5,0.5,0.5,1],friction=[1,1,1]).get_obj()
# mount.set('pos', '0.5 0 0')
# world.worldbody.append(mount)



model = world.get_model(mode="mujoco_py")

sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

viewer.render()



t_final = 20000

t = 0
timestep= 0.002

sim_state = sim.get_state()


while t<t_final:
	sim.step()
	
	if True:
		viewer.render()
	sim.data.set_joint_qvel('box_joint0', [0.2, 0, 0, 0, 0, 0])
	sim.data.set_joint_qpos('robot0_joint_1', -1)
	sim.data.set_joint_qpos('robot0_joint_2', -0.5)
	sim.data.set_joint_qpos('robot0_joint_3', 0.5)
	sim.data.set_joint_qpos('robot0_joint_4', 0.5)
	sim.data.set_joint_qpos('robot0_joint_5', 5)
	sim.data.set_joint_qpos('robot0_joint_6', 5)
	sim.data.set_joint_qpos('robot0_base_left_short_joint', 0.025)
	sim.data.set_joint_qpos('robot0_base_right_short_joint', -0.025)

	# sim.data.set_joint_qpos('robot0_joint_1', -1)
	# sim.data.set_joint_qpos('robot0_joint_2', -0.5)
	# sim.data.set_joint_qpos('robot0_joint_3', 0.5)
	# sim.data.set_joint_qpos('robot0_joint_4', 0.5)
	# sim.data.set_joint_qpos('robot0_joint_5', 5)
	# sim.data.set_joint_qpos('robot0_joint_6', -4)
	# sim.data.set_joint_qpos('robot0_base_left_short_joint', 0.025)
	# sim.data.set_joint_qpos('robot0_base_right_short_joint', -0.025)
	# sim.data.ctrl[0] =0
	# sim.data.ctrl[1] =0.5
	# sim.data.ctrl[2] =0
	# sim.data.ctrl[3] =0
	# sim.data.ctrl[4] =0
	# sim.data.ctrl[5] =0
	t=t+1



