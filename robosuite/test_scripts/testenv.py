from robosuite.models import MujocoWorldBase
import ipdb
import numpy as np
world = MujocoWorldBase()

from robosuite.models.robots import Panda

mujoco_robot = Panda()

from robosuite.models.grippers import gripper_factory

gripper = gripper_factory('PandaGripper')
# gripper.hide_visualization()
mujoco_robot.add_gripper(gripper)

mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)

# from robosuite.models.objects import MujocoXMLObject
# from robosuite.models.objects import CanObject
from robosuite.models.objects import BallObject
from robosuite.models.objects import BoxObject
# from robosuite.models.objects import iPhone12ProMaxObject
# from robosuite.models.objects import CanVisualObject
from robosuite.utils.mjcf_utils import new_joint
# can = CanObject(name="can").get_obj()
# # can_visual = CanVisualObject(name="can_visual")
# # # ipdb.set_trace()
# can.set("pos", '2.0 0 0.0')
# # can_visual.get_obj().set('pos', '0.0 0 0.0')
# # print("can_joints", can.joints)
# # world.merge(can)
# world.worldbody.append(can)

from robosuite.models.arenas import TableArena
from robosuite.models.arenas import EmptyArena

# mujoco_arena = TableArena()
mujoco_arena =EmptyArena()
#mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)


iphonebox = BoxObject(name="iphonebox",size=[0.04,0.04,0.02],rgba=[0,0.5,0.5,1],friction=[1,1,1]).get_obj()
iphonebox.set('pos', '0.5 -2 1')
world.worldbody.append(iphonebox)
# sphere = BallObject(
#    name="sphere",
#    size=[0.04],
#    rgba=[0, 0.5, 0.5, 1],friction=[1,1,1]).get_obj()
# sphere.append(new_joint(name='sphere_free_joint', type='free'))
## to set the sphere position
# sphere.set('pos', '1.0 -2 0.5')
# world.worldbody.append(sphere)
# world.merge(sphere)
# phone = iPhone12ProMaxObject(
#    name="phone").get_obj()
# # sphere.append(new_joint(name='sphere_free_joint', type='free'))
# phone.set('pos', '2.0 0 1.0')
# world.worldbody.append(phone)

# world.merge(can_visual)

box = BoxObject(name="box",size=[0.4,2.7,0.47],rgba=[0.5,0.5,0.5,1],friction=[1,1,1]).get_obj()
box.set('pos', '0.5 -2 0')
world.worldbody.append(box)


model = world.get_model(mode="mujoco_py")
# ipdb.set_trace()
from mujoco_py import MjSim, MjViewer

sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh
# sim.model.geom_pos[sim.model.geom_name2id("Conveyor_belt_collisio")] = ([0, 0 + np.random.uniform(-1,1), 0])
##print the sphere joint position
# print(sim.data.get_joint_qpos('sphere_joint0'))
# ##print the sphere joint velocity
# print(sim.data.get_joint_qvel('sphere_joint0'))
# print(sim.data.get_joint_qpos('sphere_joint0'))
# sim.data.set_joint_qpos('sphere_joint0',[2 , 0, 0, 1, 0, 0, 0])

for i in range(10000):
  sim.data.ctrl[:] = 0
  sim.step()
  viewer.render()
  ##to change the sphere joint values during runtime
  # sim.data.set_joint_qpos('sphere_joint0',[2 + np.random.uniform(-0.3,0.3), 0, 0, 1, 0, 0, 0])
  ##to change the sphere joint velocities during runtime
  # sim.data.set_joint_qvel('sphere_joint0', [0, 0, 0, 0, 0, 0])
  # if i<100:
  #   sim.data.set_joint_qvel('sphere_joint0',[0 , 0, 0, 0, 0, 0])
  # print(sim.data.get_joint_qvel('sphere_joint0'))
  sim.data.set_joint_qvel('box_joint0', [0, 0.4, 0, 0, 0, 0])