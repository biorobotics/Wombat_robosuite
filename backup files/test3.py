from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Wombat_arm
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.arenas import BinsArena
from robosuite.models.objects import BallObject
from robosuite.models.objects import CanObject
from robosuite.utils.mjcf_utils import new_joint
from mujoco_py import MjSim, MjViewer

world = MujocoWorldBase()
mujoco_robot = Wombat_arm()

#gripper = gripper_factory('PandaGripper')
gripper = gripper_factory('Robotiq85Gripper')
#gripper = gripper_factory('RethinkGripper')
#gripper = gripper_factory('Robotiq140Gripper')
#gripper.hide_visualization()
mujoco_robot.add_gripper(gripper)

mujoco_robot.set_base_xpos([0.9, 0.06, 1.6])
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
#can = CanObject
#can.set('pos', '1.0 0 1.0')
#world.worldbody.append(can)

model = world.get_model(mode="mujoco_py")

sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

for i in range(10000):
  sim.data.ctrl[:] = 0
  #sim.step()
  viewer.render()
