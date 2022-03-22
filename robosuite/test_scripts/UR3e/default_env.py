import numpy as np
import robosuite as suite
import matplotlib.pyplot as plt
import time
from robosuite import load_controller_config

import transforms3d as t3d
import ipdb
from IPython import embed
from ur_ikfast import ur_kinematics 

ur3e_arm = ur_kinematics.URKinematics('ur3e')


is_render = True
controller_names = ["OSC_POSE","OSC_POSITION","JOINT_POSITION"]
controller_config = load_controller_config(default_controller=controller_names[0])
controller_config['control_delta'] = False
# print(controller_config)
# quit()
# create environment instance


def ik(pose):
    print(f"pose {pose}")
    return ur3e_arm.inverse(pose, False)

def move_robot(env, joint_qpos):
    for i in range(len(joint_qpos)):
        joint_name = "robot0_joint_" + str(i+1)
        env.sim.data.set_joint_qpos(joint_name, joint_qpos[i])



env = suite.make(
	env_name="PickiPhone", # try with other tasks like "Stack" and "Door"
	robots="UR3e",  # try with other robots like "Sawyer" and "Jaco"
    gripper_types="PandaGripper",
	controller_configs=controller_config,            
	has_renderer=is_render,
	has_offscreen_renderer=not is_render,
	use_camera_obs=not is_render,
	render_camera='frontview',
	camera_names = 'birdview',  					# visualize the "frontview" camera
)

# reset the environment
env.reset() 

# joint_angles = [-3.1, -1.6, 1.6, -1.6, -1.6, 0.] 
# home_robot(env)

ee_pose= np.zeros(7)
# ee_pose = [0, 0, -1.5708, 0, 1.50797, np.pi *20/180.0, 0]
# ee_pose = np.array([0.1, 0.1, 0.5,0,0,0,1])
observations = env._get_observations()
# print(observations['robot0_eef_quat'])
# print(observations['robot0_eef_pos'])
# pos: [-0.11685664  0.14085101  0.78299397]
# quat: [-0.93662904  0.34983983  0.01160817  0.01426127]
# pos: [-0.10655796  0.13434555  0.79233005]
# quat: [0.57185142 0.81973174 0.02626598 0.01832835]

print(f"pos: {observations['robot0_eef_pos']}")
print(f"quat: {observations['robot0_eef_quat']}")


ee_pose = np.zeros(7)
joint_values = [0, 0, -1.5708, 0, 1.50797, np.pi *20/180.0]
ee_pose_start = ur3e_arm.forward(joint_values)
ee_pose[0:6] = [-0.315, 0.05, 0.3, 0, 0, 0]
# action[0] = 0

for i in range(100000):
    # action = np.random.randn(env.robots[0].dof) # sample random action
    #time.sleep(2)
    env.sim.data.set_joint_qvel('iphone_joint0', [0,0.1,0,0,0,0])

    # joint_angles = ik(ee_pose)
    # action_joint = np.zeros(7)
    # action_joint[0:6] = joint_angles
    # print(f"action_joint {action_joint}")
    obs, reward, done, info = env.step(ee_pose)
    
      # take action in the environment
    # env.sim.data.set_joint_qpos('robot0_joint_1', -1)
    # env.sim.data.set_joint_qpos('robot0_joint_2', -0.5)
    # env.sim.data.set_joint_qpos('robot0_joint_3', 0.5)
    # env.sim.data.set_joint_qpos('robot0_joint_4', 0.5)
    # env.sim.data.set_joint_qpos('robot0_joint_5', 5)
    # env.sim.data.set_joint_qpos('robot0_joint_6', 5)
    # print(f" test2.py -> observation {obs}")

    ee_pose[0] += 0.00001

    # action[0] +=0.00001
    # print(f"robot eef pos {current_eef_pos}")
    # print(f"test2.py -> Control Signals {env.sim.data.ctrl}")
    # print(obs.keys())
    # time.sleep(200)
    # state_dim = 13#env.observation_space.shape[0]
    # action_dim = 7#env.action_space.shape[0]
    # print(state_dim)
    # print(action_dim)
    # object_pos = obs['phone_pos']
    # object_or = obs['phone_quat']
    #ipdb.set_trace()
    #embed()
    #print(i)

    if is_render:
        env.render()  # render on display


	# else:
	# 	ipdb.set_trace()



# controller_config['control_delta'] = True
# action = np.zeros(7)

# object_pos = 0
# robot_pos = 0.2

# for i in range(500):
# 	# print("in stage 2",obs['robot0_eef_pos'][2]-obs['iPhone_pos'][2])

# 	print(controller_config)	
# 	obs, reward, done, info = env.step(action)
# 	if is_render:
# 		env.render()  # render on display
# 	object_pos = obs['iPhone_pos'][2]
# 	robot_pos = obs['robot0_eef_pos'][2]
