import numpy as np
import robosuite as suite
import matplotlib.pyplot as plt
import time
from robosuite import load_controller_config

import transforms3d as t3d
import ipdb
from IPython import embed

is_render = True
controller_names = ["OSC_POSE","OSC_POSITION","JOINT_POSITION"]
controller_config = load_controller_config(default_controller=controller_names[0])
controller_config['control_delta'] = False
# print(controller_config)
# quit()
# create environment instance
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

action= np.zeros(7)
action = np.array([0.1, 0.1, 0.9,0,0,0,0])
R_ie = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
for i in range(100000):
    # action = np.random.randn(env.robots[0].dof) # sample random action
    #time.sleep(2)
    env.sim.data.set_joint_qvel('iphone_joint0', [0,0.1,0,0,0,0])
    obs, reward, done, info = env.step(action)  # take action in the environment
    # env.sim.data.set_joint_qpos('robot0_joint_1', -1)
    # env.sim.data.set_joint_qpos('robot0_joint_2', -0.5)
    # env.sim.data.set_joint_qpos('robot0_joint_3', 0.5)
    # env.sim.data.set_joint_qpos('robot0_joint_4', 0.5)
    # env.sim.data.set_joint_qpos('robot0_joint_5', 5)
    # env.sim.data.set_joint_qpos('robot0_joint_6', 5)
    # print(obs.keys())
    # time.sleep(200)
    # state_dim = 13#env.observation_space.shape[0]
    # action_dim = 7#env.action_space.shape[0]
    # print(state_dim)
    # print(action_dim)
    object_pos = obs['phone_pos']
    object_or = obs['phone_quat']
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
