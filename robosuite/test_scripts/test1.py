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
	env_name="PickPlaceiPhone", # try with other tasks like "Stack" and "Door"
	robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
	controller_configs=controller_config,            
	has_renderer=is_render,
	has_offscreen_renderer=not is_render,
	use_camera_obs=not is_render,
	render_camera='frontview',
	camera_names = 'birdview',  					# visualize the "frontview" camera
)

# reset the environment
env.reset()
action = np.zeros(7)
R_ie = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
for i in range(100000):
	# action = np.random.randn(env.robots[0].dof) # sample random action
	#time.sleep(2)
	obs, reward, done, info = env.step(action)  # take action in the environment
	print(obs.keys())
	# print(obs.values())
	# print(obs.values())
	object_pos = obs['iPhone12ProMax_pos']
	object_or = obs['iPhone12ProMax_quat']
	#ipdb.set_trace()
	#embed()
	#print(i)
	if True:
		#print(obs.get_keys())
		print("Object posit.: ",object_pos)
		print("Object ori.: ",t3d.quaternions.quat2axangle(object_or))
		R_bi = t3d.quaternions.quat2mat(object_or)
		R_br= np.matmul(R_bi,R_ie)

		ax_r = t3d.axangles.mat2axangle(R_br)


		action[0:3] = object_pos
		print("action position given", action[0:6])
		#print("object_pos", object_pos)
		action[2] += 0.2	
		action[3:6] = np.array([-ax_r[0][1]*ax_r[1],-ax_r[0][2]*ax_r[1],ax_r[0][0]*ax_r[1]])
		# action[3:6] = 0
		# object_or = obs['iPhone_quat']
		# print("actual robot pose: ",t3d.quaternions.quat2axangle(obs['robot0_eef_quat']))
		# print("diff in z: ",obs['robot0_eef_pos'][2]-obs['iPhone12ProMax_pos'][2])
		# print("given robot pose: ",action[3:6])
		# action[6] = 0
		plt.show()
		if is_render:
			env.render()  # render on display
		else:
			plt.imshow(obs['image-state'])
			print("action: ",action)

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
