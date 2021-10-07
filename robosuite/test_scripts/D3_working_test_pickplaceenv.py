import numpy as np
import robosuite as suite
import matplotlib.pyplot as plt
from robosuite import load_controller_config
from mujoco_py import load_model_from_path, MjSim, MjViewer
import time
import transforms3d as t3d
import invK
import trajGenerator as trg
from interpolateTraj import interpolateTraj
import math

def traj(j1):
	# j1 = np.array([0.1,0.05,-0.8,0,0,0])
	target_wp1 = np.array([[0.0,0.0,0.6,0,0,0],
					[j1[0],j1[1],-j1[2],0,0,0]])
	dist = math.sqrt(j1[0]**2 + j1[1]**2 + (-j1[2]-0.6)**2)
	# dist = 0.22913
	# multiplier = int(dist*196.4)
	multiplier = int(dist*630)
	# print("dist", dist)
	# print("target_wp1",target_wp1[1])
	# print("multiplier",multiplier)
	# time.sleep(200)
	target_traj1=interpolateTraj(target_wp1,multiplier)
	return target_traj1


is_render = True
controller_names = ["OSC_POSE", "WOMBAT_ARM_IK"]#["OSC_POSE","OSC_POSITION"]
controller_config = load_controller_config(default_controller=controller_names[1])
controller_config['control_delta'] = False
print("controller loaded test.py")
# quit()
# create environment instance
env = suite.make(
	env_name="PickPlaceiPhone",#"PickPlaceiPhone", # try with other tasks like "Stack" and "Door"
	robots="Wombat_arm",  # try with other robots like "Sawyer" and "Jaco"
	controller_configs=controller_config,            
	has_renderer=is_render,
	has_offscreen_renderer=not is_render,
	use_camera_obs=not is_render,
	render_camera='frontview',
	camera_names = 'birdview',  					# visualize the "frontview" camera
)
print("environment instance created test.py")
#model=load_model_from_path("/home/yashraghav/robosuite/robosuite/models/assets/robots/wombat_arm/wombat_arm.xml")
#sim = MjSim(model)
#sim_state = sim.get_state()
#print(sim.get_joint_qpos("robot0_branch1_joint"))
# reset the environment
# env.render()
#time.sleep(1000)
env.reset()
# env.render()
t = 0
timestep= 0.0005
t_final = 450
# sim_state = sim.get_state()
q_pos_last = [0]*6
# vertical circle
# r=0.08
# dt=1.0/(t_final-t)
# target_traj=np.array([[r*np.sin(i*dt*np.pi*2),0.0,0.7-r*np.cos(i*dt*np.pi*2),0,0,0] for i in range(t,t_final)])

r=0.1
tLin=50
dt=1.0/(t_final-t-tLin)
target_traj1=np.array([[0,r*np.sin(np.pi/2*(float(i)/tLin)),0.6,0,0,0] for i in range(0,tLin)])
target_traj2=np.array([[r*np.sin((i)*dt*np.pi*2),r*np.cos(i*dt*np.pi*2),0.6,0,0,0] for i in range(t,t_final-tLin)])
target_traj=np.block([[target_traj1],[target_traj2]])
# target_wp1=np.array([[0.0,0.0,0.6,0,0,0],
# 					[-0.04046,-0.21,0.84831,0,0,0]])
# # 					# [-0.2,0.1,0.7,0,0,0]])
# # #					[0.15,0.3,0.65,-0.4,0.1,0],
# # #					[0.15,0.3,0.65,-0.4,0.1,0]])
# target_wp2=np.array([[0.1,0.05,0.8,0,0,0],
# 					[0.0,0.0,0.6,0,0,0]])
# target_wp3=np.array([[0.0,0.0,0.6,0,0,0],
# 					[-0.1,-0.05,0.8,0,0,0]])
# target_wp4=np.array([[-0.1,-0.05,0.8,0,0,0],
# 					[0.0,0.0,0.6,0,0,0]])
# target_wp5=np.array([[0.0,0.0,0.6,0,0,0],
# 					[0.1,-0.05,0.8,0,0,0]])
# target_wp6=np.array([[0.1,-0.05,0.8,0,0,0],
# 					[0.0,0.0,0.6,0,0,0]])
# target_wp7=np.array([[0.0,0.0,0.6,0,0,0],
# 					[-0.1,0.05,0.8,0,0,0]])
# target_wp8=np.array([[-0.1,0.05,0.8,0,0,0],
# 					[0.0,0.0,0.6,0,0,0]])
# target_traj1=interpolateTraj(target_wp1,45)
# target_traj2=interpolateTraj(target_wp2,45)
# target_traj3=interpolateTraj(target_wp3,45)
# target_traj4=interpolateTraj(target_wp4,45)
# target_traj5=interpolateTraj(target_wp5,45)
# target_traj6=interpolateTraj(target_wp6,45)
# target_traj7=interpolateTraj(target_wp7,45)
# target_traj8=interpolateTraj(target_wp8,45)
# target_traj=np.block([[target_traj1],[target_traj2],[target_traj3],[target_traj4],[target_traj5],[target_traj6],[target_traj7],[target_traj8]])
# target_wp1=np.array([[0.0,0.0,0.6,0,0,0],
# 					[0.0,0,0.75,0,0,0]])
# target_wp2=np.array([[0.0,0,0.75,0,0,0],
# 					[0.0,0.0,0.6,0,0,0]])
# target_wp3=np.array([[0.0,0.0,0.6,0,0,0],
# 					[-0.0,0.15,0.75,0,0,0]])
# target_wp4=np.array([[-0.0,0.15,0.75,0,0,0],
# 					[0.0,0.0,0.6,0,0,0]])
# target_traj1=interpolateTraj(target_wp1,35)
# target_traj2=interpolateTraj(target_wp2,35)
# target_traj3=interpolateTraj(target_wp3,35)
# target_traj4=interpolateTraj(target_wp4,55)
# target_traj=np.block([[target_traj1],[target_traj2],[target_traj3],[target_traj4]])
# print("shape",target_traj.shape)
# time.sleep(200)
#convert to joint trajectory
# joint_target_traj=trg.jointTraj(target_traj)
# #current target joint values, in IK frame
# joint_real=joint_target_traj[t]
# joint_sim=invK.ik_wrapper(joint_real)
# action = np.array([joint_sim[0],joint_sim[1],joint_sim[2],joint_sim[3],joint_sim[4],joint_sim[5]])
temp = np.zeros(6)
# action = np.zeros(6)
##next 1 line when gripper added externally
# action = np.zeros(7)
action1 = np.zeros(6)
target_sim = np.zeros(6)
temp = invK.real2sim_wrapper([0,0,0.6,0,0,0])
action = invK.real2sim_wrapper([0,0,0.6,0,0,0])
##next 3 lines when gripper added externally
# temp = invK.real2sim_wrapper([0,0,0.6,0,0,0])
# action[0:6] = temp
# action[6] = 0
R_ie = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
i=0
# R_ie = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
for i in range(1000):
	#action = np.random.randn(env.robots[0].dof) # sample random action
	#print(env.robots[0].dof)
	# if i%2==0:
	# 	action = np.array([0.5,0.5,0.5,0.5,0.5,0.5])
	# else:
	# 	action = np.array([1,1,1,1,1,1])
	# time.sleep(2)
	#action = target_traj[i]
	print("i",i)
	# action = invK.real2sim_wrapper(target_traj[i])
	if i>0:
		print("sim action taken in test.py", action)
		print("real action taken in test.py", target_traj[i])

	action=np.array([0.1,0.1,0.6,0,0,0,0])
	# action = invK.real2sim_wrapper([0.1,0.1,0.6,0,0,0])
	print("action",action)
	obs, reward, done, info = env.step(action)  # take action in the environment
	object_pos = obs['iPhone12ProMax_pos']
	object_or = obs['iPhone12ProMax_quat']
	obj1 = obs['robot0_eef_pos']
	obj2 = obs['robot0_eef_quat']
	obj3 = obs['robot0_gripper_qpos']
	print("robot0_eef_pos",obj1)
	print("robot0_eef_quat",obj2)
	print("robot0_gripper_qpos",obj3)
	print("object_pos",object_pos)
	# time.sleep(100)
	R_bi = t3d.quaternions.quat2mat(object_or)
	R_br = np.matmul(R_bi,R_ie)
	ax_r = t3d.axangles.mat2axangle(R_br)
	##next if block when gripper added externally
	# if i==0:
	# 	target_sim[0:3] = object_pos
	# 	target_sim[2] += -0.12
	# 	target_sim[3:6] = np.array([-ax_r[0][1]*ax_r[1],-ax_r[0][2]*ax_r[1],ax_r[0][0]*ax_r[1]])
	# 	# target_sim[3:6] = np.array([0,0,0])
	# 	target_real = invK.sim2real_wrapper(target_sim)
	# 	target_traj = traj(target_real)
	# 	print("target_sim", target_sim)
	# 	print("target_real", target_real)
	# 	time.sleep(200)
	i=i+1
	##next 3 lines when gripper added externally
	# action1 = invK.real2sim_wrapper(target_traj[i])
	# action[0:6] = action1
	# action[6] = 0
	action = invK.real2sim_wrapper(target_traj[i])
	print(obs.keys())
	# env.sim.model.geom_pos[env.sim.model.geom_name2id("iPhone12ProMax_g0")] = ([0.145, 0.195 + np.random.uniform(-0.5,0.5), 0.82])
	# env.sim.model.geom_pos[env.sim.model.geom_name2id("iPhone12ProMax_g0_visual")] = ([0.145, 0.195 + np.random.uniform(-0.5,0.5), 0.82])
	# env.sim.model.geom_pos[env.sim.model.geom_name2id("Conveyor_belt_collision1")] = ([0, 0 + np.random.uniform(-1,1), 0])
	# env.sim.model.geom_pos[env.sim.model.geom_name2id("Conveyor_belt_visual1")] = ([0, 0 + np.random.uniform(-1,1), 0])
	# print(obs['object-state'])
	# object_pos = obs['iPhone_pos']
	# object_or = obs['iPhone_quat']
	# object_pos = obs['iPhone12ProMax_pos']
	# object_or = obs['iPhone12ProMax_quat']
	# object_pos = obs['Can_pos']
	# object_or = obs['Can_quat']
	#print(i)
	# if i==2:
	# 	time.sleep(200)
	if True:
		# print("Object pos: ",object_pos)
		# print("Object or: ",t3d.quaternions.quat2axangle(object_or))
		# R_bi = t3d.quaternions.quat2mat(object_or)
		# R_br= np.matmul(R_bi,R_ie)

		# ax_r = t3d.axangles.mat2axangle(R_br)


		# action[0:3] = object_pos
		# action[2] += 0.2	
		# action[3:6] = np.array([-ax_r[0][1]*ax_r[1],-ax_r[0][2]*ax_r[1],ax_r[0][0]*ax_r[1]])
		#action[3:6] = 0
		##object_or = obs['iPhone12ProMax_quat']
		#print("actual robot pose: ",t3d.quaternions.quat2axangle(obs['robot0_eef_quat']))
		#print("diff in z: ",obs['robot0_eef_pos'][2]-obs['iPhone_pos'][2])
		#print("given robot pose: ",action[3:6])
		#action[6] = 0
		plt.show()
		if is_render:
			env.render()  # render on display
			# print(target_traj[719])
		else:
			plt.imshow(obs['image-state'])
			#print("action: ",action)
			print(7)



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