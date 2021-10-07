import numpy as np
import ipdb
import matplotlib.pyplot as plt

from D3_pick_place import D3_pick_place_env

def plot_variables(var_readings1 = None,var_readings2 = None,is_show = True,choice = 1):
	
	if choice==1:
		fig,a = plt.subplots(2,3)
		a[0][0].plot(var_readings1[0],label="lin1 force_act")
		a[0][0].plot(var_readings2[0],label="lin1 force_cmd")
		a[0][0].legend(loc='upper right')
		a[0][1].plot(var_readings1[1],label="lin2 force_act")
		a[0][1].plot(var_readings2[1],label="lin2 force_cmd")
		a[0][1].legend(loc='upper right')
		a[0][2].plot(var_readings1[2],label="lin3 force_act")
		a[0][2].plot(var_readings2[2],label="lin3 force_cmd")
		a[0][2].legend(loc='upper right')
		a[1][0].plot(var_readings1[3],label="rot1 torque_act")
		a[1][0].plot(var_readings2[3],label="rot1 torque_cmd")
		a[1][0].legend(loc='upper right')
		a[1][1].plot(var_readings1[4],label="rot2 torque_act")
		a[1][1].plot(var_readings2[4],label="rot2 torque_cmd")
		a[1][1].legend(loc='upper right')
		a[1][2].plot(var_readings1[5],label="rot3 torque_act")
		a[1][2].plot(var_readings2[5],label="rot3 torque_cmd")
		a[1][2].legend(loc='upper right')

	if choice==2:
		fig1,b = plt.subplots(2,3)
		b[0][0].plot(var_readings1[0],label="lin1_des")
		b[0][0].plot(var_readings2[0],label="lin1_act")
		b[0][0].legend(loc='upper right')
		b[0][1].plot(var_readings1[1],label="lin2_des")
		b[0][1].plot(var_readings2[1],label="lin2_act")
		b[0][1].legend(loc='upper right')
		b[0][2].plot(var_readings1[2],label="lin3_des")
		b[0][2].plot(var_readings2[2],label="lin3_act")
		b[0][2].legend(loc='upper right')
		b[1][0].plot(var_readings1[3],label="rot1_des")
		b[1][0].plot(var_readings2[3],label="rot1_act")
		b[1][0].legend(loc='upper right')
		b[1][1].plot(var_readings1[4],label="rot2_des")
		b[1][1].plot(var_readings2[4],label="rot2_act")
		b[1][1].legend(loc='upper right')
		b[1][2].plot(var_readings1[5],label="rot3_des")
		b[1][2].plot(var_readings2[5],label="rot3_act")
		b[1][2].legend(loc='upper right')

	if choice==3:
		fig,a = plt.subplots(2,3)
		a[0][0].plot(var_readings1[0],label="lin1 pos_error")
		a[0][0].legend(loc='upper right')
		a[0][1].plot(var_readings1[1],label="lin2 pos_error")
		a[0][1].legend(loc='upper right')
		a[0][2].plot(var_readings1[2],label="lin3 pos_error")
		a[0][2].legend(loc='upper right')
		a[1][0].plot(var_readings1[3],label="rot1 pos_error")
		a[1][0].legend(loc='upper right')
		a[1][1].plot(var_readings1[4],label="rot2 pos_error")
		a[1][1].legend(loc='upper right')
		a[1][2].plot(var_readings1[5],label="rot3 pos_error")
		a[1][2].legend(loc='upper right')


if __name__ == '__main__':

	timestep = 10000
	t = 0


	D3_pp = D3_pick_place_env(True)
	D3_pp.set_env()
	action_zero = np.array([0,0,0.6,0,0,0,-0.2,-0.2])
	obs_current = np.zeros(28)
	obs_last = np.zeros(28)
	gripper_to_finger = 0.09

	torque_reading = [[],[],[],[],[],[],[]]
	torque_cmd = [[],[],[],[],[],[],[]]
	desired_reading = [[],[],[],[],[],[],[]]
	actual_reading = [[],[],[],[],[],[],[]]
	pos_error_reading = [[],[],[],[],[],[],[]]
	plot_variables = 0
	pos_y = 0
	pp = 0 # pp=0, go for pick-place and pp=0 means don't go
	
	while t<timestep:
		print("action_zero", action_zero)
		obs_current,reward,done,_ = D3_pp.step(action_zero)
		if t>=0 and t<1400 and pp==0:
			action_zero[1] -= 0.0002 #motion happening here
			if np.linalg.norm(obs_current[19]-obs_current[12])>0.002:
				pos_x_dir = (obs_current[19]-obs_current[12])/np.abs(obs_current[19]-obs_current[12])
				pos_x += 0.0001*pos_x_dir  #motion happening here
				action_zero[0] = pos_x
				obs_current,reward,done,_ = D3_pp.step(action_zero)
				# print("Stage 1 ",pp)

		### Calculate variables to pick ###
		elif t>=1400 and t<1500 and pp==0:
			obs_current,reward,done,_ = D3_pp.step(action_zero) 
			vel_iPhone = (obs_current[13] - obs_last[13])
			steps_to_reach = ((0.0 - obs_current[13])/vel_iPhone)
			vel_iPhone_rt = (obs_current[13] - obs_last[13])/(0.002) #rt ==> real_time
			# print("Stage 2 ",pp)

		### calculate target velocity ###
		elif t == 1500 and pp==0:
			vel_z = (obs_current[21] - 0.81)/int(steps_to_reach)
			vel_y = (obs_current[20] - 0.0)/int(steps_to_reach)
			# print("Stage 3 ",pp)

		### Start snatch motion ###
		elif np.linalg.norm(obs_current[20]-obs_current[13])<0.001 or pp == 1:
			# print("Stage 4 ",pp)
			obs_current,reward,done,_ = D3_pp.step(action_zero)
			#residual trajectory added here
			#action_zero[0]+= action_network[0] #adding del_x to current_motion
			action_zero[1]+= (vel_iPhone_rt)*0.002 #+ action_network[1] #adding del_y to motion
			action_zero[2]+= vel_z*10 #+ action_network[2]  #adding del_z to motion
			action_zero[2] = min(0.73,action_zero[2])
			action_zero[1] = min(0.6,action_zero[1])
		
			if np.linalg.norm(obs_current[21]-0.83)<0.001 and pp == 1 and obs_current[26]< 0.1:
				# print("stay!!")
				action_zero[2] = pos_z
				action_zero[6] = 0.4
				action_zero[7] = 0.4
				action_zero[2] = min(0.73,action_zero[2])
				action_zero[1] = min(0.6,action_zero[1])
								
			elif obs_current[26]> 0.1 and action_zero[2]>0.55:
				# print("go up!! ship is sinking")
				pos_z -= 0.0005 #motion happening here
				action_zero[2] = pos_z
				action_zero[6] = 0.4
				action_zero[7] = 0.4
				action_zero[2] = min(0.73,action_zero[2])
				action_zero[1] = min(0.6,action_zero[1])
								
			elif action_zero[2]<0.55:
				# print("breaking up of the loop")
				break
			pos_z = action_zero[2]
			pp =1
		else:
			obs_current,reward,done,_ = D3_pp.step(action_zero)

		obs_last = obs_current
			# pass
		#if pick and place

		# if t>=0 and t<10000:


		# 	obs_current,reward,done,_ = D3_pp.step(action_zero) 

		# 	vel_iPhone = (obs_current[13] - obs_last[13])/(0.002/1.2)
		# 	# print("vel_iPhone: ",vel_iPhone)
		# 	steps_to_reach = ((obs_current[20] - obs_current[13])/vel_iPhone)/(0.002/1.2)
		# 	# print("time_to_reach: ",steps_to_reach,t)
		# 	obs_current,reward,done,_ = D3_pp.step(np.array([0,pos_y,0.6,0,0,0,-0.2,-0.2])) 
		# 	pos_y -= 0.00001
		# 	print("pos_y: ",pos_y )
		# elif t == 10000:
		# 	vel_z = (obs_current[21] - 0.81)/int(steps_to_reach+1250)
		# 	print("vel_z: ",vel_z,steps_to_reach)
		# else:
		# 	if np.linalg.norm(obs_current[21]-0.81) > 0.035:
		# 		pos_z = 0.6+vel_z*(t-1000)
		# 		obs_current,reward,done,_ = D3_pp.step(np.array([0,pos_y,pos_z,0,0,0,-0.2,-0.2])) 
		# 		print("Phase 1: vel_z: ",np.linalg.norm(obs_current[21]-0.81))
		# 	elif np.linalg.norm(obs_current[21]-0.81) > 0.0001 and np.linalg.norm(obs_current[21]-0.81)<0.035:
		# 		pos_z = 0.6+vel_z*(t-1000)
		# 		obs_current,reward,done,_ = D3_pp.step(np.array([0,pos_y,pos_z,0,0,0,0.1,0.1])) 
		# 		print("Phase 2: vel_z: ",np.linalg.norm(obs_current[21]-0.81))
		# 		t_up = t
		# 	if obs_current[26]>0.0:
		# 		D3_pp.step(np.array([0,pos_y,pos_z-1e-4*(t-t_up),0,0,0,0.1,0.1])) 
		# 		print("Phase 3: vel_z: ",pos_z-1e-4*(t-t_up))

		# else:
		# 	if obs_current[26] < -0.1:
		# 		pos_z = 0.6+vel_z*(t-1000)
		# 		obs_current,reward,done,_ = D3_pp.step(np.array([0,0,pos_z,0,0,0,-0.2,-0.2])) 
		# 		print("Phase 1: vel_z: ",np.linalg.norm(obs_current[21]-0.81))
		# 	elif obs_current[26] > 0.1:
		# 		pos_z = 0.6+vel_z*(t-1000)
		# 		obs_current,reward,done,_ = D3_pp.step(np.array([0,0,pos_z,0,0,0,0.2,0.2])) 
		# 		print("Phase 2: vel_z: ",np.linalg.norm(obs_current[21]-0.81))
		# 		t_up = t
			# else:
			# 	D3_pp.step(np.array([0,0,pos_z-1e-4*(t-t_up),0,0,0,0.1,0.1])) 
			# 	print("Phase 3: vel_z: ",pos_z-1e-4*(t-t_up))
			# print("observation: ",obs_current[26:28])

		# if t<2000:
		# 	grip_ctrl = -0.2
		# if t>=2000 and t<5000:
		# 	grip_ctrl = 0.2
		# if t>=5000 and t<8000:
		# 	grip_ctrl = -0.2
		# if t>=8000 and t<10000:
		# 	grip_ctrl = 0.2
		# ##if move sideways and test
		# action = np.array([0.0,0,0.7,0,0,0,grip_ctrl,grip_ctrl])
		
		# if t>=0 :
		# 	action[1] -= t*0.0005
		# 	# action[2] += t*0.0004
		# 	# action[0] += t*0.00005
		# 	# action = action_zero
		# else:
		# 	pass

		# D3_pp.step(action)

		# print(t,action_zero)
		t += 1
		for t_n in range(6):
			torque_reading[t_n].append(D3_pp.sim.data.actuator_force[t_n])
			torque_cmd[t_n].append(D3_pp.sim.data.ctrl[t_n])
			desired_reading[t_n].append(D3_pp.joint_sim[t_n])
			actual_reading[t_n].append(D3_pp.sim.data.get_joint_qpos(D3_pp.joint_names[t_n]))
			pos_error_reading[t_n].append(D3_pp.joint_sim[t_n]-D3_pp.sim.data.get_joint_qpos(D3_pp.joint_names[t_n]))
			# print("D3_pp.joint_names[t_n]",str(D3_pp.joint_names[t_n]))

	if plot_variables:
		plot_variables(torque_reading,torque_cmd,choice=1)
		plot_variables(desired_reading,actual_reading,choice=2)
		# plot_variables(actual_reading)
		plot_variables(pos_error_reading,choice=3)

	plt.show()
