import numpy as np
import ipdb
import matplotlib.pyplot as plt
import time

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
	obs_current = np.zeros(19)
	obs_last = np.zeros(19)
	gripper_to_finger = 0.09

	torque_reading = [[],[],[],[],[],[],[]]
	torque_cmd = [[],[],[],[],[],[],[]]
	desired_reading = [[],[],[],[],[],[],[]]
	actual_reading = [[],[],[],[],[],[],[]]
	pos_error_reading = [[],[],[],[],[],[],[]]
	iterations= 350
	mult_factor=0.03
	add_factor=650
	start = False
	y=np.zeros(timestep)
	t_arr=np.linspace(0,timestep,timestep)
	while t<timestep:

		##original if pick and place

		# if t==0:
		# 	obs_current,reward,done,_ = D3_pp.step(action_zero) 
		# if (obs_current[20] - obs_current[13])>0.2:
		# 	obs_current,reward,done,_ = D3_pp.step(action_zero) 
		# 	# print("gripper y position", obs_current[20])
		# 	# print("vel. iphone_sim", D3_pp.sim.data.get_joint_qvel('iphonebox_joint0'))
		# 	# vel_iPhone = (obs_current[13] - obs_last[13])/(0.002/1.2)
		# 	vel_iPhone = D3_pp.sim.data.get_joint_qvel('iphonebox_joint0')
		# 	print("vel_iPhone: ",vel_iPhone[1])
		# 	steps_to_reach = ((obs_current[20] - obs_current[13])/(vel_iPhone[1]/1.05))#/(0.002/1.2)
		# 	# print("time_to_reach: ",steps_to_reach,t)
		# 	t_last = t
		# if (obs_current[20] - obs_current[13])<=0.2:
		# 	if np.linalg.norm(obs_current[21]-obs_current[14]) > (0.085 + vel_iPhone[1]*mult_factor) and iterations>0:
		# 		vel_z = (obs_current[21] - 0.73)/int(steps_to_reach + add_factor - vel_iPhone[1]*600)
		# 		pos_z = min(0.6+vel_z*(t-t_last),0.725)
		# 		pos_y = min(vel_z*(t-t_last),0.12)
		# 		obs_current,reward,done,_ = D3_pp.step(np.array([0,pos_y,pos_z,0,0,0,-0.2,-0.2])) 
		# 		# print("Phase 1: vel_z: ",np.linalg.norm(obs_current[21]-0.73))
		# 		vel_iPhone = D3_pp.sim.data.get_joint_qvel('iphonebox_joint0')
		# 		print("vel_iPhone: ",vel_iPhone[1]/1.05)
		# 		steps_to_reach = ((obs_current[20] - obs_current[13])/(vel_iPhone[1]/1.05))#/(0.002/1.2)
		# 		print("time_to_reach: ",steps_to_reach,t)
		# 		print(obs_current[21]-obs_current[14])
		# 	elif np.linalg.norm(obs_current[21]-obs_current[14])<=(0.085 + vel_iPhone[1]*mult_factor) and iterations>0:
		# 		pos_z = min(0.6+vel_z*(t-t_last),0.725)
		# 		pos_y = min(vel_z*(t-t_last),0.12)
		# 		if pos_z==0.725:
		# 			obs_current,reward,done,_ = D3_pp.step(np.array([0,pos_y,pos_z,0,0,0,0.2,0.2])) 
		# 		else:
		# 			obs_current,reward,done,_ = D3_pp.step(np.array([0,pos_y,pos_z,0,0,0,-0.2,-0.2])) 
		# 		# print("Phase 2: vel_z: ",np.linalg.norm(obs_current[21]-0.73))
		# 		vel_iPhone = D3_pp.sim.data.get_joint_qvel('iphonebox_joint0')
		# 		print("vel_iPhone: ",vel_iPhone[1]/1.05)
		# 		steps_to_reach = ((obs_current[20] - obs_current[13])/(vel_iPhone[1]/1.05))#/(0.002/1.2)
		# 		print("time_to_reach: ",steps_to_reach,t)
		# 		t_up = t
		# 		iterations-= 1
		# 	elif iterations<=0:
		# 		pos_z = max(0.725-0.0001*(t-t_up),0.51)
		# 		pos_y = max(0.12-0.0001*(t-t_up),0.0)
		# 		obs_current,reward,done,_ = D3_pp.step(np.array([0,pos_y,pos_z,0,0,0,0.2,0.2])) 
		# 		# print("Phase 2: vel_z: ",np.linalg.norm(obs_current[21]-0.73))
		# 		vel_iPhone = D3_pp.sim.data.get_joint_qvel('iphonebox_joint0')
		# 		print("vel_iPhone: ",vel_iPhone[1]/1.05)
		# 		steps_to_reach = ((obs_current[20] - obs_current[13])/(vel_iPhone[1]/1.05))#/(0.002/1.2)
		# 		print("time_to_reach: ",steps_to_reach,t)
				
		##modified pick and place
		
		if t==0 or (not start):
			obs_current,reward,done,_ = D3_pp.step(action_zero) 
			t_last = t
			vel_iPhone = np.round(D3_pp.sim.data.get_joint_qvel('iphonebox_joint0'),1)
			print("phase 1",t)
		if (obs_current[20] - obs_current[13])<=0 or start:
			start = True
			print("vel_iPhone", vel_iPhone)
			if np.linalg.norm(obs_current[21]-obs_current[14]) > 0.075 and iterations>0:
				pos_z = min(0.6+vel_iPhone[1]*0.004*(t-t_last),0.74)
				pos_y = min(vel_iPhone[1]*0.002*(t-t_last),0.3)
				# print("pos_y", pos_y)
				y[t-t_last-1]=pos_y
				obs_current,reward,done,_ = D3_pp.step(np.array([0,pos_y,pos_z,0,0,0,-0.2,-0.2])) 
				print("phase 2",t)
			elif np.linalg.norm(obs_current[21]-obs_current[14])<=0.075 and iterations>0:
				pos_z = min(0.6+vel_iPhone[1]*0.004*(t-t_last),0.74)
				pos_y = min(vel_iPhone[1]*0.002*(t-t_last),0.3)
				obs_current,reward,done,_ = D3_pp.step(np.array([0,pos_y,pos_z,0,0,0,0.2,0.2])) 
				y[t-t_last-1]=pos_y
				t_up = t
				y_last = pos_y
				iterations-= 1
				print("phase 3",t)
			elif iterations<=0:
				pos_z = max(0.74-vel_iPhone[1]*0.001*(t-t_up),0.51)
				pos_y = max(y_last-vel_iPhone[1]*0.001*(t-t_up),0.0)
				y[t-t_last-1]=pos_y
				obs_current,reward,done,_ = D3_pp.step(np.array([0,pos_y,pos_z,0,0,0,0.2,0.2])) 
				print("phase 4",t)
				

		##y-axis std. trajectory
		# target_traj1=np.array([[0,0,0.6,0,0,0] for i in range(0,1000)])
		# target_traj2=np.array([[0,min(0+i*0.0005,0.45),0.6,0,0,0] for i in range(0,9000)])
		# target_traj=np.block([[target_traj1],[target_traj2]])
		# print("iterations:",iterations)
		# if (obs_current[20]- obs_current[13])<0.01:
		# 	print("now close",t)
		# 	# time.sleep(10)
		# if t<2000:
		# 	grip_ctrl = -0.2
		# if t>=2000 and t<5000:
		# 	grip_ctrl = 0.2
		# if t>=5000 and t<8000:
		# 	grip_ctrl = -0.2
		# if t>=8000 and t<10000:
		# 	grip_ctrl = 0.2
		# # if move sideways and test
		# action = np.array([0.0,0,0.51,0,0,0,grip_ctrl,grip_ctrl])
		
		# if t>=1000:
		# 	action[0] += (t-1000)*0.00003#*0.001#*0.01
		# 	if action[0]>0.4:
		# 		action[0] = 0.4
		# 	action[1] -= (t-1000)*0.00003#*0.01*0.1
		# 	if action[1]<-0.4:
		# 		action[1] = -0.4
		# 	action[2] += (t-1000)*0.00003#*0.1*0.01
		# 	if action[2]>0.72:
		# 		action[2] = 0.72
			# action[2] += t*0.0004
			# action[0] += t*0.00005
			# action = action_zero
		# else:
		# 	pass
		# if t<1000:
		# 	action = np.array([0,0,0.6,0,0,0,-0.2,-0.2])
		# else:
		# 	action = np.array([0,min((t-999)*0.0005,0.45),0.6,0,0,0,-0.2,-0.2])

		# D3_pp.step(action)

		obs_last = obs_current
		# print(t,action)
		t += 1
		for t_n in range(6):
			torque_reading[t_n].append(D3_pp.sim.data.actuator_force[t_n])
			torque_cmd[t_n].append(D3_pp.sim.data.ctrl[t_n])
			desired_reading[t_n].append(D3_pp.joint_sim[t_n])
			actual_reading[t_n].append(D3_pp.sim.data.get_joint_qpos(D3_pp.joint_names[t_n]))
			pos_error_reading[t_n].append(D3_pp.joint_sim[t_n]-D3_pp.sim.data.get_joint_qpos(D3_pp.joint_names[t_n]))
			# print("D3_pp.joint_names[t_n]",str(D3_pp.joint_names[t_n]))

	# plot_variables(torque_reading,torque_cmd,choice=1)
	# plot_variables(desired_reading,actual_reading,choice=2)
	# # plot_variables(actual_reading)
	# plot_variables(pos_error_reading,choice=3)
	plt.figure(1)
	plt.plot(t_arr, y)
	

	plt.show()


