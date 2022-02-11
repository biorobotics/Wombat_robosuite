import trajGenerator as trg
from interpolateTraj import interpolateTraj
import robosuite
from robosuite.models.objects import BoxObject
from robosuite.models.robots import Wombat_arm
from robosuite.models.arenas import EmptyArena
from robosuite.models.grippers import gripper_factory
import invK
import math
import copy
from mujoco_py import MjSim, MjViewer
import numpy as np
import time
import matplotlib as mpl
import ipdb
mpl.use('TkAgg')
import matplotlib.pyplot as plt 
from robosuite.models import MujocoWorldBase
import pyautogui
import cv2


##PD controller below
def PD_controller_rot(des,current,q_pos_last,scale):
	
	#kp = 10
	#kp=10
	#kp = 1
	#kd = 0.3
	#kp = 5
	#kd = 0.6
	kp=10*scale#20*scale ####10
	kd=4#0.6 ####4
	qpos = des+kp*(des-current)-kd*(current-q_pos_last)
	# print(kp*(des-current))
	return qpos

	# return np.array(points)
def PD_controller_lin(des,current,q_pos_last,scale):
	
	#kp = 10
	#kd = 0.8
	#kp=10
	#kd=0.1
	kp=800#150 ####1000
	kd=1500 ####1500
	qpos = des+kp*(des-current)-kd*(current-q_pos_last)
	# print(kp*(des-current))
	return qpos

##Gripper PD controller below
def Gripper_PD_controller(des,current,last,state):
	if state=='left':
		kp=10
		kd=2
		pos = des+kp*(des-current)-kd*(current-last)
		return pos
	if state=='right':
		kp=10
		kd=2
		pos = des+kp*(des-current)-kd*(current-last)
		return pos


#scales the PD signal based on the ee pos or joint values; wombat_arm needs
#different PD values depending on where it is, position-wise
def PD_signal_scale(ee_pos,joint_vals):
	ee_xy_disp=np.array([math.sqrt(ee_pos[0]**2+ee_pos[1]**2)]*6)+1.0
	lin_vals=np.array([joint_vals[2],joint_vals[0],joint_vals[1]]*2)+1.0
	scale=7
	PD_scale_factor=((np.multiply(ee_xy_disp,lin_vals)**2)-1)*scale
	#print("PD_scale_factor:",PD_scale_factor)
	#PD_scale_factor=np.array([1,1,1,1,1,1])
	return PD_scale_factor

##Building a custom world, importing wombat_arm and adding gripper below
world = MujocoWorldBase()
mujoco_robot = Wombat_arm()

# gripper = gripper_factory(None)
# # # gripper = gripper_factory('D3_gripper')
# # #gripper.hide_visualization()
# mujoco_robot.add_gripper(gripper)

# mujoco_robot.set_base_xpos([0.4, 0.06, 0])
mujoco_robot.set_base_xpos([0, 0.0, 0])
world.merge(mujoco_robot)

#mujoco_arena = TableArena()
# mujoco_arena = BinsArena()
mujoco_arena =EmptyArena()
# mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

##adding iphone as a box and conveyor belt as a box below
iphonebox = BoxObject(name="iphonebox",size=[0.04,0.11,0.0069],rgba=[0,0,0,1],friction=[1,1,1]).get_obj()
iphonebox.set('pos', '0.6 2 1')
world.worldbody.append(iphonebox)

box = BoxObject(name="box",size=[0.35,9.7,0.37],rgba=[0.5,0.5,0.5,1],friction=[1,1,1]).get_obj()
box.set('pos', '0.6 -2 0')
world.worldbody.append(box)

model = world.get_model(mode="mujoco_py")

sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

viewer.render()

t_final = 5000#10000
torque1=[[None]]*t_final
torque2=[[None]]*t_final
torque3=[[None]]*t_final
torque4=[[None]]*t_final
torque5=[[None]]*t_final
torque6=[[None]]*t_final
gripper_torque_left=[[None]]*t_final
gripper_torque_right=[[None]]*t_final
gripper_joint_left=[[None]]*t_final
gripper_joint_right=[[None]]*t_final
quat1=[[None]]*t_final
quat2=[[None]]*t_final
quat3=[[None]]*t_final
quat4=[[None]]*t_final
torque_act1=[[None]]*t_final
torque_act2=[[None]]*t_final
torque_act3=[[None]]*t_final
torque_act4=[[None]]*t_final
torque_act5=[[None]]*t_final
torque_act6=[[None]]*t_final
t = 0
timestep= 0.002

sim_state = sim.get_state()
q_pos_last = [0]*6

t_arr=np.linspace(timestep*t,timestep*t_final,t_final-t)


##horizontal circle trajectory
# r=0.3
# tLin=2000
# dt=1.0/(t_final-t-tLin)
# target_traj1=np.array([[0,-r*np.sin(np.pi/2*(float(i)/tLin)),0.6,0,0,0] for i in range(0,tLin)])
# target_traj2=np.array([[-r*np.sin((i)*dt*np.pi*2),-r*np.cos(i*dt*np.pi*2),0.6,0,0,0] for i in range(t,t_final- tLin)])
# target_traj=np.block([[target_traj1],[target_traj2]])
#y-axis back and forth trajectory
# target_traj1=np.array([[0,0,0.6,0,0,0] for i in range(0,200)])
# target_traj2=np.array([[0,0-i*0.00004,0.6,0,0,0] for i in range(0,4900)])
# target_traj3=np.array([[0,-0.19596+i*0.000008,0.6,0,0,0] for i in range(0,4900)])
# target_traj=np.block([[target_traj1],[target_traj2],[target_traj3]])
##y-axis back and forth fast trajectory
target_traj1=np.array([[0,0,0.6,0,0,0] for i in range(0,1000)])
target_traj2=np.array([[0,0-i*0.0002,0.6,0,0,0] for i in range(0,2000)])
target_traj3=np.array([[0,-0.4+i*0.0002,0.6,0,0,0] for i in range(0,2000)])
target_traj=np.block([[target_traj1],[target_traj2],[target_traj3]])
##y-axis std. trajectory
# target_traj1=np.array([[0,0,0.6,0,0,0] for i in range(0,1000)])
# target_traj2=np.array([[0,min(0+i*0.0005,0.45),0.6,0,0,0] for i in range(0,9000)])
# target_traj=np.block([[target_traj1],[target_traj2]])
##x-axis back and forth trajectory
# target_traj1=np.array([[0,0,0.6,0,0,0] for i in range(0,200)])
# target_traj2=np.array([[0+i*0.00008,0,0.6,0,0,0] for i in range(0,4900)])
# target_traj3=np.array([[0.39192-i*0.00008,0,0.6,0,0,0] for i in range(0,4900)])
# target_traj=np.block([[target_traj1],[target_traj2],[target_traj3]])
##x-axis fast back and forth trajectory
# target_traj1=np.array([[0,0,0.6,0,0,0] for i in range(0,500)])
# target_traj2=np.array([[0+i*0.0004,0,0.6,0,0,0] for i in range(0,1000)])
# target_traj3=np.array([[0.4-i*0.0004,0,0.6,0,0,0] for i in range(0,1000)])
# target_traj=np.block([[target_traj1],[target_traj2],[target_traj3]])
##z-axis back and forth trajectory
# target_traj1=np.array([[0,0,0.6,0,0,0] for i in range(0,200)])
# target_traj2=np.array([[0,0,min(0.6+i*0.0000316,0.72),0,0,0] for i in range(0,4900)])
# target_traj3=np.array([[0,0,0.72-i*0.0000316,0,0,0] for i in range(0,4900)])
# target_traj=np.block([[target_traj1],[target_traj2],[target_traj3]])
##z-axis fast back and forth trajectory
# target_traj1=np.array([[0,0,0.6,0,0,0] for i in range(0,500)])
# target_traj2=np.array([[0,0,0.6+i*0.000155,0,0,0] for i in range(0,1000)])
# target_traj3=np.array([[0,0,0.755-i*0.000155,0,0,0] for i in range(0,1000)])
# target_traj=np.block([[target_traj1],[target_traj2],[target_traj3]])
##vertical circle
# r=0.1
# dt=1.0/(t_final-t)
# target_traj=np.array([[r*np.sin(i*dt*np.pi*2),0.0,0.6+r*np.cos(i*dt*np.pi*2),0,0,0] for i in range(t,t_final)])
# target_traj=np.array([[-0.2,0,0.7,0,0,0] for i in range(t,t_final)])
#convert to joint trajectory
joint_target_traj,qu =trg.jointTraj(target_traj)
# joint_target_traj=trg.jointTraj(target_traj)
# print(target_traj)
# print(joint_target_traj)
sim.set_state(sim_state)
j_actual=np.zeros((t_final-t,6))
j_goal=np.zeros((t_final-t,6))
branch2_ee_pos=np.zeros((t_final-t,3))
branch3_ee_pos=np.zeros((t_final-t,3))
ee_plate_2_pos=np.zeros((t_final-t,3))
ee_plate_3_pos=np.zeros((t_final-t,3))
grip_signal=[-0.2]*2
ee_pose_des = []
ee_pose_current = []
# sim.data.set_joint_qpos('robot0_gripper_left_finger_joint', -0.287884)
# sim.data.set_joint_qpos('robot0_gripper_right_finger_joint', -0.295456)


while t<t_final:
	#rotary
	q_pos_last[0] = sim.data.get_joint_qpos("robot0_branch1_joint")
	q_pos_last[1] = sim.data.get_joint_qpos("robot0_branch2_joint")
	q_pos_last[2] = sim.data.get_joint_qpos("robot0_branch3_joint")
	#linear
	q_pos_last[3] = sim.data.get_joint_qpos("robot0_branch1_linear_joint")
	q_pos_last[4] = sim.data.get_joint_qpos("robot0_branch2_linear_joint")
	q_pos_last[5] = sim.data.get_joint_qpos("robot0_branch3_linear_joint")
	# print("q_pos_last",q_pos_last)
	# last=[sim.data.get_joint_qpos('robot0_gripper_left_finger_joint'),sim.data.get_joint_qpos("robot0_branch3_linear_joint")]
	# gripper_joint_left[t] = sim.data.get_joint_qpos('robot0_gripper_left_finger_joint')
	# gripper_joint_right[t] = sim.data.get_joint_qpos('robot0_gripper_right_finger_joint')
	quat1[t],quat2[t],quat3[t],quat4[t]=sim.data.sensordata[12:16]
	branch2_ee_pos[t],branch3_ee_pos[t],ee_plate_2_pos[t],ee_plate_3_pos[t]=sim.data.sensordata[0:3],sim.data.sensordata[3:6],sim.data.sensordata[16:19],sim.data.sensordata[19:22]
	sim.step()
	#print(sim.data.get_joint_qpos("branch1_joint"),sim.data.get_joint_qpos("branch2_joint"),sim.data.get_joint_qpos("branch3_joint"),sim.data.get_joint_qpos("branch1_linear_joint"),sim.data.get_joint_qpos("branch2_linear_joint"),sim.data.get_joint_qpos("branch3_linear_joint"))
	if True:
		viewer.render()

	#current target joint values, in IK frame
	joint_real=joint_target_traj[t]
	ee_pose=invK.real2sim_wrapper(target_traj[t])
	ee_pose_des.append(ee_pose)
	
	#convert current target joint values, in sim frame
	joint_sim=invK.ik_wrapper(joint_real)
	
	j_goal[t,:]=np.array(joint_sim)
	#calculate/send PD control signal to the motors
	# q_pos_last = np.array([sim.data.get_joint_qpos("robot0_branch1_joint"),sim.data.get_joint_qpos("robot0_branch2_joint"),sim.data.get_joint_qpos("robot0_branch3_joint"),sim.data.get_joint_qpos("robot0_branch1_linear_joint"),sim.data.get_joint_qpos("robot0_branch2_linear_joint"),sim.data.get_joint_qpos("robot0_branch3_linear_joint")])
	PD_scale=PD_signal_scale(target_traj[t],joint_target_traj[t])
	
	PD_signal=[PD_controller_rot(joint_sim[3],sim.data.get_joint_qpos("robot0_branch1_joint"),q_pos_last[0],PD_scale[0]),
			   PD_controller_rot(joint_sim[4],sim.data.get_joint_qpos("robot0_branch2_joint"),q_pos_last[1],PD_scale[1]),
			   PD_controller_rot(joint_sim[5],sim.data.get_joint_qpos("robot0_branch3_joint"),q_pos_last[2],PD_scale[2]),
			   PD_controller_lin(joint_sim[0],sim.data.get_joint_qpos("robot0_branch1_linear_joint"),q_pos_last[3],PD_scale[3]),
			   PD_controller_lin(joint_sim[1],sim.data.get_joint_qpos("robot0_branch2_linear_joint"),q_pos_last[4],PD_scale[4]),
			   PD_controller_lin(joint_sim[2],sim.data.get_joint_qpos("robot0_branch3_linear_joint"),q_pos_last[5],PD_scale[5])]
		
	sim.data.ctrl[0]=PD_signal[0]
	torque4[t]=PD_signal[0]
	sim.data.ctrl[1]=PD_signal[1]
	torque5[t]=PD_signal[1]
	sim.data.ctrl[2]=PD_signal[2]
	torque6[t]=PD_signal[2]
	sim.data.ctrl[3]=PD_signal[3]
	torque1[t]=PD_signal[3]
	sim.data.ctrl[4]=PD_signal[4]
	torque2[t]=PD_signal[4]
	sim.data.ctrl[5]=PD_signal[5]
	torque3[t]=PD_signal[5]
	
	torque_act4[t],torque_act5[t],torque_act6[t],torque_act1[t],torque_act2[t],torque_act3[t]=sim.data.actuator_force[0:6]
	# if False:
	# 	left_finger_open = -0.287884
	# 	right_finger_open = -0.295456

	# 	grip_signal=[max(Gripper_PD_controller(left_finger_open,sim.data.get_joint_qpos('robot0_gripper_left_finger_joint'),gripper_joint_left[t],'left'),-1.5),
	# 		       max(Gripper_PD_controller(right_finger_open,sim.data.get_joint_qpos("robot0_gripper_right_finger_joint"),gripper_joint_right[t],'right'),-1.5)]
	# 	# sim.data.ctrl[6]= grip_signal[0]
	# 	# sim.data.ctrl[7]= grip_signal[1]
	# 	# print("robot0_gripper_left_finger_joint",sim.data.get_joint_qpos('robot0_gripper_left_finger_joint'))
	# 	# print("robot0_gripper_right_finger_joint",sim.data.get_joint_qpos('robot0_gripper_right_finger_joint'))
	# if t==2000:
	# 	left_finger_close = 0.246598
	# 	right_finger_close = 0.241764

	# 	grip_signal=[min(Gripper_PD_controller(left_finger_close,sim.data.get_joint_qpos('robot0_gripper_left_finger_joint'),gripper_joint_left[t],'left'),1.5),
	# 		       min(Gripper_PD_controller(right_finger_close,sim.data.get_joint_qpos("robot0_gripper_right_finger_joint"),gripper_joint_right[t],'right'),1.5)]
	# 	# sim.data.ctrl[6]= grip_signal[0]
	# 	# sim.data.ctrl[7]= grip_signal[1] 
	# 	# print("robot0_gripper_left_finger_joint",sim.data.get_joint_qpos('robot0_gripper_left_finger_joint'))
	# 	# print("robot0_gripper_right_finger_joint",sim.data.get_joint_qpos('robot0_gripper_right_finger_joint'))
	# sim.data.ctrl[6]= grip_signal[0]
	# sim.data.ctrl[7]= grip_signal[1]
	j_actual[t,:]=np.array([sim.data.get_joint_qpos("robot0_branch1_linear_joint"),
						 sim.data.get_joint_qpos("robot0_branch2_linear_joint"),
						 sim.data.get_joint_qpos("robot0_branch3_linear_joint"),
						 sim.data.get_joint_qpos("robot0_branch1_joint"),
						 sim.data.get_joint_qpos("robot0_branch2_joint"),
						 sim.data.get_joint_qpos("robot0_branch3_joint")])
	# gripper_torque_left[t] = grip_signal[0]
	# gripper_torque_right[t] = grip_signal[1]
	# print("robot0_gripper_left_finger_joint",sim.data.get_joint_qpos('robot0_gripper_left_finger_joint'))
	# print("robot0_gripper_right_finger_joint",sim.data.get_joint_qpos('robot0_gripper_right_finger_joint'))
	# if sim.data.get_joint_qpos('robot0_gripper_left_finger_joint')>0.25:
	# 	sim.data.set_joint_qpos('robot0_gripper_left_finger_joint', 0.246598)
	# if sim.data.get_joint_qpos('robot0_gripper_right_finger_joint')>0.25:
	# 	sim.data.set_joint_qpos('robot0_gripper_right_finger_joint', 0.241764)
	# if sim.data.get_joint_qpos('robot0_gripper_left_finger_joint')<-0.29:
	# 	sim.data.set_joint_qpos('robot0_gripper_left_finger_joint', -0.287884)
	# if sim.data.get_joint_qpos('robot0_gripper_right_finger_joint')<-0.30:
	# 	sim.data.set_joint_qpos('robot0_gripper_right_finger_joint', -0.295456)

	# if t>2000 and t<4000:
	# 	sim.data.ctrl[6]= 0.2 
	# 	sim.data.ctrl[7]= 0.2
	# if t>4000 and t<6000:
	# 	sim.data.ctrl[6]= -0.2 
	# 	sim.data.ctrl[7]= -0.2
	# if t>6000 and t<8000:
	# 	sim.data.ctrl[6]= 0.2 
	# 	sim.data.ctrl[7]= 0.2
	# if t>8000:
	# 	sim.data.ctrl[6]= -0.2 
	# 	sim.data.ctrl[7]= -0.2
	sim.data.set_joint_qvel('box_joint0', [0, -0.4, 0, 0, 0, 0])
	##iphonebox pose
	print("iphonebox_pose: ",sim.data.get_joint_qpos('iphonebox_joint0'))
	print("branch3_ee_joint: ",sim.data.get_joint_qpos('robot0_branch3_ee_joint'))
	ee_current_xyz=copy.copy(sim.data.sensordata[9:12])
	ee_current=np.append(ee_current_xyz,
				copy.copy(sim.data.sensordata[12:16]))
	ee_pose_current.append(ee_current)
	##setting gripper fingers values
	# sim.data.set_joint_qpos('gripper0_left_finger_joint', -0.18)
	# sim.data.set_joint_qpos('gripper0_right_finger_joint',-0.18)
	##printing gripper fingers velocities
	# print(sim.data.get_joint_qvel('gripper0_left_finger_joint'))
	# print(sim.data.get_joint_qvel('gripper0_right_finger_joint'))
	"""Available "joint" names = ('robot0_branch1_joint', 'robot0_branch1_pivot_joint', 'robot0_branch1_linear_joint', 
	'robot0_branch1_linear_revolute_joint', 'robot0_branch1_clevis_joint', 'robot0_branch1_ee_joint', 'gripper0_left_finger_joint',
	 'gripper0_right_finger_joint', 'robot0_branch2_joint', 'robot0_branch2_pivot_joint', 'robot0_branch2_linear_joint', 
	 'robot0_branch2_linear_revolute_joint', 'robot0_branch2_clevis_joint', 'robot0_branch2_ee_joint', 'robot0_branch3_joint',
	  'robot0_branch3_pivot_joint', 'robot0_branch3_linear_joint', 'robot0_branch3_linear_revolute_joint', 
	  'robot0_branch3_clevis_joint', 'robot0_branch3_ee_joint', 'iphonebox_joint0', 'box_joint0')"""
	##Just like the code between lines 195-203 you can either set the 'joint pos' and 'joint_vel' or you can get/obtain the 'joint_pos' and 'joint_vel'

	
	t=t+1
##plotting
ee_pose_des_x=[(li[0]) for li in ee_pose_des]
ee_pose_current_x=[(li[0]-0.6) for li in ee_pose_current]
ee_pose_des_y=[(li[1]) for li in ee_pose_des]
ee_pose_current_y=[(li[1]-0.09-0.009+0.004) for li in ee_pose_current]
ee_pose_des_z=[(li[2]) for li in ee_pose_des]
ee_pose_current_z=[(li[2]-1.59-0.012) for li in ee_pose_current]

ee_pose_des_orientx=[(li[0]-0.27-0.007+0.02) for li in qu]
ee_pose_current_orientx=[li[3] for li in ee_pose_current]
ee_pose_des_orienty=[(li[1]-0.645-0.005) for li in qu]
ee_pose_current_orienty=[li[4] for li in ee_pose_current]
ee_pose_des_orientz=[(li[2]+0.65) for li in qu]
ee_pose_current_orientz=[li[5] for li in ee_pose_current]
ee_pose_des_orientw=[(li[3]) for li in qu]
ee_pose_current_orientw=[(li[6]+1.25+0.027-0.02) for li in ee_pose_current]
##till above
# print(ee_pose_des_x[5000]-ee_pose_current_x[5000])
# print(ee_pose_des_y[5000]-ee_pose_current_y[5000])
# print(ee_pose_des_z[5000]-ee_pose_current_z[5000])
# print(ee_pose_des_orientx[5000]-ee_pose_current_orientx[5000])
# print(ee_pose_des_orienty[5000]-ee_pose_current_orienty[5000])
# print(ee_pose_des_orientz[5000]-ee_pose_current_orientz[5000])
# print(ee_pose_des_orientw[5000]-ee_pose_current_orientw[5000])

# fig,a =plt.subplots(2,2)
# a[0][0].plot(t_arr,quat1,label="Gripper w-quat",color='orange')
# a[0][0].legend(loc='upper right')
# a[0][1].plot(t_arr,quat2,label="Gripper x-quat",color='orange')
# a[0][1].legend(loc='upper right')
# a[1][0].plot(t_arr,quat3,label="Gripper y-quat",color='orange')
# a[1][0].legend(loc='upper right')
# a[1][1].plot(t_arr,quat4,label="Gripper z-quat",color='orange')
# a[1][1].legend(loc='upper right')

# fig1,b =plt.subplots(2,3)
# b[0][0].plot(t_arr,torque1,label="lin1 force_cmd")
# b[0][0].plot(t_arr,torque_act1,label="lin1 force_act",color='orange')
# b[0][0].legend(loc='upper right')
# b[0][1].plot(t_arr,torque2,label="lin2 force_cmd")
# b[0][1].plot(t_arr,torque_act2,label="lin2 force_act",color='orange')
# b[0][1].legend(loc='upper right')
# b[0][2].plot(t_arr,torque3,label="lin3 force_cmd")
# b[0][2].plot(t_arr,torque_act3,label="lin3 force_act",color='orange')
# b[0][2].legend(loc='upper right')
# b[1][0].plot(t_arr,torque4,label="rot1 torque_cmd")
# b[1][0].plot(t_arr,torque_act4,label="rot1 torque_act",color='orange')
# b[1][0].legend(loc='upper right')
# b[1][1].plot(t_arr,torque5,label="rot2 torque_cmd")
# b[1][1].plot(t_arr,torque_act5,label="rot2 torque_act",color='orange')
# b[1][1].legend(loc='upper right')
# b[1][2].plot(t_arr,torque6,label="rot3 torque_cmd")
# b[1][2].plot(t_arr,torque_act6,label="rot3 torque_act",color='orange')
# b[1][2].legend(loc='upper right')

# fig2,c =plt.subplots(2,3)
# c[0][0].plot(t_arr,j_goal[:,0],label="lin1_des")
# c[0][0].plot(t_arr,j_actual[:,0],label="lin1_act",color='orange')
# c[0][0].legend(loc='upper right')
# c[0][1].plot(t_arr,j_goal[:,1],label="lin2_des")
# c[0][1].plot(t_arr,j_actual[:,1],label="lin2_act",color='orange')
# c[0][1].legend(loc='upper right')
# c[0][2].plot(t_arr,j_goal[:,2],label="lin3_des")
# c[0][2].plot(t_arr,j_actual[:,2],label="lin3_act",color='orange')
# c[0][2].legend(loc='upper right')
# c[1][0].plot(t_arr,j_goal[:,3],label="rot1_des")
# c[1][0].plot(t_arr,j_actual[:,3],label="rot1_act",color='orange')
# c[1][0].legend(loc='upper right')
# c[1][1].plot(t_arr,j_goal[:,4],label="rot2_des")
# c[1][1].plot(t_arr,j_actual[:,4],label="rot2_act",color='orange')
# c[1][1].legend(loc='upper right')
# c[1][2].plot(t_arr,j_goal[:,5],label="rot3_des")
# c[1][2].plot(t_arr,j_actual[:,5],label="rot3_act",color='orange')
# c[1][2].legend(loc='upper right')

# fig3,d =plt.subplots(2,3)
# d[0][0].plot(t_arr,branch2_ee_pos[:,0],label="branch2_ee_pos_x")
# d[0][0].plot(t_arr,ee_plate_2_pos[:,0],label="ee_plate_2_pos_x",color='orange')
# d[0][0].legend(loc='upper right')
# d[0][1].plot(t_arr,branch2_ee_pos[:,1],label="branch2_ee_pos_y")
# d[0][1].plot(t_arr,ee_plate_2_pos[:,1],label="ee_plate_2_pos_y",color='orange')
# d[0][1].legend(loc='upper right')
# d[0][2].plot(t_arr,branch2_ee_pos[:,2],label="branch2_ee_pos_z")
# d[0][2].plot(t_arr,ee_plate_2_pos[:,2],label="ee_plate_2_pos_z",color='orange')
# d[0][2].legend(loc='upper right')
# d[1][0].plot(t_arr,branch3_ee_pos[:,0],label="branch3_ee_pos_x")
# d[1][0].plot(t_arr,ee_plate_3_pos[:,0],label="ee_plate_3_pos_x",color='orange')
# d[1][0].legend(loc='upper right')
# d[1][1].plot(t_arr,branch3_ee_pos[:,1],label="branch3_ee_pos_y")
# d[1][1].plot(t_arr,ee_plate_3_pos[:,1],label="ee_plate_3_pos_y",color='orange')
# d[1][1].legend(loc='upper right')
# d[1][2].plot(t_arr,branch3_ee_pos[:,2],label="branch3_ee_pos_z")
# d[1][2].plot(t_arr,ee_plate_3_pos[:,2],label="ee_plate_3_pos_z",color='orange')
# d[1][2].legend(loc='upper right')

# plt.figure(1)
# plt.title("left finger torque vs time")
# plt.xlabel('time(s)')
# plt.ylabel('left finger torque')
# plt.plot(t_arr,gripper_torque_left,label="left finger torque")

# plt.figure(2)
# plt.title("right finger torque vs time")
# plt.xlabel('time(s)')
# plt.ylabel('right finger torque')
# plt.plot(t_arr,gripper_torque_right,label="right finger torque")

# plt.figure(3)
# plt.title("gripper_joint_left vs time")
# plt.xlabel('time(s)')
# plt.ylabel('gripper_joint_left')
# plt.plot(t_arr,gripper_joint_left,label="gripper_joint_left")

# plt.figure(4)
# plt.title("gripper_joint_right vs time")
# plt.xlabel('time(s)')
# plt.ylabel('gripper_joint_right')
# plt.plot(t_arr,gripper_joint_right,label="gripper_joint_right")

# plt.show()

# plt.figure(1)
# plt.title("Gripper w-quat vs time")
# plt.xlabel('time(s)')
# plt.ylabel('Gripper w-quat')
# plt.plot(t_arr,quat1,label="Gripper w-quat",color='orange')
# plt.legend(loc='upper right')

# plt.figure(2)
# plt.title("Gripper x-quat vs time")
# plt.xlabel('time(s)')
# plt.ylabel('Gripper x-quat')
# plt.plot(t_arr,quat2,label="Gripper x-quat",color='orange')
# plt.legend(loc='upper right')

# plt.figure(3)
# plt.title("Gripper y-quat vs time")
# plt.xlabel('time(s)')
# plt.ylabel('Gripper y-quat')
# plt.plot(t_arr,quat3,label="Gripper y-quat",color='orange')
# plt.legend(loc='upper right')

# plt.figure(4)
# plt.title("Gripper z-quat vs time")
# plt.xlabel('time(s)')
# plt.ylabel('Gripper z-quat')
# plt.plot(t_arr,quat4,label="Gripper z-quat",color='orange')
# plt.legend(loc='upper right')

# plt.figure(5)
# plt.title("rot1 torque vs time")
# plt.xlabel('time(s)')
# plt.ylabel('rot1 torque')
# plt.plot(t_arr,torque1,label="rot1 torque",color='orange')
# plt.legend(loc='upper right')

# plt.figure(6)
# plt.title("rot2 torque vs time")
# plt.xlabel('time(s)')
# plt.ylabel('rot2 torque')
# plt.plot(t_arr,torque2,label="rot2 torque",color='orange')
# plt.legend(loc='upper right')

# plt.figure(7)
# plt.title("rot3 torque vs time")
# plt.xlabel('time(s)')
# plt.ylabel('rot3 torque')
# plt.plot(t_arr,torque3,label="rot3 torque",color='orange')
# plt.legend(loc='upper right')

# plt.figure(8)
# plt.title("lin1 force vs time")
# plt.xlabel('time(s)')
# plt.ylabel('lin1 force')
# plt.plot(t_arr,torque4,label="lin1 force",color='orange')
# plt.legend(loc='upper right')

# plt.figure(9)
# plt.title("lin2 force vs time")
# plt.xlabel('time(s)')
# plt.ylabel('lin2 force')
# plt.plot(t_arr,torque5,label="lin2 force",color='orange')
# plt.legend(loc='upper right')

# plt.figure(10)
# plt.title("lin3 force vs time")
# plt.xlabel('time(s)')
# plt.ylabel('lin3 force')
# plt.plot(t_arr,torque6,label="lin3 force",color='orange')
# plt.legend(loc='upper right')

# plt.figure(11)
# plt.title("linear1 joint vs time")
# plt.xlabel('time(s)')
# plt.ylabel('linear1 joint')
# plt.plot(t_arr,j_goal[:,0],label="lin1_des")
# plt.plot(t_arr,j_actual[:,0],label="lin1_act",color='orange')
# plt.legend(loc='upper right')

# plt.figure(12)
# plt.title("linear2 joint vs time")
# plt.xlabel('time(s)')
# plt.ylabel('linear2 joint')
# plt.plot(t_arr,j_goal[:,1],label="lin2_des")
# plt.plot(t_arr,j_actual[:,1],label="lin2_act",color='orange')
# plt.legend(loc='upper right')

# plt.figure(13)
# plt.title("linear3 joint vs time")
# plt.xlabel('time(s)')
# plt.ylabel('linear3 joint')
# plt.plot(t_arr,j_goal[:,2],label="lin3_des")
# plt.plot(t_arr,j_actual[:,2],label="lin3_act",color='orange')
# plt.legend(loc='upper right')

# plt.figure(14)
# plt.title("rotary1 joint vs time")
# plt.xlabel('time(s)')
# plt.ylabel('rotary1 joint')
# plt.plot(t_arr,j_goal[:,3],label="rot1_des")
# plt.plot(t_arr,j_actual[:,3],label="rot1_act",color='orange')
# plt.legend(loc='upper right')

# plt.figure(15)
# plt.title("rotary2 joint vs time")
# plt.xlabel('time(s)')
# plt.ylabel('rotary2 joint')
# plt.plot(t_arr,j_goal[:,4],label="rot2_des")
# plt.plot(t_arr,j_actual[:,4],label="rot2_act",color='orange')
# plt.legend(loc='upper right')

# plt.figure(16)
# plt.title("rotary3 joint vs time")
# plt.xlabel('time(s)')
# plt.ylabel('rotary3 joint')
# plt.plot(t_arr,j_goal[:,5],label="rot3_des")
# plt.plot(t_arr,j_actual[:,5],label="rot3_act",color='orange')
# plt.legend(loc='upper right')
plt.figure(1)
plt.title("des_x_coord vs actual_x_coord")
plt.xlabel('time(s)')
plt.ylabel('x value')
plt.ylim([-0.65,0.65])
plt.plot(t_arr,ee_pose_des_x,label="Desired x-coordinate")
plt.plot(t_arr,ee_pose_current_x,label="Actual x-coordinate")
plt.legend(["Desired x-coordinate","Actual x-coordinate"])

plt.figure(2)
plt.title("des_y_coord vs actual_y_coord")
plt.xlabel('time(s)')
plt.ylabel('y value')
plt.ylim([-0.65,0.65])
plt.plot(t_arr,ee_pose_des_y,label="Desired y-coordinate")
plt.plot(t_arr,ee_pose_current_y,label="Actual y-coordinate")
plt.legend(["Desired y-coordinate","Actual y-coordinate"])
	
plt.figure(3)
plt.title("des_z_coord vs actual_z_coord")
plt.xlabel('time(s)')
plt.ylabel('z value')
plt.ylim([-0.65,0.65])
plt.plot(t_arr,ee_pose_des_z,label="Desired z-coordinate")
plt.plot(t_arr,ee_pose_current_z,label="Actual z-coordinate")
plt.legend(["Desired z-coordinate","Actual z-coordinate"])

plt.figure(4)
plt.title("des_x_quaternion vs actual_x_quaternion")
plt.xlabel('time(s)')
plt.ylabel('quaternion_x value')
plt.ylim([-1.57,1.57])
plt.plot(t_arr,ee_pose_current_orientx,label="Actual x-quaternion")
plt.plot(t_arr,ee_pose_des_orientx,label="Desired x-quaternion")
plt.legend(["Actual x-quaternion","Desired x-quaternion"])

plt.figure(5)
plt.title("des_y_quaternion vs actual_y_quaternion")
plt.xlabel('time(s)')
plt.ylabel('quaternion_y value')
plt.ylim([-1.57,1.57])
plt.plot(t_arr,ee_pose_current_orienty,label="Actual y-quaternion")
plt.plot(t_arr,ee_pose_des_orienty,label="Desired y-quaternion")
plt.legend(["Actual y-quaternion","Desired y-quaternion"])

plt.figure(6)
plt.title("des_z_quaternion vs actual_z_quaternion")
plt.xlabel('time(s)')
plt.ylabel('quaternion_z value')
plt.ylim([-1.57,1.57])
plt.plot(t_arr,ee_pose_current_orientz,label="Actual z-quaternion")
plt.plot(t_arr,ee_pose_des_orientz,label="Desired z-quaternion")
plt.legend(["Actual z-quaternion","Desired z-quaternion"])

plt.figure(7)
plt.title("des_w_quaternion vs actual_w_quaternion")
plt.xlabel('time(s)')
plt.ylabel('quaternion_w value')
plt.ylim([-1.57,1.57])
plt.plot(t_arr,ee_pose_current_orientw,label="Actual w-quaternion")
plt.plot(t_arr,ee_pose_des_orientw,label="Desired w-quaternion")
plt.legend(["Actual w-quaternion","Desired w-quaternion"])


plt.figure(8)
plt.title("linear1 joint vs time")
plt.xlabel('time(s)')
plt.ylabel('linear1 joint')
plt.plot(t_arr,j_goal[:,0],label="lin1_des")
plt.plot(t_arr,j_actual[:,0],label="lin1_act",color='orange')
plt.legend(loc='upper right')

plt.figure(9)
plt.title("linear2 joint vs time")
plt.xlabel('time(s)')
plt.ylabel('linear2 joint')
plt.plot(t_arr,j_goal[:,1],label="lin2_des")
plt.plot(t_arr,j_actual[:,1],label="lin2_act",color='orange')
plt.legend(loc='upper right')

plt.figure(10)
plt.title("linear3 joint vs time")
plt.xlabel('time(s)')
plt.ylabel('linear3 joint')
plt.plot(t_arr,j_goal[:,2],label="lin3_des")
plt.plot(t_arr,j_actual[:,2],label="lin3_act",color='orange')
plt.legend(loc='upper right')

plt.figure(11)
plt.title("rotary1 joint vs time")
plt.xlabel('time(s)')
plt.ylabel('rotary1 joint')
plt.plot(t_arr,j_goal[:,3],label="rot1_des")
plt.plot(t_arr,j_actual[:,3],label="rot1_act",color='orange')
plt.legend(loc='upper right')

plt.figure(12)
plt.title("rotary2 joint vs time")
plt.xlabel('time(s)')
plt.ylabel('rotary2 joint')
plt.plot(t_arr,j_goal[:,4],label="rot2_des")
plt.plot(t_arr,j_actual[:,4],label="rot2_act",color='orange')
plt.legend(loc='upper right')

plt.figure(13)
plt.title("rotary3 joint vs time")
plt.xlabel('time(s)')
plt.ylabel('rotary3 joint')
plt.plot(t_arr,j_goal[:,5],label="rot3_des")
plt.plot(t_arr,j_actual[:,5],label="rot3_act",color='orange')
plt.legend(loc='upper right')


plt.show()

			