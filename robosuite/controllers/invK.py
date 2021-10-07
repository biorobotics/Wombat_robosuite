
import numpy as np
import transforms3d as t3d

#gets matrix representing rotation about Z axis
def rotZ(theta):
	return np.array([[np.cos(theta), -np.sin(theta), 0, 0],\
					[np.sin(theta), np.cos(theta), 0, 0],\
					[0, 0, 1, 0],\
					[0, 0, 0, 1]])
#gets matrix representing rotation about Y axis
def rotY(theta):
	return np.array([[np.cos(theta), 0, np.sin(theta), 0],\
					[0, 1, 0, 0],\
					[-np.sin(theta), 0, np.cos(theta), 0],\
					[0, 0, 0, 1]])
#gets matrix representing rotation about X axis
def rotX(theta):
	return np.array([[1, 0, 0, 0],\
					[0, np.cos(theta), -np.sin(theta), 0],\
					[0, np.sin(theta), np.cos(theta), 0],\
					[0, 0, 0, 1]])
#gets matrix representing translation about pos=[x,y,z]
def trans(pos):
	return np.block([[np.identity(3),np.transpose([pos])],[0,0,0,1]])
#takes in units of [m,m,m,rad,rad,rad]
#					x,y,z,rX,rY,rZ
#rotation is assumed to be in z-y-x order/roll pitch yaw
#given target, the end effector position, returns the active joint values
#required to reach it. Also returns whether the joint values are valid.
#return format is [LD1,LD2,LD3,RM1,RM2,RM3]
def invK(target):
	#total length of arm, from clevis joint axis to pivot axis of lin. mot.
	armLen=1.17685
	#length of the linear motor, along the arm
	linMotLen=0.324
	#lower linear motor actuator limit
	lowerJointLim=-0.010
	#distance between clevis joint axis to the arm's linear 0 position.
	upperLimDist=0.335-lowerJointLim+linMotLen/2
	#upper linear motor actuator limit
	upperJointLim=armLen-upperLimDist-linMotLen/2
	#rotation of end effector
	#print(target)
	Rz=rotZ(target[5])
	Ry=rotY(target[4])
	Rx=rotX(target[3])
	rTarget=np.matmul(Rz,np.matmul(Ry,Rx))
	cartPos=trans(target[0:3])
	#calculate position of end effector in matrix form
	tMat=np.matmul(cartPos,rTarget)

	#defining some constants for translating
	#effFromPlate=trans([0, 0.831*0.0254,0.75*0.0254/2])
	effFromPlate=trans([0,0.02111,0.75*0.0254/2])
	rotXUniv=rotX(45*np.pi/180)
	#transZUniv=trans([0,0,-1.108*0.0254])
	transZUniv=trans([0,0,-0.02928])
	transYAct=trans([0,6.5*0.0254,0])

	#locations of the actuator "centers" on the stationary plate; this is
	#the intersection between the slider and pivot axis of the lin. mot.
	#orientation will be wrong but we only care about position.
	rB=[np.matmul(rotZ(2*np.pi/3),transYAct),\
		transYAct,\
		np.matmul(rotZ(4*np.pi/3),transYAct)]

	#location of the "bottom" of the universal joint, extended to
	#bottom of the end effector plate
	rUBot=[np.matmul(np.matmul(tMat,rotZ(2*np.pi/3)), effFromPlate),\
		np.matmul(tMat,effFromPlate),\
		np.matmul(np.matmul(tMat,rotZ(4*np.pi/3)), effFromPlate)]
	#location of the "top" of the universal joint, where the clevis joint is
	transToTop=np.matmul(rotXUniv,transZUniv)
	rUTop=[np.matmul(rUBot[0],transToTop),\
		np.matmul(rUBot[1],transToTop),\
		np.matmul(rUBot[2],transToTop)]
	# print(rUTop)
	#calculate distance from the clevis joints to the actuator centers;
	linVec1=np.subtract(rB[0][0:3,3],rUTop[0][0:3,3])
	linVec2=np.subtract(rB[1][0:3,3],rUTop[1][0:3,3])
	linVec3=np.subtract(rB[2][0:3,3],rUTop[2][0:3,3])
	distToAct=[[np.linalg.norm(linVec1)],\
	[np.linalg.norm(linVec2)],\
	[np.linalg.norm(linVec3)]]
	#convert the distance 
	jointD=np.subtract(distToAct,np.array([[upperLimDist] for i in range(3)]))

	#project the locations of the clevis joints onto the xy plane and
	#calculate the rotation of the rotary motors based on that. Similar to
	#looking from top-down on the robot and taking the angle of the arm based
	#on where it's facing.
	dxArm=np.subtract([rUTop[0][0][3],rUTop[1][0][3],rUTop[2][0][3]],\
		[rB[0][0][3],rB[1][0][3],rB[2][0][3]])
	dyArm=np.subtract([rUTop[0][1][3],rUTop[1][1][3],rUTop[2][1][3]],\
		[rB[0][1][3],rB[1][1][3],rB[2][1][3]])
	angArm=np.arctan2(dyArm,dxArm)

	#final answer for joint positions
	joints=np.block([np.transpose(jointD),angArm])
	#joints = [jointD[0][0],jointD[1][0],jointD[2][0],angArm[0],angArm[1],angArm[2]]

	#check if joint positions are valid
	valid=1
	for i in range(3):
		#check linear motor limits
		if valid==0 or jointD[i]<lowerJointLim or jointD[i]>upperJointLim:
			print("exceeded linear actuator limits")
			valid=0
	#check that the arm is not in "danger" poses, where the arm is positioned such that
	#two forward kinematics solutions are close to each other. Also, if the end effector is
	#upside down.
	#define the plane with the 3 clevis joint positions
	e1=np.subtract(rUTop[0][0:3,3],rUTop[1][0:3,3])
	e2=np.subtract(rUTop[0][0:3,3],rUTop[2][0:3,3])
	normal=np.cross(e1,e2)
	pConst=-1*np.sum(np.multiply(normal,rUTop[0][0:3,3]))
	#determine which side of this plane the end effector, and the center
	#of the linear actuators are.
	eeSide=np.sign(np.sum(np.multiply(target[0:3],normal))+pConst)
	P1Side=np.sign(np.sum(np.multiply(rB[0][0:3,3],normal))+pConst)
	P2Side=np.sign(np.sum(np.multiply(rB[1][0:3,3],normal))+pConst)
	P3Side=np.sign(np.sum(np.multiply(rB[2][0:3,3],normal))+pConst)
	if valid==0 or eeSide*P1Side!=-1 or eeSide*P2Side!=-1 or eeSide*P3Side!=-1:
		valid=0

	#end effector should also be a certain "angle" away from the 
	#danger positions. This is arbitrarily chosen but for now this will
	#be 10 degrees(converted to rad)
	dangerDeg=10*np.pi/180
	normMag=np.linalg.norm(normal)
	#calculate the angle between the plane and the arm
	cAng1=np.arcsin(abs(np.dot(normal,linVec1))/\
		(np.linalg.norm(linVec1)*normMag))
	cAng2=np.arcsin(abs(np.dot(normal,linVec2))/\
		(np.linalg.norm(linVec2)*normMag))
	cAng3=np.arcsin(abs(np.dot(normal,linVec3))/\
		(np.linalg.norm(linVec3)*normMag))
	if valid==0 or cAng1<dangerDeg or cAng2<dangerDeg or cAng3<dangerDeg:
		valid=0
	
	return (joints,valid)

#note: base position is offset [0,0.03175,0.0015]
def sim2real_wrapper(target_sim):

	#gets sim values and converts for real robot
	T_sim_real = np.eye(4)
	T_real_target = np.eye(4)
	T_sim_target = np.eye(4)
	t_real = [0]*6

	#T_sim_real[0:3,0:3] = np.array([[0,1,0],
	#								 [1,0,0],
	#								 [0,0,-1]])
	T_sim_real[0:3,0:3] = np.array([[-1,0,0],
									[0,1,0],
									[0,0,-1]])
	T_sim_real[0:3,3] = [0,0.03175,0]

	T_sim_target[0:3,0:3] = t3d.euler.euler2mat(target_sim[3],target_sim[4],target_sim[5],'szyx')
	T_sim_target[0:3,3] = [target_sim[0],target_sim[1],target_sim[2]]

	T_real_target = np.matmul(np.linalg.inv(T_sim_real),T_sim_target)

	t_real[0],t_real[1],t_real[2] = T_real_target[0:3,3]
	t_real[3],t_real[4],t_real[5] = t3d.euler.mat2euler(T_real_target[0:3,0:3])
	#t_real[3],t_real[4],t_real[5] = [0,0,0]
	return t_real


	pass

def real2sim_wrapper(target_real):
	#gets real value and converts to sim robot model
	T_sim_real = np.eye(4)
	T_real_target = np.eye(4)
	T_sim_target = np.eye(4)
	t_sim = [0]*6

	#T_sim_real[0:3,0:3] = np.array([[0,1,0],
	#								[1,0,0],
	#								[0,0,-1]])
	T_sim_real[0:3,0:3] = np.array([[-1,0,0],
									[0,1,0],
									[0,0,-1]])
	#T_sim_real[0:3,3] = [0,0,0]
	T_sim_real[0:3,3] = [0,0.03175,0]
	T_real_target[0:3,0:3] = t3d.euler.euler2mat(target_real[3],target_real[4],target_real[5],'szyx')
	T_real_target[0:3,3] = [target_real[0],target_real[1],target_real[2]]
	T_sim_target = np.matmul(T_sim_real,T_real_target)
	#print(T_sim_target)
	t_sim[0],t_sim[1],t_sim[2] = T_sim_target[0:3,3]
	t_sim[3],t_sim[4],t_sim[5] = t3d.euler.mat2euler(T_sim_target[0:3,0:3])
	#t_sim[3],t_sim[4],t_sim[5] = [0,0,0]

	#print(t_sim)
	return t_sim

def ik_wrapper(joint):
	joint_real = joint[0:6]
	joint_sim = [0]*6
	#for i in range(3):
	#	joint_sim[i] = -(joint_real[i]-0.1)
	
	#joint_sim[3] = joint_real[3]-0.5235
	#joint_sim[4] = joint_real[4]-(-1.5707)
	#joint_sim[5] = joint_real[5]-2.6179
	#switch order of arms from the hardware/real robot convention, to sim
	#arm 1,2,3->3,1,2
	joint_sim[2]=-(joint_real[0]-0.0951)
	joint_sim[0]=-(joint_real[1]-0.0951)
	joint_sim[1]=-(joint_real[2]-0.0951)
	joint_sim[5] = joint_real[3]-0.5235
	joint_sim[3] = joint_real[4]-(-1.5707)
	joint_sim[4] = joint_real[5]-2.6179
	return joint_sim

def ik_wrapper_reverse(joint):
	joint_sim = joint[0:6]
	joint_real = [0]*6
	#for i in range(3):
	#	joint_sim[i] = -(joint_real[i]-0.1)
	
	#joint_sim[3] = joint_real[3]-0.5235
	#joint_sim[4] = joint_real[4]-(-1.5707)
	#joint_sim[5] = joint_real[5]-2.6179
	#switch order of arms from the hardware/real robot convention, to sim
	#arm 1,2,3->3,1,2
	joint_real[0]=-(joint_sim[2]+0.0951)
	joint_real[1]=-(joint_sim[0]+0.0951)
	joint_real[2]=-(joint_sim[1]+0.0951)
	joint_real[3] = joint_sim[5]+0.5235
	joint_real[4] = joint_sim[3]+(-1.5707)
	joint_real[5] = joint_sim[4]+2.6179
	return joint_real

if __name__ == "__main__":

	target_current_sim = [0,0.0,-0.8,0,0,0]
	target = sim2real_wrapper(target_current_sim)
	# print(target)
	joint,valid = invK(target)
	if valid:
		joint_real = ik_wrapper(joint)
	else:
		print("Not a valid trajectory !!!")
	# real2sim_wrapper(target_current)
	print(joint_real,joint)
