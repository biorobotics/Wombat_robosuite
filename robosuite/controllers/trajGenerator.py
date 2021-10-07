import numpy as np
#from interpolateTraj import interpolateTraj
#from invK import invK
import robosuite.controllers.interpolateTraj
import robosuite.controllers.invK

#puts the traj in an hdf5 file at location fileLoc
def outputToFile(traj,name,fileLoc):
	hf=h5py.File(name,'w')
	hf.create_dataset("trajectory",data=traj)
	hf.close()
	return

def angDiff(a1,a2):
	d1=(a1-a2)%(2*np.pi)
	d2=(a2-a1)%(2*np.pi)
	if d1<d2:
		return -d1
	return d2
#given two pairs of joint values, finds the rotary values for j2 that
#are closest to j1. Assumes values are in rad
def nextClosestJointRad(j1,j2):
	j2c=np.copy(j2)
	for i in range(3,6):
		aDiff1=angDiff(j1[i],j2[i])
		aDiff2=angDiff(j1[i],j2[i]+np.pi)
		if abs(aDiff1)<abs(aDiff2):
			j2c[i]=j1[i]+aDiff1
		else:
			j2c[i]=j1[i]+aDiff2
	return j2c
#given two pairs of joint values, finds the rotary values for j2 that
#are closest to j1. Assumes values are in deg
def nextClosestJointDeg(j1,j2):
	j1R=np.copy(j1)
	j2R=np.copy(j2)
	j1R[3:6]=j1R[3:6]*np.pi/180.
	j2R[3:6]=j2R[3:6]*np.pi/180.
	jAdj=nextClosestJointRad(j1R,j2R)
	jAdj[3:6]=jAdj[3:6]*180./np.pi
	return jAdj

#adjusts traj so that it reflect's the robot's reference
#frames.
#jTraj: the joint trajectory [LD1,LD2,LD3,RM1,RM2,RM3] in m and rad
#homeRotPos: rotary motor positions at homing position
#startJoints: rotary motor positions [RM1,RM2,RM3], before start of traj
#motorOrient: orientation of the rotation axes of the rotary motors
#(1=axis pointing down,-1=axis pointing up)
#isRelative: whether the trajectory is absolute or relative. true=relative
#returns the adjusted joint trajectory [LD1,LD2,LD3,RM1,RM2,RM3], in mm and rad
def adjustJointTrajToRobot(
	jTraj,homeRotPos,startJoints,motorOrient,isRelative=False):

	jTrajAdj=np.zeros(jTraj.shape)
	rotOffset=(homeRotPos-motorOrient*np.array([30., -90., 150.]))
	prevJoints=np.copy(startJoints)
	for i in range(jTraj.shape[0]):
		#adjust to mm and deg
		jTrajAdj[i]=np.copy(jTraj[i])
		jTrajAdj[i][0:3]=jTrajAdj[i][0:3]*1000.
		jTrajAdj[i][3:6]=jTrajAdj[i][3:6]*180./np.pi
		#adjust for motor orientation and offset
		jTrajAdj[i][3:6]=np.multiply(jTrajAdj[i][3:6],motorOrient)+rotOffset
		#adjust joint positions to match the initial starting joints
		jTrajAdj[i]=nextClosestJointDeg(prevJoints,jTrajAdj[i])
		prevJoints=np.copy(jTrajAdj[i])
	#if isRelative is active, convert the whole trajectory to relative commands
	if isRelative:
		prevJoints=np.copy(startJoints)
		jTrajAdjRelative=np.zeros(jTrajAdj.shape)
		for i in range(jTrajAdj.shape[0]):
			jTrajAdjRelative[i]=jTrajAdj[i]-prevJoints
			prevJoints=jTrajAdj[i]
		return jTrajAdjRelative
	return jTrajAdj

#given a trajectory for the ee, calculates the joint values
#needed to follow the same trajectory
#eeTraj assumes that all waypoints are in [m,m,m,rad,rad,rad] units
def jointTraj(poseTraj):
	jTraj=np.zeros(np.asarray(poseTraj).shape)
	prevJoints=None
	#use IK on all traj points to solve for the joint values
	for i in range(np.asarray(poseTraj).shape[0]):
		(ji,vi)=robosuite.controllers.invK.invK(poseTraj[i])
		ji=ji.flatten()
		if vi==0:
			print("IK resulted in joints that are invalid!\n")
			return None
		
		#smooth joint angles to prevent jumps in solutions for angular vals;
		#usually happens whenever the robot passes near/through singularity
		if prevJoints is None:
			prevJoints=np.copy(ji)
		jTraj[i]=nextClosestJointRad(prevJoints,ji)
		prevJoints=np.copy(jTraj[i])
	return jTraj

#given a set of waypoints for the ee, calculates the joint values
#needed to follow the same trajectory.
#assumes that all waypoints are in [m,m,m,rad,rad,rad] units
def jointTrajFromWp(wp,segments):
	#linearly interpolate waypoints
	poseTraj=interpolateTraj(wp,segments)
	return jointTraj(poseTraj)


class TrajGenerator:
	def __init__(self,homeRotPos,startJoints,isRelative=False):
		self.isRelative=isRelative
		self.startJoints=np.copy(startJoints)
		self.homeRotPos=np.copy(homeRotPos)
		#TODO: confirm that motor axes are all pointing upwards
		self.motorOrient=np.array([-1,-1,1])
		self.rotOffset=(self.homeRotPos-self.motorOrient*np.array([30., -90., 150.]))*np.pi/180.
	def setHomeRotPos(self,homeRotPos):
		self.homeRotPos=np.copy(homeRotPos)
		self.rotOffset=(self.homeRotPos-self.motorOrient*np.array([30., -90., 150.]))*np.pi/180.
	def setStartJoints(self,startJoints):
		self.startJoints=np.copy(startJoints)
	def setRelative(self,isRelative):
		self.isRelative=isRelative
	def setMotorOrient(self,motorOrient):
		self.motorOrient=np.copy(motorOrient)
		self.rotOffset=(self.homeRotPos-self.motorOrient*np.array([30., -90., 150.]))*np.pi/180.
	#given a trajectory for the ee, calculates the joint values
	#needed to follow the same trajectory
	#eeTraj assumes that all waypoints are in [m,m,m,rad,rad,rad] units
	def jointTraj(self,eeTraj):
		jointTraj=np.zeros(eeTraj.shape)
		prevJoints=np.copy(self.startJoints)
		prevJoints[3:6]=prevJoints[3:6]*np.pi/180.
		#use IK on all traj points to solve for the joint values
		for i in range(eeTraj.shape[0]):
			(ji,vi)=invK(eeTraj[i])
			ji=ji.flatten()
			if vi==0:
				print("IK resulted in joints that are invalid!\n")
				return None

			#adjust for the difference in IK frame assumptions and the real
			#robot frames.
			#TODO: confirm that the rotation axis of the motors are all upwards
			ji[3:6]=np.multiply(ji[3:6],self.motorOrient)+self.rotOffset
			#smooth joint angles to prevent jumps in solutions for angular vals
			jointTraj[i]=nextClosestJointRad(prevJoints,ji)
			prevJoints=np.copy(jointTraj[i])
			#convert from m/rad to mm/degrees
			jointTraj[i][0:3]=ji[0:3]*1000.
			jointTraj[i][3:6]=ji[3:6]*180./np.pi

		#if isRelative is active, then convert to relative position
		prevJoints=np.copy(self.startJoints)
		if self.isRelative:
			relativeJointTraj=np.zeros(jointTraj.shape)
			for i in range(jointTraj.shape[0]):
				relativeJointTraj[i]=jointTraj[i]-prevJoints
				prevJoints=jointTraj[i]
			return relativeJointTraj
		return jointTraj
	#given a set of waypoints for the ee, calculates the joint values
	#needed to follow the same trajectory.
	#assumes that all waypoints are in [m,m,m,rad,rad,rad] units
	def jointTrajFromWp(self,wp,segments):
		#linearly interpolate the waypoints
		poseTraj=interpolateTraj(wp,segments)
		return self.jointTraj(poseTraj)
	
