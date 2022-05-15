#!/usr/bin/env python3

import numpy as np
import rospy
from geometry_msgs.msg import PoseArray, Pose
from scipy.spatial.transform import Rotation as R
from supereclipse import SuperEclipse

def generate_traj(mode):
    s = str(mode)
    n=100
    pose_arr = PoseArray()
    lc = SuperEclipse()
    lX,lY,lZ = lc.construct_curve(np.array([0,0,0.65]),[-0.15,0,0.65])
    if (s == "supereclipse"):
        n = len(lX)
    for i in range(n):
        pose = Pose()
        if(s=="Sine_XY"):
            pose.position.x = 0.0-0.15*i/100.0
            pose.position.y = 0.0+0.00*i/100
            pose.position.z = 0.65 - 0.1*np.sin(4*np.pi*i/100.)
        if(s=="Linear_X"):
            pose.position.x = 0-0.15*i/100
            pose.position.y = 0.0
            pose.position.z = 0.65
        if(s=="Linear_Y"):
            pose.position.x = 0
            pose.position.y = 0.0-0.1*i/100
            pose.position.z = 0.65
        if(s=="supereclipse"):
            pose.position.x = lX[i]
            pose.position.y = lY[i]
            pose.position.z = lZ[i]
            # pose.position.z = 0.65 + (0.65-lZ[i])
        if(s=="pick"):
            pose.position.x = -0.436
            pose.position.y = 0.04 +0.1*i/100
            pose.position.z = 0.1

        # else:
        #     pose.position.x = -0.1
        #     pose.position.y = -0.1
        #     pose.position.z = 0.55
        

        theta = 0.0#0.0 + (np.pi/5.0)*i/100.0
        Ri=R.from_euler('xyz',[theta,0,0],degrees=False)
        qi=Ri.as_quat()
        pose.orientation.x = qi[0]
        pose.orientation.y = qi[1]
        pose.orientation.z = qi[2]
        pose.orientation.w = qi[3]

        pose_arr.poses.append(pose)
    print("Desired Trajectory Calculated")
    return pose_arr

