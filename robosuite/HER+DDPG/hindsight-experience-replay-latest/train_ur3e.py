import numpy as np 
import os, sys 
from arguments import get_args
from mpi4py import MPI

import random 
import torch 

from rl_modules.grasping_agent import grasping_agent
from rl_modules.ddpg_agent import ddpg_agent
import math 


import robosuite as suite 
from robosuite import load_controller_config
from robosuite.models.objects import BoxObject
from robosuite.models.objects import CylinderObject
from robosuite.models.robots import UR3e
from robosuite.models.arenas import EmptyArena
from robosuite.models import MujocoWorldBase
from mujoco_py import MjSim, MjViewer

# UR stuff 
from ur_ikfast import ur_kinematics

# DMP Stuff
from wombat_dmp.srv import *
import rospy 
from geometry_msgs import Pose, Vector3
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R

class UR3e_env(object):
    def __init__(self, args=None, is_render=False):
        self.is_render = is_render 
        self.action_dim = 6
        self.action_network_dim = 3
        self.obs_dim = 34 #TODO: Verify this
        self.q_pos_last = np.zeros(self.action_dim)
        self.observation_current = None
        self.observation_last = None 
        self.observation_last2last = None 
        self.joint_sim_last = None
        self.done = False
        self.action_high = None #TODO:
        self.action_low = np.array([-0.00005]*self.action_dim)
        self.action_network_high = np.array([0.00005]*self.action_network_dim)
        self.action_network_low = np.array([-0.00005]*self.action_network_dim)	 

        self._max_episode_steps = 11000
        self.ur3e_arm = ur_kinematics.URKinematics('ur3e')

    def angDiff(self,a1,a2):
        d1=(a1-a2)%(2*np.pi)
        d2=(a2-a1)%(2*np.pi)
        if d1<d2:
            return -d1
        return d2
		#given two pairs of joint values, finds the rotary values for j2 that
		#are closest to j1. Assumes values are in rad
	
    def nextClosestJointRad(self,j1,j2):
        j2c=np.copy(j2)
        for i in range(0,3):
            aDiff1=self.angDiff(j1[i],j2[i])
            aDiff2=self.angDiff(j1[i],j2[i]+np.pi)
            if abs(aDiff1)<abs(aDiff2):
                j2c[i]=j1[i]+aDiff1
            else:
                j2c[i]=j1[i]+aDiff2
        return j2c

    def quat_to_euler(self, quat):
        r_quat = R.from_quat([quat[0],quat[1],quat[2],quat[3]])
        e_angles = r_quat.as_euler('xyz', degrees=False)
        return e_angles
    
    def euler_to_quat(self, euler):
        rot = R.from_euler('xyz',[euler[0], euler[1], euler[2]], degrees=False)
        quat = rot.as_quat()
        return quat
        
    def set_env(self, phone_x, phone_speed, phone_orient):
        self.phone_x = phone_x
        self.phone_speed = phone_speed
        self.phone_orient = phone_orient
        
        self.world = MujocoWorldBase()
        self.mujoco_robot = UR3e()
        self.mujoco_robot.set_base_xpos([0.5, 0.0, 0.35])
        self.world.merge(self.mujoco_robot)

        self.mujoco_arena = EmptyArena()
        self.world.merge(self.mujoco_arena)

        self.iphonebox = BoxObject(name="iphonebox",size=[0.08,0.039,0.0037],rgba=[0,0,0,1],friction=[1,1,5]).get_obj()
        self.iphonebox.set('pos', '1.2 {} 0.8'.format(self.phone_x))
        self.iphonebox.set('quat', '1 {} 0 0'.format(self.phone_orient)) #0
        self.world.worldbody.append(self.iphoneself.box)

        self.box = BoxObject(name="box",size=[9.7,0.35,0.37],rgba=[0.9,0.9,0.9,1],friction=[1,1,1]).get_obj()
        self.box.set('pos', '1 0.4 0')
        self.world.worldbody.append(self.box)

        self.model = self.world.get_model(mode="mujoco_py")
        self.sim = MjSim(self.model)

        if self.is_render:
            self.viewer = MjViewer(self.sim)
            self.viewer.vopt.geomgroup[0] = 0
            self.viewer.render()

        self.timestep =0.0005 #?Is this needed
        self.robot_joints_no = 6
        self.joint_names = ['robot0_joint_'+ str(i+1) for i in range(self.robot_joints_no)]
        self.sim_state = self.sim.get_state()

        self.observation_current = self.get_observation()
        return self.observation_current

    
    def reset(self, phone_x = 0.2, phone_speed = -0.2, phone_orient = 0): #TODO: Set the default values here 
        obs_new = self.set_env(phone_x, phone_speed, phone_orient)
        return obs_new

    '''
    TODO: Set the params for DR: fricton for phone, table, gripper fingers 
        Gripper finger open/close position should be decided from the phone_width
    '''

    def get_observation(self):

        observation = np.zeros(self.obs_dim)

        #* For observations 0:6 : Robot Joint Qpos
        for i in range(self.robot_joints_no):
            observation[i] = self.sim.data.get_joint_qpos(self.joint_name[i])

        #* For observation 6:12: Robot Joint Qvels 
        for i in range(self.robot_joints_no):
            observation[i] = self.sim.data.get_joint_vel(self.joint_name[i])

        #* Phone Observations [cartesian :(x, y, z), quat: (w, x, y, z)]
        observation[12:19] = self.sim.data.get_joint_qpos('iphonebox_joint0')
        
        # * EE Pose :- Gripper Base Link
        # TODO: First debug and see the dimensionality of this
        observation[19:26] = self.sim.data.sensordata[0:7]	
        observation[19] = observation[19] + 0.02

        #* Gripper Finger Joints:
        observation[26] = self.sim.data.get_joint_qpos('robot0_gripper_left_finger_joint')
        observation[27] = self.sim.data.get_joint_qpos('robot0_gripper_right_finger_joint')

        #*  Phone Velocity 
        observation[28:34] = self.sim.data.get_joint_qvel('iphonebox_joint0')

        # TODO: Check these values for the new environment 
        goal = 0.88#0.83		#z value of the phone should be at 0.83 m from floor
       
        # Z value of the phone
        achieved_goal = observation[14] 
        
        observation = {'observation':observation,'desired_goal':np.array([goal]),'achieved_goal': np.array([achieved_goal])}

        return observation

    def step(self, action):
        if(self.observation_last is not None):
            self.observation_last2last = self.observation_last
        else:
            self.observation_last2last = np.zeros_like(self.observation_current)

        
        self.observation_last = self.observation_current['observation']
        # ?Debug this here 
        q_guess = self.observation_last[19:26]

        ee_pose = np.zeros(7)
        ee_pose[0:3] = action[0:3]
        ee_pose[3:] = self.euler_to_quat(action[3:])
        ee_pose = self.clip_robot_joints(ee_pose)

        joint_values = self.ur3e_arm.inverse(ee_pose, False, q_guess = q_guess)


        for i in range(len(joint_values)):
            self.sim.data.set_joint_qpos(self.joint_names[i], joint_values[i])


        if action[6] >0:
            des_state = 'close'
        else:
            des_state = 'open'

        self.sim.data.ctrl[6:8] = self.grip_signal(des_state,self.observation_last,self.observation_last2last)
        self.clip_grip_action()

        self.sim.step()

        if self.is_render:
            self.viewer.render()
        self.observation_current = self.get_observation()
        self.reward = self.compute_reward(self.observation_current)
        self.sim.data.set_joint_qvel('box_joint0',[0,self.phone_speed, 0, 0, 0, 0])

        self.done = self.is_done()
        return self.observation_current, self.reward, self.done


    def compute_reward(self, obs):
        reward_grasp = []
        for i in range(obs.shape[0]):
            if obs[14]>=0.88 and (obs[21]-obs[14] <= 0.12) and (abs(obs[19]-obs[12]) <= 0.04) and obs[26] > -0.29:
                reward_grasp.append(50)
                # print("Reward: +50")
            elif obs[14] >= 0.83 and obs[26] > -0.29 and (obs[21]-obs[14] <= 0.18):
                reward_grasp.append(5)
                # print("Reward: +5")
            else:
                reward_grasp.append(0)
        reward_grasp = np.array(reward_grasp)
        return reward_grasp

    def is_done(self, obs):
        # Check if the episode ended in sucess, give the higher reward here, if true
        if ((obs[21]-obs[14]) < 0.14) and (obs[21] > 0.99):
            return True
        # else:
        return False


    def Gripper_PD_controller(self,des,current,last):
        kp=10
        kd=2
        pos = des+kp*(des-current)-kd*(current-last)
        return pos

    def clip_grip_action(self):
        if self.sim.data.get_joint_qpos('robot0_gripper_left_finger_joint')>0.25:
            self.sim.data.set_joint_qpos('robot0_gripper_left_finger_joint', 0.246598)
        if self.sim.data.get_joint_qpos('robot0_gripper_right_finger_joint')>0.25:
            self.sim.data.set_joint_qpos('robot0_gripper_right_finger_joint', 0.241764)
        if self.sim.data.get_joint_qpos('robot0_gripper_left_finger_joint')<-0.29:
            self.sim.data.set_joint_qpos('robot0_gripper_left_finger_joint', -0.287884)
        if self.sim.data.get_joint_qpos('robot0_gripper_right_finger_joint')<-0.30:
            self.sim.data.set_joint_qpos('robot0_gripper_right_finger_joint', -0.295456)
	
    def clip_robot_joints(self,ee_pose):
        x = ee_pose[0]
        y = ee_pose[1]
        z = ee_pose[2]

        # First clip in the sphere 
        r = np.sqrt(x**2 + y**2 + z**2)
        if(r>=0.5):
            ee_pose[0:3] -=0.0005
            ee_pose =self.clip_robot_joints(ee_pose)
            self.limit_flag = True
        
        # Cyclindrical Space
        r_cyl= np.sqrt(x**2, y**2)
        if(r<=0.12):
            ee_pose[0:2]+=0.0005
            ee_pose=self.clip_robot_joints(ee_pose)
            self.limit_flag = True

        return ee_pose


    def grip_signal(self,des_state,obs_last,obs_last2last):
        if des_state=='open':
            left_finger_open = -0.287884
            right_finger_open = -0.295456

            grip_signal=[self.Gripper_PD_controller(left_finger_open,obs_last[26],obs_last2last[26]),
                            self.Gripper_PD_controller(right_finger_open,obs_last[27],obs_last2last[27])]
        if des_state=='close':
            left_finger_close = 0.246598
            right_finger_close = 0.241764

            grip_signal=[self.Gripper_PD_controller(left_finger_close,obs_last[26],obs_last2last[26]),
                            self.Gripper_PD_controller(right_finger_close,obs_last[27],obs_last2last[27])]
        return grip_signal



    def seed(self, seed=None):
        """
        Utility function to set numpy seed
        Args:
            seed (None or int): If specified, numpy seed to set
        Raises:
            TypeError: [Seed must be integer]
        """
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")


def get_env_params(env):
	# ipdb.set_trace()
	obs_dict = env.reset()	#not yet written
	# ipdb.set_trace()
	# close the environment
	# ipdb.set_trace()
	params = {'obs': obs_dict['observation'].shape[0],
			'goal': obs_dict['desired_goal'].shape[0],
			'action': env.action_network_dim,
			'action_max': env.action_high[0],
			}
	params['max_timesteps'] = env._max_episode_steps
	return params

def launch(args,env):
	# create the ddpg_agent
	
	# set random seeds for reproduce
	env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
	random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
	np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
	torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
	if args.cuda:
		torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
	# get the environment parameters
	env_params = get_env_params(env)
	grasping_trainer = grasping_agent(args, env, env_params)
	grasping_trainer.learn()
	# create the ddpg agent to interact with the environment 
	#####################################################works till this point############################################
	####### not yet there ##########

if __name__ == '__main__':
	# take the configuration for the HER
	os.environ['OMP_NUM_THREADS'] = '1'
	os.environ['MKL_NUM_THREADS'] = '1'
	os.environ['IN_MPI'] = '1'
	# get the params
	args = get_args()
	# print(args)


	from time import localtime, strftime

	current_time = strftime("%Y-%m-%d-%H-%M-%S", localtime())
	print("Current Time =", current_time)
	# args = parser.parse_args()
	# args.log = os.path.join(args.log)
	pick_place_env = UR3e_env(args,is_render=True)
	# pick_place_env = PickPlace_env(args)
	# pick_place_env.run()
	launch(args,pick_place_env)