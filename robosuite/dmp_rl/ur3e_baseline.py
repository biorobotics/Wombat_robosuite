import numpy as np
import robosuite as suite
import matplotlib.pyplot as plt
import time
from robosuite import load_controller_config
from geometry_msgs.msg import Pose
import transforms3d as t3d
import ipdb
from IPython import embed
from scipy.spatial.transform import Rotation as R
from ur_ikfast import ur_kinematics 
import ipdb 
from dmp_server import DMP
import time



ur3e_arm = ur_kinematics.URKinematics('ur3e')

dmp_object = DMP(num_dmps = 3, num_bfs=40, K_val=150, dtime = 0.006, a_s = 4, tolerance=0.3, rescale='rotodilation', T = 1.5)


# print(controller_config)
# quit()
# create environment instance


def quat_to_euler(  quat):
    r_quat = R.from_quat([quat.x,quat.y,quat.z,quat.w])
    e_angles = r_quat.as_euler('xyz', degrees=False)
    return e_angles


def euler_to_quat(euler):
    rot = R.from_euler('xyz',[euler[0], euler[1], euler[2]], degrees=False)
    quat = rot.as_quat()
    return quat

def ik(pose):
    return ur3e_arm.inverse(pose, False)

def dyn_rand_phone():
    phone_x = np.random.uniform(0.232, 0.532)#0.382 is taken as the middle point
    
    phone_speed = np.random.uniform(0.20, 0.35)
    phone_orient = 0.0
    # phone_orient = np.random.uniform(-0.05, 0.05)
    return phone_x, phone_speed, phone_orient

def main(num_episodes=1):

    is_render = True
    controller_names = ["OSC_POSE"]
    controller_config = load_controller_config(default_controller=controller_names[0])

    # controller delta takes in actions as delta between current and desired ee pose for OSC Controller 
    controller_config['control_delta'] = True
    controller_config['impedance_mode'] = "fixed"
    controller_config['kp'] = 10000
    env = suite.make(
        env_name="LiftiPhone", 
        robots="UR3e",  
        gripper_types="DDR_gripper",
        controller_configs=controller_config,            
        has_renderer=is_render,
        has_offscreen_renderer=not is_render,
        use_camera_obs=not is_render,
        render_camera=None,
        camera_names = None,
    )

    

    for i in range(num_episodes):
        env.reset() 
        time.sleep(0.5)
        # Initialize the robot position and open gripper
        action= np.zeros(7)
        action[6] = 1

        observations = env._get_observations()
        gripper_pos = observations['robot0_eef_pos']
        prev_ee_pose = gripper_pos
        gripper_quat = observations['robot0_eef_quat']
        print(f"obs {observations.keys()}")                
        phone_velocity = 0.1
        pre_grasp_pos = 0.1
        promximal_tol  = 0.1
        path_executed = False
        plan_flag = True
        pick_ht = 0.1
        wait_flag = False
        resume_flag = False
        # action[:3]  = gripper_pos
        # action[3:7] = gripper_quat

        for i in range(10000):
            env.sim.data.set_joint_qvel('conveyor_joint0', [phone_velocity,0.0,0.1,0,0,0])
            # print(f"action {action}")
            observations, reward, done, info = env.step(action)
            # print(f"obs: eef_pose {observations['robot0_eef_pos']}")        
            # print(f"obs: gripper_qpos {observations['robot0_gripper_qpos']}")  
            # print(f"obs: phone_pos {observations['phone_pos']}")      
            phone_pos = observations['phone_pos']
            # print(f"phone_pos {phone_pos[:3]}")
            # print(f"gripper_pos {gripper_pos[:3]}")

            gripper_pos = observations['robot0_eef_pos']
            if phone_pos[0] < pre_grasp_pos:
                # print(f"stage0: Waiting for phone")
                pass
            if(phone_pos[0]>pre_grasp_pos and path_executed == False):
                # print(f"stage1: Executing path")
                if(plan_flag):
                    prev_ee_pose = gripper_pos
                    gripper_quat = observations['robot0_eef_quat']
                    phone_quat = observations['phone_quat']
                    start_pose = Pose()
                    goal_pose = Pose()

                    start_pose.position.x = gripper_pos[0]
                    start_pose.position.y = gripper_pos[1]
                    start_pose.position.z = gripper_pos[2]
                    start_pose.orientation.x = gripper_quat[0]
                    start_pose.orientation.y = gripper_quat[1]
                    start_pose.orientation.z = gripper_quat[2]
                    start_pose.orientation.w = gripper_quat[3]

                    goal_pose.position.x = phone_pos[0]
                    goal_pose.position.y = phone_pos[1]
                    goal_pose.position.z = phone_pos[2]
                    goal_pose.orientation.x = phone_quat[0]
                    goal_pose.orientation.y = phone_quat[1]
                    goal_pose.orientation.z = phone_quat[2]
                    goal_pose.orientation.w = phone_quat[3]

                    print(f"start_pose {start_pose}")
                    print(f"goal_pose {goal_pose}")

                    traj = dmp_object.handle_dmp_path(start_pose, goal_pose, "pick", phone_velocity = 0)
                    time.sleep(5)
                    
                    
                    desired_traj = np.zeros((len(traj), 6))
                    for i in range(len(traj)):
                        desired_traj[i][0] =traj[i].position.x 
                        desired_traj[i][1] =traj[i].position.y 
                        desired_traj[i][2] =traj[i].position.z
                        desired_traj[i][5] =quat_to_euler(traj[i].orientation)[0]
                        desired_traj[i][4] =quat_to_euler(traj[i].orientation)[1]
                        desired_traj[i][3] =quat_to_euler(traj[i].orientation)[2]
                    traj_index = 0
                    # desired_traj = np.zeros((100, 6))
                    # for j in range(100):
                    #     desired_traj[j][0] = 0.17*(1 - (j/100)) 

                    plan_flag = False

                    print(f"desired_traj {desired_traj}")
                skip_factor = 1
                max_idx = ((len(traj))/skip_factor)*skip_factor 
                # max_idx-=3
                # max_idx = desired_traj.shape[0]                   
                path_executed = traj_index >=max_idx
                if(path_executed==True):
                    print(f"Path Executed {path_executed}")
                    action[0:3] = np.zeros(3)

                if(path_executed==False):
                    ee_pose = desired_traj[traj_index, :3]
                    ee_pose[2] = np.clip(ee_pose[2], a_min = pick_ht, a_max = None)
                    gripper_pos[2] = np.clip(gripper_pos[2], a_min = pick_ht, a_max = None)
                    
                    traj_index+=skip_factor
                    delta = ee_pose - gripper_pos 
                    action[:3] =1* delta
                    print(f"delta {delta}")
                    prev_ee_pose = ee_pose


            if(path_executed):
                if(wait_flag == False and resume_flag==False):
                    completion_time = i
                    wait_flag = True

                if( (i-completion_time)>100 and resume_flag):
                    resume_flag = True

                if(resume_flag):
                    action[7] = -1
                    grip_flag = True
                    grip_time = i


                    if(grip_flag and (i-grip_time)>100):
                        # action[2]-=0.0001
                        pass

                    if(grip_flag and (i-grip_time)>500):
                        return

                
                





            
            if is_render:
                env.render()  # render on display


if __name__ == '__main__':
    main()

