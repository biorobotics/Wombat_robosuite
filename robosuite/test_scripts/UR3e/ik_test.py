# %%
from ur_ikfast import ur_kinematics 

ur3e_arm = ur_kinematics.URKinematics('ur3e')



import numpy as np 
# %%

def ik(pose):
    print(f"pose {pose}")
    return ur3e_arm.inverse(pose, False)

joint_values = [0, 0, -1.5708, 0, 1.50797, np.pi *20/180.0]

ee_pose_out = ur3e_arm.forward(joint_values)

vals = ur3e_arm.inverse(ee_pose_out, False)
# joint_angles = ik(ee_pose)

print(vals)