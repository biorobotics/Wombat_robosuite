import wombat_arm_ik
import numpy as np

from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder

mujoco_loc="/home/yashraghav/mujoco-py/xmls/apple_arm/"
#model = load_model_from_path(mujoco_loc+"/xmls/wombat_arm/wombat_arm.xml")
model=load_model_from_path("/home/yashraghav/mujoco-py/xmls/wombat_arm/wombat_arm.xml")
sim = MjSim(model)

#flag for simulating
simulate=True
viewer = MjViewer(sim)
while 1:
	print("w")