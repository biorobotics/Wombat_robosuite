#!/usr/bin/env python3
"""
Shows how to toss a capsule to a container.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os

#Load the model and environment from its xml file
common_path = "/home/yashraghav/robosuite/robosuite/models/assets/objects"
models_to_load = ["iPhone12ProMax.xml","iPhone12ProMax-visual.xml"]
models = []
models = load_model_from_path(os.path.join(common_path,models_to_load[0]))
sim = MjSim(models)

#the time for each episode of the simulation
sim_horizon = 1000

#initialize the simulation visualization
viewer = MjViewer(sim)

#get initial state of simulation


#repeat indefinitely
while True:
	viewer.render()
	sim.step()
