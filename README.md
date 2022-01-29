1. This "version_4" branch has the iPhone 12 pro max in the simulation, not added through wombat_arm.xml but as a separate xml, + most updated model of "D3_gripper" i.e. it has 4-DoF whereas the earlier model had only 2-DoF
2. Go to "robosuite/HER+DDPG/hindsight-experience-replay-latest/" and run the "D3_demo_dyn_rand.py" to see the phone pick-up success rate, corresponding to the trained weights. 
3. If you want to train the policy then run the "D3_train_dyn_rand.py" and the trained weights will be stored in the "saved_models/PickplaceiPhone/" 
4. The IK controller files, DDPG script, HER script, the training script and the demo script all lie in the directory "robosuite/HER+DDPG/hindsight-experience-replay-latest/" 
