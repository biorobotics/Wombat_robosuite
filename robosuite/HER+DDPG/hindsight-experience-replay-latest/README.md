1. Run the "train_with_dyn_rand.py" script to train the D3 control policy(DDPG+HER).
2. The trained weights will be stored in the "saved_models/PickPlaceiPhone" directory. 
3. Currently the best trained weights are with the name "Model_dyn_rand_epoch_50.pt". 
4. To check how good the policy is run the "D3_demo_dyn_rand.py" script. It will run the arm for some 20 attempts(this number can be changed, in each attempt the arm will try to pick up the phone).
5. "D3_controller" houses the IK controller files 
6. "rl_modules" has the DDPG algo. script
7. "her_modules" has the HER algo. script
8. In the "arguments.py" script, one can change the no. of epochs, batch size for training, no of runs for demo, learning rate, and various other parameters
9. Breakdown of the training script "train_with_dyn_rand.py":
- We first create the environment class "D3_pick_place_env"
- In the "_init_" function, we set the action_dimensions(6 i.e. position+orientation), no. of observations (eg. phone position, etc.), no. of steps per episode, maxm. action per step and few other parameters
- The "angDiff" and "nextClosestJointRad" functions make sure that the property of angle repeatability doesn't interfere with the working of PD controller, basically making sure PD controller perceives 360 deg. and 0 deg. as the same angle
- The "set_env" function primarily calls the D3 arm in sim. world, sets the phone's initial position, size, speed and orientation. Also, sets the conveyor size, speed. And returns the current state of observations
- "get_signal", "grip_signal", "PD_controller_rot", "PD_controller_lin", "PD_signal_scale", "Gripper_PD_controller" are basically the PD controller functions
- "reset" sets the phone position, speed and orientation to default values
- "clip_action" makes sure the commanded action values doesn't make the end-effector to try reaching outside the workspace, or hitting the conveyor, in place case the IK controller throws an error
- "action2robot" doesn't have any use as of now
- "step" commands the 6 active joints and returns the updated state values of observations and reward
- "get_observation" returns the state values of observations
- "compute_reward" returns the reward
- "get_env_params" returns 5 parameters associated with the environment
- "launch" sets the environment randomness and calls the ddpg algo. script to start the training/learning process
