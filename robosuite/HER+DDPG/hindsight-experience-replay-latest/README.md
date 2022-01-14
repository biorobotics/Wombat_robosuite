1. Run the "train_with_dyn_rand.py" script to train the D3 control policy(DDPG+HER).
2. The trained weights will be stored in the "saved_models/PickPlaceiPhone" directory. 
3. Currently the best trained weights are with the name "Model_dyn_rand_epoch_50.pt". 
4. To check how good the policy is run the "D3_demo_dyn_rand.py" script. It will run the arm for some 20 attempts(this number can be changed, in each attempt the arm will try to pick up the phone).
5. "D3_controller" houses the IK controller files 
