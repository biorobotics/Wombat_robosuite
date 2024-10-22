import tensorflow as tf
import numpy as np
import gym
from replay_buffer import ReplayBuffer
from actor import ActorNetwork
from critic import CriticNetwork
from ou_noise import OUNoise
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import robosuite as suite
from robosuite import load_controller_config
import transforms3d as t3d
import ipdb
from IPython import embed

is_render = True
controller_names = ["OSC_POSE","OSC_POSITION","JOINT_POSITION"]
controller_config = load_controller_config(default_controller=controller_names[0])
controller_config['control_delta'] = False
# print(controller_config)
# quit()


# np.set_printoptions(precision=4)


# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE =  0.001
# Soft target update param
TAU = 0.001
RANDOM_SEED = 1234
EXPLORE = 70
DEVICE = '/cpu:0'

def trainer(epochs=1000, MINIBATCH_SIZE=40, GAMMA = 0.99, epsilon=1.0, min_epsilon=0.01, BUFFER_SIZE=100000, train_indicator=True, render=False):
    with tf.Session() as sess:

        
        # configuring environment
        # create environment instance
        env = suite.make(
            env_name="PickPlaceiPhone", # try with other tasks like "Stack" and "Door"
            robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
            controller_configs=controller_config,            
            has_renderer=is_render,
            has_offscreen_renderer=not is_render,
            use_camera_obs=not is_render,
            render_camera='frontview',
            camera_names = 'birdview',                      # visualize the "frontview" camera
        )
        # configuring the random processes
        # np.random.seed(RANDOM_SEED)
        # tf.set_random_seed(RANDOM_SEED)
        # env.seed(RANDOM_SEED)
        # info of the environment to pass to the agent
        state_dim = 12#env.observation_space.shape[0]
        action_dim = 7#env.action_space.shape[0]
        action_bound = np.array([np.float64(10),np.float64(10),np.float64(10),np.float64(10),np.float64(10),np.float64(10),np.float64(10)]) # I choose this number since the mountain continuos does not have a boundary
        # Creating agent
        ruido = OUNoise(action_dim, mu = 0.4) # this is the Ornstein-Uhlenbeck Noise
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, TAU, DEVICE)
        critic = CriticNetwork(sess, state_dim, action_dim, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars(), DEVICE)


        sess.run(tf.global_variables_initializer())
        
        # Initialize target network weights
        actor.update_target_network()
        critic.update_target_network()
        # Initialize replay memory
        replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

        goal = 0
        max_state = -1.
        try:
            critic.recover_critic()
            actor.recover_actor()
            print('********************************')
            print('models restored succesfully')
            print('********************************')
        except :
            print('********************************')
            print('Failed to restore models')
            print('********************************')

        temp = [[None]]*int(epochs)
        t_arr = np.linspace(1,epochs,int(epochs))
        for i in range(epochs):
            
            state_ = env.reset()
            temp1 = state_['robot0_joint_pos_cos']
            temp2 = state_['robot0_joint_pos_sin']
            state = np.hstack((temp1,temp2))
            ep_reward = 0
            ep_ave_max_q = 0
            done = False
            step = 0
            max_state_episode = -1
            epsilon -= (epsilon/EXPLORE)
            epsilon = np.maximum(min_epsilon,epsilon)

            while (not done):

                if render:
                    env.render()
                    
                #print('step', step)
                # 1. get action with actor, and add noise
                action_original = actor.predict(np.reshape(state,(1,state_dim))) # + (10. / (10. + i))* np.random.randn(1)
                action = action_original + max(epsilon,0)*ruido.noise()

                
                # remove comment if you want to see a step by step update
                # print(step,'a',action_original, action,'s', state[0], 'max state', max_state_episode)
                
                # 2. take action, see next state and reward :
                action = action.flatten() 
                print("action",action)
                next_state_, reward, done, info = env.step(action)
                print("reward",reward)
                temp1 = next_state_['robot0_joint_pos_cos']
                temp2 = next_state_['robot0_joint_pos_sin']
                next_state = np.hstack((temp1,temp2))

                if train_indicator:
                    # 3. Save in replay buffer:
                    replay_buffer.add(np.reshape(state, (actor.s_dim,)), np.reshape(action, (actor.a_dim,)), reward,
                                      done, np.reshape(next_state, (actor.s_dim,)))

                    # Keep adding experience to the memory until
                    # there are at least minibatch size samples
                    if replay_buffer.size() > MINIBATCH_SIZE:

                        # 4. sample random minibatch of transitions: 
                        s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)

                        # Calculate targets
                        
                        # 5. Train critic Network (states,actions, R + gamma* V(s', a')): 
                        # 5.1 Get critic prediction = V(s', a')
                        # the a' is obtained using the actor prediction! or in other words : a' = actor(s')
                        target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                        # 5.2 get y_t where: 
                        y_i = []
                        for k in range(MINIBATCH_SIZE):
                            if t_batch[k]:
                                y_i.append(r_batch[k])
                            else:
                                y_i.append(r_batch[k] + GAMMA * target_q[k])

                        
                        # 5.3 Train Critic! 
                        predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
                        
                        ep_ave_max_q += np.amax(predicted_q_value)
                        
                        # 6 Compute Critic gradient (depends on states and actions)
                        # 6.1 therefore I first need to calculate the actions the current actor would take.
                        a_outs = actor.predict(s_batch)
                        # 6.2 I calculate the gradients 
                        grads = critic.action_gradients(s_batch, a_outs)
                        actor.train(s_batch, grads[0])

                        # Update target networks
                        actor.update_target_network()
                        critic.update_target_network()

            
                state = next_state
                if next_state[0] > max_state_episode:
                    max_state_episode = next_state[0]

                ep_reward = ep_reward + reward
                step +=1
            temp[i] = round(ep_reward,3)
            
            if done:
                ruido.reset() 
                if state[0] > 0.45:
                    #print('****************************************')
                    #print('got it!')
                    #print('****************************************')
                    goal += 1
            
            
            if max_state_episode > max_state:
                max_state = max_state_episode
            print('th',i+1,'n steps', step,'R:', round(ep_reward,3),'Pos', round(epsilon,3),'Efficiency', round(100.*((goal)/(i+1.)),3) )
           
            
            # print('Efficiency', 100.*((goal)/(i+1.)))
            

        print('*************************')
        print('now we save the model')
        critic.save_critic()
        actor.save_actor()
        print('model saved succesfuly')
        print('*************************')
        plt.figure(1)
        plt.title("Reward vs episodes")
        plt.xlabel('episodes')
        plt.ylabel('Reward')
        plt.plot(t_arr,temp)
        plt.show()
                


if __name__ == '__main__':
    trainer(epochs=4, epsilon = 1., render = True)