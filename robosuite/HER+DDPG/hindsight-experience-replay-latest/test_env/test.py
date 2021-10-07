import numpy as np
import gym


env = gym.make('FetchReach-v1')
obs = env.reset()   
done = False

def policy(observation, desired_goal):
    # Here you would implement your smarter policy. In this case,
    # we just sample random actions.
    action = np.zeros(4)
    action[2] = 0.0
    print("Action: ",action)
    # action[3] = 0
    return action

while not done:
    action = policy(obs['observation'], obs['desired_goal'])
    obs, reward, done, info = env.step(action)
    env.render()
    # If we want, we can substitute a goal here and re-compute
    # the reward. For instance, we can just pretend that the desired
    # goal was what we achieved all along.
    substitute_goal = obs['achieved_goal'].copy()
    print("Desired goal: {}, Achieved goal: {}".format(obs['desired_goal'],obs['achieved_goal']))
    substitute_reward = env.compute_reward(
        obs['achieved_goal'], substitute_goal, info)
    print('reward is {}, substitute_reward is {}'.format(
        reward, substitute_reward))