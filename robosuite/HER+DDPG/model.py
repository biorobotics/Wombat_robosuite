import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import torch.optim as optim

from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
import os
import matplotlib.pyplot as plt
import glob

import numpy as np
import ipdb
# from torchsummary import summary

# from pushover import notify
# from utils import makegif
from random import randint



class CEM():
	''' cross-entropy method, as optimization of the action policy 
	the policy weights theta are stored in this CEM class instead of in the policy
	'''
	def __init__(self, theta_dim, ini_mean_scale=3, ini_std_scale=1.0):
		self.theta_dim = theta_dim
		self.initialize(ini_mean_scale=ini_mean_scale, ini_std_scale=ini_std_scale)

		
	def sample(self):
		# theta = self.mean + np.random.randn(self.theta_dim) * self.std
		theta = self.mean + np.random.normal(size=self.theta_dim) * self.std
		return theta

	def initialize(self, ini_mean_scale=0.0, ini_std_scale=1.0):
		# self.mean = ini_mean_scale*np.ones(self.theta_dim)
		# self.std = ini_std_scale*np.ones(self.theta_dim)
		self.mean = np.array([0,0,0,0])
		self.std = np.array([0.5,0.5,0.5,1])

	def sample_multi(self, n):
		theta_list=[]
		for i in range(n):
			theta_list.append(self.sample())
		return np.array(theta_list)


	def update(self, selected_samples):
		self.mean = np.mean(selected_samples, axis = 0)
		# print('mean: ', self.mean)
		self.std = np.std(selected_samples, axis = 0) # plus the entropy offset, or else easily get 0 std
		# print('mean std: ', np.mean(self.std))

		return self.mean, self.std


class FeatureExtractor(nn.Module):
	def __init__(self,image_channels):
		super(FeatureExtractor, self).__init__()

		self.image_channels = image_channels

		self.image_network = nn.Sequential(
								nn.Conv2d(self.image_channels,4, kernel_size=2, stride=2),
								nn.ReLU(),
								nn.Conv2d(4, 8, kernel_size=2, stride=2),
								nn.ReLU(),
								nn.Conv2d(8, 16, kernel_size=2, stride=2),
								nn.ReLU(),
								nn.Conv2d(16, 32, kernel_size=2, stride=2),
								nn.ReLU(),
								nn.Conv2d(32, 64, kernel_size=2, stride=2),
								nn.ReLU(),
								nn.Conv2d(64, 64, kernel_size=2, stride=2),
								nn.ReLU(),
								nn.Conv2d(64, 64, kernel_size=2, stride=2),
								nn.ReLU())

	def forward(self,image):

		image = image.permute(0,3,1,2)
		x = self.image_network(image)
		return x





class QNetwork(nn.Module):
	def __init__(self):
		super(QNetwork, self).__init__()

		self.image_channels = 1

		self.action_dim = 4
		self.state_dim = 24
		self.goal_dim = 3

		self.combined_dim = self.action_dim+self.state_dim+self.goal_dim

		self.action_state_network = nn.Sequential(
										nn.Linear(self.action_dim+self.state_dim,256),
										nn.ReLU(),
										nn.Linear(256,64),
										nn.ReLU(),
										)

		self.image_network = FeatureExtractor(self.image_channels)

		self.combined_network = nn.Sequential(
								nn.Linear(self.combined_dim,256),
								nn.ReLU(),
								nn.Linear(256,256),
								nn.ReLU(),
								nn.Linear(256,256),
								nn.ReLU(),
								nn.Linear(256,1))




	def forward(self,state,action,goal):

		x1 = self.combined_network(torch.cat((state,action,goal),dim=1))

		return x1




class QT_Opt():
	def __init__(self, replay_buffer, qnet, target_qnet1, target_qnet2, action_min,action_max, 
										hidden_dim=64, q_lr=3e-3, cem_update_itr=4, select_num=6, num_samples=64):
		
		self.state_dim = 24
		self.action_dim = 4
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.num_samples = num_samples
		self.select_num = select_num
		self.cem_update_itr = cem_update_itr
		self.replay_buffer = replay_buffer
		self.qnet = qnet.to(self.device) # gpu
		self.target_qnet1 = target_qnet1.to(self.device)
		self.target_qnet2 = target_qnet2.to(self.device)
		self.cem = CEM(theta_dim = self.action_dim)  # cross-entropy method for updating
		theta = self.cem.sample()

		self.action_min = action_min
		self.action_max = action_max
		
		self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=q_lr)
		self.step_cnt = 0
		self.writer = SummaryWriter()

	def update(self, batch_size, epoch, episode_reward,  gamma=0.9, soft_tau=1e-2, update_delay=100):
		state, action, reward, next_state, goal, done = self.replay_buffer.sample(batch_size)
		self.step_cnt+=1

		# state = obs_normalizer.normalize(state)
		# next_state = obs_normalizer.normalize(next_state)
		# goal = goal_normalizer.normalize(goal)

		state_      = torch.FloatTensor(state).to(self.device)
		next_state_ = torch.FloatTensor(next_state).to(self.device)
		action     = torch.FloatTensor(action).to(self.device)
		goal 		= torch.FloatTensor(goal).to(self.device)
		reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
		done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
		#ipdb.set_trace()
		predict_q = self.qnet(state_, action, goal) # predicted Q(s,a) value


		# get argmax_a' from the CEM for the target Q(s', a'), together with updating the CEM stored weights
		# print(next_state_.shape)
		new_next_action = []
		for i in range(batch_size):      # batch of states, use them one by one
			new_next_action.append(self.cem_optimal_action(next_state_[i],goal[i]))

		new_next_action=torch.FloatTensor(new_next_action).to(self.device)

		target_q_min = torch.min(self.target_qnet1(next_state_, new_next_action, goal)[0], self.target_qnet2(next_state_, new_next_action, goal)[0])
		target_q = reward + (1-done)*gamma*target_q_min	#NOT SURE

		# print("predict_q: and target_q",predict_q,target_q.detach())
		q_loss = ((predict_q - target_q.detach())**2).mean()  # MSE loss, note that original paper uses cross-entropy loss
		self.writer.add_scalar("Loss/train", q_loss, epoch)
		self.writer.add_scalar("self.episode_reward/train", episode_reward, epoch)
		print('Q Loss: ',q_loss)
		self.q_optimizer.zero_grad()
		q_loss.backward()
		self.q_optimizer.step()

		# update the target nets, according to original paper:
		# one with Polyak averaging, another with lagged/delayed update
		self.target_qnet1=self.target_soft_update(self.qnet, self.target_qnet1, soft_tau)
		self.target_qnet2=self.target_delayed_update(self.qnet, self.target_qnet2, update_delay)

		# #ipdb.set_trace()
		return q_loss.item()
	

	def cem_optimal_action(self, state, goal):
		''' evaluate action wrt Q(s,a) to select the optimal using CEM
		return the only one largest, very gready
		state_image: gripper_states+image_feature vector
		'''
		# numpy_state = state_image.cpu().detach().numpy()
		states = torch.vstack([state]*self.num_samples)
		goals= torch.vstack([goal.unsqueeze(0)]*self.num_samples)

		''' the following line is critical:
		every time use a new/initialized cem, and cem is only for deriving the argmax_a', 
		but not for storing the weights of the policy.
		Without this line, the Q-network cannot converge, the loss will goes to infinity through time.
		I think the reason is that if you try to use the cem (gaussian distribution of policy weights) fitted 
		to the last state for the next state, it will generate samples mismatched to the global optimum for the 
		current state, which will lead to a local optimum for current state after cem iterations. And there may be
		several different local optimum for a similar state using cem from different last state, which will cause
		the optimal Q-value cannot be learned and even have a divergent loss for Q learning.
		'''
		self.cem.initialize(ini_mean_scale=0.0,ini_std_scale=0.1)  # the critical line
		for itr in range(self.cem_update_itr):

			actions = self.cem.sample_multi(self.num_samples)
			actions = np.clip(actions,a_min=self.action_min,a_max=self.action_max)
			# print("Actions sampled: ",actions)
			#ipdb.set_trace()
			q_values = self.target_qnet1(states, torch.FloatTensor(actions).to(self.device), goals) # 2 dim to 1 dim
			#ipdb.set_trace()
			q_values = q_values[0].detach().cpu().numpy().reshape(-1)
			max_idx=q_values.argsort()[-1]  # select maximal one q
			idx = q_values.argsort()[-int(self.select_num):]  # select top maximum q
			selected_actions = actions[idx]
			_,_= self.cem.update(selected_actions)  # mean as the theta for argmax_a'
		#ipdb.set_trace()
		optimal_action = actions[max_idx]
		return optimal_action

	def target_soft_update(self, net, target_net, soft_tau):
		''' Soft update the target net '''
		for target_param, param in zip(target_net.parameters(), net.parameters()):
			target_param.data.copy_(  # copy data value into target parameters
				target_param.data * (1.0 - soft_tau) + param.data * soft_tau
			)

		return target_net

	def target_delayed_update(self, net, target_net, update_delay):
		''' delayed update the target net '''
		if self.step_cnt%update_delay == 0:
			for target_param, param in zip(target_net.parameters(), net.parameters()):
				target_param.data.copy_(  # copy data value into target parameters
					param.data 
				)

		return target_net

	def save_model(self, path):
		torch.save(self.qnet.state_dict(), path)
		torch.save(self.target_qnet1.state_dict(), path)
		torch.save(self.target_qnet2.state_dict(), path)

	def load_model(self, path):
		self.qnet.load_state_dict(torch.load(path))
		self.target_qnet1.load_state_dict(torch.load(path))
		self.target_qnet2.load_state_dict(torch.load(path))
		self.qnet.eval()
		self.target_qnet1.eval()
		self.target_qnet2.eval()