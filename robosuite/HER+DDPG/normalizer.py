
import numpy as np


class Normalizer:
	def __init__(self, eps=1e-2, default_clip_range=np.inf):
		self.eps = eps
		self.default_clip_range = default_clip_range
		# some local information
		self.mean = None
		self.std = None
		# thread locker
	
	# update the parameters of the normalizer
	def update(self, v):

		self.mean = np.mean(v,axis=0)
		self.std = np.std(v,axis=0)
		self.std[np.where(self.std==0)[0]] = self.eps

	# normalize the observation
	def normalize(self, v, clip_range=None):
		# import ipdb
		# ipdb.set_trace()
		if clip_range is None:
			clip_range = self.default_clip_range
		if self.mean is not None and self.std is not None:
			return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)
		else:
			return v
