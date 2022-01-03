import random
from rl_trainer.algo.network import CNNCategoricalActor, CNNGaussianActor
from gym.spaces import Box, Discrete
import torch 
import torch.nn as nn
import numpy as np 


class random_agent:
    def __init__(self, action_space, seed=0):

        self.action_space = action_space
        self.seed(seed)

    def seed(self, seed = None):
        random.seed(seed)

    def act(self, obs):
        actions = []
        if isinstance(self.action_space, Discrete):
            for _ in range(obs.shape[0]):
                a = random.randint(0, self.action_space.n-1)
                actions.append(a)
        if isinstance(self.action_space, Box):
            for _ in range(obs.shape[0]):
                a =  np.random.uniform([-1, -1], [1, 1])
                actions.append(a)
        return actions

class rl_agent:
    
    def __init__(self, state_shape, action_space, device):
        
        if isinstance(action_space, Box):
            self.actor = CNNGaussianActor(state_shape, action_space.shape[0], nn.ReLU).to(device)
        elif isinstance(action_space, Discrete):
            self.actor = CNNCategoricalActor(state_shape, action_space.n, nn.ReLU).to(device)

    def act(self, obs):

        pi, _ = self.actor(obs)
        a_raw = pi.sample()
        
        return a_raw.detach().cpu().numpy()

    def load_model(self, pth):

        self.actor.load_model(pth)
