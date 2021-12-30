import random
from rl_trainer.algo.network import CNNCategoricalActor, CNNGaussianActor
from gym.spaces import Box, Discrete
import torch 
import torch.nn as nn


class random_agent:
    def __init__(self, seed=None):
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]
        #self.seed(seed)

    def seed(self, seed = None):
        random.seed(seed)

    def act(self, obs):
        actions = []
        for _ in range(obs.shape[0]):
            a = random.randint(0, 35)
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
