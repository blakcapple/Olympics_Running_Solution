import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_trainer.algo.cnn import CNNLayer
from torch.distributions.normal import Normal
import os 
import numpy as np 

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

## RL MODEL

        

class CNNCritic(nn.Module):

    def __init__(self, input_shape, activation):
        super().__init__()
        self.input_shape = input_shape
        self.cnn_layer = CNNLayer(input_shape)
        self.linear_layer = mlp([64]+[256]+[1], activation)
        self.v_net = nn.Sequential(self.cnn_layer, self.linear_layer)

    def forward(self, obs):

        return torch.squeeze(self.v_net(obs), -1)

    def save_model(self, pth):
        torch.save(self.state_dict(), pth)

    def load_model(self, pth):
        self.load_state_dict(torch.load(pth))

class CNNActorCritic(nn.Module):
    
    def __init__(self, state_shape, action_shape, activation=nn.ReLU):
        super().__init__()
        self.pi = CNNGaussianActor(state_shape, action_shape, activation)

        self.v = CNNCritic(state_shape, activation)
    
    def step(self, obs):

        with torch.no_grad():
            pi = self.pi.distribution(obs)
            a = pi.sample()
            logp_a = self.pi.log_prob_from_distribution(pi, a)
            v = self.v(obs)

        return a.detach().cpu().numpy(), v.detach().cpu().numpy(), logp_a.detach().cpu().numpy()

    def act(self, obs, phase='train'):
        if phase == 'test':
            return self.pi.eval(obs)
        elif phase == 'train':
            return self.step(obs)[0]
        else:
            raise NotImplementedError