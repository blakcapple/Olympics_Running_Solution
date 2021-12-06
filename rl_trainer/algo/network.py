import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from rl_trainer.algo.cnn import CNNLayer

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class CNNCategoricalActor(nn.Module):

    def __init__(self, input_shape, act_dim, activation):

        self.input_shape = input_shape
        self.act_dim = act_dim 
        self.cnn_layer = CNNLayer(input_shape)
        self.linear_layer = mlp([64]+[256]+[act_dim], activation)
        self.logits_net = nn.Sequential(self.cnn_layer, self.linear_layer)
        
    def distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs):
        prob = self.logits_net(obs)

        return prob

class CNNCritic(nn.Module):

    def __init__(self, input_shape, activation):

        self.input_shape = input_shape
        self.cnn_layer = CNNLayer(input_shape)
        self.linear_layer = mlp([64]+[256]+[1], activation)
        self.v_net = nn.Sequential(self.cnn_layer, self.linear_layer)

    def forward(self, obs):

        return torch.squeeze(self.v_net(obs), -1)

class CNNActorCritic(nn.Module):

    def __init__(self, state_shape, action_shape, activation=nn.ReLU):
        
        self.pi = CNNCategoricalActor(state_shape, action_shape, activation)

        self.v = CNNCritic(state_shape, activation)
    
    def step(self, obs):

        with torch.no_grad():
            pi = self.pi.distribution(obs)
            a = pi.sample()
            logp_a = self.pi.log_prob_from_distribution(pi, a)
            v = self.v(obs)

        return a.detach().numpy(), v.detach().numpy(), logp_a.detach().numpy()

    def act(self, obs, phase='train'):
        if phase == 'test':
            prob = self.pi(obs)
            return torch.argmax(prob).item()
        elif phase == 'train':
            return self.step(obs)[0]
        else:
            raise NotImplementedError






    




