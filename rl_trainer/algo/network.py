import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from rl_trainer.algo.cnn import CNNLayer
import os 
## CNN 
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNNLayer(nn.Module):

    out_channel = 32
    hidden_size = 64
    kernel_size = 3
    stride = 1
    use_Relu = True
    use_orthogonal = True
    
    def __init__(self, state_shape):
        
        super().__init__()
        
        active_func = [nn.Tanh(), nn.ReLU()][self.use_Relu]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self.use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][self.use_Relu])
        input_channel = state_shape[0]
        input_width = state_shape[1]
        input_height = state_shape[2]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        cnn_out_size = self.out_channel * (input_width - self.kernel_size + self.stride) * (input_height - self.kernel_size + self.stride)
        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=self.out_channel,
                            kernel_size=self.kernel_size,
                            stride=self.stride)
                  ),
            active_func,
            Flatten(),
            init_(nn.Linear(cnn_out_size,
                            self.hidden_size)),
                            active_func,
                            )
    def forward(self, input):
        output = self.cnn(input)
        return output

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

## RL MODEL
class CNNCategoricalActor(nn.Module):

    def __init__(self, input_shape, act_dim, activation):
        super().__init__()
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

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self.distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self.log_prob_from_distribution(pi, act.view(-1))
        return pi, logp_a

    def save_model(self, pth):
        torch.save(self.state_dict(), pth)

    def load_model(self, pth):
        self.load_state_dict(torch.load(pth))

    def eval(self, obs):
        """
        return the best action
        """
        logits = self.logits_net(obs).view(-1)
        return torch.argmax(logits).item()
        

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
        self.pi = CNNCategoricalActor(state_shape, action_shape, activation)

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








    




