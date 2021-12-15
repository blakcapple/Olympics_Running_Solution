import torch.nn as nn
import torch 
import os
from torch.distributions import Categorical

actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}           #dicretise action space

###
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

###
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


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
        return torch.argmax(logits)
####
class RLAgent:

    def __init__(self, state_shape, action_shape):
        
        self.actor = CNNCategoricalActor(state_shape, action_shape, nn.ReLU)

    def choose_action(self, obs):

        state = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0)
        pi, _ = self.actor(state)
        a_raw = pi.sample()

        return a_raw

    def load_model(self, pth):

        self.actor.load_model(pth)


state_shape = [1, 25, 25]
action_shape = 35
load_pth = os.path.dirname(os.path.abspath(__file__)) + "/actor_1700.pth"
agent = RLAgent(state_shape, action_shape)
agent.load_model(load_pth)
load_path2 = os.path.dirname(os.path.abspath(__file__)) + "/actor_700.pth"
agent_base = RLAgent(state_shape, action_shape)
agent_base.load_model(load_path2)

def my_controller(observation_list, action_space_list, is_act_continuous):
    obs = observation_list['obs'].copy()
    actions_raw = agent.choose_action(obs)
    actions = actions_map[actions_raw.item()]
    wrapped_actions = [[actions[0]], [actions[1]]]
    return wrapped_actions

