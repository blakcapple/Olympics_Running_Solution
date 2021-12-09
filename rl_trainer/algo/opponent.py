import random
from rl_trainer.algo.network import CNNCategoricalActor
import torch 
import torch.nn as nn

#dicretise action space
actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
            7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
            14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
            21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
            28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
            35: [200, 30]} 

class random_agent:
    def __init__(self, seed=None):
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]
        #self.seed(seed)

    def seed(self, seed = None):
        random.seed(seed)

    def act(self, obs):
        force = random.uniform(self.force_range[0], self.force_range[1])
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        return [[force], [angle]]

class rl_agent:
    
    def __init__(self, state_shape, action_shape):
        
        self.actor = CNNCategoricalActor(state_shape, action_shape, nn.ReLU)

    def act(self, obs):

        pi, _ = self.actor(obs)
        a_raw = pi.sample()
        a = actions_map[a_raw.item()]
        wrap_a = [[a[0]], [a[1]]]
        return wrap_a

    def load_model(self, pth):

        self.actor.load_model(pth)
