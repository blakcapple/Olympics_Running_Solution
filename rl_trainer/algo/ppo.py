from algo.buffer import PPOBuffer
from network import CNNActorCritic


class PPO:

    def __init__(self, state_shape, action_shape, device):

        self.ac = CNNActorCritic(state_shape, action_shape).to(device)

