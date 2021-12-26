from rl_trainer.algo.network import CNNActorCritic
import torch
from torch.optim import Adam
import os
from copy import deepcopy
import wandb
import numpy as np 

class PPO:

    def __init__(self, state_shape, action_shape, pi_lr, v_lr, device, logger, max_size, batch_size, clip_ratio=0.2, 
                train_pi_iters=80, train_v_iters=80, target_kl=0.01, max_grad_norm=0.5, save_dir='data/models'):

        self.ac = CNNActorCritic(state_shape, action_shape).to(device)
        self.clip_ratio = clip_ratio
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.v_optimizer = Adam(self.ac.v.parameters(), lr=v_lr)
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        self.logger = logger
        self.check_point_dir = os.path.join(save_dir, 'models')
        os.makedirs(self.check_point_dir, exist_ok=True)
        self.max_grad_norm = max_grad_norm
        self.max_size = max_size
        self.batch_size = batch_size

    def compute_loss_pi(self, data):
        
        data = deepcopy(data)
        idxs = np.random.randint(0, self.max_size, self.batch_size)
        obs, act, adv, logp_old = data['obs'][idxs], data['act'][idxs], data['adv'][idxs], data['logp'][idxs]
        obs = obs.view(-1, obs.shape[2], obs.shape[3], obs.shape[4])
        act = act.view(-1, act.shape[2])
        adv = adv.view(-1)
        logp_old = logp_old.view(-1)
        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(self, data):
        
        data = deepcopy(data)
        idxs = np.random.randint(0, self.max_size, self.batch_size)
        obs, ret = data['obs'][idxs], data['ret'][idxs]
        obs = obs.view(-1, obs.shape[2], obs.shape[3], obs.shape[4])
        ret = ret.view(-1)

        return ((self.ac.v(obs) - ret)**2).mean()

    def learn(self, data):

        # pi_l_old, pi_info_old = self.compute_loss_pi(data)
        # pi_l_old = pi_l_old.item()
        # v_l_old = self.compute_loss_v(data).item()
        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                self.logger.info('Early stopping at step %d due to reaching max kl.'%i)
                break

            loss_pi.backward()
            # torch.nn.utils.clip_grad_norm_(self.ac.pi.parameters(), self.max_grad_norm)
            self.pi_optimizer.step()
        wandb.log({'stop_pi':i})
        # Value function learning
        for i in range(self.train_v_iters):
            self.v_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.v.parameters(), self.max_grad_norm)
            self.v_optimizer.step()

        # Log changes from update
        kl = pi_info['kl']
        wandb.log({'Loss_pi':loss_pi, 'Loss_v':loss_v, 'KL': kl})

    def save_models(self, index=None):
        
        if index is not None:
            actor_pth = os.path.join(self.check_point_dir, f'actor_{index}.pth')
            critic_pth = os.path.join(self.check_point_dir, f'critic_{index}.pth')
        elif index == None:
            actor_pth = os.path.join(self.check_point_dir, 'actor.pth')
            critic_pth = os.path.join(self.check_point_dir, 'critic.pth')
        self.ac.v.save_model(critic_pth)
        self.ac.pi.save_model(actor_pth)

    def load_models(self, load_path, index=None):

        if index is not None:
            actor_pth = os.path.join(load_path, f'actor_{index}.pth')
            critic_pth = os.path.join(load_path, f'critic_{index}.pth')
        elif index == None:
            actor_pth = os.path.join(load_path, 'actor.pth')
            critic_pth = os.path.join(load_path, 'critic.pth')
        self.ac.v.load_model(critic_pth)
        self.ac.pi.load_model(actor_pth)

    def select_action(self, obs, phase='train'):

        a = self.ac.act(obs, phase)
        
        return a 

    def step(self, obs):

        a, v, logp = self.ac.step(obs)

        return a, v, logp 


