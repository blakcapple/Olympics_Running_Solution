from algo.network import CNNActorCritic
import torch
from torch.optim import Adam
from spinup.utils.mpi_pytorch import mpi_avg_grads
from spinup.utils.mpi_tools import mpi_avg
import os

class PPO:

    def __init__(self, state_shape, action_shape, pi_lr, v_lr, device, logger, clip_ratio=0.2, 
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

    def compute_loss_pi(self, data):

        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
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

        obs, ret = data['obs'], data['ret']

        return ((self.ac.v(obs) - ret)**2).mean()

    def learn(self, data):

        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()
        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break

            loss_pi.backward()
            # torch.nn.utils.clip_grad_norm_(self.ac.pi.parameters(), self.max_grad_norm)
            mpi_avg_grads(self.ac.pi)    # average grads across MPI processes
            self.pi_optimizer.step()
        # Value function learning
        for i in range(self.train_v_iters):
            self.v_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(self.ac.v)    # average grads across MPI processes
            torch.nn.utils.clip_grad_norm_(self.ac.v.parameters(), self.max_grad_norm)
            self.v_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        self.logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    def save_models(self):

        actor_pth = os.path.join(self.check_point_dir, 'actor.pth')
        self.ac.pi.save_model(actor_pth)
        critic_pth = os.path.join(self.check_point_dir, 'critic.pth')
        self.ac.v.save_model(critic_pth)

    def load_models(self):

        actor_pth = os.path.join(self.check_point_dir, 'actor.pth')
        self.ac.pi.load_model(actor_pth)
        critic_pth = os.path.join(self.check_point_dir, 'critic.pth')
        self.ac.v.load_model(critic_pth)

    def select_action(self, obs, phase='train'):

        a = self.ac.act(obs, phase)
        
        return a 

    def step(self, obs):

        a, v, logp = self.ac.step(obs)

        return a, v, logp 


