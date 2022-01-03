import numpy as np
from numpy.core.numeric import indices
from algo.utils import combined_shape, discount_cumsum
import torch 

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, n_rollout, device, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, n_rollout, *obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, n_rollout, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros((size, n_rollout), dtype=np.float32)
        self.rew_buf = np.zeros((size, n_rollout), dtype=np.float32)
        self.ret_buf = np.zeros((size, n_rollout), dtype=np.float32)
        self.val_buf = np.zeros((size, n_rollout), dtype=np.float32)
        self.logp_buf = np.zeros((size, n_rollout), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = 0, size
        self.path_start_idx = np.zeros(n_rollout, dtype=int) # every rollout have different idx
        self.device = device
        self.n_rollout = n_rollout
        self.generate_ready = False
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, n_rollout, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx[n_rollout], self.ptr)
        rews = np.append(self.rew_buf[path_slice, n_rollout], last_val)
        vals = np.append(self.val_buf[path_slice, n_rollout], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice, n_rollout] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice, n_rollout] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx[n_rollout] = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr = 0
        self.path_start_idx = np.zeros(self.n_rollout, dtype=int)
        # the next two lines implement the advantage normalization trick
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.var(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in data.items()}
    
    def generate_data(self, batch_size):
        """
        randomly sample data from buffer , call this at the end of the epoch 
        """
        assert self.ptr == self.max_size
        if not self.generate_ready:
            # the next two lines implement the advantage normalization trick
            adv_mean = np.mean(self.adv_buf)
            adv_std = np.var(self.adv_buf)
            self.adv_buf = (self.adv_buf - adv_mean) / adv_std
            _tensor_names = [
                "obs_buf",
                "act_buf",
                "ret_buf",
                "adv_buf",
                "logp_buf",
            ]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generate_ready = True
        indices = np.random.permutation(self.max_size*self.n_rollout)
        # indices = np.arange(self.max_size*self.n_rollout)
        start_idx = 0 
        while start_idx < self.max_size*self.n_rollout:
            yield self._get_samples(indices[start_idx:start_idx+batch_size])
            start_idx += batch_size

    def swap_and_flatten(self, arr):
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
    
    def _get_samples(self, indices):
        
        data = dict(obs=self.obs_buf[indices], 
                    act=self.act_buf[indices], 
                    ret=self.ret_buf[indices].flatten(),
                    adv=self.adv_buf[indices].flatten(), 
                    logp=self.logp_buf[indices].flatten(),)
        
        return {k: self.to_torch(v) for k,v in data.items()}
    
    def to_torch(self, array, copy=True):
        
        """
        Convert a numpy array to a Pytorch Tensor
        """
        
        if copy:
            return torch.tensor(array).to(self.device)
        return torch.as_tensor(array).to(self.device)
    
    def reset(self):
        """
        Call this to reset the buffer
        """
        self.obs_buf = np.zeros((self.max_size, self.n_rollout, *self.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((self.max_size, self.n_rollout, self.act_dim), dtype=np.float32)
        self.adv_buf = np.zeros((self.max_size, self.n_rollout), dtype=np.float32)
        self.rew_buf = np.zeros((self.max_size, self.n_rollout), dtype=np.float32)
        self.ret_buf = np.zeros((self.max_size, self.n_rollout), dtype=np.float32)
        self.val_buf = np.zeros((self.max_size, self.n_rollout), dtype=np.float32)
        self.logp_buf = np.zeros((self.max_size, self.n_rollout), dtype=np.float32)
        self.ptr = 0
        self.path_start_idx = np.zeros(self.n_rollout, dtype=int)
        self.generate_ready = False
