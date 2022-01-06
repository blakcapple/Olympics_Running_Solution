from random import random
import torch 
import numpy as np 
from rl_trainer.algo.opponent import random_agent, rl_agent
import time
import os 
import pdb
import wandb
from copy import deepcopy
from gym.spaces import Box, Discrete
import re

class Runner:

    def __init__(self, env, policy, opponent, buffer, local_epoch_step, logger, device, 
                load_dir, n_rollout, load_index, action_space, act_dim):
        
        self.env = env
        self.policy = policy 
        self.buffer = buffer
        self.total_epoch_step = local_epoch_step * n_rollout
        self.local_steps_per_epoch = local_epoch_step
        self.logger = logger 
        self.ep_ret_history = [] 
        self.best_ep_ret = -np.inf
        self.ctrl_agent_index = 1
        self.device = device
        self.load_index = load_index
        self.action_space = action_space
        self.act_dim = act_dim
        self.opponet = opponent
        self.load_dir = os.path.join(load_dir, 'models') # where to load models for opponent
        self.n_rollout = n_rollout
        self.save_index = []
        if isinstance(self.opponet, rl_agent):
            self.random_play_flag = False
            self.self_play_flag = True  
        else:
            self.self_play_flag = False
            self.random_play_flag = True
        if self.self_play_flag:
            self.begin_self_play = True
        else:
            self.begin_self_play = False
        if isinstance(action_space, Discrete):
            self.actions_map = self._set_actions_map(action_space.n)
        else:
            self.actions_map = None
        self._read_history_models() # read history models from dir
        self.last_epoch = 0

    def _read_history_models(self):
        
        number = re.compile(r'\d+')
        files = os.listdir(self.load_dir)
        for file in files:
            index = number.findall(file)
            self.save_index.append(int(index[0]))
        self.save_index.sort() # from low to high sorting

    def _set_actions_map(self, action_num):
        #dicretise action space
        forces = np.linspace(-100, 200, num=int(np.sqrt(action_num)), endpoint=True)
        thetas = np.linspace(-30, 30, num=int(np.sqrt(action_num)), endpoint=True)
        actions = [[force, theta] for force in forces for theta in thetas]
        actions_map = {i:actions[i] for i in range(action_num)}
        return actions_map
    
    def _wrapped_action(self, actions, opponent_actions):
        
        wrapped_actions = []
        for action, opponent_action in zip(actions, opponent_actions):
            if isinstance(self.action_space, Discrete):
                real_action = self.actions_map[action]
                real_opponent_action = self.actions_map[opponent_action]
            elif isinstance(self.action_space, Box):
                action = np.clip(action, -1, 1)
                opponent_action = np.clip(opponent_action, -1, 1)
                high = self.action_space.high
                low = self.action_space.low
                real_action = low + 0.5*(action + 1.0)*(high - low)
                real_opponent_action = low + 0.5*(opponent_action + 1.0)*(high - low)
            wrapped_action = [[real_action[0]], [real_action[1]]]
            wrapped_opponent_action = [[real_opponent_action[0]], [real_opponent_action[1]]]
            if self.ctrl_agent_index == 1:
                wrapped_actions.append([wrapped_opponent_action, wrapped_action])
            elif self.ctrl_agent_index == 0:
                wrapped_actions.append([wrapped_action, wrapped_opponent_action])

        return wrapped_actions

    def rollout(self, epochs):

        o= self.env.reset()
        ep_lens = np.zeros(self.n_rollout)
        ep_rets = np.zeros(self.n_rollout)
        obs_ctrl = o[f'{self.ctrl_agent_index}']
        obs_oppo = o[f'{1-self.ctrl_agent_index}']
        obs_ctrl_agent = obs_ctrl.reshape(self.n_rollout, 1, 25, 25)
        obs_oppo_agent = obs_oppo.reshape(self.n_rollout, 1, 25, 25)
        start_time = time.time()
        episode = 0
        record_win = []
        record_win_op = []
        epoch_reward = []
    # Main loop: collect experience in env and update/log each epoch
        for epoch in range(epochs):
            if (self.load_index+epoch) > 2000 and not self.begin_self_play:
                self.begin_self_play = True
                self.self_play_flag = True
            for t in range(self.local_steps_per_epoch):
                a, v, logp = self.policy.step(torch.as_tensor(obs_ctrl_agent, dtype=torch.float32, device=self.device))
                action_opponent = self.opponet.act(torch.as_tensor(obs_oppo_agent, dtype=torch.float32, device=self.device))
                env_a = self._wrapped_action(a, action_opponent)
                next_o, r, d, info = self.env.step(env_a)
                for i, done in enumerate(d):
                    if not done:
                        r[i] = [-1, -1]
                    if done:
                        if r[i][0] == r[i][1]:
                            r[i]=[-1., -1.]
                        elif r[i][1] > r[i][0]:
                            r[i][0] -= 100
                        elif r[i][0] > r[i][1]:
                            r[i][1] -=100

                for i in range(self.n_rollout):
                    ep_rets[i] += (r[i, self.ctrl_agent_index])
                    ep_lens[i] +=1 

                # save and log
                self.buffer.store(obs_ctrl_agent, a.reshape(self.n_rollout, self.act_dim), r[:, self.ctrl_agent_index], v, logp)

                terminal = d 
                epoch_ended = t==(self.local_steps_per_epoch-1)

                for index, done in enumerate(terminal):
                    if done or epoch_ended:
                        if epoch_ended and not(done):
                            print('Warning: trajectory cut off by epoch at %d steps.'%ep_lens[index], flush=True)
                        # if trajectory didn't reach terminal state, bootstrap value target
                        if epoch_ended:
                            _, v, _ = self.policy.step(torch.as_tensor([obs_ctrl_agent[index]], dtype=torch.float32, device=self.device))
                        else:
                            v = 0
                        self.buffer.finish_path(index, v)
                        if done:
                            episode +=1
                            win_is = 1 if r[index][self.ctrl_agent_index] > r[index][1-self.ctrl_agent_index] else 0
                            win_is_op = 1 if r[index][self.ctrl_agent_index] < r[index][1-self.ctrl_agent_index] else 0
                            record_win.append(win_is)
                            record_win_op.append(win_is_op)
                            epoch_reward.append(ep_rets[index])
                            # only save EpRet / EpLen if trajectory finished
                        ep_rets[index], ep_lens[index] = 0, 0
                if epoch_ended:
                    self.ctrl_agent_index = np.random.randint(0,2) # random ctrl index
                    # reset the env
                    next_o = self.env.reset()
                # Update obs (critical!)
                next_obs_ctrl_agent = next_o[f'{self.ctrl_agent_index}'].reshape(self.n_rollout, 1, 25, 25)
                next_obs_oppo_agent = next_o[f'{1-self.ctrl_agent_index}'].reshape(self.n_rollout, 1, 25, 25)
                obs_ctrl_agent = next_obs_ctrl_agent
                obs_oppo_agent = next_obs_oppo_agent

            # update policy
            self.policy.learn(epoch)
            # Log info about epoch
            wandb.log({'WinR':np.mean(record_win[-100:]), 'Reward':np.mean(epoch_reward)}, step=epoch)
            epoch_reward = []
            self.logger.info(f'epoch:{epoch}, WinR:{np.mean(record_win[-100:])}, LoseR:, {np.mean(record_win_op[-100:])}, time:{time.time() - start_time}')
            if epoch % 50 == 0 or epoch == (epochs-1):
                self.policy.save_models(self.load_index+epoch)
                self.save_index.append(self.load_index+epoch)

            if self.begin_self_play and epoch > 0:
    
                if self.self_play_flag:
                    if (epoch - self.last_epoch) == 40:
                        self.opponet = random_agent(self.action_space) # give agent a break
                        self.self_play_flag = False
                        self.random_play_flag = True
                        self.last_epoch = epoch

                elif self.random_play_flag:
                    if (epoch - self.last_epoch) == 10:
                        p = np.random.rand(1)
                        low_number = max((len(self.save_index) - 40), 0) # the oldest model to self-play
                        median_number = max((len(self.save_index) - 20), 1)
                        high_number = max((len(self.save_index) - 10), 2) # the newest model to self-play
                        if p > 0.8:
                            number = np.random.randint(median_number, high_number)
                            index = self.save_index[number]  # load the newer model 
                        else:
                            number = np.random.randint(low_number, median_number) # load the older model 
                            index = self.save_index[number]
                        state_shape = [1, 25, 25] 
                        self.opponet = rl_agent(state_shape, self.action_space, self.device) 
                        load_pth = os.path.join(self.load_dir, f'actor_{index}.pth')
                        self.opponet.load_model(load_pth)
                        self.self_play_flag = True
                        self.random_play_flag = False
                        self.last_epoch = epoch

