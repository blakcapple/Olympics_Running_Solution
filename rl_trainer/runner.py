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

    def __init__(self, args, env, policy, opponent, buffer, logger, device, 
                action_space, act_dim):
        
        self.total_epoch_step = args.epoch_step
        self.n_rollout = args.cpu
        self.load_index = args.load_index
        self.local_steps_per_epoch = int(self.total_epoch_step / args.cpu)
        self.eval_step = args.eval_step
        self.randomplay_epoch = args.randomplay_epoch
        self.randomplay_interval = args.randomplay_interval
        self.selfplay_interval = args.selfplay_interval
        self.save_interval = args.save_interval
        self.eval_interval = args.eval_interval
        self.env = env
        self.policy = policy 
        self.buffer = buffer
        self.logger = logger 
        self.ep_ret_history = [] 
        self.best_ep_ret = -np.inf
        self.device = device
        self.action_space = action_space
        self.act_dim = act_dim
        self.opponet = opponent
        self.load_dir = os.path.join(args.save_dir, 'models') # where to load models for opponent
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
        
        patten = re.compile(r'actor_(?P<index>\d+)')
        files = os.listdir(self.load_dir)
        for file in files:
            index = patten.findall(file)
            if len(index) > 0 :
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
            wrapped_actions.append([wrapped_action, wrapped_opponent_action])

        return wrapped_actions

    def _wrapped_eval_action(self, actions):

        wrapped_actions = []
        for action in actions:
            if isinstance(self.action_space, Discrete):
                real_action = self.actions_map[action]
            elif isinstance(self.action_space, Box):
                action = np.clip(action, -1, 1)
                high = self.action_space.high
                low = self.action_space.low
                real_action = low + 0.5*(action + 1.0)*(high - low)
            wrapped_action = [[real_action[0]], [real_action[1]]]
            wrapped_opponent_action = [[0], [0]]
            wrapped_actions.append([wrapped_action, wrapped_opponent_action])

        return wrapped_actions

    def rollout(self, epochs):

        ep_lens = np.zeros(self.n_rollout)
        ep_rets = np.zeros(self.n_rollout)
        start_time = time.time()
        episode = 0
        record_win = []
        record_win_op = []
        epoch_reward = []
        o = self.env.reset()
        obs_ctrl_agent = o['0'].reshape(self.n_rollout, 4, 25, 25)
        obs_oppo_agent = o['1'].reshape(self.n_rollout, 4, 25, 25)
    # Main loop: collect experience in env and update/log each epoch
        for epoch in range(epochs):
            if (self.load_index+epoch) > self.randomplay_epoch and not self.begin_self_play:
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
                    ep_rets[i] += (r[i, 0])
                    ep_lens[i] +=1 

                # save and log
                self.buffer.store(obs_ctrl_agent, a.reshape(self.n_rollout, self.act_dim), r[:, 0], v, logp)
                next_obs_ctrl_agent = next_o['0'].reshape(self.n_rollout, 4, 25, 25)
                next_obs_oppo_agent = next_o['1'].reshape(self.n_rollout, 4, 25, 25)
                obs_ctrl_agent = next_obs_ctrl_agent
                obs_oppo_agent = next_obs_oppo_agent
                terminal = d 
                epoch_ended = t==(self.local_steps_per_epoch-1)

                for index, done in enumerate(terminal):
                    if done or epoch_ended:
                        if epoch_ended and not(done):
                            print('Warning: trajectory cut off by epoch at %d steps.'%ep_lens[index], flush=True)
                        # if trajectory didn't reach terminal state, bootstrap value target
                        if done:
                            v = 0
                        else:
                            _, v, _ = self.policy.step(torch.as_tensor([obs_ctrl_agent[index]], dtype=torch.float32, device=self.device))
                        self.buffer.finish_path(index, v)
                        if done:
                            episode +=1
                            win_is = 1 if r[index][0] > r[index][1] else 0
                            win_is_op = 1 if r[index][0] < r[index][1] else 0
                            record_win.append(win_is)
                            record_win_op.append(win_is_op)
                            epoch_reward.append(ep_rets[index])
                            # only save EpRet / EpLen if trajectory finished
                        ep_rets[index], ep_lens[index] = 0, 0
            
            # update policy
            self.policy.learn(epoch)
            # Log info about epoch
            wandb.log({'WinR':np.mean(record_win[-100:]), 'Reward':np.mean(epoch_reward)}, step=epoch)
            epoch_reward = []
            self.logger.info(f'epoch:{epoch}, WinR:{np.mean(record_win[-100:])}, LoseR:, {np.mean(record_win_op[-100:])}, time:{time.time() - start_time}')
            if epoch % self.save_interval == 0 or epoch == (epochs-1):
                self.policy.save_models(self.load_index+epoch)
                self.save_index.append(self.load_index+epoch)

            # eval the agent(assume the opponent is static)
            if epoch % self.eval_interval == 0 or epoch == (epochs-1):
                self.eval(self.eval_step, epoch)
                # reset the env
                o = self.env.reset()
                obs_ctrl_agent = o['0'].reshape(self.n_rollout, 4, 25, 25)
                obs_oppo_agent = o['1'].reshape(self.n_rollout, 4, 25, 25)
                
            if self.begin_self_play and epoch > 0:
    
                if self.self_play_flag:
                    if (epoch - self.last_epoch) == self.selfplay_interval:
                        self.opponet = random_agent(self.action_space) # give agent a break
                        self.self_play_flag = False
                        self.random_play_flag = True
                        self.last_epoch = epoch

                elif self.random_play_flag:
                    if (epoch - self.last_epoch) == self.randomplay_interval:
                        p = np.random.rand(1)
                        low_number = max((len(self.save_index) - 60), 0) # the oldest model to self-play
                        median_number = max((len(self.save_index) - 40), 1)
                        high_number = max((len(self.save_index) - 20), 2) # the newest model to self-play
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
    

    def eval(self, total_step, epoch):
        """
        evaluate the performence of agent 
        """
        ep_rets = np.zeros(self.n_rollout)
        start_time = time.time()
        eval_win = []
        eval_reward = []
        o = self.env.reset()
        obs_ctrl = o['0']
        obs_ctrl_agent = obs_ctrl.reshape(self.n_rollout, 4, 25, 25)
        for _ in range(total_step):
            a, _, _ = self.policy.step(torch.as_tensor(obs_ctrl_agent, dtype=torch.float32, device=self.device))
            env_a = self._wrapped_eval_action(a)
            next_o, r, d, _ = self.env.step(env_a)
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
                ep_rets[i] += (r[i, 0])
            # Update obs (critical!)
            next_obs_ctrl_agent = next_o['0'].reshape(self.n_rollout, 4, 25, 25)
            obs_ctrl_agent = next_obs_ctrl_agent
            terminal = d
            for index, done in enumerate(terminal):
                if done:
                    win_is = 1 if r[index][0] > r[index][1] else 0
                    eval_win.append(win_is)
                    eval_reward.append(ep_rets[index])
        wandb.log({'EvalWinR':np.mean(eval_win), 'EvalReward':np.mean(eval_reward)}, step=epoch)
        self.logger.info(f'epoch:{epoch}, EvalWinR:{np.mean(eval_win)}, EvalReward:, {np.mean(eval_reward)}, time:{time.time() - start_time}')

                






