from random import random
import torch 
import numpy as np 
from rl_trainer.algo.opponent import random_agent, rl_agent
import time
import os 
import pdb
import wandb
from copy import deepcopy
#dicretise action space
actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
            7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
            14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
            21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
            28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
            35: [200, 30]} 

def wrapped_action(actions, opponent_actions, ctrl_agent_index):
    wrapped_actions = []
    for action, opponent_action in zip(actions, opponent_actions):
        real_action = actions_map[action]
        real_opponent_action = actions_map[opponent_action]
        wrapped_action = [[real_action[0]], [real_action[1]]]
        wrapped_opponent_action = [[real_opponent_action[0]], [real_opponent_action[1]]]
        if ctrl_agent_index == 1:
            wrapped_actions.append([wrapped_opponent_action, wrapped_action])
        elif ctrl_agent_index == 0:
            wrapped_actions.append([wrapped_action, wrapped_opponent_action])
    return wrapped_actions 


class Runner:

    def __init__(self, env, policy, buffer, local_epoch_step, logger, device, load_dir, n_rollout, load_index):
        
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
        if self.load_index > 0:
            state_shape = [1, 25, 25] 
            action_shape = 35
            self.opponet = rl_agent(state_shape, action_shape, self.device)
            self.opponet.actor = deepcopy(policy.ac.pi)
        else:
            self.opponet = random_agent()
        self.load_dir = load_dir
        self.n_rollout = n_rollout
        self.save_index = []

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
            for t in range(self.local_steps_per_epoch):
                a, v, logp = self.policy.step(torch.as_tensor(obs_ctrl_agent, dtype=torch.float32, device=self.device))
                action_opponent = self.opponet.act(torch.as_tensor(obs_oppo_agent, dtype=torch.float32, device=self.device))
                env_a = wrapped_action(a, action_opponent, self.ctrl_agent_index)
                next_o, r, d, info = self.env.step(env_a)
                next_obs_ctrl_agent = next_o[f'{self.ctrl_agent_index}'].reshape(self.n_rollout, 1, 25, 25)
                next_obs_oppo_agent = next_o[f'{1-self.ctrl_agent_index}'].reshape(self.n_rollout, 1, 25, 25)
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
                self.buffer.store(obs_ctrl_agent, a.reshape(self.n_rollout,1), r[:, self.ctrl_agent_index], v, logp)

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
                    next_obs_ctrl_agent = next_o[f'{self.ctrl_agent_index}'].reshape(self.n_rollout, 1, 25, 25)
                    next_obs_oppo_agent = next_o[f'{1-self.ctrl_agent_index}'].reshape(self.n_rollout, 1, 25, 25)
                # Update obs (critical!)
                obs_ctrl_agent = next_obs_ctrl_agent
                obs_oppo_agent = next_obs_oppo_agent

            # update policy
            data = self.buffer.get()
            self.policy.learn(data, epoch)
            # Log info about epoch
            wandb.log({'WinR':np.mean(record_win[-100:]), 'Reward':np.mean(epoch_reward)}, step=epoch)
            self.logger.info(f'epoch:{epoch}, WinR:{np.mean(record_win[-100:])}, LoseR:, {np.mean(record_win_op[-100:])}, time:{time.time() - start_time}')
            if epoch % 50 == 0 or epoch == (epochs-1):
                self.policy.save_models(self.load_index+epoch)
                self.save_index.append(self.load_index+epoch)

            # if self.load_index == 0: # if train from zero
            #     if epoch == 1000:   # change the random agent to rl agent
            #         state_shape = [1, 25, 25] 
            #         action_shape = 35
            #         self.opponet = rl_agent(state_shape, action_shape, self.device)
            #         load_pth = os.path.join(self.load_dir, 'models/actor_500.pth')
            #         self.opponet.load_model(load_pth)

            # elif self.load_index > 0: # if train from load model
            #     if epoch == 0: # load the model 
            #         state_shape = [1, 25, 25] 
            #         action_shape = 35
            #         self.opponet = rl_agent(state_shape, action_shape, self.device)
            #         load_pth = os.path.join(self.load_dir, f'models/actor_{self.load_index}.pth')
            #         self.opponet.load_model(load_pth)

            # if epoch > 1000 and epoch % 50 == 0:
            #     p = np.random.rand(1)
            #     low_number = max((len(self.save_index) - 10), 0) # the oldest model to self-play
            #     if p > 0.7:
            #         index = self.save_index[-1]  # load the latest model
            #     else:
            #         number = np.random.randint(low_number, len(self.save_index)-1) # load the history model
            #         index = self.save_index[number]
            #     load_pth = os.path.join(self.load_dir, f'models/actor_{index}.pth')
            #     self.opponet.load_model(load_pth)  # load past model to self-play
            # load new model every 1000 times
            if epoch % 1000 == 0 and  epoch > 0: 
                load_index = self.save_index[-10]
                state_shape = [1, 25, 25] 
                action_shape = 35
                self.opponet = rl_agent(state_shape, action_shape, self.device)
                load_pth = os.path.join(self.load_dir, f'models/actor_{load_index}.pth')
                self.opponet.load_model(load_pth)
            

