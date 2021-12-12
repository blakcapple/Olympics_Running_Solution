import torch 
import numpy as np 
from rl_trainer.algo.opponent import random_agent, rl_agent
import wandb
import time
import os 
from spinup.utils.mpi_pytorch import sync_params
from spinup.utils.mpi_tools import num_procs, proc_id
#dicretise action space
actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
            7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
            14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
            21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
            28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
            35: [200, 30]} 

def wrapped_action(action):

    action = actions_map[action.item()]
    wrapped_action = [[action[0]], [action[1]]] 

    return wrapped_action 


class Runner:

    def __init__(self, env, policy, buffer, total_epoch_step, logger, device, load_dir):
        
        self.env = env
        self.policy = policy 
        self.buffer = buffer
        self.total_epoch_step = total_epoch_step
        self.local_steps_per_epoch = int(total_epoch_step / num_procs())
        self.logger = logger 
        self.ep_ret_history = [] 
        self.best_ep_ret = -np.inf
        self.ctrl_agent_index = 1
        self.device = device
        self.opponet = random_agent()
        self.id = proc_id()
        self.load_pth = os.path.join(load_dir, 'models/actor.pth')

    def rollout(self, epochs):

        o, ep_ret, ep_len = self.env.reset(), 0, 0
        obs_ctrl_agent = np.array(o[self.ctrl_agent_index]['obs']).reshape(1, 25, 25)
        obs_oppo_agent = np.array(o[1-self.ctrl_agent_index]['obs']).reshape(1, 25, 25)
        start_time = time.time()
        episode = 0
        record_win = []
        record_win_op = []
    # Main loop: collect experience in env and update/log each epoch
        for epoch in range(epochs):
            for t in range(self.local_steps_per_epoch):
                a, v, logp = self.policy.step(torch.as_tensor([obs_ctrl_agent], dtype=torch.float32, device=self.device))
                action_opponent = self.opponet.act(torch.as_tensor([obs_oppo_agent], dtype=torch.float32, device=self.device))
                action_ctrl = wrapped_action(a)
                action = [action_opponent, action_ctrl]
                next_o, r, d, _, info = self.env.step(action)
                next_obs_ctrl_agent = np.array(next_o[self.ctrl_agent_index]['obs']).reshape(1, 25, 25)
                next_obs_oppo_agent = np.array(next_o[1-self.ctrl_agent_index]['obs']).reshape(1, 25, 25)
                if not d:
                    r = [-1., -1.]
                else:
                    if r[0] == r[1]:
                        r=[-1., -1.]
                    elif r[0] > r[1]:
                        r[1] -= 100
                    elif r[1] > r[0]:
                        r[0] -=100

                ep_ret += r[self.ctrl_agent_index]
                ep_len += 1

                # save and log
                self.buffer.store(obs_ctrl_agent, a, r[self.ctrl_agent_index], v, logp)
                self.logger.store(VVals=v)
                
                # Update obs (critical!)
                obs_ctrl_agent = next_obs_ctrl_agent
                obs_oppo_agent = next_obs_oppo_agent

                terminal = d 
                epoch_ended = t==(self.local_steps_per_epoch-1)

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if epoch_ended:
                        _, v, _ = self.policy.step(torch.as_tensor([obs_ctrl_agent], dtype=torch.float32, device=self.device))
                    else:
                        v = 0
                    self.buffer.finish_path(v)
                    if terminal:
                        episode +=1
                        win_is = 1 if r[self.ctrl_agent_index] > r[1-self.ctrl_agent_index] else 0
                        win_is_op = 1 if r[self.ctrl_agent_index]<r[1-self.ctrl_agent_index] else 0
                        record_win.append(win_is)
                        record_win_op.append(win_is_op)
                        self.logger.store(WinR=win_is)
                        self.logger.store(LoseR=win_is_op)
                        # only save EpRet / EpLen if trajectory finished
                        self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0
                    next_obs_ctrl_agent = np.array(o[self.ctrl_agent_index]['obs'])
                    next_obs_oppo_agent = np.array(o[1-self.ctrl_agent_index]['obs'])
            # update policy
            data = self.buffer.get()
            self.policy.learn(data)
            # Log info about epoch
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('WinR', average_only=True)
            self.logger.log_tabular('LoseR', average_only=True)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('VVals', with_min_and_max=True)
            self.logger.log_tabular('TotalEnvInteracts', (epoch+1)*self.total_epoch_step)
            self.logger.log_tabular('LossPi', average_only=True)
            self.logger.log_tabular('LossV', average_only=True)
            self.logger.log_tabular('DeltaLossPi', average_only=True)
            self.logger.log_tabular('DeltaLossV', average_only=True)
            self.logger.log_tabular('Entropy', average_only=True)
            self.logger.log_tabular('KL', average_only=True)
            self.logger.log_tabular('ClipFrac', average_only=True)
            self.logger.log_tabular('Time', time.time()-start_time)
            self.logger.dump_tabular()
            if epoch > 100 and epoch % 100 == 0:
                self.opponet.load_model(self.load_pth)  # load past model to self-play
            if epoch % 100 == 0 or epoch == (epochs-1):
                sync_params(self.policy.ac)
                if self.id == 0:
                    self.policy.save_models()
            if epoch == 500: # change the random agent to rl agent
                state_shape = [1, 25, 25]
                action_shape = 35
                self.opponet = rl_agent(state_shape, action_shape)
                self.opponet.load_model(self.load_pth)
            

