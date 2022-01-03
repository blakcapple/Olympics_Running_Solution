import json
from pathlib import Path
from random import random
import sys
import pdb
base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)
from utils.arguments import read_args
from env.chooseenv import make
from algo.ppo import PPO
from algo.buffer import PPOBuffer
import torch
from runner import Runner
import numpy as np
import os
from utils.log import init_log 
from env.vec_env.schmem_vec_env import ShmemVecEnv
from algo.opponent import rl_agent, random_agent
import wandb
from gym.spaces import Box, Dict, Discrete


def build_env(args):
    def get_env_fn(rank):
        def init_env():
            env = make(args.game_name, args.seed + rank*1000)
            # if not args.shuffle:
            #     env.specify_a_map(args.map) # specify a difficult map
            return env 
        return init_env 
    # if args.cpu == 1:
    #     return DummyVecEnv(get_env_fn(0))
    # else:
    space = Box(low=0, high=1, shape=(25, 25), dtype=np.float32)
    dict = Dict({'0': space, '1': space})
    return ShmemVecEnv([get_env_fn(i) for i in range(args.cpu)], spaces=dict)

def main(args):
    # Random seed
    args.seed += 100
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = build_env(args)

    state_shape = [1, 25, 25]
    action_num = args.action_num
    if args.action_type == 1:
        action_space = Box(low=np.array([-100, -30]), high=np.array([200, 30]))
        act_dim = 2
    elif args.action_type == 0:
        action_space = Discrete(action_num)
        act_dim = 1
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('device', device)
    local_epoch_step = args.epoch_step / args.cpu
    buffer = PPOBuffer(state_shape, act_dim, int(local_epoch_step), args.cpu, device, args.gamma, args.lamda)
    policy = PPO(state_shape, action_space, buffer, pi_lr=args.pi_lr, v_lr=args.v_lr, device=device, 
                entropy_c = args.entropy_c,logger=logger, clip_ratio=args.clip_ratio, 
                train_pi_iters=args.train_pi_iters, train_v_iters=args.train_v_iters, 
                target_kl=args.target_kl, save_dir=args.save_dir, max_grad_norm=args.max_grad_norm, 
                max_size=int(local_epoch_step), batch_size=int(args.epoch_step/args.mini_batch))
    if args.load:
        policy.load_models(args.load_dir, args.load_index)
        opponent = rl_agent(state_shape, action_space, device)
        load_path = os.path.join(args.load_dir, f'actor_{args.load_opponent_index}.pth')
        opponent.load_model(load_path)
    else:
        opponent = random_agent(action_space)
    runner = Runner(env, policy, opponent, buffer, int(local_epoch_step), logger, 
                    device, args.save_dir, args.cpu, args.load_index, action_space, act_dim)

    runner.rollout(args.train_epoch)

if __name__ == '__main__':
    wandb.init(project="Olympics_Running", entity="the-one")
    args = read_args()
    logger, save_path, log_file = init_log(args.save_dir)
    with open(save_path+'/arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # with open(save_path+'/arguments.txt', 'r') as f:
    #     args.__dict__ = json.load(f)
    args.save_dir = save_path
    main(args)

