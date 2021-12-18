import json
from pathlib import Path
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
from utils.log import init_log 
from env.vec_env.subproc_vec_env import SubprocVecEnv
import wandb


def build_env(args):
    def get_env_fn(rank):
        def init_env():
            env = make(args.game_name, args.seed + rank*1000)
            return env 
        return init_env 
    # if args.cpu == 1:
    #     return DummyVecEnv(get_env_fn(0))
    # else:
    return SubprocVecEnv([get_env_fn(i) for i in range(args.cpu)])

def main(args):
    # Random seed
    args.seed += 100
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = build_env(args)

    state_shape = [1, 25, 25]
    action_shape = 35
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('device', device)
    policy = PPO(state_shape, action_shape, pi_lr=args.pi_lr, v_lr=args.v_lr, device=device,
                logger=logger, clip_ratio=args.clip_ratio, train_pi_iters=args.train_pi_iters, 
                train_v_iters=args.train_v_iters, target_kl=args.target_kl, save_dir=args.save_dir, 
                max_grad_norm=args.max_grad_norm)
    local_epoch_step = args.epoch_step / args.cpu
    buffer = PPOBuffer(state_shape, 1, int(local_epoch_step), args.cpu, device, args.gamma, args.lamda)

    runner = Runner(env, policy, buffer, int(local_epoch_step), logger, device, args.save_dir, args.cpu)

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

