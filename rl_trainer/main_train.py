import json
from pathlib import Path
import sys
base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)
from utils.arguments import read_args
from env.chooseenv import make
from algo.ppo import PPO
from algo.buffer import PPOBuffer
import torch 
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from runner import Runner
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params
from spinup.utils.mpi_tools import mpi_fork, proc_id, num_procs
import numpy as np
import wandb
from utils.log import init_log 
    
def main(args):
    # Special function to avoid certain slowdowns from PyTorch + MPI combo. 
    setup_pytorch_for_mpi()
    # Random seed
    args.seed += 100 * proc_id()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = make(args.game_name)
    state_shape = [1, 25, 25]
    action_shape = 35
    device = 'cpu' # spinning up mpi tools only suppor cpu 
    logger_kwargs = setup_logger_kwargs(args.algo, args.seed, data_dir=args.save_dir)
    logger = EpochLogger(**logger_kwargs)
    policy = PPO(state_shape, action_shape, pi_lr=args.pi_lr, v_lr=args.v_lr, device=device,
                logger=logger, clip_ratio=args.clip_ratio, train_pi_iters=args.train_pi_iters, 
                train_v_iters=args.train_v_iters, target_kl=args.target_kl, save_dir=args.save_dir, 
                max_grad_norm=args.max_grad_norm)
    sync_params(policy.ac) # Sync params across processes
    logger.setup_pytorch_saver(policy.ac)
    epoch_step = args.epoch_step
    local_epoch_step = int(args.epoch_step / num_procs())
    buffer = PPOBuffer(state_shape, 1, local_epoch_step, device, args.gamma, args.lamda)

    runner = Runner(env, policy, buffer, epoch_step, logger, device, args.save_dir)

    runner.rollout(args.train_epoch)

if __name__ == '__main__':
    # wandb.init(project="Olympics_Running", entity="the-one")
    args = read_args()
    logger, save_path, log_file = init_log(args.save_dir)
    with open(save_path+'/arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # with open(save_path+'/arguments.txt', 'r') as f:
    #     args.__dict__ = json.load(f)
    args.save_dir = save_path
    mpi_fork(args.cpu)
    main(args)

