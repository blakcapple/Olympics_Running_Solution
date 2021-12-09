import argparse

def read_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default="olympics-running", type=str)
    parser.add_argument('--algo', default="ppo", type=str, help="ppo/sac")
    parser.add_argument('--train_epoch', default=700, type=int)
    parser.add_argument('--map', default=1, type = int)
    parser.add_argument('--shuffle_map', action='store_true')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--cpu', type=int, default=1)
    # ppo parameters
    parser.add_argument('--pi_lr', default=3e-4, type=float)
    parser.add_argument('--v_lr', default=1e-3, type=float)
    parser.add_argument('--train_pi_iters', default=80, type=int)
    parser.add_argument('--train_v_iters', default=80, type=int)
    parser.add_argument('--lamda', default=0.97, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--target_kl', default=0.05, type=float)
    parser.add_argument('--clip_ratio', default=0.2, type=float)
    parser.add_argument('--max_grad_norm', default=1, type=float)
    parser.add_argument('--epoch_step', default=500, type=int)
    parser.add_argument('--save_dir', default='data', type=str)

    args = parser.parse_args()

    return args