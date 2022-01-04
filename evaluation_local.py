
import numpy as np
import torch
import random
from agents.rl.submission import agent, agent_base
from env.chooseenv import make
from tabulate import tabulate
import argparse
from torch.distributions import Categorical
import os
from gym.spaces import Box


def get_join_actions(state, algo_list):

    joint_actions = []

    for agent_idx in range(len(algo_list)):
        if algo_list[agent_idx] == 'random':
            driving_force = random.uniform(-100, 200)
            turing_angle = random.uniform(-30, 30)
            joint_actions.append([[driving_force], [turing_angle]])

        elif algo_list[agent_idx] == 'rl_base':

            obs = state[agent_idx]['obs']
            actions_raw = agent_base.choose_action(obs, False)
            if agent_base.is_act_continuous:
                actions_raw = actions_raw.detach().cpu().numpy().reshape(-1)
                action = np.clip(actions_raw, -1, 1)
                high = agent.action_space.high
                low = agent.action_space.low
                actions = low + 0.5*(action + 1.0)*(high - low)
            else:
                actions = agent_base.actions_map[actions_raw.item()]
            joint_actions.append([[actions[0]], [actions[1]]])

        elif algo_list[agent_idx] == 'rl':
            obs = state[agent_idx]['obs']
            actions_raw = agent.choose_action(obs, True)
            if agent.is_act_continuous:
                actions_raw = actions_raw.detach().cpu().numpy().reshape(-1)
                action = np.clip(actions_raw, -1, 1)
                high = agent.action_space.high
                low = agent.action_space.low
                actions = low + 0.5*(action + 1.0)*(high - low)
            else:
                actions = agent.actions_map[actions_raw.item()]
            joint_actions.append([[actions[0]], [actions[1]]])

    return joint_actions






RENDER = True

def run_game(env, algo_list, episode, shuffle_map,map_num, verbose=False):
    total_reward = np.zeros(2)
    num_win = np.zeros(3)       #agent 1 win, agent 2 win, draw
    episode = int(episode)
    for i in range(1, int(episode)+1):
        episode_reward = np.zeros(2)

        state = env.reset(shuffle_map)
        if RENDER:
            env.env_core.render()

        step = 0

        while True:
            joint_action = get_join_actions(state, algo_list)
            next_state, reward, done, _, info = env.step(joint_action)
            reward = np.array(reward)
            episode_reward += reward
            if RENDER:
                env.env_core.render()

            if done:
                if reward[0] != reward[1]:
                    if reward[0]==100:
                        num_win[0] +=1
                    elif reward[1] == 100:
                        num_win[1] += 1
                    else:
                        raise NotImplementedError
                else:
                    num_win[2] += 1

                if not verbose:
                    print('.', end='')
                    if i % 100 == 0 or i==episode:
                        print()
                break
            state = next_state
            step += 1
        total_reward += episode_reward
    total_reward/=episode
    print("total reward: ", total_reward)
    print('Result in map {} within {} episode:'.format(map_num, episode))
    #print(f'\nResult in base on {episode} in map {map_num} ', end='')

    header = ['Name', algo_list[0], algo_list[1]]
    data = [['score', np.round(total_reward[0], 2), np.round(total_reward[1], 2)],
            ['win', num_win[0], num_win[1]]]
    print(tabulate(data, headers=header, tablefmt='pretty'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--my_ai", default='rl', help='rl/random')
    parser.add_argument("--opponent", default='rl_base', help='rl_base/random')
    parser.add_argument("--episode", default=20)
    parser.add_argument("--map", default='6', help='1/2/3/4/all')
    args = parser.parse_args()

    env_type = "olympics-running"
    game = make(env_type, conf=None, seed = 1)

    if args.map != 'all':
        game.specify_a_map(int(args.map))
        shuffle = False
    else:
        shuffle = True

    torch.manual_seed(3)
    np.random.seed(3)
    random.seed(3)

    agent_list = [args.opponent, args.my_ai]        #your are controlling agent green
    # agent_list = [args.my_ai, args.opponent]
    run_game(game, algo_list=agent_list, episode=args.episode, shuffle_map=shuffle,map_num=args.map,verbose=False)

