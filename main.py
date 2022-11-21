
import argparse
import logging

import torch
import numpy as np

from env import make_env
from agent import RandomAgent, TopPopAgent
from utils import ReplayMemory, UserHistory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # env
    parser.add_argument(
        '--env_type',
        type=str,
        default='interest_evolution',
        choices=['interest_evolution', 'interest_exploration']
    )

    parser.add_argument('--num_users', type=int, default=2000)
    parser.add_argument('--num_candidates', type=int, default=1500)
    parser.add_argument('--slate_size', type=int, default=3)

    # user state
    parser.add_argument('--state_window_size', type=int, default=20)

    # policy length
    parser.add_argument('--oldp_length', type=int, default=100)
    parser.add_argument('--expl_length', type=int, default=20)
    parser.add_argument('--test_length', type=int, default=30)

    # old policy
    parser.add_argument(
        '--old_policy',
        type=str,
        default='random',
        choices=['random', 'user_toppop', 'toppop']
    )
    parser.add_argument('--toppop_stochasticity', type=float, default=0.1)
    parser.add_argument('--toppop_windowsize', type=int, default=20)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


def main():
    # set seed
    torch.manual_seed(1234)
    np.random.seed(1234)

    # load args
    args = parse_args()

    # build env
    env = make_env(**vars(args))
    available_item_ids = env.reset()
    user_history = UserHistory(args.num_users, args.state_window_size)

    # run old policy
    if args.old_policy == 'random':
        oldp_agent = RandomAgent(
            args.num_users, args.num_candidates, args.slate_size
        )
    elif args.old_policy == 'toppop':
        oldp_agent = TopPopAgent(
            args.num_users, args.num_candidates, args.slate_size,
            args.toppop_windowsize, args.toppop_stochasticity, local=False
        )
    elif args.old_policy == 'user_toppop':
        oldp_agent = TopPopAgent(
            args.num_users, args.num_candidates, args.slate_size,
            args.toppop_windowsize, args.toppop_stochasticity, local=True
        )
    else:
        raise NotImplementedError

    memory_old = ReplayMemory()
    state = user_history.get_state()
    num_total_responses = 0
    for i_step in range(args.oldp_length):
        # Run policy
        slates = oldp_agent.select_action(available_item_ids)

        # Run simulator
        available_item_ids, responses, done = env.step(slates)
        if done:
            logging.warning(f'Whole users are terminated at {i_step}-th step.')
            break

        num_total_responses += np.sum(responses)
        for i_slate in range(args.slate_size):
            user_history.push(slates[:, i_slate], responses[:, i_slate])
            next_state = user_history.get_state()
            for i_user in range(args.num_users):
                memory_old.push(state[i_user], slates[i_user, i_slate], next_state[i_user], responses[i_user, i_slate])
            state = next_state

        # update agent
        oldp_agent.update_policy(i_step, slates, responses)

    # run exploration policy
    print(num_total_responses)

    # test


if __name__ == '__main__':
    main()
