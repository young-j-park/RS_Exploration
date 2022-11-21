
import argparse
import logging
import copy

import torch
import numpy as np

from env import make_env
from agent import build_oldp_agent
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
    # Set seed
    torch.manual_seed(1234)
    np.random.seed(1234)

    # Load args
    args = parse_args()

    # 0. Initialize an experiment
    env = make_env(**vars(args))
    available_item_ids = env.reset()
    memory = ReplayMemory()
    user_history = UserHistory(args.num_users, args.state_window_size)
    state = user_history.get_state()

    # 1. Run old policy
    oldp_agent = build_oldp_agent(args)
    num_total_responses = 0
    for i_step in range(args.oldp_length):
        # Run policy
        slates = oldp_agent.select_action(available_item_ids)

        # Run simulator
        available_item_ids, responses, done = env.step(slates)
        if done:
            logging.warning(f'Whole users are terminated at {i_step}-th step.')
            break

        # Restore data
        num_total_responses += np.sum(responses)
        for i_slate in range(args.slate_size):
            user_history.push(slates[:, i_slate], responses[:, i_slate])
            next_state = user_history.get_state()
            for i_user in range(args.num_users):
                memory.push(state[i_user], slates[i_user, i_slate], next_state[i_user], responses[i_user, i_slate])
            state = next_state

        # Update agent
        oldp_agent.update_policy(i_step, slates, responses)

    # 2. Run exploration policy
    memory_old = copy.copy(memory)  # backup with a shallow copy (for the memory efficiency)
    print(num_total_responses)

    # 3. Evaluate


if __name__ == '__main__':
    main()
