
import argparse
import logging
from pprint import pformat
import copy

import torch
import numpy as np

from env import make_env
from agent import build_oldp_agent, build_newp_agent
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

    # user state (model)
    parser.add_argument('--state_window_size', type=int, default=20)
    parser.add_argument('--state_emb_dim', type=int, default=32)
    parser.add_argument('--agg_method', type=str, default='mean', choices=['mean', 'gru'])

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

    # new policy
    parser.add_argument(
        '--new_policy',
        type=str,
        default='dqn',
        choices=['dqn', 'cdqn']
    )
    parser.add_argument('--exploration_rate', type=float, default=0.1)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


def main():
    # Set seed
    torch.manual_seed(1234)
    np.random.seed(1234)

    # Initialize logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(levelname)s - %(message)s")

    # Load args
    args = parse_args()

    # 0. Initialize an experiment
    env = make_env(**vars(args))
    available_item_ids = env.reset()
    memory = ReplayMemory()
    user_history = UserHistory(args.num_users, args.state_window_size)
    state = user_history.get_state()

    # 1. Run an old policy (oldp)
    oldp_agent = build_oldp_agent(args)
    hit = {}
    num_total_responses = 0
    num_total_exposed = 0
    for i_step in range(args.oldp_length):
        # Run the policy
        slates = oldp_agent.select_action(available_item_ids)

        # Run simulator
        available_item_ids, responses, done = env.step(slates)
        if done:
            logging.warning(f'Whole users are terminated at {i_step}-th step')
            break

        # Restore data
        num_total_responses += np.sum(responses)
        num_total_exposed += slates.size
        hit[i_step] = num_total_responses / num_total_exposed
        if i_step % 10 == 0:
            logging.info(f'[{i_step:02d}/{args.oldp_length}] Cum Hit Ratio: {hit[i_step]:.4f}')

        for i_slate in range(args.slate_size):
            user_history.push(slates[:, i_slate], responses[:, i_slate])
            next_state = user_history.get_state()
            for i_user in range(args.num_users):
                memory.push_np(state[i_user], [slates[i_user, i_slate]], next_state[i_user], responses[i_user, i_slate])
            state = next_state

        # Update the agent
        oldp_agent.update_policy(i_step, slates, responses)

    # 2. Evaluate the old policy
    memory_old = copy.copy(memory)  # backup with a shallow copy (for the memory efficiency)

    # 3. Build & pre-train a new policy (newp)
    newp_agent = build_newp_agent(args)

    logging.info(f'Pre-train a {args.new_policy.upper()} agent.')
    newp_agent.update_policy(memory_old)

    # 4. Run exploration
    hit = {}
    num_total_responses = 0
    num_total_exposed = 0
    for i_step in range(args.expl_length):
        # Run the policy
        slates = newp_agent.select_action(state)

        # Run simulator
        available_item_ids, responses, done = env.step(slates)
        if done:
            logging.warning(f'Whole users are terminated at {i_step}-th step')
            break

        # Restore data
        num_total_responses += np.sum(responses)
        num_total_exposed += slates.size
        hit[i_step] = num_total_responses / num_total_exposed
        if i_step % 10 == 0:
            logging.info(f'[{i_step:02d}/{args.oldp_length}] Cum Hit Ratio: {hit[i_step]:.4f}')

        for i_slate in range(args.slate_size):
            user_history.push(slates[:, i_slate], responses[:, i_slate])
            next_state = user_history.get_state()
            for i_user in range(args.num_users):
                memory.push_np(state[i_user], [slates[i_user, i_slate]], next_state[i_user], responses[i_user, i_slate])
            state = next_state

        # Update the agent
        newp_agent.update_policy(i_step, slates, responses)

    # 5. Evaluate


if __name__ == '__main__':
    main()
