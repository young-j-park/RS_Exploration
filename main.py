
import argparse
import logging
import random
import copy

import torch
import numpy as np

from env import make_env
from agent import build_oldp_agent, build_newp_agent
from utils import ReplayMemory, UserHistory
from evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # seed
    parser.add_argument('--seed', type=int, default=1234)

    # env
    parser.add_argument(
        '--env_type',
        type=str,
        default='interest_exploration',
        choices=['interest_evolution', 'interest_exploration']
    )

    parser.add_argument('--num_users', type=int, default=1000)
    parser.add_argument('--num_candidates', type=int, default=1700)
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
        choices=['dqn', 'cdqn', 'random', 'user_toppop', 'toppop']
    )
    parser.add_argument('--exploration_rate', type=float, default=0.1)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


def main():
    # Initialize logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(levelname)s - %(message)s")

    # Load args
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 0. Initialize an experiment
    env = make_env(**vars(args))
    available_item_ids = env.reset()
    memory = ReplayMemory()
    user_history = UserHistory(args.num_users, args.state_window_size)
    state = user_history.get_state()
    evaluators = {
        'oldp': Evaluator(),
        'newp': Evaluator(),
        'test': Evaluator(),
    }

    # 1. Run an old policy (oldp)
    oldp_agent = build_oldp_agent(args)
    for i_step in range(args.oldp_length):
        slates, responses = step(i_step, env, user_history, state, oldp_agent, memory, evaluators['oldp'], args)
        oldp_agent.update_policy(slates, responses, memory)

        if oldp_agent.p is None and i_step >= oldp_agent.window_size:
            oldp_agent.begin_partial_exploring()

    # 2. Evaluate the old policy
    memory_old = copy.copy(memory)  # backup with a shallow copy (for the memory efficiency)

    # 3. Build & pre-train a new policy (newp)
    if 'dqn' in args.new_policy:
        newp_agent = build_newp_agent(args)
        logging.info(f'Pre-train a {args.new_policy.upper()} agent.')
        newp_agent.update_policy(None, None, memory, train_epoch=10_000, log_interval=1000)
    else:
        newp_agent = oldp_agent

    # 4. Run exploration
    for i_step in range(args.expl_length):
        slates, responses = step(i_step, env, user_history, state, newp_agent, memory, evaluators['newp'], args)
        newp_agent.update_policy(slates, responses, memory, train_epoch=1_000, log_interval=1000)

    # 5. Evaluate
    newp_agent.undo_exploration()
    for i_step in range(args.test_length):
        step(i_step, env, user_history, state, newp_agent, memory, evaluators['test'], args)
        # newp_agent.update_policy(memory, train_epoch=1_000, log_interval=1000)


def step(i_step, env, user_history, state, agent, memory, evaluator, args):
    # Run the policy
    slates = agent.select_action(state)

    # Run simulator
    available_item_ids, responses, done = env.step(slates)
    if done:
        logging.warning(f'Whole users are terminated at {i_step}-th step')
        exit()

    # Restore data
    evaluator.add(slates, responses)
    if i_step % 10 == 0:
        logging.info(f'[{i_step:02d}/{args.oldp_length}] Cum Hit Ratio: {evaluator.hit_ratio():.4f}')

    for i_slate in range(args.slate_size):
        user_history.push(slates[:, i_slate], responses[:, i_slate])
        next_state = user_history.get_state()
        for i_user in range(args.num_users):
            memory.push(state[i_user], [slates[i_user, i_slate]], next_state[i_user], responses[i_user, i_slate])
        state = next_state

    return slates, responses


if __name__ == '__main__':
    main()
