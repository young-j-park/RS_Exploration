
import uuid
import os
import argparse
import logging
import random
import pickle as pkl

import torch
import numpy as np
import yaml

from env import make_env
from agent import build_agent
from utils import ReplayMemory, UserHistory
from evaluator import Evaluator
from config.config import SAVE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # seed
    parser.add_argument('--seed', type=int, default=0)

    # env
    parser.add_argument('--env_config', type=str, default='ie_debug')

    # user state (model)
    parser.add_argument('--state_window_size', type=int, default=28)
    parser.add_argument('--state_emb_dim', type=int, default=32)
    parser.add_argument('--agg_method', type=str, default='gru', choices=['mean', 'gru'])

    # policy length
    parser.add_argument('--warmup_length', type=int, default=7*4)
    parser.add_argument('--oldp_length', type=int, default=7*8)
    parser.add_argument('--expl_length', type=int, default=7*4)
    parser.add_argument('--test_length', type=int, default=7*4)

    # common
    parser.add_argument('--exploration_rate', type=float, default=0.05)

    # old policy
    parser.add_argument(
        '--old_policy',
        type=str,
        default='user_toppop',
        choices=['random', 'toppop', 'user_toppop']
    )
    parser.add_argument('--toppop_stochastic', type=bool, default=True)
    parser.add_argument('--toppop_windowsize', type=int, default=28)

    # new policy
    parser.add_argument(
        '--new_policy',
        type=str,
        default='dqn',
        choices=['dqn', 'cdqn', 'random', 'user_toppop', 'toppop']
    )

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.seed is None:
        args.seed = random.randint(0, 10000)
        logging.info(f'Set seed as {args.seed}.')
    with open(f'./config/env_{args.env_config}.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        args.env_type = data['env_type']
        args.num_users = int(data['num_users'])
        args.num_candidates = int(data['num_candidates'])
        args.slate_size = int(data['slate_size'])
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
    memory = ReplayMemory()
    user_history = UserHistory(args.num_users, args.state_window_size)
    state = user_history.get_state()
    evaluators = {
        'warmup': Evaluator(),
        'oldp': Evaluator(),
        'newp': Evaluator(),
        'test': Evaluator(),
    }

    # 1. Warm-Up
    random_agent = build_agent(args, 'random')
    for i_step in range(args.warmup_length):
        slates, responses, state = step(i_step, env, user_history, state, random_agent, memory, evaluators['warmup'], args)

    # 2. Run an old policy (oldp)
    oldp_agent = build_agent(args, args.old_policy)
    for i_step in range(args.oldp_length):
        slates, responses, state = step(i_step, env, user_history, state, oldp_agent, memory, evaluators['oldp'], args)
        oldp_agent.update_policy(slates, responses, memory)

    # 3. Build & pre-train a new policy (newp)
    if 'dqn' in args.new_policy:
        newp_agent = build_agent(args, args.new_policy)
        logging.info(f'Pre-train a {args.new_policy.upper()} agent.')
        newp_agent.update_policy(None, None, memory, train_steps=10_000, log_interval=1000)
    elif args.old_policy == args.new_policy:
        newp_agent = oldp_agent
    else:
        raise NotImplementedError

    # 4. Run exploration
    for i_step in range(args.expl_length):
        slates, responses, state = step(i_step, env, user_history, state, newp_agent, memory, evaluators['newp'], args)
        newp_agent.update_policy(slates, responses, memory, train_steps=1_000, log_interval=1000)

    # 5. Evaluate
    newp_agent.undo_exploration()
    for i_step in range(args.test_length):
        slates, responses, state = step(i_step, env, user_history, state, newp_agent, memory, evaluators['test'], args)
        newp_agent.update_policy(slates, responses, memory, train_steps=1_000, log_interval=1000)

    def _get_save_path(args):
        pth = None
        while pth is None or os.path.isfile(pth):
            experiment_id = uuid.uuid4()
            pth = f'{SAVE_DIR}/{args.env_config}_{args.new_policy}_seed{args.seed}_{experiment_id}.pkl'
        return pth

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = _get_save_path(args)
    with open(save_path, 'wb') as f:
        pkl.dump({'evaluators': evaluators, **vars(args)}, f)
    logging.info('Results are saved.')


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
    if i_step % 7 == 0:
        logging.info(f'[{i_step:02d}/{args.oldp_length}] Cum Hit Ratio: {evaluator.hit_ratio():.4f}')

    for i_slate in range(args.slate_size):
        user_history.push(slates[:, i_slate], responses[:, i_slate])
        next_state = user_history.get_state()
        for i_user in range(args.num_users):
            memory.push(state[i_user], [slates[i_user, i_slate]], next_state[i_user], responses[i_user, i_slate])
        state = next_state

    logging.debug([slates, responses])
    return slates, responses, state


if __name__ == '__main__':
    main()
