import copy
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
from config.config import SAVE_DIR, BATCH_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # seed
    parser.add_argument('--seed', type=int, default=None)

    # env
    parser.add_argument('--env_config', type=str, default='ie_debug')

    # user state (model)
    parser.add_argument('--state_window_size', type=int, default=28)
    parser.add_argument('--state_emb_dim', type=int, default=32)
    parser.add_argument('--agg_method', type=str, default='gru', choices=['mean', 'gru'])

    # policy length
    # parser.add_argument('--warmup_length', type=int, default=7*4)
    parser.add_argument('--oldp_length', type=int, default=7*8*3)
    parser.add_argument('--expl_length', type=int, default=7*8*3*100)
    parser.add_argument('--test_length', type=int, default=7*8*3)

    # common
    parser.add_argument('--exploration_rate', type=float, default=0.05)

    # old policy
    parser.add_argument(
        '--old_policy',
        type=str,
        default='user_toppop',
        choices=['random', 'toppop', 'user_toppop', 'mab00', 'mab01', 'mab10']
    )
    parser.add_argument('--toppop_stochastic', type=bool, default=True)
    parser.add_argument('--toppop_windowsize', type=int, default=28)

    # new policy
    parser.add_argument(
        '--new_policy',
        type=str,
        default='dqn',
        choices=['dqn', 'cdqn', 'random', 'user_toppop', 'toppop', 'mab', 'mab00', 'mab01', 'mab10']
    )
    parser.add_argument('--batch_exploration', action='store_true')

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
    memory = ReplayMemory(capacity=args.oldp_length*args.num_users*args.slate_size)
    user_history = UserHistory(args.num_users, args.state_window_size)
    state = user_history.get_state()
    evaluators = {
        # 'warmup': Evaluator(),
        'oldp': Evaluator(env),
        'newp': Evaluator(env),
        'test': Evaluator(env),
    }

    # # 1. Warm-Up
    # random_agent = build_agent(args, 'random')
    # for i_step in range(args.warmup_length):
    #     slates, responses, state = step(i_step, env, user_history, state, random_agent, memory, evaluators['warmup'], args)

    # 2. Run an old policy (oldp)
    oldp_agent = build_agent(args, args.old_policy)
    for i_step in range(args.oldp_length):
        slates, responses, state = step(i_step, env, user_history, state, oldp_agent, memory, evaluators['oldp'], args)
        oldp_agent.update_policy(slates, responses, memory)

    # 3. Build & pre-train a new policy (newp)
    pretrain_steps = min(len(memory)*100 // BATCH_SIZE, 2000)
    if 'dqn' in args.new_policy:
        newp_agent = build_agent(args, args.new_policy)
        logging.info(f'Pre-train a {args.new_policy.upper()} agent.')
        newp_agent.update_policy(None, None, memory, train_steps=pretrain_steps, log_interval=pretrain_steps//10)
    elif args.old_policy == args.new_policy:
        newp_agent = oldp_agent
    else:
        raise NotImplementedError

    # 4. Run exploration
    logging.info(f'New policy {args.new_policy} starts exploring.')
    finetune_steps = pretrain_steps // 10
    step_begin = args.oldp_length
    for i_step in range(step_begin, step_begin+args.expl_length):
        slates, responses, state = step(i_step, env, user_history, state, newp_agent, memory, evaluators['newp'], args)
        if i_step % 3 == 0:
            newp_agent.update_policy(slates, responses, memory, train_steps=finetune_steps, log_interval=finetune_steps)

    # 5. Evaluate
    logging.info(f'New policy {args.new_policy} is evaluated.')
    newp_agent.undo_exploration()
    step_begin = args.oldp_length+args.expl_length
    for i_step in range(step_begin, step_begin+args.test_length):
        slates, responses, state = step(i_step, env, user_history, state, newp_agent, memory, evaluators['test'], args)
        if i_step % 3 == 0:
            newp_agent.update_policy(slates, responses, memory, train_steps=len(memory)*10//BATCH_SIZE, log_interval=finetune_steps)

    def _get_save_path(args):
        pth = None
        while pth is None or os.path.isfile(pth):
            experiment_id = uuid.uuid4()
            policy_name = args.new_policy
            if args.batch_exploration:
                policy_name += '_batch'
            pth = f'{SAVE_DIR}/{args.env_config}_{policy_name}_seed{args.seed}_{experiment_id}.pkl'
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
        logging.info(f'[Step {i_step:02d}] Cum Hit Ratio: {evaluator.hit_ratio():.4f}')
        logging.info(f'[Step {i_step:02d}] Cum MRR: {np.mean(evaluator.ranking()):.4f}')

    for i_slate in range(args.slate_size):
        user_history.push(slates[:, i_slate], responses[:, i_slate])
        next_state = user_history.get_state()
        for i_user in range(args.num_users):
            memory.push(
                max(0, i_step-args.oldp_length), state[i_user], [slates[i_user, i_slate]],
                next_state[i_user], responses[i_user, i_slate]
            )
        state = next_state

    return slates, responses, state


if __name__ == '__main__':
    main()
