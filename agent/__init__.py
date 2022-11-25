
import logging

from .random import RandomAgent
from .toppop import TopPopAgent
from .dqn import DQNAgent


def build_oldp_agent(args):
    logging.info(f'Old Agent: {args.old_policy.upper()}')

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

    return oldp_agent


def build_newp_agent(args):
    logging.info(f'New Agent: {args.new_policy.upper()}')

    if args.new_policy == 'dqn':
        newp_agent = DQNAgent(
            args.num_candidates, args.state_emb_dim, args.agg_method,
            args.exploration_rate, args.device
        )
    else:
        raise NotImplementedError

    return newp_agent
