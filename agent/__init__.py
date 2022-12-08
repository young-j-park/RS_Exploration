
import logging

from .random import RandomAgent
from .toppop import TopPopAgent
from .mab import MABAgent
from .dqn import DQNAgent
from .cdqn import CDQNAgent


def build_agent(args, policy_name: str):
    logging.info(f'Agent: {policy_name}')

    if policy_name == 'random':
        agent = RandomAgent(
            args.num_users, args.num_candidates, args.slate_size
        )
    elif policy_name == 'mab':
        agent = MABAgent(
            args.num_users, args.num_candidates, args.slate_size
        )
    elif policy_name == 'toppop':
        agent = TopPopAgent(
            args.num_users, args.num_candidates, args.slate_size, args.toppop_windowsize,
            args.toppop_stochastic, args.exploration_rate, local=False
        )
    elif policy_name == 'user_toppop':
        agent = TopPopAgent(
            args.num_users, args.num_candidates, args.slate_size, args.toppop_windowsize,
            args.toppop_stochastic, args.exploration_rate, local=True
        )
    elif args.new_policy == 'dqn':
        agent = DQNAgent(
            args.num_candidates, args.slate_size, args.state_emb_dim,
            args.agg_method, args.exploration_rate, args.device
        )
    elif args.new_policy == 'cdqn':
        agent = CDQNAgent(
            args.num_candidates, args.slate_size, args.state_emb_dim,
            args.agg_method, args.exploration_rate, args.device
        )
    else:
        raise NotImplementedError

    return agent
