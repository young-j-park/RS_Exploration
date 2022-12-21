
import logging

from .random import RandomAgent
from .toppop import TopPopAgent
from .mab import MABAgent
from .dqn import DQNAgent
from .dqn_v1 import DQNAgentV1


def build_agent(args, policy_name: str):
    logging.info(f'Agent: {policy_name}')

    if policy_name == 'random':
        agent = RandomAgent(
            args.num_users, args.num_candidates, args.slate_size
        )
    elif 'mab' in policy_name:
        rho = int(policy_name[-2:]) / 10.0
        agent = MABAgent(
            args.num_users, args.num_candidates, args.slate_size, rho
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
    elif args.new_policy in {'dqn', 'cdqn'}:
        conservative = 'cdqn' in args.new_policy
        agent = DQNAgent(
            args.num_users, args.num_candidates, args.slate_size, args.state_emb_dim,
            args.agg_method, args.exploration_rate, args. batch_exploration,
            conservative, args.device
        )
    elif args.new_policy in {'dqn_v1', 'cdqn_v1'}:
        conservative = 'cdqn' in args.new_policy
        agent = DQNAgentV1(
            args.num_users, args.num_candidates, args.slate_size, args.state_emb_dim,
            args.agg_method, args.exploration_rate, args.batch_exploration,
            conservative, args.device
        )
    else:
        raise NotImplementedError

    return agent
