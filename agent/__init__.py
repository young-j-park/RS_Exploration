
from .random import RandomAgent
from .toppop import TopPopAgent


def build_oldp_agent(args):
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
