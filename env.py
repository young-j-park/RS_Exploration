
import logging

import numpy as np

from recsim.environments.interest_evolution import create_multiuser_environment as iv_create_environment
from recsim.environments.interest_exploration import create_multiuser_environment as ie_create_environment
from recsim.simulator import environment


class MyEnv:
    def __init__(
            self,
            env: environment.MultiUserEnvironment
    ):
        super().__init__()
        self.env = env
        self.num_users = env.num_users
        self.slate_size = env.slate_size
        self.num_candidates = env.num_candidates

    def step(self, action: np.ndarray):
        """
        Args:
            action: (num_users, slate_size)
        Returns:
            response: (num_users, )
            done: bool
        """
        _, next_documents, responses, done = self.env.step(slates=action)
        response = np.array([[int(r.clicked) for r in res] for res in responses])
        return next_documents, response, done

    def get_num_candidates(self):
        return self.num_candidates

    def reset(self):
        _, docs = self.env.reset()
        return docs


def make_env(
        env_type: str,
        num_users: int,
        num_candidates: int,
        slate_size: int,
        **kwargs
):
    env_config = {
        'seed': 1234,
        'resample_documents': False,
        'num_users': num_users,
        'slate_size': slate_size,
        'num_candidates': num_candidates,
    }

    if env_type == 'interest_evolution':
        env = iv_create_environment(env_config)
    elif env_type == 'interest_exploration':
        env = ie_create_environment(env_config)
    else:
        raise ValueError(f'{env_type} is not supported.')

    env = MyEnv(env)
    env.reset()
    logging.info(f'Environment -- {env_type} -- has been built.')
    return env
