
from typing import List

import numpy as np

from utils import ReplayMemory, select_random_action


class RandomAgent:
    def __init__(
            self,
            num_users: int,
            num_candidates: int,
            slate_size: int,
    ):
        self.num_users = num_users
        self.num_candidates = num_candidates
        self.slate_size = slate_size

    def update_policy(
            self,
            slates: np.ndarray,
            responses: np.ndarray,
            memory: ReplayMemory,
            **kwargs
    ):
        return

    def select_action(self, state_arr: np.ndarray) -> np.ndarray:
        return select_random_action(self.num_users, self.num_candidates, self.slate_size)

    def undo_exploration(self):
        return

    def do_exploration(self):
        return
