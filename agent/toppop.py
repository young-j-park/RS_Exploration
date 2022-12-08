
from typing import List
import logging

import numpy as np

from utils import ReplayMemory, select_random_action, add_random_action


class TopPopAgent:
    def __init__(
            self,
            num_users: int,
            num_candidates: int,
            slate_size: int,
            window_size: int,
            stochastic: bool,
            exploration_rate: float,
            local: bool = False,
    ):
        self.num_users = num_users
        self.num_candidates = num_candidates
        self.slate_size = slate_size
        self.window_size = window_size
        self.stochastic = stochastic
        self.exploration_rate = exploration_rate
        self.local = local
        self.counts = np.zeros((window_size, num_users, num_candidates))
        self.p = [self.exploration_rate, 1-self.exploration_rate]

    def update_policy(
            self,
            slates: np.ndarray,
            responses: np.ndarray,
            memory: ReplayMemory,
            **kwargs
    ):
        cur_count = np.zeros((self.num_users, self.num_candidates))
        for i_user in range(self.num_users):
            for i_slate in range(self.slate_size):
                cur_count[i_user, slates[i_user, i_slate]] += responses[i_user, i_slate]
        self.counts = np.concatenate((self.counts[1:], [cur_count]), 0)

    def select_action(self, state_arr: np.ndarray) -> np.ndarray:
        cum_counts = np.sum(self.counts, axis=0)
        if not self.local:
            cum_counts = np.sum(cum_counts, axis=0, keepdims=True)
            cum_counts = np.repeat(cum_counts, self.num_users, axis=0)

        top_recs = []
        for i_user in range(self.num_users):
            popularity = cum_counts[i_user] + 1e-10
            popularity = popularity / np.sum(popularity)
            if self.stochastic:
                rec_ids = np.random.choice(
                    np.arange(self.num_candidates),
                    self.slate_size,
                    replace=False,
                    p=popularity,
                )
            else:
                rec_ids = np.argsort(-1 * popularity)[:self.slate_size]
            top_recs.append(rec_ids)
        top_recs = np.array(top_recs)

        # Random Exploration with p
        if self.p is not None:
            recs = add_random_action(top_recs, self.num_candidates, self.slate_size, self.p)
            return recs
        else:
            return top_recs

    def undo_exploration(self):
        self.p = None
        self.stochastic = False

    def do_exploration(self):
        self.p = [self.exploration_rate, 1 - self.exploration_rate]




