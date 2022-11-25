
from typing import List
import logging

import numpy as np

from agent.random import RandomAgent


class TopPopAgent(RandomAgent):
    def __init__(
            self,
            num_users: int,
            num_candidates: int,
            slate_size: int,
            window_size: int,
            stochasticity: float,
            local: bool = False,
    ):
        super(TopPopAgent, self).__init__(num_users, num_candidates, slate_size)
        self.window_size = window_size
        self.stochasticity = stochasticity
        self.local = local
        self.counts = np.zeros((window_size, num_users, num_candidates))
        self.p = None
        self.begin_full_exploring()

    def update_policy(self, i_step: int, slates: np.ndarray, responses: np.ndarray):
        cur_count = np.zeros((self.num_users, self.num_candidates))
        for i_user in range(self.num_users):
            for i_slate in range(self.slate_size):
                cur_count[i_user, slates[i_user, i_slate]] += responses[i_user, i_slate]
        self.counts = np.concatenate((self.counts[1:], [cur_count]), 0)

        if self.p is None and i_step >= self.window_size:
            self.begin_partial_exploring()

    def select_action(
            self,
            available_item_ids: List[int],
    ) -> np.ndarray:
        if self.p is None:
            return super().select_action(available_item_ids)

        cum_counts = np.sum(self.counts, axis=0)
        if not self.local:
            cum_counts = np.sum(cum_counts, axis=0, keepdims=True)
            cum_counts = np.repeat(cum_counts, self.num_users, axis=0)

        top_recs = []
        for i_user in range(self.num_users):
            popularity = cum_counts[i_user] + 1e-10
            popularity = popularity / np.sum(popularity)
            if self.stochasticity > 0.0:
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
        random_recs = super().select_action(available_item_ids)
        a_idx = np.zeros((self.num_users, 1), dtype=int)
        recs = np.zeros_like(top_recs)
        for i_slate in range(self.slate_size):
            random_mask = np.random.choice([True, False], self.num_users, p=self.p)
            recs[random_mask, i_slate] = random_recs[random_mask, i_slate]
            recs[~random_mask, i_slate] = np.take_along_axis(top_recs, a_idx.astype(int), axis=1)[~random_mask].squeeze()
            a_idx[~random_mask] += 1

        return np.array(recs)

    def begin_full_exploring(self):
        logging.info('Begin with random exploring.')
        self.p = None

    def begin_partial_exploring(self):
        logging.info(f'Start partial exploring: p={self.stochasticity}.')
        self.p = [self.stochasticity, 1-self.stochasticity]



