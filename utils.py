
import random
from collections import namedtuple, deque

import numpy as np
import torch

from config import PAD_IDX

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity=None):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        s, a, ns, r = args
        s = torch.as_tensor(s, dtype=torch.long).unsqueeze(0)
        a = torch.as_tensor(a, dtype=torch.long).unsqueeze(0)
        ns = torch.as_tensor(ns, dtype=torch.long).unsqueeze(0)
        r = torch.as_tensor(r, dtype=torch.float).unsqueeze(0)
        self.memory.append(Transition(s, a, ns, r))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class UserHistory:

    def __init__(self, num_users, window_size):
        self.num_users = num_users
        self.window_size = window_size
        self.pos_reactions = [deque([], maxlen=window_size) for _ in range(num_users)]
        self.neg_reactions = [deque([], maxlen=window_size) for _ in range(num_users)]

    def push(self, items, responses):
        assert len(items) == len(responses) == self.num_users
        for i_user in range(self.num_users):
            if responses[i_user]:
                self.pos_reactions[i_user].append(items[i_user])
            else:
                self.neg_reactions[i_user].append(items[i_user])

    def get_state(self):
        return np.stack(
            ([self._pad(x) for x in self.pos_reactions], [self._pad(x) for x in self.neg_reactions]), axis=1
        )

    def _pad(self, x):
        if len(x) >= self.window_size:
            return np.array(x)
        else:
            out = np.full(self.window_size, PAD_IDX)
            out[:len(x)] = x
            return out


def select_random_action(num_users, num_candidates, slate_size):
    recs = [
        np.random.choice(
            np.arange(num_candidates),
            slate_size,
            replace=False
        )
        for _ in range(num_users)
    ]
    return np.array(recs)


def add_random_action(original_recs, num_candidates, slate_size, p):
    num_users = len(original_recs)
    random_recs = select_random_action(num_users, num_candidates, slate_size)
    a_idx = np.zeros((num_users, 1), dtype=int)
    recs = np.zeros_like(original_recs, dtype=int)
    for i_slate in range(slate_size):
        random_mask = np.random.choice([True, False], num_users, p=p)
        recs[random_mask, i_slate] = random_recs[random_mask, i_slate]
        recs[~random_mask, i_slate] = np.take_along_axis(original_recs, a_idx.astype(int), axis=1)[~random_mask].squeeze()
        a_idx[~random_mask] += 1
    return recs
