
import random
from collections import namedtuple, deque

import numpy as np

from config import PAD_IDX

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity=None):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

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
        # self.pos_reactions = np.full((num_users, window_size), PAD_IDX)
        # self.neg_reactions = np.full((num_users, window_size), PAD_IDX)

    def push(self, items, responses):
        assert len(items) == len(responses) == self.num_users
        # cur_state = (np.copy(self.pos_reactions), np.copy(self.neg_reactions))
        for i_user in range(self.num_users):
            if responses[i_user]:
                self.pos_reactions[i_user].append(items[i_user])
                # self.pos_reactions[i_user] = np.concatenate((self.pos_reactions[i_user][1:], [items[i_user]]))
            else:
                self.neg_reactions[i_user].append(items[i_user])
                # self.neg_reactions[i_user] = np.concatenate((self.neg_reactions[i_user][1:], [items[i_user]]))
        # next_state = (np.copy(self.pos_reactions), np.copy(self.neg_reactions))
        # return cur_state, next_state

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
