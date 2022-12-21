
import random
from collections import namedtuple, deque

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import pairwise_distances

from config.config import PAD_IDX

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity=None):
        self.memory = deque([], maxlen=capacity)
        self.weights = deque([], maxlen=capacity)

    def push(self, i_step, *args):
        """Save a transition"""
        s, a, ns, r = args
        s = torch.as_tensor(s, dtype=torch.long).unsqueeze(0)
        a = torch.as_tensor(a, dtype=torch.long).unsqueeze(0)
        ns = torch.as_tensor(ns, dtype=torch.long).unsqueeze(0)
        r = torch.as_tensor(r, dtype=torch.float).unsqueeze(0)
        self.memory.append(Transition(s, a, ns, r))
        self.weights.append(i_step+1)

    # def sample(self, batch_size):
    #     _batch_size = min(batch_size, len(self.memory))
    #     return random.sample(self.memory, _batch_size)

    def sample(self, batch_size):
        max_step = max(self.weights)
        population = self.memory
        weights = np.exp(2*(np.array(self.weights) + np.random.randn()*1e-5 - max_step) / max_step)
        k = min(batch_size, len(self.memory))

        v = [random.random() ** (1 / w) for w in weights]
        order = sorted(range(len(population)), key=lambda i: v[i])
        return [population[i] for i in order[-k:]]

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
            if len(x) > 0:
                out[-len(x):] = x
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


def init_centers(X, K, mu0_ind=None):
    if mu0_ind is None:
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    else:
        mu0_ind = mu0_ind.copy()
        ind = mu0_ind.pop(0)
    mu = [X[ind]]
    indsAll = [ind]

    # print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    D2[i] = newD[i]
        # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)

        if len(mu0_ind) == 0:
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
        else:
            ind = mu0_ind.pop(0)
        mu.append(X[ind])
        indsAll.append(ind)
    return indsAll
