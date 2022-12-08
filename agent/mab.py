
import numpy as np

from utils import ReplayMemory


class MABAgent:
    def __init__(
            self,
            num_users: int,
            num_candidates: int,
            slate_size: int,
            rho: float = 1.0,
    ):
        self.num_users = num_users
        self.num_candidates = num_candidates
        self.slate_size = slate_size
        self.rho = rho
        self.round = 1
        self.clicks = np.zeros((num_users, num_candidates))
        self.rewards = np.zeros((num_users, num_candidates))
        self.avg_rewards = np.zeros(num_candidates)
        self.q = np.full(num_candidates, np.inf)
        
    def update_policy(
            self,
            slates: np.ndarray,
            responses: np.ndarray,
            memory: ReplayMemory,
            **kwargs
    ):
        for i_user in range(self.num_users):
            for i_slate in range(self.slate_size):
                self.rewards[i_user, slates[i_user, i_slate]] += responses[i_user, i_slate]
                self.clicks[i_user, slates[i_user, i_slate]] += 1
        
        self.avg_rewards = np.where(self.clicks != 0, self.rewards / self.clicks, self.clicks)
        
    def select_action(self, state_arr: np.ndarray) -> np.ndarray:
        self.q = np.where(self.clicks != 0, self.avg_rewards + np.sqrt(self.rho * np.log10(self.round) / self.clicks), self.q)
        recs = []
        for i_user in range(self.num_users):
            rec_ids = np.argsort(-1 * self.q[i_user])[:self.slate_size]
            recs.append(rec_ids)
        recs = np.array(recs)
        self.round += self.slate_size
        return recs
        
    def undo_exploration(self):
        return None

    def do_exploration(self):
        return None




