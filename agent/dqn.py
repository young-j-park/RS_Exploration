
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.user import UserModel
from utils import Transition, add_random_action
from config import BATCH_SIZE, GAMMA, TARGET_UPDATE


class DQNAgent(nn.Module):

    def __init__(
            self,
            num_candidates: int,
            slate_size: int,
            emb_dim: int,
            aggregate: str,
            exploration_rate: float,
            device: "torch.device",
    ):
        super(DQNAgent, self).__init__()
        self.num_candidates = num_candidates
        self.slate_size = slate_size
        self.exploration_rate = exploration_rate
        self.p = [exploration_rate, 1 - exploration_rate]
        self.policy_net = DQN(num_candidates, emb_dim, aggregate).to(device)
        self.target_net = DQN(num_candidates, emb_dim, aggregate).to(device)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=1e-3)
        self.device = device

    def undo_exploration(self):
        self.p = None

    def do_exploration(self):
        self.p = [self.exploration_rate, 1 - self.exploration_rate]

    def update_policy(self, memory, train_epoch=1000, log_interval=100):
        self.policy_net.train()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        for i_epoch in range(1, train_epoch+1):
            transitions = memory.sample(BATCH_SIZE)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(
                tuple(map(lambda s: s is not None, batch.next_state)),
                device=self.device,
                dtype=torch.bool
            ).to(self.device)
            non_final_next_states = torch.cat(
                [s for s in batch.next_state if s is not None]
            ).to(self.device)
            state_batch = torch.cat(batch.state).to(self.device)
            action_batch = torch.cat(batch.action).to(self.device)
            reward_batch = torch.cat(batch.reward).to(self.device)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * GAMMA) + reward_batch

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            if i_epoch % log_interval == 0:
                logging.info(f'[{i_epoch:03d}/{train_epoch}] Training Loss: {loss}')

            if i_epoch % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state_arr: np.ndarray) -> torch.Tensor:
        self.policy_net.eval()
        bs = len(state_arr)
        state = torch.tensor(state_arr, dtype=torch.long, device=self.device)
        with torch.no_grad():
            q = self.policy_net(state)
            q_recs = torch.argsort(q)

        q_recs = q_recs.detach().cpu().numpy()[:, :self.slate_size]
        if self.p:
            recs = add_random_action(q_recs, self.num_candidates, self.slate_size, self.p)
            return recs
        else:
            return q_recs


class DQN(nn.Module):

    def __init__(
            self,
            num_candidates: int,
            emb_dim: int,
            aggregate: str,
    ):
        super(DQN, self).__init__()
        self.state_model = UserModel(num_candidates, emb_dim, aggregate)
        # self.bn = nn.BatchNorm1d(emb_dim)

        # # Simple linear
        self.head = nn.Sequential(
            nn.Linear(emb_dim, num_candidates)
        )

        # 2-MLP
        # self.head = nn.Sequential(
        #     nn.Linear(emb_dim, num_candidates//2),
        #     nn.ReLU(),
        #     nn.Linear(num_candidates//2, num_candidates)
        # )

        # # 3-MLP
        # self.head = nn.Sequential(
        #     nn.Linear(emb_dim, num_candidates//4),
        #     nn.ReLU(),
        #     nn.Linear(num_candidates//4, num_candidates//2),
        #     nn.ReLU(),
        #     nn.Linear(num_candidates//2, num_candidates)
        # )

    def forward(self, state):
        """
        Args:
            state: (N, 2, W)

        Returns:
            q-values: (N, A)

        """
        emb = self.state_model(state)  # self.bn()
        q_vals = self.head(emb)
        return q_vals
