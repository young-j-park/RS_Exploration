
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.user import UserModel
from utils import Transition
from config import BATCH_SIZE, GAMMA


class DQNAgent(nn.Module):

    def __init__(
            self,
            num_candidates: int,
            emb_dim: int,
            aggregate: str,
            exploration_rate: float,
            device: "torch.device",
    ):
        super(DQNAgent, self).__init__()
        self.num_candidates = num_candidates
        self.exploration_rate = exploration_rate
        self.policy_net = DQN(num_candidates, emb_dim, aggregate).to(device)
        self.target_net = DQN(num_candidates, emb_dim, aggregate).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.device = device

    def update_policy(self, memory, train_epoch=1000):
        for i_epoch in range(train_epoch+1):
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
            )
            non_final_next_states = torch.cat(
                [s for s in batch.next_state if s is not None]
            )
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

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

            if i_epoch % 100 == 0:
                logging.info(f'[{i_epoch:03d}/{train_epoch}] Training Loss: {loss}')


    def select_action(self, state):
        pass

    def select_one_action(self, state_arr: np.ndarray) -> torch.Tensor:
        state = torch.tensor(state_arr, dtype=torch.long, device=self.device)
        sample = random.random()
        if sample > self.exploration_rate:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self.num_candidates)]],
                device=self.device,
                dtype=torch.long
            )


class DQN(nn.Module):

    def __init__(
            self,
            num_candidates: int,
            emb_dim: int,
            aggregate: str,
    ):
        super(DQN, self).__init__()
        self.state_model = UserModel(num_candidates, emb_dim, aggregate)
        self.bn = nn.BatchNorm1d(emb_dim)

        # # Simple linear
        # self.head = nn.Sequential(
        #     nn.Linear(emb_dim, num_candidates)
        # )

        # 2-MLP
        self.head = nn.Sequential(
            nn.Linear(emb_dim, num_candidates//2),
            nn.ReLU(),
            nn.Linear(num_candidates//2, num_candidates)
        )

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
        emb = self.bn(self.state_model(state))
        q_vals = self.head(emb)
        return q_vals
