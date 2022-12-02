
import logging

import torch
import torch.nn as nn
import numpy as np

from agent import DQNAgent
from utils import Transition, ReplayMemory
from config import BATCH_SIZE, GAMMA, TARGET_UPDATE


class CDQNAgent(DQNAgent):

    def update_policy(
            self,
            slates: np.ndarray,
            responses: np.ndarray,
            memory: ReplayMemory,
            train_epoch: int = 1000,
            log_interval: int = 100,
    ):
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
            q_values = self.policy_net(state_batch)
            state_action_values = q_values.gather(1, action_batch)

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

            # Add CQN loss
            dataset_expec = torch.mean(state_action_values)
            negative_sampling = torch.mean(torch.logsumexp(q_values, 1))
            min_q_loss = (negative_sampling - dataset_expec)
            loss += min_q_loss

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
