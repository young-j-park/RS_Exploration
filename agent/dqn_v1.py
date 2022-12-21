
import logging
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import autograd_hacks
from utils import Transition, ReplayMemory, add_random_action, init_centers
from agent.user import UserModel
from config.config import BATCH_SIZE, GAMMA, TARGET_UPDATE


class DQNAgentV1(nn.Module):

    def __init__(
            self,
            num_users: int,
            num_candidates: int,
            slate_size: int,
            emb_dim: int,
            aggregate: str,
            exploration_rate: float,
            batch_exploration: bool,
            conservative: bool,
            device: "torch.device",
    ):
        super(DQNAgentV1, self).__init__()
        self.num_users = num_users
        self.num_candidates = num_candidates
        self.slate_size = slate_size
        self.exploration_rate = exploration_rate
        self.batch_exploration = batch_exploration
        self.p = [exploration_rate, 1 - exploration_rate]
        self.policy_net = DQN(num_candidates, emb_dim, aggregate).to(device)
        self.target_net = DQN(num_candidates, emb_dim, aggregate).to(device)
        self.conservative = conservative
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=1e-3)
        self.device = device

        if self.batch_exploration:
            self.policy_net_grad = DQN(num_candidates, emb_dim, aggregate).to(device)
            autograd_hacks.add_hooks(self.policy_net_grad.head)

    def undo_exploration(self):
        self.p = None

    def do_exploration(self):
        self.p = [self.exploration_rate, 1 - self.exploration_rate]

    def update_policy(
            self,
            slates: np.ndarray or None,
            responses: np.ndarray or None,
            memory: ReplayMemory,
            train_steps: int = 1000,
            log_interval: int = 100,
    ):
        self.policy_net.train()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        for i_step in range(1, train_steps + 1):
            transitions = memory.sample(BATCH_SIZE)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            # non_final_mask = torch.tensor(
            #     tuple(map(lambda s: s is not None, batch.next_state)),
            #     device=self.device,
            #     dtype=torch.bool
            # ).to(self.device)
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
            # next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
            next_state_values = self.target_net(non_final_next_states).max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * GAMMA) + reward_batch

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Add CQN loss
            if self.conservative:
                dataset_expec = torch.mean(state_action_values)
                negative_sampling = torch.mean(torch.logsumexp(q_values, 1))
                min_q_loss = (negative_sampling - dataset_expec)
                loss += min_q_loss

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # for param in self.policy_net.parameters():
            #     param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            if i_step % log_interval == 0:
                logging.info(f'[{i_step:03d}/{train_steps}] Training Loss: {loss}')

            if i_step % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.target_net.eval()

    def select_action(self, state_arr: np.ndarray) -> np.ndarray:
        self.policy_net.eval()
        state = torch.tensor(state_arr, dtype=torch.long, device=self.device)
        with torch.no_grad():
            q = self.policy_net(state)
            q_recs = torch.argsort(q, descending=True)

        q_recs = q_recs.detach().cpu().numpy()[:, :self.slate_size]
        logging.debug(q[0][q_recs[0]])
        if self.p:
            if self.batch_exploration:
                return self.explore_batch(q_recs, state)
            else:
                return add_random_action(q_recs, self.num_candidates, self.slate_size, self.p)
        else:
            return q_recs

    def explore_batch(self, q_recs, state_batch):
        # Hypothetical positives
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        with torch.no_grad():
            state_emb, num_users = self.target_net.emb_sa(state_batch)
            next_state_values = self.target_net.head(state_emb)
        expected_state_action_values = (next_state_values * GAMMA) + 1.0

        # Forward
        self.policy_net_grad.load_state_dict(self.policy_net.state_dict())
        self.policy_net_grad.eval()
        autograd_hacks.clear_backprops(self.policy_net_grad.head)
        q = self.policy_net_grad.head(state_emb)

        # Backward
        for param in self.policy_net_grad.head.parameters():
            if param.grad is not None:
                param.grad.zero_()
            if hasattr(param, 'grad1') and param.grad1 is not None:
                param.grad1.zero_()
        criterion = nn.SmoothL1Loss()
        loss = criterion(q, expected_state_action_values)
        loss.backward()
        autograd_hacks.compute_grad1(self.policy_net_grad.head)

        # Gradient Embedding
        gradient_emb = []
        for param in self.policy_net_grad.head.parameters():
            grad1 = param.grad1
            if len(grad1.shape) == 2:
                grad1 = grad1.unsqueeze(2)
            gradient_emb.append(grad1)
        gradient_emb = torch.cat(gradient_emb, -1).squeeze(1).detach().cpu().numpy()
        # gradient_emb = gradient_emb.reshape((self.num_users*self.num_candidates, -1))

        # Sort
        q_rec_indices = [
            i_user*self.num_candidates + i_doc
            for i_user, slate in enumerate(q_recs) for i_doc in slate
        ]
        num_explores = np.sum(np.random.rand(q_recs.size) < self.exploration_rate)
        if num_explores == 0:
            return q_recs
        else:
            new_q_recs = [deque(recs, maxlen=self.slate_size) for recs in q_recs]
            explore_indices = init_centers(gradient_emb, len(q_rec_indices) + num_explores, q_rec_indices)
            explore_indices = explore_indices[len(q_rec_indices):]
            for explore_ind in explore_indices:
                i_user, i_doc = divmod(explore_ind, self.num_candidates)
                if i_doc not in new_q_recs[i_user]:
                    new_q_recs[i_user].append(i_doc)
            return np.array(new_q_recs)


class DQN(nn.Module):

    def __init__(
            self,
            num_candidates: int,
            emb_dim: int,
            aggregate: str,
    ):
        super(DQN, self).__init__()
        self.num_candidates = num_candidates
        self.emb_dim = emb_dim

        self.state_model = UserModel(num_candidates, emb_dim, aggregate)
        # self.bn = nn.BatchNorm1d(emb_dim)

        self.item_emb = nn.Parameter(torch.empty(num_candidates, emb_dim), requires_grad=True)
        nn.init.normal_(self.item_emb)

        # # Simple linear
        # self.head = nn.Linear(emb_dim*2, 1)

        # 2-MLP
        # self.head = nn.Sequential(
        #     nn.Linear(emb_dim*2, emb_dim),
        #     nn.ReLU(),
        #     nn.Linear(emb_dim, 1)
        # )

        # # 3-MLP
        self.head = nn.Sequential(
            nn.Linear(emb_dim*2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim//2),
            nn.ReLU(),
            nn.Linear(emb_dim//2, 1)
        )

    def forward(self, state):
        """
        Args:
            state: (N, 2, W)

        Returns:
            q-values: (N, A)

        """
        embs, num_users = self.emb_sa(state)
        q_vals = self.head(embs).view(num_users, self.num_candidates)
        return q_vals

    def emb_sa(self, state):
        num_users = len(state)
        user_emb = self.state_model(state).unsqueeze(1).expand(-1, self.num_candidates, -1)
        item_emb = self.item_emb.unsqueeze(0).expand(num_users, -1, -1)
        embs = torch.cat([user_emb, item_emb], -1).view(-1, self.emb_dim * 2)
        return embs, num_users

