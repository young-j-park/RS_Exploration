
import torch
import torch.nn as nn

from config import PAD_IDX


class UserModel(nn.Module):
    def __init__(
            self,
            num_candidates: int,
            emb_dim: int,
            aggregate: str,
    ):
        super(UserModel, self).__init__()
        self.num_candidates = num_candidates
        self.item_pos_emb = nn.Embedding(num_candidates+1, emb_dim, PAD_IDX)
        self.item_neg_emb = nn.Embedding(num_candidates+1, emb_dim, PAD_IDX)
        if aggregate == 'mean':
            self.aggregate_fn = lambda x: torch.mean(x, dim=1)
        elif aggregate == 'gru':
            self.aggregate_fn = nn.GRU(
                input_size=emb_dim,
                hidden_size=emb_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.1,
                bidirectional=False
            )

    def forward(self, state):
        """
        Args:
            state: (N, 2, W)

        Returns:
            emb: (N, D)

        """
        state += 1
        emb_pos = self.item_pos_emb(state[:, 0, :])
        emb_neg = self.item_neg_emb(state[:, 0, :])
        emb = emb_pos + emb_neg
        emb = self.aggregate_fn(emb)
        return emb
