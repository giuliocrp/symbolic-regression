# src/models/param_decoder.py

import torch
import torch.nn as nn

class ParamDecoderMLP(nn.Module):
    """
    Simple feed-forward network to learn u(x, mu).
    Input dimension = 2   (x and mu)
    Output dimension = 1  (u)
    """

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, 2]
        return self.model(x)  # shape: [batch_size, 1]
