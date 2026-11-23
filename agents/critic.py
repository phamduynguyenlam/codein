from torch import nn


class InitCritic(nn.Module):
    """Value baseline for the state before the seed grades are chosen."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, archive_context):
        return self.value_head(archive_context).squeeze(-1)
