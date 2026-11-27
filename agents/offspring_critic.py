import torch
from torch import nn


class OffspringCritic(nn.Module):
    """Remaining discounted return within the current offspring cycle."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(3 * hidden_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, archive, parents, selected, progress, cycle_step, cycle_length):
        batch_size = archive.size(0)
        progress = torch.as_tensor(progress, device=archive.device, dtype=archive.dtype).reshape(batch_size, 1)
        step = torch.as_tensor(cycle_step, device=archive.device, dtype=archive.dtype).reshape(batch_size, 1)
        length = torch.as_tensor(cycle_length, device=archive.device, dtype=archive.dtype).reshape(batch_size, 1)
        if (length <= 0).any() or (step < 0).any() or (step >= length).any():
            raise ValueError("Cycle states require 0 <= cycle_step < cycle_length.")
        features = torch.cat((archive, parents, selected, progress, step / length, (length - step) / length), dim=-1)
        return self.value_head(features).squeeze(-1)
