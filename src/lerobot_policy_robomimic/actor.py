"""Networks that map encoded observations to actions."""

import torch
from torch import Tensor, nn

from .observation_encoder import ObservationEncoder


class MLPActor(nn.Module):
    """An MLP on top of the observation encoder, with a linear action head bounded by tanh."""

    def __init__(self, encoder: ObservationEncoder, layer_dims: tuple[int, ...], action_dim: int):
        super().__init__()
        self.encoder = encoder
        layers = []
        input_dim = encoder.output_dim
        for dim in layer_dims:
            layers += [nn.Linear(input_dim, dim), nn.ReLU()]
            input_dim = dim
        self.mlp = nn.Sequential(*layers)
        self.action_head = nn.Linear(input_dim, action_dim)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Maps a batch of observations to (B, action_dim) actions in [-1, 1]."""
        return torch.tanh(self.action_head(self.mlp(self.encoder(batch))))
