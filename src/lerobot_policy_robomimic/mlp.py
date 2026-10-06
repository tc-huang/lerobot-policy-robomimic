"""The MLP between the observation encoder and the action head."""

from torch import Tensor, nn


class MLP(nn.Module):
    """Linear layers that each end in a ReLU."""

    def __init__(self, input_dim: int, layer_dims: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        for dim in layer_dims:
            layers += [nn.Linear(input_dim, dim), nn.ReLU()]
            input_dim = dim
        self.layers = nn.Sequential(*layers)
        self.output_dim = input_dim

    def forward(self, features: Tensor) -> Tensor:
        """Maps (B, input_dim) features to (B, output_dim) features."""
        return self.layers(features)
