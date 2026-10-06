from collections.abc import Iterator
from typing import Any

import torch
from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from torch import Tensor, nn

from .action_heads import make_action_head
from .configuration_robomimic_bc import RobomimicBCConfig
from .mlp import MLP
from .observation_encoder import ObservationEncoder


class RobomimicBCPolicy(PreTrainedPolicy):
    """robomimic's BC policy: an MLP from encoded observations to an action head."""

    config_class = RobomimicBCConfig
    name = "robomimic_bc"

    def __init__(self, config: RobomimicBCConfig, **kwargs: Any):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.encoder = ObservationEncoder.from_config(config)
        self.mlp = MLP(self.encoder.output_dim, config.actor_layer_dims)
        self.action_head = make_action_head(config, self.mlp.output_dim)

    def reset(self) -> None:
        """BC keeps no state between steps."""

    def get_optim_params(self) -> Iterator[nn.Parameter]:
        return self.parameters()

    def features(self, batch: dict[str, Tensor]) -> Tensor:
        """Returns the MLP output that the action head reads."""
        return self.mlp(self.encoder(batch))

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Returns a (B, 1, action_dim) chunk with the action for the current observation."""
        return self.select_action(batch).unsqueeze(1)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Returns the (B, action_dim) action for the current observation."""
        return self.action_head.act(self.features(batch))

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Returns the action head's loss and its values to log."""
        return self.action_head.loss(self.features(batch), batch[ACTION])
