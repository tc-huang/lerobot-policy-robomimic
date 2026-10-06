from collections.abc import Iterator
from typing import Any

import torch
from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from torch import Tensor, nn
from torch.nn.functional import cosine_similarity, mse_loss, smooth_l1_loss

from .actor import MLPActor
from .configuration_robomimic_bc import RobomimicBCConfig
from .observation_encoder import ObservationEncoder


class RobomimicBCPolicy(PreTrainedPolicy):
    """robomimic's BC policy: an MLP actor that regresses the demonstrated action."""

    config_class = RobomimicBCConfig
    name = "robomimic_bc"

    def __init__(self, config: RobomimicBCConfig, **kwargs: Any):
        super().__init__(config)
        config.validate_features()
        self.config = config
        encoder = ObservationEncoder(
            config.observation_features,
            config.crop_shape,
            config.spatial_softmax_num_kp,
            config.image_feature_dim,
        )
        self.actor = MLPActor(encoder, config.actor_layer_dims, config.action_feature.shape[0])

    def reset(self) -> None:
        """BC keeps no state between steps."""

    def get_optim_params(self) -> Iterator[nn.Parameter]:
        return self.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Returns a (B, 1, action_dim) chunk with the action for the current observation."""
        return self.actor(batch).unsqueeze(1)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Returns the (B, action_dim) action for the current observation."""
        return self.actor(batch)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Returns the weighted action loss and each loss term for logging."""
        losses = self.action_losses(self.actor(batch), batch[ACTION])
        loss = (
            self.config.l2_weight * losses["l2_loss"]
            + self.config.l1_weight * losses["l1_loss"]
            + self.config.cos_weight * losses["cos_loss"]
        )
        return loss, {name: value.item() for name, value in losses.items()}

    @staticmethod
    def action_losses(predicted: Tensor, target: Tensor) -> dict[str, Tensor]:
        """Returns the mean squared, smooth L1, and translation cosine losses between actions."""
        return {
            "l2_loss": mse_loss(predicted, target),
            "l1_loss": smooth_l1_loss(predicted, target),
            "cos_loss": (1 - cosine_similarity(predicted[..., :3], target[..., :3], dim=-1)).mean(),
        }
