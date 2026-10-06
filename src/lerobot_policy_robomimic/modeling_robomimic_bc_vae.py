from collections.abc import Iterator
from typing import Any

import torch
from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from torch import Tensor, nn

from .configuration_robomimic_bc_vae import RobomimicBCVAEConfig
from .vae import ActionVAE


class RobomimicBCVAEPolicy(PreTrainedPolicy):
    """robomimic's BC-VAE policy: a conditional VAE that decodes an action from a prior latent."""

    config_class = RobomimicBCVAEConfig
    name = "robomimic_bc_vae"

    def __init__(self, config: RobomimicBCVAEConfig, **kwargs: Any):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.vae = ActionVAE.from_config(config)

    def reset(self) -> None:
        """BC-VAE keeps no state between steps."""

    def get_optim_params(self) -> Iterator[nn.Parameter]:
        return self.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Returns a (B, 1, action_dim) chunk with the action for the current observation."""
        return self.select_action(batch).unsqueeze(1)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Returns the (B, action_dim) action decoded from a latent drawn from the prior."""
        batch_size = batch[next(iter(self.config.observation_features))].shape[0]
        return self.vae.sample(batch, batch_size)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Returns the reconstruction loss plus `vae_kl_weight` times the KL loss, and the values to log."""
        losses = self.vae.losses(batch, batch[ACTION])
        loss = losses["reconstruction_loss"] + self.config.vae_kl_weight * losses["kl_loss"]
        return loss, {name: value.item() for name, value in losses.items()}
