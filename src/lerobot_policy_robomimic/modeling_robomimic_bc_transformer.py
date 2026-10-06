from collections import deque
from collections.abc import Iterator
from typing import Any

import torch
from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from torch import Tensor, nn

from .action_heads import make_action_head
from .configuration_robomimic_bc_transformer import RobomimicBCTransformerConfig
from .observation_encoder import ObservationEncoder
from .transformer import Transformer


class RobomimicBCTransformerPolicy(PreTrainedPolicy):
    """robomimic's BC-Transformer policy: a causal transformer over recent observations, acting on the last one."""

    config_class = RobomimicBCTransformerConfig
    name = "robomimic_bc_transformer"

    def __init__(self, config: RobomimicBCTransformerConfig, **kwargs: Any):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.encoder = ObservationEncoder.from_config(config, camera_activation=False)
        self.transformer = Transformer(
            self.encoder.output_dim,
            config.transformer_context_length,
            config.transformer_embed_dim,
            config.transformer_num_layers,
            config.transformer_num_heads,
            config.transformer_emb_dropout,
            config.transformer_attn_dropout,
            config.transformer_block_output_dropout,
        )
        self.action_head = make_action_head(config, config.transformer_embed_dim)
        self.reset()

    def reset(self) -> None:
        """Forgets the observations of the previous episode."""
        self._observations: deque[dict[str, Tensor]] = deque(maxlen=self.config.transformer_context_length)

    def get_optim_params(self) -> Iterator[nn.Parameter]:
        return self.parameters()

    def features(self, batch: dict[str, Tensor]) -> Tensor:
        """Returns the transformer output at the last of (B, context_length, ...) observations."""
        return self.transformer(self.encoder(batch))[:, -1]

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Not supported: actions come from `select_action`, which keeps the observation history."""
        raise NotImplementedError("robomimic_bc_transformer predicts one action per step with select_action.")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Returns the (B, action_dim) action for the latest observation.

        The first observation of an episode fills the whole context, as robomimic pads it.
        """
        step = {key: batch[key] for key in self.encoder.keys}
        if not self._observations:
            self._observations.extend([step] * self.config.transformer_context_length)
        else:
            self._observations.append(step)
        sequence = {key: torch.stack([obs[key] for obs in self._observations], dim=1) for key in step}
        return self.action_head.act(self.features(sequence))

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Returns the action head's loss for the action at the last step of each context."""
        return self.action_head.loss(self.features(batch), batch[ACTION])
