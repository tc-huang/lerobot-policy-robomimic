from collections.abc import Iterator
from typing import Any

import torch
from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from torch import Tensor, nn

from .action_heads import make_action_head
from .configuration_robomimic_bc_rnn import RobomimicBCRNNConfig
from .mlp import MLP
from .observation_encoder import ObservationEncoder

LSTMState = tuple[Tensor, Tensor]


class RobomimicBCRNNPolicy(PreTrainedPolicy):
    """robomimic's BC-RNN policy: an LSTM over encoded observations, then an MLP and an action head per step."""

    config_class = RobomimicBCRNNConfig
    name = "robomimic_bc_rnn"

    def __init__(self, config: RobomimicBCRNNConfig, **kwargs: Any):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.encoder = ObservationEncoder.from_config(config)
        self.lstm = nn.LSTM(
            self.encoder.output_dim, config.rnn_hidden_dim, config.rnn_num_layers, batch_first=True
        )
        self.mlp = MLP(config.rnn_hidden_dim, config.actor_layer_dims)
        self.action_head = make_action_head(config, self.mlp.output_dim)
        self.reset()

    def reset(self) -> None:
        """Clears the LSTM state at the start of an episode."""
        self._lstm_state: LSTMState | None = None
        self._steps = 0

    def get_optim_params(self) -> Iterator[nn.Parameter]:
        return self.parameters()

    def features(self, batch: dict[str, Tensor], state: LSTMState | None = None) -> tuple[Tensor, LSTMState]:
        """Returns the (B, T, D) MLP outputs for (B, T, ...) observations, and the final LSTM state."""
        outputs, state = self.lstm(self.encoder(batch), state)
        return self.mlp(outputs), state

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Not supported: every action advances the LSTM state, so actions come from `select_action`."""
        raise NotImplementedError("robomimic_bc_rnn predicts one action per step with select_action.")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Returns the (B, action_dim) action, resetting the LSTM state every `rnn_horizon` steps."""
        if self._steps % self.config.rnn_horizon == 0:
            self._lstm_state = None
        self._steps += 1
        step = {key: batch[key].unsqueeze(1) for key in self.encoder.keys}
        features, self._lstm_state = self.features(step, self._lstm_state)
        return self.action_head.act(features[:, 0])

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Returns the action head's loss over every step of the sequences."""
        features, _ = self.features(batch)
        return self.action_head.loss(features, batch[ACTION])
