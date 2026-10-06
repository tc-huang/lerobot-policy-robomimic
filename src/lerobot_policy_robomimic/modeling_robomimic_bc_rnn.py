from typing import Any

from lerobot.policies import PreTrainedPolicy
from torch import Tensor

from .configuration_robomimic_bc_rnn import RobomimicBCRNNConfig


class RobomimicBCRNNPolicy(PreTrainedPolicy):
    """robomimic's BC-RNN policy."""

    config_class = RobomimicBCRNNConfig
    name = "robomimic_bc_rnn"

    def __init__(self, config: RobomimicBCRNNConfig, **kwargs: Any):
        super().__init__(config)
        config.validate_features()
        self.config = config

    def reset(self) -> None:
        raise NotImplementedError

    def get_optim_params(self):
        raise NotImplementedError

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise NotImplementedError

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise NotImplementedError

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError
