from typing import Any

from lerobot.policies import PreTrainedPolicy
from torch import Tensor

from .configuration_robomimic_bc import RobomimicBCConfig


class RobomimicBCPolicy(PreTrainedPolicy):
    """robomimic's BC policy."""

    config_class = RobomimicBCConfig
    name = "robomimic_bc"

    def __init__(self, config: RobomimicBCConfig, dataset_stats: dict[str, Any] | None = None):
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config

    def reset(self) -> None:
        raise NotImplementedError

    def get_optim_params(self) -> dict:
        raise NotImplementedError

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise NotImplementedError

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise NotImplementedError

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError
