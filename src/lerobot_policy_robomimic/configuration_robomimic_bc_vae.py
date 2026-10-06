from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamConfig


@PreTrainedConfig.register_subclass("robomimic_bc_vae")
@dataclass
class RobomimicBCVAEConfig(PreTrainedConfig):
    """Configuration for robomimic's BC-VAE policy."""

    def validate_features(self) -> None:
        pass

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig()

    def get_scheduler_preset(self) -> None:
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
