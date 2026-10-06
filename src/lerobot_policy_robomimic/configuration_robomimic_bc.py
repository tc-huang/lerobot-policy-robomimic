from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from .base_config import RobomimicPolicyConfig


@PreTrainedConfig.register_subclass("robomimic_bc")
@dataclass
class RobomimicBCConfig(RobomimicPolicyConfig):
    """Configuration for robomimic's BC policy.

    Args:
        actor_layer_dims: Hidden sizes of the MLP between the observation encoder and the action head.
    """

    actor_layer_dims: tuple[int, ...] = (1024, 1024)

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None
