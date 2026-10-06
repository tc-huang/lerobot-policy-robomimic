from typing import Any

import torch
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_pre_post_processors

from .configuration_robomimic_bc import RobomimicBCConfig


def make_robomimic_bc_pre_post_processors(
    config: RobomimicBCConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Builds the pipelines that run before and after `RobomimicBCPolicy`.

    LeRobot's default pipelines add the batch dimension, move tensors between devices, and
    normalize; `config.normalization_mapping` leaves every feature as it is, like robomimic.
    """
    return make_default_pre_post_processors(config, dataset_stats, normalizer_device=config.device)
