from typing import Any

import torch
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .configuration_robomimic_bc_vae import RobomimicBCVAEConfig


def make_robomimic_bc_vae_pre_post_processors(
    config: RobomimicBCVAEConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Builds the pipelines that run before and after `RobomimicBCVAEPolicy`."""
    raise NotImplementedError
