from typing import Any

import torch
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from ..common.processors import make_robomimic_pre_post_processors
from .configuration_robomimic_bc_transformer import RobomimicBCTransformerConfig


def make_robomimic_bc_transformer_pre_post_processors(
    config: RobomimicBCTransformerConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Builds the pipelines that run before and after `RobomimicBCTransformerPolicy`, shared by every robomimic policy."""
    return make_robomimic_pre_post_processors(config, dataset_stats)
