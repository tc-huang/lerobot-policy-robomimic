from typing import Any

import torch
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_pre_post_processors

from .configuration_robomimic_bc_transformer import RobomimicBCTransformerConfig


def make_robomimic_bc_transformer_pre_post_processors(
    config: RobomimicBCTransformerConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Builds the pipelines that run before and after `RobomimicBCTransformerPolicy`.

    They are the same default pipelines as BC's, which leave every feature unnormalized.
    """
    return make_default_pre_post_processors(config, dataset_stats, normalizer_device=config.device)
