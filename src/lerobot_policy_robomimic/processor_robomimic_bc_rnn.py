from typing import Any

import torch
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .configuration_robomimic_bc_rnn import RobomimicBCRNNConfig
from .processors import make_robomimic_pre_post_processors


def make_robomimic_bc_rnn_pre_post_processors(
    config: RobomimicBCRNNConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Builds the pipelines that run before and after `RobomimicBCRNNPolicy`, shared by every robomimic policy."""
    return make_robomimic_pre_post_processors(config, dataset_stats)
