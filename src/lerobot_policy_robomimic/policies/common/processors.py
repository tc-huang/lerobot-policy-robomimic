"""The pipelines that run before and after every robomimic policy."""

from typing import Any

import torch
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.factory import make_default_policy_processor_steps, make_policy_processor_pipelines

from .config import RobomimicPolicyConfig
from .language import CLIPTaskEmbeddingStep


def make_robomimic_pre_post_processors(
    config: RobomimicPolicyConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Builds LeRobot's default pipelines, with the task's CLIP embedding added for language conditioning.

    The default pipelines add the batch dimension, move tensors between devices, and normalize;
    `config.normalization_mapping` leaves every feature as it is, like robomimic. The CLIP
    embedding is computed on the CPU after batching and then moved with the observations.
    """
    steps = make_default_policy_processor_steps(config, dataset_stats, normalizer_device=config.device)
    language = (
        [
            CLIPTaskEmbeddingStep(
                model_name=config.clip_model_name, embedding_dim=config.language_embedding_dim
            )
        ]
        if config.language_conditioning is not None
        else []
    )
    return make_policy_processor_pipelines(
        input_steps=[
            steps.rename_observations,
            steps.add_batch_dim,
            *language,
            steps.to_device,
            steps.normalize,
        ],
        output_steps=[steps.unnormalize, steps.to_cpu],
    )
