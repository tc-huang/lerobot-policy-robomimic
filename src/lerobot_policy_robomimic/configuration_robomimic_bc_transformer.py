from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig

from .base_config import RobomimicPolicyConfig
from .schedulers import RobomimicLinearSchedulerConfig


@PreTrainedConfig.register_subclass("robomimic_bc_transformer")
@dataclass
class RobomimicBCTransformerConfig(RobomimicPolicyConfig):
    """Configuration for robomimic's BC-Transformer policy.

    Transformer, optimizer, and schedule defaults follow robomimic's tuned template,
    `robomimic/config/default_templates/bc_transformer.json`.

    Args:
        transformer_context_length: Number of steps the transformer sees: the current one and
            the ones before it.
        transformer_embed_dim: Width of the transformer.
        transformer_num_layers: Number of transformer blocks.
        transformer_num_heads: Number of attention heads in each block.
        transformer_emb_dropout: Dropout on the input embeddings.
        transformer_attn_dropout: Dropout on the attention weights.
        transformer_block_output_dropout: Dropout on the attention and MLP outputs of each block.
        scheduler_decay_epochs: Epochs over which the learning rate falls linearly.
        scheduler_decay_factor: Fraction of the initial learning rate reached after the decay.
        scheduler_steps_per_epoch: Optimizer steps in one robomimic epoch, the period of the
            learning rate updates.
    """

    optimizer_weight_decay: float = 0.01

    transformer_context_length: int = 10
    transformer_embed_dim: int = 512
    transformer_num_layers: int = 6
    transformer_num_heads: int = 8
    transformer_emb_dropout: float = 0.1
    transformer_attn_dropout: float = 0.1
    transformer_block_output_dropout: float = 0.1

    scheduler_decay_epochs: int = 100
    scheduler_decay_factor: float = 0.1
    scheduler_steps_per_epoch: int = 100

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.transformer_embed_dim % self.transformer_num_heads != 0:
            raise ValueError(
                f"transformer_embed_dim {self.transformer_embed_dim} must be divisible by "
                f"transformer_num_heads {self.transformer_num_heads}."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> RobomimicLinearSchedulerConfig:
        return RobomimicLinearSchedulerConfig(
            decay_epochs=self.scheduler_decay_epochs,
            decay_factor=self.scheduler_decay_factor,
            steps_per_epoch=self.scheduler_steps_per_epoch,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.transformer_context_length, 1))

    @property
    def action_delta_indices(self) -> None:
        return None
