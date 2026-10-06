from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from .base_config import RobomimicActorConfig


@PreTrainedConfig.register_subclass("robomimic_bc_rnn")
@dataclass
class RobomimicBCRNNConfig(RobomimicActorConfig):
    """Configuration for robomimic's BC-RNN policy.

    For robomimic's low-dim experiments, also set `rnn_hidden_dim=400`.

    Args:
        actor_layer_dims: Hidden sizes of the MLP between the LSTM and the action head.
        rnn_hidden_dim: Hidden size of each LSTM layer.
        rnn_num_layers: Number of stacked LSTM layers.
        rnn_horizon: Length of the training sequences, and the number of steps after which
            the LSTM state is reset at inference.
    """

    actor_layer_dims: tuple[int, ...] = ()
    rnn_hidden_dim: int = 1000
    rnn_num_layers: int = 2
    rnn_horizon: int = 10

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.use_gaussian:
            raise ValueError("robomimic has no Gaussian BC-RNN; use the GMM or deterministic head.")

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(self.rnn_horizon))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.rnn_horizon))
