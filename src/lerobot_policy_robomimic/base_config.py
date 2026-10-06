"""Settings shared by every robomimic policy."""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamConfig
from lerobot.optim.schedulers import LRSchedulerConfig


@dataclass
class RobomimicPolicyConfig(PreTrainedConfig):
    """Observation, encoder, action head, and optimizer settings of robomimic's policies.

    Defaults follow robomimic's image experiments; set `use_env_state` for its low-dim experiments.

    Args:
        use_env_state: Whether the policy reads `observation.environment_state`, the
            privileged object state that only robomimic's low-dim experiments use.
        crop_shape: (height, width) of the random crop taken from each camera during training,
            and of the center crop at inference; None disables cropping.
        random_crop_at_inference: Whether cameras are also cropped at random positions outside
            training, as in robomimic v0.1, whose checkpoints were evaluated that way.
        spatial_softmax_num_kp: Number of keypoints each camera's spatial softmax extracts.
        image_feature_dim: Size of the feature each camera is projected to.
        use_gaussian: Whether the policy outputs one diagonal Gaussian over actions instead of
            one action; it takes precedence over `use_gmm`, as in robomimic.
        gaussian_fixed_std: Whether the Gaussian's standard deviation stays at `gaussian_init_std`
            instead of being learned.
        gaussian_init_std: The Gaussian's standard deviation when the network outputs zero.
        gaussian_min_std: Lower bound of the Gaussian's standard deviation in training.
        gaussian_low_noise_eval: Whether the Gaussian's mean is the action outside training.
        use_gmm: Whether the policy outputs a Gaussian mixture over actions instead of one action.
        gmm_num_modes: Number of Gaussians in the mixture.
        gmm_min_std: Lower bound added to every Gaussian's standard deviation in training.
        gmm_low_noise_eval: Whether the standard deviations shrink to 1e-4 outside training.
        l2_weight: Without a distribution, weight of the mean squared error on the action.
        l1_weight: Without a distribution, weight of the smooth L1 error on the action.
        cos_weight: Without a distribution, weight of the cosine loss on the first three action
            dimensions.
        optimizer_lr: Adam learning rate, kept constant.
        optimizer_weight_decay: Adam weight decay.
        optimizer_grad_clip_norm: Gradient norm limit; 0 disables clipping.
    """

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ENV": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    use_env_state: bool = False

    crop_shape: tuple[int, int] | None = (76, 76)
    random_crop_at_inference: bool = False
    spatial_softmax_num_kp: int = 32
    image_feature_dim: int = 64

    use_gaussian: bool = False
    gaussian_fixed_std: bool = False
    gaussian_init_std: float = 0.1
    gaussian_min_std: float = 0.01
    gaussian_low_noise_eval: bool = True

    use_gmm: bool = True
    gmm_num_modes: int = 5
    gmm_min_std: float = 1e-4
    gmm_low_noise_eval: bool = True

    l2_weight: float = 1.0
    l1_weight: float = 0.0
    cos_weight: float = 0.0

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_obs_steps != 1:
            raise ValueError(
                f"{self.type} reads one observation per step, got n_obs_steps={self.n_obs_steps}."
            )

    @property
    def observation_features(self) -> dict[str, PolicyFeature]:
        """Input features the policy reads, in the order they are given."""
        return {
            key: feature
            for key, feature in self.input_features.items()
            if feature.type is not FeatureType.ENV or self.use_env_state
        }

    def validate_features(self) -> None:
        if self.action_feature is None:
            raise ValueError(f"{self.type} requires an 'action' output feature.")
        if not self.observation_features:
            raise ValueError(f"{self.type} requires at least one input feature it reads.")
        if self.use_env_state and self.env_state_feature is None:
            raise ValueError("use_env_state is set, but there is no 'observation.environment_state' input.")
        for key, feature in self.image_features.items():
            if self.crop_shape is not None and any(
                crop >= size for crop, size in zip(self.crop_shape, feature.shape[1:], strict=True)
            ):
                raise ValueError(f"crop_shape {self.crop_shape} must be smaller than {key} {feature.shape}.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
