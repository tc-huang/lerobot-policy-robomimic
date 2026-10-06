from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from .base_config import RobomimicPolicyConfig


@PreTrainedConfig.register_subclass("robomimic_bc_vae")
@dataclass
class RobomimicBCVAEConfig(RobomimicPolicyConfig):
    """Configuration for robomimic's BC-VAE policy, a conditional VAE over actions.

    The paper trains no BC-VAE, so the VAE settings follow robomimic's defaults in
    `robomimic/config/bc_config.py`.

    Args:
        vae_latent_dim: Size of the latent; robomimic suggests twice the action size.
        vae_latent_clip: Bound on each latent dimension sampled from the prior; None disables it.
        vae_kl_weight: Weight of the KL loss against the reconstruction loss.
        vae_encoder_layer_dims: Hidden sizes of the MLP that encodes an action and its
            observation into the posterior.
        vae_decoder_layer_dims: Hidden sizes of the MLP that decodes a latent into an action.
        vae_decoder_is_conditioned: Whether the decoder also reads the observation.
        vae_reconstruction_sum_across_elements: Whether the reconstruction loss sums the squared
            errors over action dimensions instead of averaging them.
        vae_prior_learn: Whether the prior is learned instead of fixed at N(0, 1).
        vae_prior_is_conditioned: Whether a learned prior depends on the observation.
        vae_prior_layer_dims: Hidden sizes of the MLP of an observation-conditioned prior.
        vae_prior_use_gmm: Whether a learned prior is a Gaussian mixture instead of one Gaussian.
        vae_prior_gmm_num_modes: Number of Gaussians in a mixture prior.
        vae_prior_gmm_learn_weights: Whether the mixture weights are learned instead of uniform.
    """

    vae_latent_dim: int = 14
    vae_latent_clip: float | None = None
    vae_kl_weight: float = 1.0
    vae_encoder_layer_dims: tuple[int, ...] = (300, 400)
    vae_decoder_layer_dims: tuple[int, ...] = (300, 400)
    vae_decoder_is_conditioned: bool = True
    vae_reconstruction_sum_across_elements: bool = False
    vae_prior_learn: bool = False
    vae_prior_is_conditioned: bool = False
    vae_prior_layer_dims: tuple[int, ...] = (300, 400)
    vae_prior_use_gmm: bool = False
    vae_prior_gmm_num_modes: int = 10
    vae_prior_gmm_learn_weights: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (self.vae_decoder_is_conditioned or self.vae_prior_is_conditioned):
            raise ValueError("The VAE must condition its decoder, its prior, or both on the observation.")
        if self.vae_prior_is_conditioned and not self.vae_prior_learn:
            raise ValueError("vae_prior_is_conditioned needs vae_prior_learn.")
        if self.vae_prior_use_gmm and not self.vae_prior_learn:
            raise ValueError("vae_prior_use_gmm needs vae_prior_learn.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None
