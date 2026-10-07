"""The conditional VAE over actions of robomimic's BC-VAE."""

import math

import torch
from torch import Tensor, distributions, nn

from ..common.mlp import MLP
from ..common.observation_encoder import ObservationEncoder
from .configuration_robomimic_bc_vae import RobomimicBCVAEConfig

LOG_STD_LIMITS = (-4.0, 15.0)
LOG_VAR_LIMITS = (-8.0, 30.0)


def reparameterize(mean: Tensor, logvar: Tensor) -> Tensor:
    """Samples from N(mean, exp(logvar)) differentiably, with the log standard deviation clamped."""
    std = torch.exp((0.5 * logvar).clamp(*LOG_STD_LIMITS))
    return mean + torch.randn_like(std) * std


def log_normal(x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    """Returns the log density of x under diagonal Gaussians, summed over the last dimension."""
    return (-0.5 * (var.log() + (x - mean).pow(2) / var + math.log(2 * math.pi))).sum(-1)


class ConditionedMLP(nn.Module):
    """An MLP over an optional input vector followed by the encoded observations.

    Each named output is a linear layer on the MLP's features.
    """

    def __init__(
        self,
        input_dim: int,
        encoder: ObservationEncoder,
        layer_dims: tuple[int, ...],
        outputs: dict[str, int],
    ):
        super().__init__()
        self.encoder = encoder
        self.mlp = MLP(input_dim + encoder.output_dim, layer_dims)
        self.outputs = nn.ModuleDict(
            {name: nn.Linear(self.mlp.output_dim, size) for name, size in outputs.items()}
        )

    def forward(self, batch: dict[str, Tensor], inputs: Tensor | None = None) -> dict[str, Tensor]:
        """Maps (B, input_dim) inputs and observations to a (B, size) tensor per output."""
        features = [] if inputs is None else [inputs]
        features.append(self.encoder(batch))
        hidden = self.mlp(torch.cat(features, dim=-1))
        return {name: layer(hidden) for name, layer in self.outputs.items()}


class Prior(nn.Module):
    """The latent prior: N(0, 1), or a learned Gaussian or Gaussian mixture.

    A learned prior is a set of parameters, or the output of an MLP over the observations when
    it has an encoder. Mixture weights are uniform unless `learn_weights`.
    """

    def __init__(
        self,
        latent_dim: int,
        learn: bool,
        encoder: ObservationEncoder | None,
        layer_dims: tuple[int, ...],
        use_gmm: bool,
        num_modes: int,
        learn_weights: bool,
        latent_clip: float | None,
    ):
        super().__init__()
        self.learn = learn
        self.use_gmm = use_gmm
        self.latent_clip = latent_clip
        self.shape = (num_modes if use_gmm else 1, latent_dim)
        sizes = {"mean": math.prod(self.shape), "logvar": math.prod(self.shape)}
        if use_gmm and learn_weights:
            sizes["weight"] = self.shape[0]
        self.network = None
        self.params = nn.ParameterDict()
        if learn and encoder is not None:
            self.network = ConditionedMLP(0, encoder, layer_dims, sizes)
        elif learn:
            for name, size in sizes.items():
                shape = (1, *self.shape) if name != "weight" else (1, self.shape[0])
                self.params[name] = nn.Parameter(torch.randn(shape) / math.sqrt(size))

    def distribution(self, batch: dict[str, Tensor], batch_size: int) -> tuple[Tensor, Tensor, Tensor | None]:
        """Returns the (B, modes, latent_dim) means and log variances and the (B, modes) log weights.

        The log weights are None when the mixture weights are uniform.
        """
        if self.network is not None:
            outputs = self.network(batch)
            means = outputs["mean"].unflatten(-1, self.shape)
            logvars = outputs["logvar"].unflatten(-1, self.shape)
            weights = outputs.get("weight")
        else:
            means = self.params["mean"].expand(batch_size, -1, -1)
            logvars = self.params["logvar"].expand(batch_size, -1, -1)
            weights = self.params["weight"].expand(batch_size, -1) if "weight" in self.params else None
        return means, logvars, None if weights is None else weights.log_softmax(dim=-1)

    def sample(self, batch: dict[str, Tensor], batch_size: int, device: torch.device) -> Tensor:
        """Draws (B, latent_dim) latents, clamped to `latent_clip` if set."""
        if not self.learn:
            z = torch.randn(batch_size, self.shape[1], device=device)
        else:
            means, logvars, logweights = self.distribution(batch, batch_size)
            if self.use_gmm:
                weights = (
                    torch.full(means.shape[:2], 1 / self.shape[0], device=means.device)
                    if logweights is None
                    else logweights.exp()
                )
                modes = distributions.Categorical(weights).sample()
                rows = torch.arange(batch_size, device=means.device)
                z = reparameterize(means[rows, modes], logvars[rows, modes])
            else:
                z = reparameterize(means[:, 0], logvars[:, 0])
        if self.latent_clip is not None:
            z = z.clamp(-self.latent_clip, self.latent_clip)
        return z

    def kl_loss(self, mean: Tensor, logvar: Tensor, z: Tensor, batch: dict[str, Tensor]) -> Tensor:
        """Returns the KL divergence of the posterior N(mean, exp(logvar)) from the prior.

        It is exact for a Gaussian prior and estimated at the posterior sample `z` for a mixture.
        """
        if not self.learn:
            return (-0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=1)).mean()
        means, logvars, logweights = self.distribution(batch, mean.shape[0])
        if not self.use_gmm:
            prior_mean, prior_logvar = means[:, 0], logvars[:, 0]
            return (
                -0.5
                * (
                    1
                    + logvar
                    - prior_logvar
                    - (prior_mean - mean).pow(2) / prior_logvar.exp()
                    - logvar.exp() / prior_logvar.exp()
                ).sum(dim=1)
            ).mean()
        var = logvar.clamp(*LOG_VAR_LIMITS).exp()
        prior_vars = logvars.clamp(*LOG_VAR_LIMITS).exp()
        mode_log_probs = log_normal(z.unsqueeze(1), means, prior_vars)
        if logweights is None:
            mixture_log_prob = mode_log_probs.logsumexp(dim=1) - math.log(self.shape[0])
        else:
            mixture_log_prob = (mode_log_probs + logweights).logsumexp(dim=1)
        return (log_normal(z, mean, var) - mixture_log_prob).mean()


class ActionVAE(nn.Module):
    """A conditional VAE over actions.

    The posterior encodes an action with its observation, and the decoder turns a latent with
    the observation into an action in [-1, 1]. The posterior, the decoder, and a conditioned
    prior each encode the observations with their own `ObservationEncoder`.
    """

    def __init__(
        self, posterior: ConditionedMLP, decoder: ConditionedMLP, prior: Prior, sum_across_elements: bool
    ):
        super().__init__()
        self.posterior = posterior
        self.decoder = decoder
        self.prior = prior
        self.sum_across_elements = sum_across_elements

    @classmethod
    def from_config(cls, config: RobomimicBCVAEConfig) -> "ActionVAE":
        """Builds the VAE that `config` describes."""
        action_dim = config.action_feature.shape[0]
        latent_dim = config.vae_latent_dim
        return cls(
            posterior=ConditionedMLP(
                action_dim,
                ObservationEncoder.from_config(config),
                config.vae_encoder_layer_dims,
                {"mean": latent_dim, "logvar": latent_dim},
            ),
            decoder=ConditionedMLP(
                latent_dim,
                ObservationEncoder.from_config(config),
                config.vae_decoder_layer_dims,
                {"action": action_dim},
            ),
            prior=Prior(
                latent_dim,
                config.vae_prior_learn,
                ObservationEncoder.from_config(config) if config.vae_prior_is_conditioned else None,
                config.vae_prior_layer_dims,
                config.vae_prior_use_gmm,
                config.vae_prior_gmm_num_modes,
                config.vae_prior_gmm_learn_weights,
                config.vae_latent_clip,
            ),
            sum_across_elements=config.vae_reconstruction_sum_across_elements,
        )

    def decode(self, batch: dict[str, Tensor], z: Tensor) -> Tensor:
        """Maps (B, latent_dim) latents to (B, action_dim) actions bounded by tanh."""
        return torch.tanh(self.decoder(batch, z)["action"])

    def losses(self, batch: dict[str, Tensor], actions: Tensor) -> dict[str, Tensor]:
        """Returns the reconstruction and KL losses of (B, action_dim) actions, and the posterior variance."""
        posterior = self.posterior(batch, actions)
        mean, logvar = posterior["mean"], posterior["logvar"]
        z = reparameterize(mean, logvar)
        errors = (self.decode(batch, z) - actions).pow(2)
        reconstruction = errors.sum() / len(actions) if self.sum_across_elements else errors.mean()
        return {
            "reconstruction_loss": reconstruction,
            "kl_loss": self.prior.kl_loss(mean, logvar, z, batch),
            "encoder_variance": logvar.detach().exp().mean(),
        }

    def sample(self, batch: dict[str, Tensor], batch_size: int) -> Tensor:
        """Decodes (B, action_dim) actions from latents drawn from the prior."""
        z = self.prior.sample(batch, batch_size, next(self.parameters()).device)
        return self.decode(batch, z)
