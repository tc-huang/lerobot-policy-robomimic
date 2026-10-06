"""Heads that turn features into actions, each with its own training loss."""

import abc
import math

import torch
from torch import Tensor, distributions, nn
from torch.nn.functional import cosine_similarity, mse_loss, smooth_l1_loss, softplus

from .base_config import RobomimicPolicyConfig


class ActionHead(nn.Module, abc.ABC):
    """Maps (B, input_dim) features to actions in [-1, 1]."""

    @abc.abstractmethod
    def loss(self, features: Tensor, actions: Tensor) -> tuple[Tensor, dict[str, float]]:
        """Returns the training loss against (B, action_dim) actions, and values to log."""

    @abc.abstractmethod
    def act(self, features: Tensor) -> Tensor:
        """Returns (B, action_dim) actions to execute."""


class DeterministicHead(ActionHead):
    """A linear layer bounded by tanh, trained by regression."""

    def __init__(
        self, input_dim: int, action_dim: int, l2_weight: float, l1_weight: float, cos_weight: float
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, action_dim)
        self.weights = {"l2_loss": l2_weight, "l1_loss": l1_weight, "cos_loss": cos_weight}

    def forward(self, features: Tensor) -> Tensor:
        return torch.tanh(self.linear(features))

    def loss(self, features: Tensor, actions: Tensor) -> tuple[Tensor, dict[str, float]]:
        """Weights the mean squared, smooth L1, and translation cosine losses."""
        predicted = self(features)
        losses = {
            "l2_loss": mse_loss(predicted, actions),
            "l1_loss": smooth_l1_loss(predicted, actions),
            "cos_loss": (1 - cosine_similarity(predicted[..., :3], actions[..., :3], dim=-1)).mean(),
        }
        loss = sum(self.weights[name] * value for name, value in losses.items())
        return loss, {name: value.item() for name, value in losses.items()}

    def act(self, features: Tensor) -> Tensor:
        return self(features)


class GaussianHead(ActionHead):
    """A diagonal Gaussian over actions, trained by maximum likelihood.

    Means are clamped to [-9, 9], then bounded by tanh. Scales are softplus outputs scaled to
    `init_std` at zero, or `init_std` itself with `fixed_std`, clamped to [`min_std`, 7.5].
    Outside training with `low_noise_eval`, every scale is `LOW_NOISE_SCALE` and the action is
    the mean.
    """

    LOW_NOISE_SCALE = 1e-4
    MEAN_LIMIT = 9.0
    MAX_STD = 7.5

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        fixed_std: bool,
        init_std: float,
        min_std: float,
        low_noise_eval: bool,
    ):
        super().__init__()
        self.init_std = init_std
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.mean = nn.Linear(input_dim, action_dim)
        self.scale = None if fixed_std else nn.Linear(input_dim, action_dim)

    def forward(self, features: Tensor) -> distributions.Independent:
        means = torch.tanh(self.mean(features).clamp(-self.MEAN_LIMIT, self.MEAN_LIMIT))
        if self.low_noise_eval and not self.training:
            scales = torch.full_like(means, self.LOW_NOISE_SCALE)
        else:
            if self.scale is None:
                scales = torch.full_like(means, self.init_std)
            else:
                scales = softplus(self.scale(features)) * (self.init_std / math.log(2))
            scales = scales.clamp(self.min_std, self.MAX_STD)
        return distributions.Independent(distributions.Normal(means, scales), 1)

    def loss(self, features: Tensor, actions: Tensor) -> tuple[Tensor, dict[str, float]]:
        """Returns the negative log-likelihood of the actions."""
        log_probs = self(features).log_prob(actions).mean()
        return -log_probs, {"log_probs": log_probs.item()}

    def act(self, features: Tensor) -> Tensor:
        """Returns the mean outside training with `low_noise_eval`, and a sample otherwise."""
        distribution = self(features)
        if self.low_noise_eval and not self.training:
            return distribution.mean
        return distribution.sample()


class GMMHead(ActionHead):
    """A mixture of diagonal Gaussians over actions, trained by maximum likelihood.

    Means are bounded by tanh. Scales are softplus outputs plus `min_std`, except outside
    training with `low_noise_eval`, where every scale is `LOW_NOISE_SCALE`.
    """

    LOW_NOISE_SCALE = 1e-4

    def __init__(self, input_dim: int, action_dim: int, num_modes: int, min_std: float, low_noise_eval: bool):
        super().__init__()
        self.shape = (num_modes, action_dim)
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.mean = nn.Linear(input_dim, num_modes * action_dim)
        self.scale = nn.Linear(input_dim, num_modes * action_dim)
        self.logits = nn.Linear(input_dim, num_modes)

    def forward(self, features: Tensor) -> distributions.MixtureSameFamily:
        means = torch.tanh(self.mean(features)).unflatten(-1, self.shape)
        if self.low_noise_eval and not self.training:
            scales = torch.full_like(means, self.LOW_NOISE_SCALE)
        else:
            scales = softplus(self.scale(features)).unflatten(-1, self.shape) + self.min_std
        return distributions.MixtureSameFamily(
            distributions.Categorical(logits=self.logits(features)),
            distributions.Independent(distributions.Normal(means, scales), 1),
        )

    def loss(self, features: Tensor, actions: Tensor) -> tuple[Tensor, dict[str, float]]:
        """Returns the negative log-likelihood of the actions."""
        log_probs = self(features).log_prob(actions).mean()
        return -log_probs, {"log_probs": log_probs.item()}

    def act(self, features: Tensor) -> Tensor:
        """Samples a mixture component, then an action from it."""
        return self(features).sample()


def make_action_head(config: RobomimicPolicyConfig, input_dim: int) -> ActionHead:
    """Returns the Gaussian, GMM, or deterministic head that `config` asks for, in that precedence."""
    action_dim = config.action_feature.shape[0]
    if config.use_gaussian:
        return GaussianHead(
            input_dim,
            action_dim,
            config.gaussian_fixed_std,
            config.gaussian_init_std,
            config.gaussian_min_std,
            config.gaussian_low_noise_eval,
        )
    if config.use_gmm:
        return GMMHead(
            input_dim, action_dim, config.gmm_num_modes, config.gmm_min_std, config.gmm_low_noise_eval
        )
    return DeterministicHead(input_dim, action_dim, config.l2_weight, config.l1_weight, config.cos_weight)
