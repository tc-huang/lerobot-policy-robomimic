"""Heads that turn features into actions, each with its own training loss."""

import abc

import torch
from torch import Tensor, distributions, nn
from torch.nn.functional import cosine_similarity, mse_loss, smooth_l1_loss, softplus


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
