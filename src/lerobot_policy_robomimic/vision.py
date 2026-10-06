"""Camera encoders shared by the robomimic policies."""

import math

import torch
from torch import Tensor, nn
from torchvision.models import resnet18


class RandomCrop(nn.Module):
    """Crops images at a random position in training and at the center otherwise.

    With `random_at_inference`, images are cropped at random positions outside training too.
    """

    def __init__(self, crop_shape: tuple[int, int], random_at_inference: bool = False):
        super().__init__()
        self.crop_shape = crop_shape
        self.random_at_inference = random_at_inference

    def forward(self, images: Tensor) -> Tensor:
        """Crops (B, C, H, W) images to (B, C, crop_height, crop_width)."""
        crop_h, crop_w = self.crop_shape
        height, width = images.shape[-2:]
        if not (self.training or self.random_at_inference):
            top, left = (height - crop_h) // 2, (width - crop_w) // 2
            return images[..., top : top + crop_h, left : left + crop_w]

        batch = images.shape[0]
        tops = (torch.rand(batch) * (height - crop_h)).long().tolist()
        lefts = (torch.rand(batch) * (width - crop_w)).long().tolist()
        return torch.stack(
            [
                image[:, top : top + crop_h, left : left + crop_w]
                for image, top, left in zip(images, tops, lefts, strict=True)
            ]
        )


class SpatialSoftmax(nn.Module):
    """Turns a feature map into the expected (x, y) position of each learned keypoint."""

    def __init__(self, in_channels: int, height: int, width: int, num_kp: int):
        super().__init__()
        self.keypoints = nn.Conv2d(in_channels, num_kp, kernel_size=1)
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height), torch.linspace(-1.0, 1.0, width), indexing="ij"
        )
        self.register_buffer("pos", torch.stack([pos_x.flatten(), pos_y.flatten()], dim=-1), persistent=False)

    def forward(self, features: Tensor) -> Tensor:
        """Maps (B, C, H, W) features to (B, num_kp, 2) keypoint coordinates in [-1, 1]."""
        attention = torch.softmax(self.keypoints(features).flatten(start_dim=2), dim=-1)
        return attention @ self.pos


class FiLM(nn.Module):
    """Feature-wise linear modulation of a feature map by a language embedding, then a ReLU."""

    def __init__(self, language_dim: int, channels: int):
        super().__init__()
        self.projection = nn.Linear(language_dim, 2 * channels)

    def forward(self, features: Tensor, language: Tensor) -> Tensor:
        """Maps (B, C, H, W) features and (B, language_dim) embeddings to (1 + gamma) * x + beta."""
        beta, gamma = self.projection(language)[..., None, None].chunk(2, dim=1)
        return torch.relu((1 + gamma) * features + beta)


class ImageEncoder(nn.Module):
    """ResNet-18 trunk, spatial softmax, and a linear projection to one feature per camera.

    With `language_dim`, a `FiLM` layer follows each of the trunk's residual blocks, and the
    encoder needs a language embedding for every image.
    """

    def __init__(
        self, height: int, width: int, num_kp: int, feature_dim: int, language_dim: int | None = None
    ):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet18(weights=None).children())[:-2])
        self.film = (
            nn.ModuleList(FiLM(language_dim, block.conv2.out_channels) for block in self.residual_blocks())
            if language_dim is not None
            else None
        )
        self.pool = SpatialSoftmax(512, math.ceil(height / 32), math.ceil(width / 32), num_kp)
        self.projection = nn.Linear(num_kp * 2, feature_dim)

    def residual_blocks(self) -> list[nn.Module]:
        """Returns the trunk's residual blocks in order."""
        return [block for stage in self.backbone[4:] for block in stage]

    def forward(self, images: Tensor, language: Tensor | None = None) -> Tensor:
        """Maps (B, 3, height, width) images, and with FiLM (B, language_dim) embeddings, to (B, feature_dim)."""
        if self.film is None:
            features = self.backbone(images)
        else:
            if language is None:
                raise ValueError("This camera encoder is conditioned on language and needs an embedding.")
            features = self.backbone[:4](images)
            for block, film in zip(self.residual_blocks(), self.film, strict=True):
                features = film(block(features), language)
        return self.projection(self.pool(features).flatten(start_dim=1))
