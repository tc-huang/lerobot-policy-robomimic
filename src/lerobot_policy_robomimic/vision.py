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


class ImageEncoder(nn.Module):
    """ResNet-18 trunk, spatial softmax, and a linear projection to one feature per camera."""

    def __init__(self, height: int, width: int, num_kp: int, feature_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet18(weights=None).children())[:-2])
        self.pool = SpatialSoftmax(512, math.ceil(height / 32), math.ceil(width / 32), num_kp)
        self.projection = nn.Linear(num_kp * 2, feature_dim)

    def forward(self, images: Tensor) -> Tensor:
        """Maps (B, 3, height, width) images to (B, feature_dim) features."""
        return self.projection(self.pool(self.backbone(images)).flatten(start_dim=1))
