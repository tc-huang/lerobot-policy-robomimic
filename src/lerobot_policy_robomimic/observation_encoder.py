"""Turns the observations a robomimic policy reads into one feature vector."""

import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import OBS_IMAGES
from torch import Tensor, nn

from .vision import ImageEncoder, RandomCrop


class ObservationEncoder(nn.Module):
    """Concatenates vector observations as they are and camera features from `ImageEncoder`.

    Features follow the order of `features`. Each camera is cropped, encoded, and passed
    through a ReLU.
    """

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        crop_shape: tuple[int, int] | None,
        num_kp: int,
        image_feature_dim: int,
    ):
        super().__init__()
        self.keys = list(features)
        self.crop = RandomCrop(crop_shape) if crop_shape is not None else nn.Identity()
        self.cameras = nn.ModuleDict()
        self.output_dim = 0
        for key, feature in features.items():
            if feature.type is FeatureType.VISUAL:
                height, width = crop_shape or feature.shape[1:]
                self.cameras[self.camera_name(key)] = nn.Sequential(
                    ImageEncoder(height, width, num_kp, image_feature_dim), nn.ReLU()
                )
                self.output_dim += image_feature_dim
            else:
                self.output_dim += feature.shape[0]

    @staticmethod
    def camera_name(key: str) -> str:
        """Returns the camera name of an `observation.images.<camera>` key."""
        return key.removeprefix(f"{OBS_IMAGES}.")

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Maps a batch of (B, ...) observations to (B, output_dim) features."""
        features = []
        for key in self.keys:
            camera = self.camera_name(key)
            if camera in self.cameras:
                features.append(self.cameras[camera](self.crop(batch[key])))
            else:
                features.append(batch[key])
        return torch.cat(features, dim=-1)
