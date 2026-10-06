"""Turns the observations a robomimic policy reads into one feature vector."""

from collections import OrderedDict

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
        self.cameras = nn.ModuleDict()
        self.output_dim = 0
        for key, feature in features.items():
            if feature.type is FeatureType.VISUAL:
                height, width = crop_shape or feature.shape[1:]
                self.cameras[self.camera_name(key)] = nn.Sequential(
                    OrderedDict(
                        crop=RandomCrop(crop_shape) if crop_shape is not None else nn.Identity(),
                        encoder=ImageEncoder(height, width, num_kp, image_feature_dim),
                        activation=nn.ReLU(),
                    )
                )
                self.output_dim += image_feature_dim
            else:
                self.output_dim += feature.shape[0]

    @staticmethod
    def camera_name(key: str) -> str:
        """Returns the camera name of an `observation.images.<camera>` key."""
        return key.removeprefix(f"{OBS_IMAGES}.")

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Maps observations with leading dimensions such as (B,) or (B, T) to (..., output_dim) features.

        Camera frames are encoded one by one, so each frame gets its own random crop.
        """
        features = []
        for key in self.keys:
            camera = self.camera_name(key)
            if camera in self.cameras:
                frames = batch[key]
                encoded = self.cameras[camera](frames.flatten(end_dim=-4))
                features.append(encoded.unflatten(0, frames.shape[:-3]))
            else:
                features.append(batch[key])
        return torch.cat(features, dim=-1)
