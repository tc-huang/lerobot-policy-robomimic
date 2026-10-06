"""Turns the observations a robomimic policy reads into one feature vector."""

from collections import OrderedDict

import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import OBS_IMAGES
from torch import Tensor, nn

from .base_config import RobomimicPolicyConfig
from .vision import ImageEncoder, RandomCrop


class ObservationEncoder(nn.Module):
    """Concatenates vector observations as they are and camera features from `ImageEncoder`.

    Features follow the order of `features`. Each camera is cropped, encoded, and, with
    `camera_activation`, passed through a ReLU. A language feature holds one embedding per
    sample, which is repeated over any further leading dimensions such as time.
    """

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        crop_shape: tuple[int, int] | None,
        num_kp: int,
        image_feature_dim: int,
        random_crop_at_inference: bool = False,
        camera_activation: bool = True,
    ):
        super().__init__()
        self.keys = list(features)
        self.language_key = next(
            (key for key, feature in features.items() if feature.type is FeatureType.LANGUAGE), None
        )
        self.cameras = nn.ModuleDict()
        self.output_dim = 0
        for key, feature in features.items():
            if feature.type is FeatureType.VISUAL:
                height, width = crop_shape or feature.shape[1:]
                self.cameras[self.camera_name(key)] = nn.Sequential(
                    OrderedDict(
                        crop=(
                            RandomCrop(crop_shape, random_crop_at_inference)
                            if crop_shape is not None
                            else nn.Identity()
                        ),
                        encoder=ImageEncoder(height, width, num_kp, image_feature_dim),
                        activation=nn.ReLU() if camera_activation else nn.Identity(),
                    )
                )
                self.output_dim += image_feature_dim
            else:
                self.output_dim += feature.shape[0]

    @classmethod
    def from_config(
        cls, config: RobomimicPolicyConfig, camera_activation: bool = True
    ) -> "ObservationEncoder":
        """Builds the encoder for the observations `config` reads."""
        return cls(
            config.observation_features,
            config.crop_shape,
            config.spatial_softmax_num_kp,
            config.image_feature_dim,
            config.random_crop_at_inference,
            camera_activation,
        )

    @staticmethod
    def camera_name(key: str) -> str:
        """Returns the camera name of an `observation.images.<camera>` key."""
        return key.removeprefix(f"{OBS_IMAGES}.")

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Maps observations with leading dimensions such as (B,) or (B, T) to (..., output_dim) features.

        Camera frames are encoded one by one, so each frame gets its own random crop.
        """
        leading = self.leading_shape(batch)
        features = []
        for key in self.keys:
            camera = self.camera_name(key)
            if camera in self.cameras:
                frames = batch[key]
                encoded = self.cameras[camera](frames.flatten(end_dim=-4))
                features.append(encoded.unflatten(0, frames.shape[:-3]))
            elif key == self.language_key:
                features.append(self.repeat_language(batch[key], leading))
            else:
                features.append(batch[key])
        return torch.cat(features, dim=-1)

    def leading_shape(self, batch: dict[str, Tensor]) -> torch.Size:
        """Returns the leading dimensions, such as (B,) or (B, T), of the observations."""
        key = next(key for key in self.keys if key != self.language_key)
        value = batch[key]
        return value.shape[:-3] if self.camera_name(key) in self.cameras else value.shape[:-1]

    @staticmethod
    def repeat_language(embedding: Tensor, leading: torch.Size) -> Tensor:
        """Returns a language embedding repeated to (*leading, embedding_dim).

        The embedding has the batch dimension and possibly some of the further leading ones.
        """
        missing = len(leading) - (embedding.dim() - 1)
        return embedding.reshape(*embedding.shape[:-1], *([1] * missing), -1).expand(*leading, -1)
