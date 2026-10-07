"""Turns the observations a robomimic policy reads into one feature vector."""

import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import OBS_IMAGES
from torch import Tensor, nn

from .config import RobomimicPolicyConfig
from .vision import ImageEncoder, RandomCrop


class CameraEncoder(nn.Module):
    """Crops a camera's images, encodes them with `ImageEncoder`, and applies an activation."""

    def __init__(self, crop: nn.Module, encoder: ImageEncoder, activation: nn.Module):
        super().__init__()
        self.crop = crop
        self.encoder = encoder
        self.activation = activation

    def forward(self, images: Tensor, language: Tensor | None = None) -> Tensor:
        """Maps (B, 3, H, W) images, and (B, language_dim) embeddings for FiLM, to (B, feature_dim)."""
        return self.activation(self.encoder(self.crop(images), language))


class ObservationEncoder(nn.Module):
    """Concatenates vector observations as they are and camera features from `ImageEncoder`.

    Features follow the order of `features`. Each camera is cropped, encoded, and, with
    `camera_activation`, passed through a ReLU. A language feature holds one embedding per
    sample, which is repeated over any further leading dimensions such as time. With `film`,
    that embedding modulates every camera's ResNet instead of joining the features.
    """

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        crop_shape: tuple[int, int] | None,
        num_kp: int,
        image_feature_dim: int,
        random_crop_at_inference: bool = False,
        camera_activation: bool = True,
        film: bool = False,
    ):
        super().__init__()
        self.keys = list(features)
        self.language_key = next(
            (key for key, feature in features.items() if feature.type is FeatureType.LANGUAGE), None
        )
        if film and self.language_key is None:
            raise ValueError("FiLM needs a language feature.")
        self.film = film
        language_dim = features[self.language_key].shape[0] if film and self.language_key else None
        self.cameras = nn.ModuleDict()
        self.output_dim = 0
        for key, feature in features.items():
            if feature.type is FeatureType.VISUAL:
                height, width = crop_shape or feature.shape[1:]
                self.cameras[self.camera_name(key)] = CameraEncoder(
                    crop=(
                        RandomCrop(crop_shape, random_crop_at_inference)
                        if crop_shape is not None
                        else nn.Identity()
                    ),
                    encoder=ImageEncoder(height, width, num_kp, image_feature_dim, language_dim),
                    activation=nn.ReLU() if camera_activation else nn.Identity(),
                )
                self.output_dim += image_feature_dim
            elif not (film and key == self.language_key):
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
            film=config.language_conditioning == "film",
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
        language = (
            self.repeat_language(batch[self.language_key], leading) if self.language_key is not None else None
        )
        film_language = language.flatten(end_dim=-2) if self.film and language is not None else None
        features = []
        for key in self.keys:
            camera = self.camera_name(key)
            if camera in self.cameras:
                frames = batch[key]
                encoded = self.cameras[camera](frames.flatten(end_dim=-4), film_language)
                features.append(encoded.unflatten(0, frames.shape[:-3]))
            elif key == self.language_key:
                if not self.film:
                    features.append(language)
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
