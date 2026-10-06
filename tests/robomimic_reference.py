"""Builds robomimic's own networks with the settings of its paper experiments."""

import pytest

pytest.importorskip("robomimic", reason="third_party/robomimic submodule is not checked out")

from robomimic.config import config_factory  # noqa: E402
from robomimic.models.base_nets import SpatialSoftmax  # noqa: E402
from robomimic.models.obs_core import CropRandomizer, VisualCore  # noqa: E402
from robomimic.scripts.generate_paper_configs import modify_config_for_default_image_exp  # noqa: E402
from robomimic.utils.obs_utils import obs_encoder_kwargs_from_config  # noqa: E402
from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict  # noqa: E402


def image_experiment_config():
    """Returns robomimic's BC config for the image experiments."""
    return modify_config_for_default_image_exp(config_factory("bc"))


def rgb_encoder_kwargs() -> dict:
    """Returns the camera encoder settings of the image experiments."""
    return obs_encoder_kwargs_from_config(image_experiment_config().observation.encoder)["rgb"]


def crop_randomizer(image_shape: list[int]) -> CropRandomizer:
    """Returns the camera crop of the image experiments."""
    return CropRandomizer(input_shape=image_shape, **rgb_encoder_kwargs()["obs_randomizer_kwargs"])


def spatial_softmax(feature_shape: list[int]) -> SpatialSoftmax:
    """Returns the spatial softmax of the image experiments."""
    pool_kwargs = rgb_encoder_kwargs()["core_kwargs"]["pool_kwargs"]
    return SpatialSoftmax(input_shape=feature_shape, **pool_kwargs)


def visual_core(image_shape: list[int]) -> VisualCore:
    """Returns the camera encoder of the image experiments."""
    core_kwargs = extract_class_init_kwargs_from_dict(
        VisualCore, dict(rgb_encoder_kwargs()["core_kwargs"]), copy=True
    )
    return VisualCore(input_shape=image_shape, **core_kwargs)
