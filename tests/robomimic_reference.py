"""Builds robomimic's own networks with the settings of its paper experiments."""

import pytest

pytest.importorskip("robomimic", reason="third_party/robomimic submodule is not checked out")

from collections import OrderedDict  # noqa: E402

import robomimic.utils.obs_utils as obs_utils  # noqa: E402
import torch  # noqa: E402
from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE  # noqa: E402
from robomimic.config import config_factory  # noqa: E402
from robomimic.models.base_nets import SpatialSoftmax  # noqa: E402
from robomimic.models.obs_core import CropRandomizer, VisualCore  # noqa: E402
from robomimic.models.obs_nets import ObservationEncoder, obs_encoder_factory  # noqa: E402
from robomimic.models.policy_nets import ActorNetwork  # noqa: E402
from robomimic.scripts.generate_paper_configs import modify_config_for_default_image_exp  # noqa: E402
from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict  # noqa: E402

STATE_KEYS = {"robot0_eef_pos": 3, "robot0_eef_quat": 4, "robot0_gripper_qpos": 2}
CAMERAS = ("agentview", "robot0_eye_in_hand")
IMAGE_SHAPE = (3, 84, 84)


def image_experiment_features() -> dict[str, PolicyFeature]:
    """Returns the LeRobot input features of an image experiment dataset."""
    features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(sum(STATE_KEYS.values()),))}
    for camera in CAMERAS:
        features[f"{OBS_IMAGES}.{camera}"] = PolicyFeature(type=FeatureType.VISUAL, shape=IMAGE_SHAPE)
    return features


def image_experiment_obs_shapes() -> dict[str, list[int]]:
    """Returns the robomimic observation shapes of an image experiment."""
    shapes = {key: [dim] for key, dim in STATE_KEYS.items()}
    return shapes | {f"{camera}_image": list(IMAGE_SHAPE) for camera in CAMERAS}


def random_image_observations(batch_size: int) -> tuple[dict, dict]:
    """Returns the same random observations keyed for robomimic and for LeRobot."""
    obs = {key: torch.randn(batch_size, dim) for key, dim in STATE_KEYS.items()}
    obs |= {f"{camera}_image": torch.rand(batch_size, *IMAGE_SHAPE) for camera in CAMERAS}
    batch = {OBS_STATE: torch.cat([obs[key] for key in STATE_KEYS], dim=-1)}
    batch |= {f"{OBS_IMAGES}.{camera}": obs[f"{camera}_image"] for camera in CAMERAS}
    return obs, batch


def lerobot_feature_order(features: torch.Tensor) -> torch.Tensor:
    """Reorders the last dimension of an image experiment from robomimic's sorted keys to LeRobot's order.

    It applies to encoder outputs and to the input columns of the first MLP layer alike.
    """
    feature_dim = rgb_encoder_kwargs()["core_kwargs"]["feature_dimension"]
    sizes = {key: STATE_KEYS.get(key, feature_dim) for key in sorted(image_experiment_obs_shapes())}
    by_key = dict(zip(sizes, features.split(list(sizes.values()), dim=-1), strict=True))
    ordered = [by_key[key] for key in STATE_KEYS] + [by_key[f"{camera}_image"] for camera in CAMERAS]
    return torch.cat(ordered, dim=-1)


def image_experiment_config():
    """Returns robomimic's BC config for the image experiments."""
    return modify_config_for_default_image_exp(config_factory("bc"))


def rgb_encoder_kwargs() -> dict:
    """Returns the camera encoder settings of the image experiments."""
    return obs_utils.obs_encoder_kwargs_from_config(image_experiment_config().observation.encoder)["rgb"]


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


def observation_encoder(obs_shapes: dict[str, list[int]]) -> ObservationEncoder:
    """Returns the observation encoder of the image experiments, with keys in sorted order.

    robomimic sorts observation keys when it reads the dataset (`robomimic/utils/file_utils.py:162`).
    """
    config = image_experiment_config()
    obs_utils.initialize_obs_utils_with_config(config)
    return obs_encoder_factory(
        obs_shapes=OrderedDict(sorted(obs_shapes.items())),
        encoder_kwargs=obs_utils.obs_encoder_kwargs_from_config(config.observation.encoder),
    )


def actor_network(obs_shapes: dict[str, list[int]], action_dim: int) -> ActorNetwork:
    """Returns BC's actor network of the image experiments, with keys in sorted order."""
    config = image_experiment_config()
    obs_utils.initialize_obs_utils_with_config(config)
    return ActorNetwork(
        obs_shapes=OrderedDict(sorted(obs_shapes.items())),
        ac_dim=action_dim,
        mlp_layer_dims=config.algo.actor_layer_dims,
        encoder_kwargs=obs_utils.obs_encoder_kwargs_from_config(config.observation.encoder),
    )


def load_visual_core(image_encoder, reference: VisualCore) -> None:
    """Copies the weights of robomimic's `VisualCore` into an `ImageEncoder`."""
    image_encoder.backbone.load_state_dict(reference.backbone.nets.state_dict())
    image_encoder.pool.keypoints.load_state_dict(reference.pool.nets.state_dict())
    image_encoder.projection.load_state_dict(reference.nets[-1].state_dict())
