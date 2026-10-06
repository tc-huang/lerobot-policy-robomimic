"""Builds robomimic's own networks with the settings of its paper experiments."""

import pytest

pytest.importorskip("robomimic", reason="third_party/robomimic submodule is not checked out")

from collections import OrderedDict  # noqa: E402

import robomimic.utils.obs_utils as obs_utils  # noqa: E402
import torch  # noqa: E402
from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE  # noqa: E402
from robomimic.algo import algo_factory  # noqa: E402
from robomimic.algo.bc import BC  # noqa: E402
from robomimic.config import config_factory  # noqa: E402
from robomimic.models.base_nets import SpatialSoftmax  # noqa: E402
from robomimic.models.obs_core import CropRandomizer, VisualCore  # noqa: E402
from robomimic.models.obs_nets import ObservationEncoder, obs_encoder_factory  # noqa: E402
from robomimic.scripts.generate_paper_configs import (  # noqa: E402
    modify_bc_config_for_dataset,
    modify_bc_rnn_config_for_dataset,
    modify_config_for_default_image_exp,
)
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


def random_image_observations(*leading: int) -> tuple[dict, dict]:
    """Returns the same random observations, with leading dimensions such as (B,) or (B, T).

    The first dict is keyed for robomimic, the second for LeRobot.
    """
    obs = {key: torch.randn(*leading, dim) for key, dim in STATE_KEYS.items()}
    obs |= {f"{camera}_image": torch.rand(*leading, *IMAGE_SHAPE) for camera in CAMERAS}
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


def bc_config(gmm: bool = True, **loss_weights: float):
    """Returns robomimic's config for BC in the image experiments on proficient-human Lift data.

    Those experiments use a GMM head (`robomimic/scripts/generate_paper_configs.py:368`);
    `gmm=False` turns it off, which gives robomimic's plain `BC`.
    """
    config = modify_bc_config_for_dataset(image_experiment_config(), "lift", "ph", "image")
    with config.algo.values_unlocked():
        config.algo.gmm.enabled = gmm
        for name, weight in loss_weights.items():
            config.algo.loss[name] = weight
    return config


def bc_rnn_config(hdf5_type: str = "image"):
    """Returns robomimic's config for BC-RNN in the experiments on proficient-human Lift data.

    These experiments use a GMM head (`robomimic/scripts/generate_paper_configs.py:415`).
    """
    return modify_bc_rnn_config_for_dataset(image_experiment_config(), "lift", "ph", hdf5_type)


def bc_rnn_algo(obs_shapes: dict[str, list[int]], action_dim: int, gmm: bool) -> BC:
    """Returns robomimic's BC-RNN algorithm built from `bc_rnn_config`, with keys in sorted order."""
    config = bc_rnn_config()
    with config.algo.values_unlocked():
        config.algo.gmm.enabled = gmm
    return make_algo(config, obs_shapes, action_dim)


def bc_algo(obs_shapes: dict[str, list[int]], action_dim: int, gmm: bool, **loss_weights: float) -> BC:
    """Returns robomimic's BC algorithm built from `bc_config`, with keys in sorted order."""
    return make_algo(bc_config(gmm, **loss_weights), obs_shapes, action_dim)


def make_algo(config, obs_shapes: dict[str, list[int]], action_dim: int) -> BC:
    """Builds the robomimic algorithm of `config` on the CPU, with keys in sorted order."""
    obs_utils.initialize_obs_utils_with_config(config)
    return algo_factory(
        algo_name="bc",
        config=config,
        obs_key_shapes=OrderedDict(sorted(obs_shapes.items())),
        ac_dim=action_dim,
        device=torch.device("cpu"),
    )


def load_policy(policy, reference) -> None:
    """Copies the weights of BC's or BC-RNN's actor network into our policy.

    The input columns of the first layer after the encoder, the LSTM's or the MLP's, are
    reordered from robomimic's sorted keys.
    """
    for camera in CAMERAS:
        load_visual_core(
            policy.encoder.cameras[camera].encoder,
            reference.nets["encoder"].nets["obs"].obs_nets[f"{camera}_image"],
        )
    reordered = False
    if "rnn" in reference.nets:
        lstm = dict(reference.nets["rnn"].nets.state_dict())
        lstm["weight_ih_l0"] = lerobot_feature_order(lstm["weight_ih_l0"])
        policy.lstm.load_state_dict(lstm)
        reordered = True
    reference_mlp = reference.nets["mlp"]._model if "mlp" in reference.nets else []
    reference_linears = [layer for layer in reference_mlp if isinstance(layer, torch.nn.Linear)]
    ours_linears = [layer for layer in policy.mlp.layers if isinstance(layer, torch.nn.Linear)]
    for ours, theirs in zip(ours_linears, reference_linears, strict=True):
        weight = theirs.weight if reordered else lerobot_feature_order(theirs.weight)
        ours.load_state_dict({"weight": weight, "bias": theirs.bias})
        reordered = True
    decoder = reference.nets["decoder"].nets
    if "action" in decoder:
        policy.action_head.linear.load_state_dict(decoder["action"].state_dict())
    else:
        for name in ("mean", "scale", "logits"):
            getattr(policy.action_head, name).load_state_dict(decoder[name].state_dict())


def load_visual_core(image_encoder, reference: VisualCore) -> None:
    """Copies the weights of robomimic's `VisualCore` into an `ImageEncoder`."""
    image_encoder.backbone.load_state_dict(reference.backbone.nets.state_dict())
    image_encoder.pool.keypoints.load_state_dict(reference.pool.nets.state_dict())
    image_encoder.projection.load_state_dict(reference.nets[-1].state_dict())
