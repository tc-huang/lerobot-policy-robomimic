"""Converts a robomimic v0.1 model zoo BC-RNN checkpoint into a LeRobot policy directory.

Usage:
    python -m lerobot_policy_robomimic.convert_checkpoint \\
        --checkpoint data/robomimic/model_zoo/lift_ph_image_epoch_500_succ_100.pth \\
        --output-dir outputs/checkpoints/lift_ph_image_bc_rnn
"""

import argparse
import json
import re
from pathlib import Path

import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import Tensor

from .configuration_robomimic_bc_rnn import RobomimicBCRNNConfig
from .convert_dataset import DEFAULT_STATE_KEYS, ENV_STATE_KEY, IMAGE_SUFFIX
from .modeling_robomimic_bc_rnn import RobomimicBCRNNPolicy

ENCODER_PREFIX = "policy.nets.encoder.nets.obs.obs_nets."

_SKIPPED = (
    re.compile(r"_image\.nets\.[01]\."),
    re.compile(r"\.pool_net\.(temperature|pos_x|pos_y)$"),
    re.compile(r"^policy\.nets\.rnn\.per_step_net\."),
)
_RENAMES = (
    (re.compile(r"^(\w+)_image\.vis_core\.nets\."), r"encoder.cameras.\1.encoder.backbone."),
    (re.compile(r"^(\w+)_image\.pool_net\.nets\."), r"encoder.cameras.\1.encoder.pool.keypoints."),
    (re.compile(r"^(\w+)_image\.nets\.3\."), r"encoder.cameras.\1.encoder.projection."),
)


def load_checkpoint(path: Path) -> dict:
    """Loads a robomimic checkpoint without running pickled code."""
    return torch.load(path, map_location="cpu", weights_only=True)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Unsupported robomimic checkpoint: {message}")


def policy_config(checkpoint: dict) -> RobomimicBCRNNConfig:
    """Builds the `RobomimicBCRNNConfig` that matches a v0.1 BC-RNN checkpoint."""
    config = json.loads(checkpoint["config"])
    algo, observation = config["algo"], config["observation"]
    rnn, gmm, encoder = algo["rnn"], algo["gmm"], observation["encoder"]
    obs = observation["modalities"]["obs"]
    _require(checkpoint["algo_name"] == "bc" and rnn["enabled"], "only BC-RNN is supported")
    _require("image" in obs, "only robomimic v0.1 configs, with an 'image' modality, are supported")
    _require(
        rnn["rnn_type"] == "LSTM" and not rnn["kwargs"].get("bidirectional", False), "needs a one-way LSTM"
    )
    _require(not rnn.get("open_loop", False), "open-loop BC-RNN is not supported")
    _require(rnn["horizon"] == config["train"]["seq_length"], "rnn.horizon must equal train.seq_length")
    _require(gmm["std_activation"] == "softplus", "GMM scales must use softplus")
    _require(config["train"].get("frame_stack", 1) == 1, "frame stacking is not supported")
    if obs["image"]:
        _require(encoder["visual_core"] == "ResNet18Conv", "cameras need a ResNet18Conv backbone")
        _require(not any(encoder["visual_core_kwargs"].values()), "pretrained or coord-conv backbones")
        _require(encoder["use_spatial_softmax"], "cameras need a spatial softmax")
        softmax = encoder["spatial_softmax_kwargs"]
        _require(
            softmax["temperature"] == 1.0
            and not softmax["learnable_temperature"]
            and not softmax["noise_std"],
            "the spatial softmax must have a fixed temperature of 1 and no noise",
        )
        _require(encoder["obs_randomizer_class"] in (None, "CropRandomizer"), "only crop randomization")

    shapes = checkpoint["shape_metadata"]["all_shapes"]
    state_keys = [key for key in DEFAULT_STATE_KEYS if key in obs["low_dim"]]
    extra = set(obs["low_dim"]) - set(state_keys) - {ENV_STATE_KEY}
    _require(not extra, f"low-dim observations {sorted(extra)} have no LeRobot feature")
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(sum(shapes[key][0] for key in state_keys),))
    }
    if ENV_STATE_KEY in obs["low_dim"]:
        input_features[OBS_ENV_STATE] = PolicyFeature(
            type=FeatureType.ENV, shape=tuple(shapes[ENV_STATE_KEY])
        )
    for key in sorted(obs["image"]):
        camera = key.removesuffix(IMAGE_SUFFIX)
        input_features[f"{OBS_IMAGES}.{camera}"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=tuple(shapes[key])
        )

    randomizer = encoder.get("obs_randomizer_kwargs") or {}
    crop = (
        (randomizer["crop_height"], randomizer["crop_width"]) if encoder.get("obs_randomizer_class") else None
    )
    optim = algo["optim_params"]["policy"]
    return RobomimicBCRNNConfig(
        input_features=input_features,
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(checkpoint["shape_metadata"]["ac_dim"],))
        },
        use_env_state=ENV_STATE_KEY in obs["low_dim"],
        crop_shape=crop,
        random_crop_at_inference=crop is not None,
        spatial_softmax_num_kp=encoder.get("spatial_softmax_kwargs", {}).get("num_kp", 32),
        image_feature_dim=encoder.get("visual_feature_dimension", 64),
        use_gmm=gmm["enabled"],
        gmm_num_modes=gmm["num_modes"],
        gmm_min_std=gmm["min_std"],
        gmm_low_noise_eval=gmm["low_noise_eval"],
        l2_weight=algo["loss"]["l2_weight"],
        l1_weight=algo["loss"]["l1_weight"],
        cos_weight=algo["loss"]["cos_weight"],
        optimizer_lr=optim["learning_rate"]["initial"],
        optimizer_weight_decay=optim["regularization"]["L2"],
        actor_layer_dims=tuple(algo["actor_layer_dims"]),
        rnn_hidden_dim=rnn["hidden_dim"],
        rnn_num_layers=rnn["num_layers"],
        rnn_horizon=rnn["horizon"],
    )


def feature_permutation(checkpoint: dict, config: RobomimicBCRNNConfig) -> Tensor:
    """Returns the robomimic feature index of each feature in the policy's order.

    robomimic concatenates observations in sorted key order, the order of `all_shapes`.
    """
    shapes = checkpoint["shape_metadata"]["all_shapes"]
    images = {key for key in shapes if key.endswith(IMAGE_SUFFIX)}
    sizes = {key: config.image_feature_dim if key in images else shapes[key][0] for key in shapes}
    starts, offset = {}, 0
    for key, size in sizes.items():
        starts[key] = offset
        offset += size

    order = []
    for feature in config.observation_features:
        if feature == OBS_STATE:
            keys = [key for key in DEFAULT_STATE_KEYS if key in shapes]
        elif feature == OBS_ENV_STATE:
            keys = [ENV_STATE_KEY]
        else:
            keys = [feature.removeprefix(f"{OBS_IMAGES}.") + IMAGE_SUFFIX]
        order += [starts[key] + i for key in keys for i in range(sizes[key])]
    return torch.tensor(order)


def convert_state_dict(checkpoint: dict, config: RobomimicBCRNNConfig) -> dict[str, Tensor]:
    """Renames the weights of a v0.1 BC-RNN checkpoint to `RobomimicBCRNNPolicy`'s names."""
    converted = {}
    for key, value in checkpoint["model"].items():
        if any(pattern.search(key) for pattern in _SKIPPED):
            continue
        if key.startswith(ENCODER_PREFIX):
            short = key.removeprefix(ENCODER_PREFIX)
            for pattern, replacement in _RENAMES:
                if pattern.match(short):
                    converted[pattern.sub(replacement, short)] = value
                    break
            else:
                raise KeyError(f"Unexpected encoder weight {key}")
        elif key.startswith("policy.nets.rnn.nets."):
            converted["lstm." + key.removeprefix("policy.nets.rnn.nets.")] = value
        elif key.startswith("policy.nets.mlp._model."):
            converted["mlp.layers." + key.removeprefix("policy.nets.mlp._model.")] = value
        elif key.startswith("policy.nets.decoder.nets."):
            converted["action_head." + key.removeprefix("policy.nets.decoder.nets.")] = value
        else:
            raise KeyError(f"Unexpected weight {key}")
    converted["lstm.weight_ih_l0"] = converted["lstm.weight_ih_l0"][
        :, feature_permutation(checkpoint, config)
    ]
    return converted


def convert(checkpoint_path: Path, output_dir: Path) -> RobomimicBCRNNPolicy:
    """Writes the policy, its config, and its processors to `output_dir`."""
    checkpoint = load_checkpoint(checkpoint_path)
    config = policy_config(checkpoint)
    policy = RobomimicBCRNNPolicy(config)
    policy.load_state_dict(convert_state_dict(checkpoint, config), strict=True)
    policy.save_pretrained(output_dir)
    preprocessor, postprocessor = make_pre_post_processors(config)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="robomimic v0.1 .pth checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True, help="LeRobot policy directory to write")
    args = parser.parse_args()
    convert(args.checkpoint, args.output_dir)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
