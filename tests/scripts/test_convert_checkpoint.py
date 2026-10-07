import json
import re
from collections import OrderedDict
from pathlib import Path

import pytest
import robomimic_reference
import torch
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from lerobot_policy_robomimic import RobomimicBCRNNConfig, RobomimicBCRNNPolicy
from lerobot_policy_robomimic.policies.common.vision import RandomCrop
from lerobot_policy_robomimic.scripts.convert_checkpoint import convert, convert_state_dict, policy_config

MODEL_ZOO_CHECKPOINT = (
    Path(__file__).resolve().parents[2] / "data/robomimic/model_zoo/lift_ph_image_epoch_500_succ_100.pth"
)

V01_OBSERVATION = {
    "modalities": {
        "obs": {
            "low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
            "image": ["agentview_image", "robot0_eye_in_hand_image"],
        },
        "goal": {"low_dim": [], "image": []},
    },
    "encoder": {
        "visual_core": "ResNet18Conv",
        "visual_core_kwargs": {"pretrained": False, "input_coord_conv": False},
        "obs_randomizer_class": "CropRandomizer",
        "obs_randomizer_kwargs": {"crop_height": 76, "crop_width": 76, "num_crops": 1, "pos_enc": False},
        "visual_feature_dimension": 64,
        "use_spatial_softmax": True,
        "spatial_softmax_kwargs": {
            "num_kp": 32,
            "learnable_temperature": False,
            "temperature": 1.0,
            "noise_std": 0.0,
        },
    },
}


def v01_checkpoint(reference) -> dict:
    """Writes a robomimic v0.5 BC-RNN network in the v0.1 checkpoint format of the model zoo.

    v0.1 named a camera's backbone `vis_core` and its spatial softmax `pool_net`.
    """
    model = {
        re.sub(
            r"_image\.pool\.", "_image.pool_net.", re.sub(r"_image\.backbone\.", "_image.vis_core.", key)
        ): value
        for key, value in reference.nets.state_dict().items()
    }
    algo = json.loads(robomimic_reference.bc_rnn_config().dump())["algo"]
    config = {"algo": algo, "observation": V01_OBSERVATION, "train": {"seq_length": 10}}
    shapes = OrderedDict(
        sorted(
            (key, tuple(shape)) for key, shape in robomimic_reference.image_experiment_obs_shapes().items()
        )
    )
    return {
        "model": model,
        "config": json.dumps(config),
        "algo_name": "bc",
        "shape_metadata": {"ac_dim": 7, "all_shapes": shapes},
    }


def test_converted_policy_matches_robomimic():
    torch.manual_seed(0)
    reference = robomimic_reference.bc_rnn_algo(
        robomimic_reference.image_experiment_obs_shapes(), action_dim=7, gmm=True
    )
    checkpoint = v01_checkpoint(reference)
    config = policy_config(checkpoint)
    config.device = "cpu"
    policy = RobomimicBCRNNPolicy(config)
    policy.load_state_dict(convert_state_dict(checkpoint, config), strict=True)
    for module in policy.modules():
        if isinstance(module, RandomCrop):
            module.random_at_inference = False
    policy.eval()
    reference.set_eval()
    reference.reset()

    for step in range(12):
        obs, batch = robomimic_reference.random_image_observations(2)
        torch.manual_seed(step)
        ours = policy.select_action(batch)
        torch.manual_seed(step)
        torch.testing.assert_close(ours, reference.get_action(obs), msg=f"step {step}")


def test_config_follows_the_checkpoint():
    reference = robomimic_reference.bc_rnn_algo(
        robomimic_reference.image_experiment_obs_shapes(), action_dim=7, gmm=True
    )
    config = policy_config(v01_checkpoint(reference))

    assert list(config.input_features) == [
        OBS_STATE,
        f"{OBS_IMAGES}.agentview",
        f"{OBS_IMAGES}.robot0_eye_in_hand",
    ]
    assert config.input_features[OBS_STATE].shape == (9,)
    assert config.output_features[ACTION].shape == (7,)
    assert config.random_crop_at_inference
    assert (config.rnn_hidden_dim, config.rnn_num_layers, config.rnn_horizon) == (1000, 2, 10)
    assert config.use_gmm and config.actor_layer_dims == ()


def test_rejects_unsupported_checkpoints():
    reference = robomimic_reference.bc_rnn_algo(
        robomimic_reference.image_experiment_obs_shapes(), action_dim=7, gmm=True
    )
    checkpoint = v01_checkpoint(reference)
    config = json.loads(checkpoint["config"])
    config["algo"]["rnn"]["rnn_type"] = "GRU"
    checkpoint["config"] = json.dumps(config)

    with pytest.raises(ValueError, match="one-way LSTM"):
        policy_config(checkpoint)


@pytest.mark.skipif(not MODEL_ZOO_CHECKPOINT.exists(), reason="model zoo checkpoint is not downloaded")
def test_converts_the_model_zoo_checkpoint(tmp_path):
    convert(MODEL_ZOO_CHECKPOINT, tmp_path)

    config = RobomimicBCRNNConfig.from_pretrained(tmp_path)
    policy = RobomimicBCRNNPolicy.from_pretrained(tmp_path)

    assert config.random_crop_at_inference
    assert sum(parameter.numel() for parameter in policy.parameters()) > 0
    assert (tmp_path / "policy_preprocessor.json").exists()
    assert (tmp_path / "policy_postprocessor.json").exists()
