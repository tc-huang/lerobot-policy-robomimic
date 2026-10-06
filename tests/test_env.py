import json
from pathlib import Path

import numpy as np
import pytest
import robomimic_reference
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

pytest.importorskip("robosuite", reason="robosuite is not installed; install it with `uv sync --extra sim`")

from lerobot_policy_robomimic import RobomimicEnvConfig  # noqa: E402

CAMERAS = ["agentview", "robot0_eye_in_hand"]
LIFT_ENV_ARGS = {
    "env_name": "Lift",
    "env_version": "1.5.1",
    "type": 1,
    "env_kwargs": {
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": True,
        "control_freq": 20,
        "controller_configs": {
            "type": "BASIC",
            "body_parts": {
                "right": {
                    "type": "OSC_POSE",
                    "input_max": 1,
                    "input_min": -1,
                    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                    "kp": 150,
                    "damping": 1,
                    "impedance_mode": "fixed",
                    "kp_limits": [0, 300],
                    "damping_limits": [0, 10],
                    "position_limits": None,
                    "orientation_limits": None,
                    "uncouple_pos_ori": True,
                    "control_delta": True,
                    "interpolation": None,
                    "ramp_ratio": 0.2,
                    "input_ref_frame": "world",
                    "gripper": {"type": "GRIP"},
                }
            },
        },
        "robots": ["Panda"],
        "camera_depths": False,
        "camera_heights": 84,
        "camera_widths": 84,
        "lite_physics": False,
        "reward_shaping": False,
        "camera_names": CAMERAS,
    },
}


@pytest.fixture
def env_config(tmp_path: Path) -> RobomimicEnvConfig:
    path = tmp_path / "robomimic_env_args.json"
    path.write_text(json.dumps(LIFT_ENV_ARGS))
    return RobomimicEnvConfig(env_args_path=str(path))


def test_config_follows_the_env_args(env_config):
    assert (env_config.task, env_config.fps, env_config.episode_length) == ("Lift", 20, 400)
    assert env_config.features["action"].shape == (7,)
    assert env_config.features["agent_pos"].shape == (9,)
    assert env_config.features["environment_state"].shape == (10,)
    assert env_config.features["pixels/agentview"].shape == (84, 84, 3)
    assert env_config.features_map == {
        "action": ACTION,
        "agent_pos": OBS_STATE,
        "environment_state": OBS_ENV_STATE,
        "pixels/agentview": f"{OBS_IMAGES}.agentview",
        "pixels/robot0_eye_in_hand": f"{OBS_IMAGES}.robot0_eye_in_hand",
    }


def test_steps_match_robomimic_env(env_config):
    ours = env_config.make_env(LIFT_ENV_ARGS)
    reference = robomimic_reference.robosuite_env(LIFT_ENV_ARGS, CAMERAS)
    np.random.seed(0)
    reference.reset()
    ours.reset(seed=0)
    np.testing.assert_array_equal(ours._env.sim.get_state().flatten(), reference.get_state()["states"])
    rng = np.random.default_rng(0)

    for step in range(5):
        action = rng.uniform(-1, 1, size=7)
        observation, reward, terminated, _, info = ours.step(action)
        expected, reference_reward, _, _ = reference.step(action)
        for camera in CAMERAS:
            np.testing.assert_array_equal(observation["pixels"][camera], expected[f"{camera}_image"])
        np.testing.assert_allclose(
            observation["agent_pos"],
            np.concatenate([expected[key] for key in robomimic_reference.STATE_KEYS]),
        )
        np.testing.assert_allclose(observation["environment_state"], expected["object"])
        assert reward == pytest.approx(reference_reward), f"step {step}"
        assert terminated == info["is_success"] == reference.is_success()["task"]
    ours.close()


def test_seeded_resets_repeat(env_config):
    env = env_config.make_env(LIFT_ENV_ARGS, camera_names=[])
    first, _ = env.reset(seed=3)
    second, _ = env.reset(seed=3)

    np.testing.assert_array_equal(first["environment_state"], second["environment_state"])
    assert env._max_episode_steps == 400
    env.close()


def test_create_envs_batches_observations(env_config):
    env_config.camera_names = []
    vec_env = env_config.create_envs(n_envs=2)["robomimic"][0]
    observation, _ = vec_env.reset(seed=[0, 1])

    assert observation["agent_pos"].shape == (2, 9)
    assert observation["environment_state"].shape == (2, 10)
    vec_env.close()
