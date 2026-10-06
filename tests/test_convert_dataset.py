import json

import h5py
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from lerobot_policy_robomimic.convert_dataset import (
    MASKS_FILE,
    VIDEO_THRESHOLD_BYTES,
    convert,
    use_videos,
)

DEMO_LENGTHS = {"demo_0": 2, "demo_1": 3, "demo_10": 4}
# robomimic's camera size; LeRobot's default libsvtav1 encoder never finishes on 8x8 frames.
IMAGE_SHAPE = (84, 84, 3)


def write_robomimic_hdf5(path, with_images=True):
    """Writes a tiny file with the layout produced by robomimic's dataset_states_to_obs.py."""
    rng = np.random.default_rng(0)
    demos = {}
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        env_args = {"env_name": "Lift", "env_kwargs": {"control_freq": 20, "robots": ["Panda"]}}
        data.attrs["env_args"] = json.dumps(env_args)
        data.attrs["total"] = sum(DEMO_LENGTHS.values())
        for key, n in DEMO_LENGTHS.items():
            obs = {
                "robot0_eef_pos": rng.normal(size=(n, 3)),
                "robot0_eef_quat": rng.normal(size=(n, 4)),
                "robot0_gripper_qpos": rng.normal(size=(n, 2)),
                "robot0_joint_pos": rng.normal(size=(n, 7)),
                "object": rng.normal(size=(n, 10)),
            }
            if with_images:
                obs["agentview_image"] = rng.integers(0, 256, size=(n, *IMAGE_SHAPE), dtype=np.uint8)
                obs["robot0_eye_in_hand_image"] = rng.integers(0, 256, size=(n, *IMAGE_SHAPE), dtype=np.uint8)
            demo = data.create_group(key)
            demo.attrs["num_samples"] = n
            demo.create_dataset("actions", data=rng.uniform(-1, 1, size=(n, 7)))
            for name, value in obs.items():
                demo.create_dataset(f"obs/{name}", data=value)
            demos[key] = {"actions": demo["actions"][()], "obs": obs}
        f.create_dataset("mask/train", data=np.array([b"demo_0", b"demo_10"]))
        f.create_dataset("mask/valid", data=np.array([b"demo_1"]))
    return demos


def test_use_videos():
    assert not use_videos("auto", VIDEO_THRESHOLD_BYTES)
    assert use_videos("auto", VIDEO_THRESHOLD_BYTES + 1)
    assert use_videos("on", 0)
    assert not use_videos("off", VIDEO_THRESHOLD_BYTES + 1)


def test_convert_image_dataset(tmp_path):
    demos = write_robomimic_hdf5(tmp_path / "image.hdf5")
    convert(tmp_path / "image.hdf5", "test/robomimic", "lift the cube", root=tmp_path / "out")
    dataset = LeRobotDataset("test/robomimic", root=tmp_path / "out")

    assert dataset.fps == 20
    assert dataset.meta.robot_type == "panda"
    assert dataset.meta.video_keys == []
    assert dataset.meta.features[OBS_STATE]["names"][:4] == [
        "robot0_eef_pos.0",
        "robot0_eef_pos.1",
        "robot0_eef_pos.2",
        "robot0_eef_quat.0",
    ]
    assert dataset.meta.total_episodes == len(DEMO_LENGTHS)

    index = 0
    for episode, (key, n) in enumerate(DEMO_LENGTHS.items()):
        demo = demos[key]
        state = np.concatenate(
            [demo["obs"][k] for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")], axis=1
        )
        for t in range(n):
            item = dataset[index]
            assert item["episode_index"] == episode
            assert item["task"] == "lift the cube"
            torch.testing.assert_close(item[OBS_STATE], torch.from_numpy(state[t]).float())
            torch.testing.assert_close(item[ACTION], torch.from_numpy(demo["actions"][t]).float())
            torch.testing.assert_close(item[OBS_ENV_STATE], torch.from_numpy(demo["obs"]["object"][t]).float())
            for camera in ("agentview", "robot0_eye_in_hand"):
                pixels = (item[f"{OBS_IMAGES}.{camera}"] * 255).round().to(torch.uint8).permute(1, 2, 0)
                np.testing.assert_array_equal(pixels.numpy(), demo["obs"][f"{camera}_image"][t])
            index += 1

    masks = json.loads((tmp_path / "out" / MASKS_FILE).read_text())
    assert masks == {"demo_keys": ["demo_0", "demo_1", "demo_10"], "masks": {"train": [0, 2], "valid": [1]}}


def test_convert_low_dim_dataset(tmp_path):
    write_robomimic_hdf5(tmp_path / "low_dim.hdf5", with_images=False)
    convert(tmp_path / "low_dim.hdf5", "test/robomimic", "lift the cube", root=tmp_path / "out")
    dataset = LeRobotDataset("test/robomimic", root=tmp_path / "out")

    assert dataset.meta.camera_keys == []
    assert set(dataset.meta.features) >= {OBS_STATE, OBS_ENV_STATE, ACTION}


def test_convert_to_video(tmp_path):
    write_robomimic_hdf5(tmp_path / "image.hdf5")
    convert(tmp_path / "image.hdf5", "test/robomimic", "lift the cube", root=tmp_path / "out", video="on")
    dataset = LeRobotDataset("test/robomimic", root=tmp_path / "out")

    assert dataset.meta.video_keys == [f"{OBS_IMAGES}.agentview", f"{OBS_IMAGES}.robot0_eye_in_hand"]
    assert dataset[0][f"{OBS_IMAGES}.agentview"].shape == (3, *IMAGE_SHAPE[:2])
