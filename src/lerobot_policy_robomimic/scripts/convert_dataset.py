"""Converts a robomimic hdf5 dataset into a LeRobotDataset.

Usage:
    robomimic-convert-dataset \\
        --hdf5 data/robomimic/lift/ph/image_v15.hdf5 \\
        --repo-id tc-huang/robomimic_lift_ph_image \\
        --task "lift the cube"
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from ..dataset_format import DEFAULT_STATE_KEYS, ENV_ARGS_FILE, ENV_STATE_KEY, IMAGE_SUFFIX, MASKS_FILE

# Above this many bytes of raw pixels, images are stored as MP4 videos instead of lossless PNG images.
VIDEO_THRESHOLD_BYTES = 2 * 1024**3


def sorted_demo_keys(data: h5py.Group) -> list[str]:
    """Returns demo keys in numeric order, so `demo_10` comes after `demo_2`."""
    return sorted(data, key=lambda key: int(key.removeprefix("demo_")))


def image_bytes(data: h5py.Group, image_keys: list[str]) -> int:
    """Returns the raw pixel size of every camera frame in the dataset."""
    demo = data[sorted_demo_keys(data)[0]]
    per_frame = sum(int(np.prod(demo["obs"][key].shape[1:])) for key in image_keys)
    return int(data.attrs["total"]) * per_frame


def use_videos(video: str, num_bytes: int) -> bool:
    """Resolves the `--video` choice; `auto` picks videos only for large datasets."""
    if video == "auto":
        return num_bytes > VIDEO_THRESHOLD_BYTES
    return video == "on"


def make_features(demo: h5py.Group, state_keys: Sequence[str], image_keys: list[str], videos: bool) -> dict:
    obs = demo["obs"]
    state_names = [f"{key}.{i}" for key in state_keys for i in range(obs[key].shape[1])]
    features = {
        OBS_STATE: {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        ACTION: {"dtype": "float32", "shape": (demo["actions"].shape[1],), "names": None},
    }
    if ENV_STATE_KEY in obs:
        features[OBS_ENV_STATE] = {"dtype": "float32", "shape": (obs[ENV_STATE_KEY].shape[1],), "names": None}
    for key in image_keys:
        camera = key.removesuffix(IMAGE_SUFFIX)
        features[f"{OBS_IMAGES}.{camera}"] = {
            "dtype": "video" if videos else "image",
            "shape": obs[key].shape[1:],
            "names": ["height", "width", "channels"],
        }
    return features


def convert(
    hdf5_path: Path,
    repo_id: str,
    task: str,
    root: Path | None = None,
    state_keys: Sequence[str] = DEFAULT_STATE_KEYS,
    video: str = "auto",
) -> LeRobotDataset:
    """Writes every demo of `hdf5_path` as one episode, in numeric demo order.

    The robomimic masks and the robosuite settings the demos were recorded with are kept next
    to LeRobot's metadata, in `MASKS_FILE` and `ENV_ARGS_FILE`. The env args also keep `task`
    under "lang", where robomimic looks for an env's language instruction.
    """
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        env_args = json.loads(data.attrs["env_args"])
        demo_keys = sorted_demo_keys(data)
        first = data[demo_keys[0]]
        image_keys = sorted(key for key in first["obs"] if key.endswith(IMAGE_SUFFIX))
        videos = bool(image_keys) and use_videos(video, image_bytes(data, image_keys))

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=env_args["env_kwargs"]["control_freq"],
            features=make_features(first, state_keys, image_keys, videos),
            root=root,
            robot_type=",".join(env_args["env_kwargs"]["robots"]).lower(),
            use_videos=videos,
            image_writer_threads=4 if image_keys else 0,
        )
        for demo_key in demo_keys:
            demo = data[demo_key]
            obs = demo["obs"]
            state = np.concatenate([obs[key][()] for key in state_keys], axis=1).astype(np.float32)
            actions = demo["actions"][()].astype(np.float32)
            env_state = obs[ENV_STATE_KEY][()].astype(np.float32) if ENV_STATE_KEY in obs else None
            images = {key.removesuffix(IMAGE_SUFFIX): obs[key][()] for key in image_keys}
            for t in range(len(actions)):
                frame = {OBS_STATE: state[t], ACTION: actions[t], "task": task}
                if env_state is not None:
                    frame[OBS_ENV_STATE] = env_state[t]
                for camera, frames in images.items():
                    frame[f"{OBS_IMAGES}.{camera}"] = frames[t]
                dataset.add_frame(frame)
            dataset.save_episode()
        dataset.finalize()

        episode_index = {key: i for i, key in enumerate(demo_keys)}
        masks = {
            name: sorted(episode_index[key.decode()] for key in f["mask"][name][()])
            for name in f.get("mask", {})
        }
    masks_path = dataset.root / MASKS_FILE
    masks_path.write_text(json.dumps({"demo_keys": demo_keys, "masks": masks}, indent=2) + "\n")
    (dataset.root / ENV_ARGS_FILE).write_text(json.dumps(env_args | {"lang": task}, indent=2) + "\n")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hdf5", type=Path, required=True, help="robomimic hdf5 file with observations")
    parser.add_argument("--repo-id", required=True, help="LeRobotDataset repo id, e.g. user/name")
    parser.add_argument("--task", required=True, help="task string stored with every frame")
    parser.add_argument("--root", type=Path, help="output directory; defaults to $HF_LEROBOT_HOME/<repo-id>")
    parser.add_argument("--state-keys", nargs="+", default=list(DEFAULT_STATE_KEYS))
    parser.add_argument(
        "--video",
        choices=["auto", "on", "off"],
        default="auto",
        help=f"store cameras as MP4; auto uses MP4 above {VIDEO_THRESHOLD_BYTES / 1024**3:.0f} GiB of raw pixels",
    )
    args = parser.parse_args()
    dataset = convert(args.hdf5, args.repo_id, args.task, args.root, args.state_keys, args.video)
    print(
        f"Wrote {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames to {dataset.root}"
    )


if __name__ == "__main__":
    main()
