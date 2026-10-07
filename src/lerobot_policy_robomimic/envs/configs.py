"""The `robomimic` env type: robosuite rebuilt from a converted dataset's env args."""

import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.envs.utils import freeze_after_episode_end
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from ..dataset_format import DEFAULT_STATE_KEYS

# Rollout horizons of robomimic's proficient-human datasets, by robosuite env name.
ROLLOUT_HORIZONS = {
    "Lift": 400,
    "PickPlaceCan": 400,
    "NutAssemblySquare": 400,
    "TwoArmTransport": 700,
    "ToolHang": 700,
}


@EnvConfig.register_subclass("robomimic")
@dataclass
class RobomimicEnvConfig(EnvConfig):
    """robosuite env of a robomimic dataset, rebuilt from the env args its demos were recorded with.

    Building the config makes one robosuite env without cameras to read the action and
    observation sizes, so robosuite must be installed.

    Args:
        env_args_path: `meta/robomimic_env_args.json` of a dataset converted by `convert_dataset`.
        episode_length: Steps before an episode is cut off; defaults to robomimic's rollout horizon.
        camera_names: Cameras to render as `observation.images.<camera>`; defaults to the env args'.
        camera_height: Height of the rendered images; defaults to the env args' or 84.
        camera_width: Width of the rendered images; defaults to the env args' or 84.
        state_keys: robosuite observations concatenated into `observation.state`, as in the dataset.
    """

    env_args_path: str | None = None
    episode_length: int | None = None
    camera_names: list[str] | None = None
    camera_height: int | None = None
    camera_width: int | None = None
    state_keys: list[str] = field(default_factory=lambda: list(DEFAULT_STATE_KEYS))

    def __post_init__(self) -> None:
        if self.env_args_path is None:
            raise ValueError("Set env_args_path to the meta/robomimic_env_args.json of a converted dataset.")
        env_args = json.loads(Path(self.env_args_path).read_text())
        env_kwargs = env_args["env_kwargs"]
        self.task = env_args["env_name"]
        self.fps = env_kwargs["control_freq"]
        if self.episode_length is None:
            if self.task not in ROLLOUT_HORIZONS:
                raise ValueError(f"Set episode_length: robomimic has no rollout horizon for {self.task}.")
            self.episode_length = ROLLOUT_HORIZONS[self.task]
        if self.camera_names is None:
            self.camera_names = list(env_kwargs.get("camera_names") or [])
        self.camera_height = self.camera_height or env_kwargs.get("camera_heights") or 84
        self.camera_width = self.camera_width or env_kwargs.get("camera_widths") or 84

        probe = self.make_env(env_args, camera_names=[])
        self.features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=probe.action_space.shape),
            "agent_pos": PolicyFeature(
                type=FeatureType.STATE, shape=probe.observation_space["agent_pos"].shape
            ),
            "environment_state": PolicyFeature(
                type=FeatureType.ENV, shape=probe.observation_space["environment_state"].shape
            ),
        }
        probe.close()
        self.features_map = {"action": ACTION, "agent_pos": OBS_STATE, "environment_state": OBS_ENV_STATE}
        for camera in self.camera_names:
            self.features[f"pixels/{camera}"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.camera_height, self.camera_width, 3)
            )
            self.features_map[f"pixels/{camera}"] = f"{OBS_IMAGES}.{camera}"
        self._env_args = env_args

    def make_env(self, env_args: dict[str, Any], camera_names: list[str] | None = None) -> gym.Env:
        """Builds one `RobomimicEnv` from `env_args` with this config's settings."""
        from .robosuite import RobomimicEnv

        assert self.camera_names is not None and self.episode_length is not None
        assert self.camera_height is not None and self.camera_width is not None
        return RobomimicEnv(
            env_args,
            self.camera_names if camera_names is None else camera_names,
            self.camera_height,
            self.camera_width,
            self.state_keys,
            self.episode_length,
        )

    @property
    def gym_kwargs(self) -> dict:
        return {}

    def create_envs(
        self, n_envs: int, use_async_envs: bool = False
    ) -> dict[str, dict[int, gym.vector.VectorEnv]]:
        """Builds `n_envs` envs that stop simulating once their episode ends, like LeRobot's own envs."""
        env_fns = [freeze_after_episode_end(partial(self.make_env, self._env_args)) for _ in range(n_envs)]
        if use_async_envs and n_envs > 1:
            vec_env = gym.vector.AsyncVectorEnv(
                env_fns, context="forkserver", autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
            )
        else:
            vec_env = gym.vector.SyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
        return {self.type: {0: vec_env}}
