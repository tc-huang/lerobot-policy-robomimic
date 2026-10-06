"""A gymnasium env around the robosuite simulator a robomimic dataset was recorded in."""

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

OBJECT_STATE_KEY = "object-state"


class RobomimicEnv(gym.Env):
    """Rebuilds a robosuite env from a robomimic dataset's `env_args`, as robomimic's rollouts do.

    Observations follow LeRobot's env convention: `pixels` holds one upright (H, W, 3) image per
    camera, `agent_pos` the `state_keys` concatenated, and `environment_state` the object state.
    An episode ends when the task succeeds or after `episode_length` steps.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_args: dict[str, Any],
        camera_names: Sequence[str],
        camera_height: int,
        camera_width: int,
        state_keys: Sequence[str],
        episode_length: int,
    ):
        try:
            import robosuite
        except ImportError as err:
            raise ImportError("robosuite is not installed; install it with `uv sync --extra sim`.") from err

        kwargs = dict(env_args["env_kwargs"])
        kwargs.update(
            has_renderer=False,
            has_offscreen_renderer=bool(camera_names),
            use_camera_obs=bool(camera_names),
            use_object_obs=True,
            ignore_done=True,
            camera_depths=False,
        )
        if camera_names:
            kwargs.update(
                camera_names=list(camera_names), camera_heights=camera_height, camera_widths=camera_width
            )
        else:
            for key in ("camera_names", "camera_heights", "camera_widths"):
                kwargs.pop(key, None)
        self._env = robosuite.make(env_args["env_name"], **kwargs)
        self.camera_names = list(camera_names)
        self.state_keys = list(state_keys)
        self.task_description = env_args["env_name"]
        self._max_episode_steps = episode_length

        low, high = self._env.action_spec
        self.action_space = spaces.Box(
            low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32
        )
        sample = self._observation(self._env.reset())
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        camera: spaces.Box(0, 255, shape=image.shape, dtype=np.uint8)
                        for camera, image in sample["pixels"].items()
                    }
                ),
                "agent_pos": spaces.Box(-np.inf, np.inf, shape=sample["agent_pos"].shape, dtype=np.float64),
                "environment_state": spaces.Box(
                    -np.inf, np.inf, shape=sample["environment_state"].shape, dtype=np.float64
                ),
            }
        )

    def _observation(self, raw: dict[str, np.ndarray]) -> dict[str, Any]:
        return {
            "pixels": {camera: raw[f"{camera}_image"][::-1].copy() for camera in self.camera_names},
            "agent_pos": np.concatenate([raw[key] for key in self.state_keys]).astype(np.float64),
            "environment_state": np.asarray(raw[OBJECT_STATE_KEY], dtype=np.float64),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Starts an episode from robosuite's random initial state.

        robosuite 1.5.1 places objects with NumPy's global generator, so a seed is applied there.
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        return self._observation(self._env.reset()), {"is_success": False}

    def step(self, action: np.ndarray):
        raw, reward, _, _ = self._env.step(action)
        is_success = bool(self._env._check_success())
        return self._observation(raw), reward, is_success, False, {"is_success": is_success}

    def render(self) -> np.ndarray | None:
        """Returns the first camera's upright image, if any camera is rendered."""
        if not self.camera_names:
            return None
        raw = self._env._get_observations(force_update=True)
        return raw[f"{self.camera_names[0]}_image"][::-1].copy()

    def close(self) -> None:
        self._env.close()
