"""Names that the converters and the env share for robomimic data and converted datasets."""

# Proprioceptive keys used by every robomimic paper config (scripts/generate_paper_configs.py).
DEFAULT_STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
ENV_STATE_KEY = "object"
IMAGE_SUFFIX = "_image"

MASKS_FILE = "meta/robomimic_masks.json"
ENV_ARGS_FILE = "meta/robomimic_env_args.json"
