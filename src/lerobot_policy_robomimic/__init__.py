"""robomimic policies for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError as err:
    raise ImportError("lerobot is not installed. Please install lerobot to use this policy package.") from err

from .configuration_robomimic_bc import RobomimicBCConfig
from .modeling_robomimic_bc import RobomimicBCPolicy
from .processor_robomimic_bc import make_robomimic_bc_pre_post_processors

__all__ = [
    "RobomimicBCConfig",
    "RobomimicBCPolicy",
    "make_robomimic_bc_pre_post_processors",
]
