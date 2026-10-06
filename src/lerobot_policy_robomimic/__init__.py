"""robomimic policies for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError as err:
    raise ImportError("lerobot is not installed. Please install lerobot to use this policy package.") from err

from .configuration_robomimic_bc import RobomimicBCConfig
from .configuration_robomimic_bc_rnn import RobomimicBCRNNConfig
from .modeling_robomimic_bc import RobomimicBCPolicy
from .modeling_robomimic_bc_rnn import RobomimicBCRNNPolicy
from .processor_robomimic_bc import make_robomimic_bc_pre_post_processors
from .processor_robomimic_bc_rnn import make_robomimic_bc_rnn_pre_post_processors

__all__ = [
    "RobomimicBCConfig",
    "RobomimicBCPolicy",
    "RobomimicBCRNNConfig",
    "RobomimicBCRNNPolicy",
    "make_robomimic_bc_pre_post_processors",
    "make_robomimic_bc_rnn_pre_post_processors",
]
