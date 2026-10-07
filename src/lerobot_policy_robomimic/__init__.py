"""robomimic policies for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError as err:
    raise ImportError("lerobot is not installed. Please install lerobot to use this policy package.") from err

from .envs.configs import RobomimicEnvConfig
from .policies.bc.configuration_robomimic_bc import RobomimicBCConfig
from .policies.bc.modeling_robomimic_bc import RobomimicBCPolicy
from .policies.bc.processor_robomimic_bc import make_robomimic_bc_pre_post_processors
from .policies.bc_rnn.configuration_robomimic_bc_rnn import RobomimicBCRNNConfig
from .policies.bc_rnn.modeling_robomimic_bc_rnn import RobomimicBCRNNPolicy
from .policies.bc_rnn.processor_robomimic_bc_rnn import make_robomimic_bc_rnn_pre_post_processors
from .policies.bc_transformer.configuration_robomimic_bc_transformer import RobomimicBCTransformerConfig
from .policies.bc_transformer.modeling_robomimic_bc_transformer import RobomimicBCTransformerPolicy
from .policies.bc_transformer.processor_robomimic_bc_transformer import (
    make_robomimic_bc_transformer_pre_post_processors,
)
from .policies.bc_vae.configuration_robomimic_bc_vae import RobomimicBCVAEConfig
from .policies.bc_vae.modeling_robomimic_bc_vae import RobomimicBCVAEPolicy
from .policies.bc_vae.processor_robomimic_bc_vae import make_robomimic_bc_vae_pre_post_processors

__all__ = [
    "RobomimicBCConfig",
    "RobomimicBCPolicy",
    "RobomimicBCRNNConfig",
    "RobomimicBCRNNPolicy",
    "RobomimicBCTransformerConfig",
    "RobomimicBCTransformerPolicy",
    "RobomimicBCVAEConfig",
    "RobomimicBCVAEPolicy",
    "RobomimicEnvConfig",
    "make_robomimic_bc_pre_post_processors",
    "make_robomimic_bc_rnn_pre_post_processors",
    "make_robomimic_bc_transformer_pre_post_processors",
    "make_robomimic_bc_vae_pre_post_processors",
]
