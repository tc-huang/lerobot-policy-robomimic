import importlib

import pytest
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.utils.import_utils import register_third_party_plugins

from lerobot_policy_robomimic import (
    RobomimicBCConfig,
    RobomimicBCPolicy,
    RobomimicBCRNNConfig,
    RobomimicBCRNNPolicy,
    RobomimicBCTransformerConfig,
    RobomimicBCTransformerPolicy,
    RobomimicBCVAEConfig,
    RobomimicBCVAEPolicy,
)

POLICIES = [
    ("robomimic_bc", RobomimicBCConfig, RobomimicBCPolicy),
    ("robomimic_bc_rnn", RobomimicBCRNNConfig, RobomimicBCRNNPolicy),
    ("robomimic_bc_transformer", RobomimicBCTransformerConfig, RobomimicBCTransformerPolicy),
    ("robomimic_bc_vae", RobomimicBCVAEConfig, RobomimicBCVAEPolicy),
]


@pytest.mark.parametrize(("policy_type", "config_class", "policy_class"), POLICIES)
def test_lerobot_discovers_policy(policy_type, config_class, policy_class):
    register_third_party_plugins()

    assert PreTrainedConfig.get_choice_class(policy_type) is config_class
    assert get_policy_class(policy_type) is policy_class


@pytest.mark.parametrize(("policy_type", "config_class", "policy_class"), POLICIES)
def test_processor_follows_naming_convention(policy_type, config_class, policy_class):
    module = importlib.import_module(config_class.__module__.replace("configuration_", "processor_"))

    assert callable(getattr(module, f"make_{policy_type}_pre_post_processors"))
