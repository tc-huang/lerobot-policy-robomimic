import importlib

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.utils.import_utils import register_third_party_plugins

from lerobot_policy_robomimic import RobomimicBCConfig, RobomimicBCPolicy


def test_lerobot_discovers_policy():
    register_third_party_plugins()

    assert PreTrainedConfig.get_choice_class("robomimic_bc") is RobomimicBCConfig
    assert get_policy_class("robomimic_bc") is RobomimicBCPolicy


def test_processor_follows_naming_convention():
    module = importlib.import_module(RobomimicBCConfig.__module__.replace("configuration_", "processor_"))

    assert callable(module.make_robomimic_bc_pre_post_processors)
