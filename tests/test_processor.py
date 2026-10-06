import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from lerobot_policy_robomimic import RobomimicBCConfig

STATS = {
    OBS_STATE: {"mean": torch.full((9,), 3.0), "std": torch.full((9,), 2.0)},
    ACTION: {"mean": torch.full((7,), 3.0), "std": torch.full((7,), 2.0)},
}


def make_config() -> RobomimicBCConfig:
    return RobomimicBCConfig(
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(9,)),
            f"{OBS_IMAGES}.agentview": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )


def test_pipelines_keep_values_unnormalized():
    preprocessor, postprocessor = make_pre_post_processors(make_config(), dataset_stats=STATS)
    observation = {OBS_STATE: torch.randn(9), f"{OBS_IMAGES}.agentview": torch.rand(3, 84, 84)}
    action = torch.rand(1, 7) * 2 - 1

    batch = preprocessor(observation)

    for key, value in observation.items():
        torch.testing.assert_close(batch[key], value.unsqueeze(0))
    torch.testing.assert_close(postprocessor(action), action)
