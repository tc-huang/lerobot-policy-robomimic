import pytest
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.optim import AdamConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from lerobot_policy_robomimic import RobomimicBCConfig

STATE = PolicyFeature(type=FeatureType.STATE, shape=(9,))
ENV_STATE = PolicyFeature(type=FeatureType.ENV, shape=(10,))
IMAGE = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84))
ACTION_FEATURE = PolicyFeature(type=FeatureType.ACTION, shape=(7,))


def make_config(**kwargs):
    inputs = {
        OBS_STATE: STATE,
        OBS_ENV_STATE: ENV_STATE,
        f"{OBS_IMAGES}.agentview": IMAGE,
    }
    return RobomimicBCConfig(
        device="cpu",
        input_features=kwargs.pop("input_features", inputs),
        output_features={ACTION: ACTION_FEATURE},
        **kwargs,
    )


def test_image_experiment_ignores_env_state():
    config = make_config()
    config.validate_features()

    assert list(config.observation_features) == [OBS_STATE, f"{OBS_IMAGES}.agentview"]


def test_low_dim_experiment_reads_env_state():
    config = make_config(input_features={OBS_STATE: STATE, OBS_ENV_STATE: ENV_STATE}, use_env_state=True)
    config.validate_features()

    assert list(config.observation_features) == [OBS_STATE, OBS_ENV_STATE]


def test_use_env_state_requires_the_feature():
    with pytest.raises(ValueError, match="use_env_state"):
        make_config(input_features={OBS_STATE: STATE}, use_env_state=True).validate_features()


def test_requires_an_input_it_reads():
    with pytest.raises(ValueError, match="at least one input"):
        make_config(input_features={OBS_ENV_STATE: ENV_STATE}).validate_features()


def test_crop_must_fit_inside_images():
    with pytest.raises(ValueError, match="crop_shape"):
        make_config(crop_shape=(84, 76)).validate_features()


def test_bc_reads_a_single_observation():
    with pytest.raises(ValueError, match="n_obs_steps"):
        make_config(n_obs_steps=2)


def test_optimizer_matches_robomimic():
    optimizer = make_config().get_optimizer_preset()

    assert optimizer == AdamConfig(lr=1e-4, weight_decay=0.0, grad_clip_norm=0.0)
    assert make_config().get_scheduler_preset() is None
