import pytest
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.optim import AdamConfig, AdamWConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from lerobot_policy_robomimic import (
    RobomimicBCConfig,
    RobomimicBCRNNConfig,
    RobomimicBCTransformerConfig,
    RobomimicBCVAEConfig,
)
from lerobot_policy_robomimic.policies.common.schedulers import RobomimicLinearSchedulerConfig

STATE = PolicyFeature(type=FeatureType.STATE, shape=(9,))
ENV_STATE = PolicyFeature(type=FeatureType.ENV, shape=(10,))
IMAGE = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84))
ACTION_FEATURE = PolicyFeature(type=FeatureType.ACTION, shape=(7,))


@pytest.fixture(
    params=[RobomimicBCConfig, RobomimicBCRNNConfig, RobomimicBCTransformerConfig, RobomimicBCVAEConfig]
)
def config_class(request):
    return request.param


def make_config(config_class=RobomimicBCConfig, **kwargs):
    inputs = {
        OBS_STATE: STATE,
        OBS_ENV_STATE: ENV_STATE,
        f"{OBS_IMAGES}.agentview": IMAGE,
    }
    return config_class(
        device="cpu",
        input_features=kwargs.pop("input_features", inputs),
        output_features={ACTION: ACTION_FEATURE},
        **kwargs,
    )


def test_image_experiment_ignores_env_state(config_class):
    config = make_config(config_class)
    config.validate_features()

    assert list(config.observation_features) == [OBS_STATE, f"{OBS_IMAGES}.agentview"]


def test_low_dim_experiment_reads_env_state(config_class):
    config = make_config(
        config_class, input_features={OBS_STATE: STATE, OBS_ENV_STATE: ENV_STATE}, use_env_state=True
    )
    config.validate_features()

    assert list(config.observation_features) == [OBS_STATE, OBS_ENV_STATE]


def test_use_env_state_requires_the_feature(config_class):
    with pytest.raises(ValueError, match="use_env_state"):
        make_config(config_class, input_features={OBS_STATE: STATE}, use_env_state=True).validate_features()


def test_requires_an_input_it_reads(config_class):
    with pytest.raises(ValueError, match="at least one input"):
        make_config(config_class, input_features={OBS_ENV_STATE: ENV_STATE}).validate_features()


def test_crop_must_fit_inside_images(config_class):
    with pytest.raises(ValueError, match="crop_shape"):
        make_config(config_class, crop_shape=(84, 76)).validate_features()


def test_reads_one_observation_per_step(config_class):
    with pytest.raises(ValueError, match="n_obs_steps"):
        make_config(config_class, n_obs_steps=2)


@pytest.mark.parametrize("config_class", [RobomimicBCConfig, RobomimicBCRNNConfig, RobomimicBCVAEConfig])
def test_bc_bc_rnn_and_bc_vae_use_constant_adam(config_class):
    optimizer = make_config(config_class).get_optimizer_preset()

    assert optimizer == AdamConfig(lr=1e-4, weight_decay=0.0, grad_clip_norm=0.0)
    assert make_config(config_class).get_scheduler_preset() is None


def test_bc_transformer_uses_adamw_with_a_linear_decay():
    config = make_config(RobomimicBCTransformerConfig)

    assert config.get_optimizer_preset() == AdamWConfig(lr=1e-4, weight_decay=0.01, grad_clip_norm=0.0)
    assert config.get_scheduler_preset() == RobomimicLinearSchedulerConfig(
        decay_epochs=100, decay_factor=0.1, steps_per_epoch=100
    )


@pytest.mark.parametrize("config_class", [RobomimicBCRNNConfig, RobomimicBCTransformerConfig])
def test_only_bc_has_a_gaussian_head(config_class):
    with pytest.raises(ValueError, match="no Gaussian"):
        make_config(config_class, use_gaussian=True)


@pytest.mark.parametrize("config_class", [RobomimicBCConfig, RobomimicBCVAEConfig])
def test_bc_and_bc_vae_read_single_steps(config_class):
    config = make_config(config_class)

    assert config.observation_delta_indices is None
    assert config.action_delta_indices is None


def test_bc_rnn_reads_sequences_of_rnn_horizon():
    config = make_config(RobomimicBCRNNConfig, rnn_horizon=4)

    assert config.observation_delta_indices == [0, 1, 2, 3]
    assert config.action_delta_indices == [0, 1, 2, 3]


def test_bc_transformer_reads_its_context_and_one_action():
    config = make_config(RobomimicBCTransformerConfig, transformer_context_length=3)

    assert config.observation_delta_indices == [-2, -1, 0]
    assert config.action_delta_indices is None


def test_bc_transformer_heads_must_divide_the_width():
    with pytest.raises(ValueError, match="divisible"):
        make_config(RobomimicBCTransformerConfig, transformer_embed_dim=100, transformer_num_heads=8)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"vae_prior_is_conditioned": True}, "vae_prior_is_conditioned needs vae_prior_learn"),
        ({"vae_prior_use_gmm": True}, "vae_prior_use_gmm needs vae_prior_learn"),
    ],
)
def test_bc_vae_rejects_what_robomimic_rejects(settings, message):
    with pytest.raises(ValueError, match=message):
        make_config(RobomimicBCVAEConfig, **settings)
