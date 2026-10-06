import pytest
import robomimic_reference
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION

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
from lerobot_policy_robomimic.language import OBS_LANGUAGE_EMBEDDING
from lerobot_policy_robomimic.observation_encoder import ObservationEncoder

LANGUAGE_DIM = 768
POLICIES = [
    (RobomimicBCConfig, RobomimicBCPolicy),
    (RobomimicBCRNNConfig, RobomimicBCRNNPolicy),
    (RobomimicBCTransformerConfig, RobomimicBCTransformerPolicy),
    (RobomimicBCVAEConfig, RobomimicBCVAEPolicy),
]


def obs_shapes() -> dict[str, list[int]]:
    return robomimic_reference.image_experiment_obs_shapes() | {robomimic_reference.LANG_EMB: [LANGUAGE_DIM]}


def make_policy(config_class, policy_class, **kwargs):
    config = config_class(
        device="cpu",
        input_features=robomimic_reference.image_experiment_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        **kwargs,
    )
    return policy_class(config)


def test_config_reads_the_embedding_last():
    config = make_policy(RobomimicBCConfig, RobomimicBCPolicy, language_conditioning="concat").config

    assert list(config.observation_features)[-1] == OBS_LANGUAGE_EMBEDDING
    assert config.observation_features[OBS_LANGUAGE_EMBEDDING] == PolicyFeature(
        type=FeatureType.LANGUAGE, shape=(LANGUAGE_DIM,)
    )


def test_needs_an_observation_besides_the_task():
    config = RobomimicBCConfig(
        device="cpu",
        input_features={},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        language_conditioning="concat",
    )

    with pytest.raises(ValueError, match="besides the task"):
        config.validate_features()


def test_concat_encoder_matches_robomimic_up_to_feature_order():
    torch.manual_seed(0)
    reference = robomimic_reference.observation_encoder(obs_shapes())
    features = robomimic_reference.image_experiment_features() | {
        OBS_LANGUAGE_EMBEDDING: PolicyFeature(type=FeatureType.LANGUAGE, shape=(LANGUAGE_DIM,))
    }
    ours = ObservationEncoder(features, crop_shape=(76, 76), num_kp=32, image_feature_dim=64)
    for camera in robomimic_reference.CAMERAS:
        robomimic_reference.load_visual_core(
            ours.cameras[camera].encoder, reference.obs_nets[f"{camera}_image"]
        )
    obs, batch = robomimic_reference.random_image_observations(4)
    language_obs, language_batch = robomimic_reference.random_language(4)

    expected = robomimic_reference.lerobot_feature_order(reference.eval()(obs | language_obs), LANGUAGE_DIM)

    assert ours.output_dim == 9 + 64 * len(robomimic_reference.CAMERAS) + LANGUAGE_DIM
    torch.testing.assert_close(ours.eval()(batch | language_batch), expected)


def test_concat_bc_rnn_loss_matches_robomimic():
    torch.manual_seed(0)
    config = robomimic_reference.with_language(robomimic_reference.bc_rnn_config())
    reference = robomimic_reference.make_algo(config, obs_shapes(), action_dim=7)
    policy = make_policy(RobomimicBCRNNConfig, RobomimicBCRNNPolicy, language_conditioning="concat")
    robomimic_reference.load_policy(policy, reference.nets["policy"], LANGUAGE_DIM)
    policy.eval()
    policy.action_head.train()
    reference.set_eval()
    reference.nets["policy"].training = True
    obs, batch = robomimic_reference.random_image_observations(2, 10)
    language_obs, language_batch = robomimic_reference.random_language(2, 10)
    actions = torch.rand(2, 10, 7) * 2 - 1
    reference_batch = {"obs": obs | language_obs, "goal_obs": None, "actions": actions}

    loss, _ = policy.forward(batch | language_batch | {ACTION: actions})
    expected = reference._compute_losses(reference._forward_training(reference_batch), reference_batch)

    torch.testing.assert_close(loss, expected["action_loss"])


@pytest.mark.parametrize(("config_class", "policy_class"), POLICIES)
def test_every_policy_acts_on_the_task(config_class, policy_class):
    torch.manual_seed(0)
    policy = make_policy(config_class, policy_class, language_conditioning="concat").eval()
    _, batch = robomimic_reference.random_image_observations(2)
    _, first = robomimic_reference.random_language(2)
    _, second = robomimic_reference.random_language(2)

    torch.manual_seed(1)
    first_actions = policy.select_action(batch | first)
    policy.reset()
    torch.manual_seed(1)
    second_actions = policy.select_action(batch | second)

    assert first_actions.shape == (2, 7)
    assert not torch.allclose(first_actions, second_actions)
