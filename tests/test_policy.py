import pytest
import robomimic_reference
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_ENV_STATE

from lerobot_policy_robomimic import RobomimicBCConfig, RobomimicBCPolicy

LOSS_WEIGHTS = {"l2_weight": 1.0, "l1_weight": 0.5, "cos_weight": 0.3}


def make_policy(features=None, **kwargs) -> RobomimicBCPolicy:
    config = RobomimicBCConfig(
        device="cpu",
        input_features=features or robomimic_reference.image_experiment_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        **kwargs,
    )
    return RobomimicBCPolicy(config)


def matching_pair(gmm: bool, **loss_weights):
    torch.manual_seed(0)
    reference = robomimic_reference.bc_algo(
        robomimic_reference.image_experiment_obs_shapes(), action_dim=7, gmm=gmm, **loss_weights
    )
    policy = make_policy(use_gmm=gmm, **loss_weights)
    robomimic_reference.load_policy(policy, reference.nets["policy"])
    return policy, reference


def losses(policy, reference, actions):
    obs, batch = robomimic_reference.random_image_observations(len(actions))
    reference_batch = {"obs": obs, "goal_obs": None, "actions": actions}
    ours = policy.forward(batch | {ACTION: actions})
    theirs = reference._compute_losses(reference._forward_training(reference_batch), reference_batch)
    return ours, theirs


def test_deterministic_head_matches_robomimic_bc():
    policy, reference = matching_pair(gmm=False, **LOSS_WEIGHTS)
    policy.eval()
    reference.set_eval()

    (loss, logs), reference_losses = losses(policy, reference, torch.rand(4, 7) * 2 - 1)

    torch.testing.assert_close(loss, reference_losses["action_loss"])
    for name in ("l2_loss", "l1_loss", "cos_loss"):
        assert logs[name] == pytest.approx(reference_losses[name].item(), rel=1e-5)
    obs, batch = robomimic_reference.random_image_observations(4)
    torch.testing.assert_close(policy.select_action(batch), reference.get_action(obs))
    torch.testing.assert_close(policy.predict_action_chunk(batch), reference.get_action(obs).unsqueeze(1))


def test_gmm_loss_matches_robomimic_bc_gmm():
    policy, reference = matching_pair(gmm=True)
    policy.eval()
    policy.action_head.train()
    reference.set_eval()
    reference.nets["policy"].training = True

    (loss, logs), reference_losses = losses(policy, reference, torch.rand(4, 7) * 2 - 1)

    torch.testing.assert_close(loss, reference_losses["action_loss"])
    assert logs["log_probs"] == pytest.approx(reference_losses["log_probs"].item(), rel=1e-5)


def test_gmm_actions_match_robomimic_bc_gmm():
    policy, reference = matching_pair(gmm=True)
    policy.eval()
    reference.set_eval()
    obs, batch = robomimic_reference.random_image_observations(4)

    torch.manual_seed(1)
    ours = policy.select_action(batch)
    torch.manual_seed(1)
    theirs = reference.get_action(obs)

    torch.testing.assert_close(ours, theirs)


def gaussian_pair(**gaussian):
    torch.manual_seed(0)
    reference = robomimic_reference.make_algo(
        robomimic_reference.bc_gaussian_config(**gaussian),
        robomimic_reference.image_experiment_obs_shapes(),
        action_dim=7,
    )
    fields = {f"gaussian_{name}": value for name, value in gaussian.items()}
    policy = make_policy(use_gaussian=True, **fields)
    robomimic_reference.load_policy(policy, reference.nets["policy"])
    return policy, reference


@pytest.mark.parametrize("fixed_std", [False, True])
def test_gaussian_loss_matches_robomimic_bc_gaussian(fixed_std):
    policy, reference = gaussian_pair(fixed_std=fixed_std)
    policy.eval()
    policy.action_head.train()
    reference.set_eval()
    reference.nets["policy"].training = True

    (loss, logs), reference_losses = losses(policy, reference, torch.rand(4, 7) * 2 - 1)

    torch.testing.assert_close(loss, reference_losses["action_loss"])
    assert logs["log_probs"] == pytest.approx(reference_losses["log_probs"].item(), rel=1e-5)


@pytest.mark.parametrize("low_noise_eval", [True, False])
def test_gaussian_actions_match_robomimic_bc_gaussian(low_noise_eval):
    policy, reference = gaussian_pair(low_noise_eval=low_noise_eval)
    policy.eval()
    reference.set_eval()
    obs, batch = robomimic_reference.random_image_observations(4)

    torch.manual_seed(1)
    ours = policy.select_action(batch)
    torch.manual_seed(1)
    theirs = reference.get_action(obs)

    torch.testing.assert_close(ours, theirs)


def test_image_policy_ignores_env_state():
    features = robomimic_reference.image_experiment_features()
    features[OBS_ENV_STATE] = PolicyFeature(type=FeatureType.ENV, shape=(10,))
    policy = make_policy(features, use_gmm=False).eval()
    _, batch = robomimic_reference.random_image_observations(2)

    first = policy.select_action(batch | {OBS_ENV_STATE: torch.zeros(2, 10)})
    second = policy.select_action(batch | {OBS_ENV_STATE: torch.ones(2, 10)})

    torch.testing.assert_close(first, second)


def test_save_and_load(tmp_path):
    torch.manual_seed(0)
    policy = make_policy().eval()
    policy.save_pretrained(tmp_path)
    loaded = RobomimicBCPolicy.from_pretrained(tmp_path).eval()
    _, batch = robomimic_reference.random_image_observations(2)

    torch.manual_seed(1)
    expected = policy.select_action(batch)
    torch.manual_seed(1)
    torch.testing.assert_close(loaded.select_action(batch), expected)
