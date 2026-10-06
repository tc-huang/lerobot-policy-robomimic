import pytest
import robomimic_reference
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_ENV_STATE

from lerobot_policy_robomimic import RobomimicBCConfig, RobomimicBCPolicy

LOSS_WEIGHTS = {"l2_weight": 1.0, "l1_weight": 0.5, "cos_weight": 0.3}


def make_policy(**kwargs) -> RobomimicBCPolicy:
    config = RobomimicBCConfig(
        device="cpu",
        input_features=robomimic_reference.image_experiment_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        **kwargs,
    )
    return RobomimicBCPolicy(config)


def test_matches_robomimic_bc():
    torch.manual_seed(0)
    reference = robomimic_reference.bc_algo(
        robomimic_reference.image_experiment_obs_shapes(), action_dim=7, **LOSS_WEIGHTS
    )
    reference.set_eval()
    policy = make_policy(**LOSS_WEIGHTS).eval()
    robomimic_reference.load_actor_network(policy.actor, reference.nets["policy"])
    obs, batch = robomimic_reference.random_image_observations(4)
    actions = torch.rand(4, 7) * 2 - 1
    reference_batch = {"obs": obs, "goal_obs": None, "actions": actions}

    loss, logs = policy.forward(batch | {ACTION: actions})
    reference_losses = reference._compute_losses(
        reference._forward_training(reference_batch), reference_batch
    )

    torch.testing.assert_close(loss, reference_losses["action_loss"])
    for name in ("l2_loss", "l1_loss", "cos_loss"):
        assert logs[name] == pytest.approx(reference_losses[name].item(), rel=1e-5)
    torch.testing.assert_close(policy.select_action(batch), reference.get_action(obs))
    torch.testing.assert_close(policy.predict_action_chunk(batch), reference.get_action(obs).unsqueeze(1))


def test_image_policy_ignores_env_state():
    torch.manual_seed(0)
    features = robomimic_reference.image_experiment_features()
    features[OBS_ENV_STATE] = PolicyFeature(type=FeatureType.ENV, shape=(10,))
    policy = RobomimicBCPolicy(
        RobomimicBCConfig(
            device="cpu",
            input_features=features,
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        )
    ).eval()
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

    torch.testing.assert_close(loaded.select_action(batch), policy.select_action(batch))
