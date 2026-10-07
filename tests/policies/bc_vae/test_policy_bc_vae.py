import pytest
import robomimic_reference
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION

from lerobot_policy_robomimic import RobomimicBCVAEConfig, RobomimicBCVAEPolicy

PRIOR = {"learn": True, "is_conditioned": True, "use_gmm": True}


def make_policy(**kwargs) -> RobomimicBCVAEPolicy:
    config = RobomimicBCVAEConfig(
        device="cpu",
        input_features=robomimic_reference.image_experiment_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        **kwargs,
    )
    return RobomimicBCVAEPolicy(config)


def matching_pair():
    torch.manual_seed(0)
    reference = robomimic_reference.make_algo(
        robomimic_reference.bc_vae_config(kl_weight=0.5, prior=PRIOR),
        robomimic_reference.image_experiment_obs_shapes(),
        action_dim=7,
    )
    policy = make_policy(vae_kl_weight=0.5, **{f"vae_prior_{key}": value for key, value in PRIOR.items()})
    robomimic_reference.load_vae(policy.vae, reference.nets["policy"]._vae)
    policy.eval()
    reference.set_eval()
    return policy, reference


def test_loss_matches_robomimic_bc_vae():
    policy, reference = matching_pair()
    obs, batch = robomimic_reference.random_image_observations(4)
    actions = torch.rand(4, 7) * 2 - 1
    reference_batch = {"obs": obs, "goal_obs": None, "actions": actions}

    torch.manual_seed(1)
    loss, logs = policy.forward(batch | {ACTION: actions})
    torch.manual_seed(1)
    predictions = reference._forward_training(reference_batch)
    reference_losses = reference._compute_losses(predictions, reference_batch)

    torch.testing.assert_close(loss, reference_losses["action_loss"])
    assert logs["reconstruction_loss"] == pytest.approx(reference_losses["recons_loss"].item(), rel=1e-5)
    assert logs["kl_loss"] == pytest.approx(reference_losses["kl_loss"].item(), rel=1e-5)
    assert logs["encoder_variance"] == pytest.approx(predictions["encoder_variance"].mean().item(), rel=1e-5)


def test_actions_match_robomimic_bc_vae():
    policy, reference = matching_pair()
    obs, batch = robomimic_reference.random_image_observations(4)

    torch.manual_seed(1)
    ours = policy.select_action(batch)
    torch.manual_seed(1)
    theirs = reference.get_action(obs)

    torch.testing.assert_close(ours, theirs)
    torch.manual_seed(1)
    torch.testing.assert_close(policy.predict_action_chunk(batch), theirs.unsqueeze(1))


def test_save_and_load(tmp_path):
    torch.manual_seed(0)
    policy = make_policy().eval()
    policy.save_pretrained(tmp_path)
    loaded = RobomimicBCVAEPolicy.from_pretrained(tmp_path).eval()
    _, batch = robomimic_reference.random_image_observations(2)

    torch.manual_seed(1)
    expected = policy.select_action(batch)
    torch.manual_seed(1)
    torch.testing.assert_close(loaded.select_action(batch), expected)
