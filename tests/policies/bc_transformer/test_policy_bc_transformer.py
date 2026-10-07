import pytest
import robomimic_reference
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION

from lerobot_policy_robomimic import RobomimicBCTransformerConfig, RobomimicBCTransformerPolicy

CONTEXT = 10


def make_policy(**kwargs) -> RobomimicBCTransformerPolicy:
    config = RobomimicBCTransformerConfig(
        device="cpu",
        input_features=robomimic_reference.image_experiment_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        **kwargs,
    )
    return RobomimicBCTransformerPolicy(config)


def matching_pair(gmm: bool):
    torch.manual_seed(0)
    reference = robomimic_reference.bc_transformer_algo(
        robomimic_reference.image_experiment_obs_shapes(), action_dim=7, gmm=gmm
    )
    policy = make_policy(use_gmm=gmm)
    robomimic_reference.load_policy(policy, reference.nets["policy"])
    return policy, reference


def context_losses(policy, reference):
    obs, batch = robomimic_reference.random_image_observations(2, CONTEXT)
    actions = torch.rand(2, 7) * 2 - 1
    reference_batch = {"obs": obs, "goal_obs": None, "actions": actions}
    ours = policy.forward(batch | {ACTION: actions})
    theirs = reference._compute_losses(reference._forward_training(reference_batch), reference_batch)
    return ours, theirs


def test_gmm_loss_matches_robomimic_bc_transformer_gmm():
    policy, reference = matching_pair(gmm=True)
    policy.eval()
    policy.action_head.train()
    reference.set_eval()

    (loss, logs), reference_losses = context_losses(policy, reference)

    torch.testing.assert_close(loss, reference_losses["action_loss"])
    assert logs["log_probs"] == pytest.approx(reference_losses["log_probs"].item(), rel=1e-5)


def test_deterministic_loss_matches_robomimic_bc_transformer():
    policy, reference = matching_pair(gmm=False)
    policy.eval()
    reference.set_eval()

    (loss, logs), reference_losses = context_losses(policy, reference)

    torch.testing.assert_close(loss, reference_losses["action_loss"])
    assert logs["l2_loss"] == pytest.approx(reference_losses["l2_loss"].item(), rel=1e-5)


def test_gmm_distribution_at_the_last_step_matches_robomimic():
    policy, reference = matching_pair(gmm=True)
    policy.eval()
    reference.set_eval()
    obs, batch = robomimic_reference.random_image_observations(2, CONTEXT)

    ours = policy.action_head(policy.features(batch))
    theirs = reference.nets["policy"].forward_train(obs)

    torch.testing.assert_close(
        ours.component_distribution.base_dist.loc, theirs.component_distribution.base_dist.loc[:, -1]
    )
    torch.testing.assert_close(
        ours.component_distribution.base_dist.scale, theirs.component_distribution.base_dist.scale[:, -1]
    )
    torch.testing.assert_close(ours.mixture_distribution.logits, theirs.mixture_distribution.logits[:, -1])


def test_actions_match_robomimic_on_padded_observation_windows():
    policy, reference = matching_pair(gmm=False)
    policy.eval()
    reference.set_eval()
    steps = [robomimic_reference.random_image_observations(2) for _ in range(13)]
    policy.reset()

    for t, (_, batch) in enumerate(steps):
        window = [steps[max(0, i)][0] for i in range(t - CONTEXT + 1, t + 1)]
        obs = {key: torch.stack([frame[key] for frame in window], dim=1) for key in window[0]}
        torch.testing.assert_close(policy.select_action(batch), reference.get_action(obs), msg=f"step {t}")


def test_predict_action_chunk_is_not_supported():
    _, batch = robomimic_reference.random_image_observations(1)

    with pytest.raises(NotImplementedError, match="select_action"):
        make_policy().predict_action_chunk(batch)


def test_save_and_load(tmp_path):
    torch.manual_seed(0)
    policy = make_policy(use_gmm=False).eval()
    policy.save_pretrained(tmp_path)
    loaded = RobomimicBCTransformerPolicy.from_pretrained(tmp_path).eval()
    _, batch = robomimic_reference.random_image_observations(2)

    torch.testing.assert_close(loaded.select_action(batch), policy.select_action(batch))
