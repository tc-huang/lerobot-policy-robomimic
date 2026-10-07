import pytest
import robomimic_reference
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION

from lerobot_policy_robomimic import RobomimicBCRNNConfig, RobomimicBCRNNPolicy


def make_policy(**kwargs) -> RobomimicBCRNNPolicy:
    config = RobomimicBCRNNConfig(
        device="cpu",
        input_features=robomimic_reference.image_experiment_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        **kwargs,
    )
    return RobomimicBCRNNPolicy(config)


def matching_pair(gmm: bool):
    torch.manual_seed(0)
    reference = robomimic_reference.bc_rnn_algo(
        robomimic_reference.image_experiment_obs_shapes(), action_dim=7, gmm=gmm
    )
    policy = make_policy(use_gmm=gmm)
    robomimic_reference.load_policy(policy, reference.nets["policy"])
    return policy, reference


def sequence_losses(policy, reference):
    obs, batch = robomimic_reference.random_image_observations(2, 10)
    actions = torch.rand(2, 10, 7) * 2 - 1
    reference_batch = {"obs": obs, "goal_obs": None, "actions": actions}
    ours = policy.forward(batch | {ACTION: actions})
    theirs = reference._compute_losses(reference._forward_training(reference_batch), reference_batch)
    return ours, theirs


def test_gmm_sequence_loss_matches_robomimic_bc_rnn_gmm():
    policy, reference = matching_pair(gmm=True)
    policy.eval()
    policy.action_head.train()
    reference.set_eval()
    reference.nets["policy"].training = True

    (loss, logs), reference_losses = sequence_losses(policy, reference)

    torch.testing.assert_close(loss, reference_losses["action_loss"])
    assert logs["log_probs"] == pytest.approx(reference_losses["log_probs"].item(), rel=1e-5)


def test_deterministic_sequence_loss_matches_robomimic_bc_rnn():
    policy, reference = matching_pair(gmm=False)
    policy.eval()
    reference.set_eval()

    (loss, logs), reference_losses = sequence_losses(policy, reference)

    torch.testing.assert_close(loss, reference_losses["action_loss"])
    assert logs["l2_loss"] == pytest.approx(reference_losses["l2_loss"].item(), rel=1e-5)


def test_actions_match_robomimic_across_state_resets():
    policy, reference = matching_pair(gmm=True)
    policy.eval()
    reference.set_eval()
    policy.reset()
    reference.reset()

    for step in range(25):
        obs, batch = robomimic_reference.random_image_observations(2)
        torch.manual_seed(step)
        ours = policy.select_action(batch)
        torch.manual_seed(step)
        theirs = reference.get_action(obs)
        torch.testing.assert_close(ours, theirs, msg=f"step {step}")


def test_lstm_state_resets_every_rnn_horizon():
    torch.manual_seed(0)
    policy = make_policy(use_gmm=False, rnn_horizon=3).eval()
    _, batch = robomimic_reference.random_image_observations(1)

    actions = [policy.select_action(batch) for _ in range(4)]

    assert not torch.allclose(actions[1], actions[0])
    torch.testing.assert_close(actions[3], actions[0])
    policy.reset()
    torch.testing.assert_close(policy.select_action(batch), actions[0])


def test_predict_action_chunk_is_not_supported():
    _, batch = robomimic_reference.random_image_observations(1)

    with pytest.raises(NotImplementedError, match="select_action"):
        make_policy().predict_action_chunk(batch)


def test_save_and_load(tmp_path):
    torch.manual_seed(0)
    policy = make_policy(use_gmm=False).eval()
    policy.save_pretrained(tmp_path)
    loaded = RobomimicBCRNNPolicy.from_pretrained(tmp_path).eval()
    _, batch = robomimic_reference.random_image_observations(2)

    torch.testing.assert_close(loaded.select_action(batch), policy.select_action(batch))


def share_one_buffer(lstm: torch.nn.LSTM) -> None:
    """Makes every LSTM weight a view of one buffer, as cuDNN's `flatten_parameters` does on CUDA."""
    buffer = torch.cat([parameter.detach().flatten() for parameter in lstm.parameters()])
    offset = 0
    for parameter in lstm.parameters():
        parameter.data = buffer[offset : offset + parameter.numel()].view_as(parameter)
        offset += parameter.numel()


def test_save_and_load_with_cudnn_weights(tmp_path):
    torch.manual_seed(0)
    policy = make_policy().eval()
    share_one_buffer(policy.lstm)
    policy.save_pretrained(tmp_path)
    loaded = RobomimicBCRNNPolicy.from_pretrained(tmp_path).eval()

    for name, tensor in policy.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor, msg=name)
