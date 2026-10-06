import pytest
import robomimic_reference
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION

from lerobot_policy_robomimic import RobomimicBCVAEConfig
from lerobot_policy_robomimic.vae import ActionVAE

VARIANTS = {
    "fixed_prior": {},
    "learned_prior": {"prior": {"learn": True}},
    "conditioned_prior": {"prior": {"learn": True, "is_conditioned": True}},
    "conditioned_mixture_prior": {"prior": {"learn": True, "is_conditioned": True, "use_gmm": True}},
    "mixture_prior_with_weights": {"prior": {"learn": True, "use_gmm": True, "gmm_learn_weights": True}},
    "conditioned_mixture_prior_with_weights": {
        "prior": {"learn": True, "is_conditioned": True, "use_gmm": True, "gmm_learn_weights": True}
    },
    "decoder_is_conditioned_ignored": {
        "decoder": {"is_conditioned": False},
        "prior": {"learn": True, "is_conditioned": True},
    },
    "clipped_latent_summed_errors": {
        "latent_clip": 0.5,
        "decoder": {"reconstruction_sum_across_elements": True},
    },
}


def config_fields(vae: dict) -> dict:
    """Maps robomimic's `algo.vae` overrides to `RobomimicBCVAEConfig` fields.

    robomimic ignores `decoder.is_conditioned` (`robomimic/models/vae_nets.py:1049-1050`),
    so it maps to no field.
    """
    fields = {}
    for name, value in vae.items():
        if name == "decoder":
            fields |= {f"vae_{key}": nested for key, nested in value.items() if key != "is_conditioned"}
        elif name == "prior":
            fields |= {f"vae_prior_{key}": nested for key, nested in value.items()}
        else:
            fields[f"vae_{name}"] = value
    return fields


def matching_pair(vae: dict):
    torch.manual_seed(0)
    reference = robomimic_reference.make_algo(
        robomimic_reference.bc_vae_config(**vae),
        robomimic_reference.image_experiment_obs_shapes(),
        action_dim=7,
    )
    config = RobomimicBCVAEConfig(
        device="cpu",
        input_features=robomimic_reference.image_experiment_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        **config_fields(vae),
    )
    ours = ActionVAE.from_config(config)
    theirs = reference.nets["policy"]._vae
    robomimic_reference.load_vae(ours, theirs)
    return ours.eval(), theirs.eval()


@pytest.mark.parametrize("vae", VARIANTS.values(), ids=VARIANTS.keys())
def test_losses_match_robomimic_vae(vae):
    ours, theirs = matching_pair(vae)
    obs, batch = robomimic_reference.random_image_observations(4)
    actions = torch.rand(4, 7) * 2 - 1

    torch.manual_seed(1)
    losses = ours.losses(batch, actions)
    torch.manual_seed(1)
    outputs = theirs.forward(inputs={"action": actions}, outputs={"action": actions}, conditions=obs)

    torch.testing.assert_close(losses["reconstruction_loss"], outputs["reconstruction_loss"])
    torch.testing.assert_close(losses["kl_loss"], outputs["kl_loss"])
    torch.testing.assert_close(losses["encoder_variance"], outputs["encoder_params"]["logvar"].exp().mean())


@pytest.mark.parametrize("vae", VARIANTS.values(), ids=VARIANTS.keys())
def test_samples_match_robomimic_vae(vae):
    ours, theirs = matching_pair(vae)
    obs, batch = robomimic_reference.random_image_observations(4)

    torch.manual_seed(1)
    actions = ours.sample(batch, batch_size=4)
    torch.manual_seed(1)
    expected = theirs.decode(conditions=obs, n=4)["action"]

    torch.testing.assert_close(actions, expected)
