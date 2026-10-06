import robomimic_reference
import torch

from lerobot_policy_robomimic.actor import MLPActor
from lerobot_policy_robomimic.observation_encoder import ObservationEncoder


def load_actor_network(actor: MLPActor, reference) -> None:
    """Copies robomimic's `ActorNetwork` weights, reordering the first layer's input columns."""
    for camera in robomimic_reference.CAMERAS:
        robomimic_reference.load_visual_core(
            actor.encoder.cameras[camera].encoder,
            reference.nets["encoder"].nets["obs"].obs_nets[f"{camera}_image"],
        )
    reference_mlp = [layer for layer in reference.nets["mlp"]._model if isinstance(layer, torch.nn.Linear)]
    ours_mlp = [layer for layer in actor.mlp if isinstance(layer, torch.nn.Linear)]
    for index, (ours, theirs) in enumerate(zip(ours_mlp, reference_mlp, strict=True)):
        weight = robomimic_reference.lerobot_feature_order(theirs.weight) if index == 0 else theirs.weight
        ours.load_state_dict({"weight": weight, "bias": theirs.bias})
    actor.action_head.load_state_dict(reference.nets["decoder"].nets["action"].state_dict())


def test_mlp_actor_matches_robomimic_actor_network():
    torch.manual_seed(0)
    reference = robomimic_reference.actor_network(
        robomimic_reference.image_experiment_obs_shapes(), action_dim=7
    )
    encoder = ObservationEncoder(
        robomimic_reference.image_experiment_features(), crop_shape=(76, 76), num_kp=32, image_feature_dim=64
    )
    actor = MLPActor(encoder, layer_dims=(1024, 1024), action_dim=7)
    load_actor_network(actor, reference)
    obs, batch = robomimic_reference.random_image_observations(4)

    torch.testing.assert_close(actor.eval()(batch), reference.eval()(obs))
