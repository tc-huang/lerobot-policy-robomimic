import robomimic_reference
import torch

from lerobot_policy_robomimic.actor import MLPActor
from lerobot_policy_robomimic.observation_encoder import ObservationEncoder


def test_mlp_actor_matches_robomimic_actor_network():
    torch.manual_seed(0)
    reference = robomimic_reference.actor_network(
        robomimic_reference.image_experiment_obs_shapes(), action_dim=7
    )
    encoder = ObservationEncoder(
        robomimic_reference.image_experiment_features(), crop_shape=(76, 76), num_kp=32, image_feature_dim=64
    )
    actor = MLPActor(encoder, layer_dims=(1024, 1024), action_dim=7)
    robomimic_reference.load_actor_network(actor, reference)
    obs, batch = robomimic_reference.random_image_observations(4)

    torch.testing.assert_close(actor.eval()(batch), reference.eval()(obs))
