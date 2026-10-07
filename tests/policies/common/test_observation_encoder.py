import robomimic_reference
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE

from lerobot_policy_robomimic.policies.common.observation_encoder import ObservationEncoder


def test_vector_observations_pass_through():
    features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(9,)),
        OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(10,)),
    }
    encoder = ObservationEncoder(features, crop_shape=(76, 76), num_kp=32, image_feature_dim=64)
    batch = {OBS_STATE: torch.randn(2, 9), OBS_ENV_STATE: torch.randn(2, 10)}

    assert encoder.output_dim == 19
    torch.testing.assert_close(encoder(batch), torch.cat([batch[OBS_STATE], batch[OBS_ENV_STATE]], dim=-1))


def test_matches_robomimic_up_to_feature_order():
    torch.manual_seed(0)
    reference = robomimic_reference.observation_encoder(robomimic_reference.image_experiment_obs_shapes())
    ours = ObservationEncoder(
        robomimic_reference.image_experiment_features(), crop_shape=(76, 76), num_kp=32, image_feature_dim=64
    )
    for camera in robomimic_reference.CAMERAS:
        robomimic_reference.load_visual_core(
            ours.cameras[camera].encoder, reference.obs_nets[f"{camera}_image"]
        )
    obs, batch = robomimic_reference.random_image_observations(4)

    expected = robomimic_reference.lerobot_feature_order(reference.eval()(obs))

    assert ours.output_dim == 9 + 64 * len(robomimic_reference.CAMERAS)
    torch.testing.assert_close(ours.eval()(batch), expected)


def test_sequences_are_encoded_frame_by_frame():
    torch.manual_seed(0)
    encoder = ObservationEncoder(
        robomimic_reference.image_experiment_features(), crop_shape=(76, 76), num_kp=32, image_feature_dim=64
    ).eval()
    steps = [robomimic_reference.random_image_observations(2)[1] for _ in range(3)]
    sequence = {key: torch.stack([step[key] for step in steps], dim=1) for key in steps[0]}

    torch.testing.assert_close(encoder(sequence), torch.stack([encoder(step) for step in steps], dim=1))
