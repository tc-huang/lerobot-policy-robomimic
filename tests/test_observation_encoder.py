import robomimic_reference
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from lerobot_policy_robomimic.observation_encoder import ObservationEncoder

STATE_KEYS = {"robot0_eef_pos": 3, "robot0_eef_quat": 4, "robot0_gripper_qpos": 2}
CAMERAS = ("agentview", "robot0_eye_in_hand")
IMAGE_SHAPE = (3, 84, 84)


def image_features():
    features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(sum(STATE_KEYS.values()),))}
    for camera in CAMERAS:
        features[f"{OBS_IMAGES}.{camera}"] = PolicyFeature(type=FeatureType.VISUAL, shape=IMAGE_SHAPE)
    return features


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
    obs_shapes = {key: [dim] for key, dim in STATE_KEYS.items()}
    obs_shapes |= {f"{camera}_image": list(IMAGE_SHAPE) for camera in CAMERAS}
    reference = robomimic_reference.observation_encoder(obs_shapes).eval()
    ours = ObservationEncoder(image_features(), crop_shape=(76, 76), num_kp=32, image_feature_dim=64).eval()
    for camera in CAMERAS:
        robomimic_reference.load_visual_core(ours.cameras[camera][0], reference.obs_nets[f"{camera}_image"])

    obs = {key: torch.randn(4, dim) for key, dim in STATE_KEYS.items()}
    obs |= {f"{camera}_image": torch.rand(4, *IMAGE_SHAPE) for camera in CAMERAS}
    batch = {OBS_STATE: torch.cat([obs[key] for key in STATE_KEYS], dim=-1)}
    batch |= {f"{OBS_IMAGES}.{camera}": obs[f"{camera}_image"] for camera in CAMERAS}

    sizes = {key: 64 if key.endswith("_image") else STATE_KEYS[key] for key in reference.obs_shapes}
    reference_features = dict(zip(sizes, reference(obs).split(list(sizes.values()), dim=-1), strict=True))
    expected = [reference_features[key] for key in STATE_KEYS]
    expected += [reference_features[f"{camera}_image"] for camera in CAMERAS]

    assert ours.output_dim == 9 + 64 * len(CAMERAS)
    torch.testing.assert_close(ours(batch), torch.cat(expected, dim=-1))
