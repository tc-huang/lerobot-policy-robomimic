import pytest
import robomimic_reference
import torch

from lerobot_policy_robomimic.vision import ImageEncoder, RandomCrop, SpatialSoftmax

CROP = (76, 76)


def test_center_crop_matches_robomimic():
    images = torch.rand(4, 3, 84, 84)
    reference = robomimic_reference.crop_randomizer([3, 84, 84]).eval()

    torch.testing.assert_close(RandomCrop(CROP).eval()(images), reference.forward_in(images))


def test_random_crop_takes_windows_like_robomimic():
    torch.manual_seed(0)
    images = torch.arange(2 * 84 * 84, dtype=torch.float32).reshape(2, 1, 84, 84).expand(2, 3, 84, 84)
    offsets = set()
    for _ in range(200):
        crops = RandomCrop(CROP).train()(images)
        for image, crop in zip(images, crops, strict=True):
            top, left = divmod(int(crop[0, 0, 0] - image[0, 0, 0]), 84)
            torch.testing.assert_close(crop, image[:, top : top + 76, left : left + 76])
            offsets.add((top, left))

    assert {top for top, _ in offsets} == set(range(8))
    assert {left for _, left in offsets} == set(range(8))


def test_spatial_softmax_matches_robomimic():
    torch.manual_seed(0)
    ours = SpatialSoftmax(in_channels=512, height=3, width=3, num_kp=32)
    reference = robomimic_reference.spatial_softmax([512, 3, 3]).eval()
    reference.nets.load_state_dict(ours.keypoints.state_dict())
    features = torch.randn(4, 512, 3, 3)

    torch.testing.assert_close(ours(features), reference(features))


@pytest.mark.parametrize("training", [False, True])
def test_image_encoder_matches_robomimic_visual_core(training):
    torch.manual_seed(0)
    reference = robomimic_reference.visual_core([3, *CROP]).train(training)
    ours = ImageEncoder(height=76, width=76, num_kp=32, feature_dim=64).train(training)
    robomimic_reference.load_visual_core(ours, reference)
    images = torch.rand(4, 3, *CROP)

    torch.testing.assert_close(ours(images), reference(images))
