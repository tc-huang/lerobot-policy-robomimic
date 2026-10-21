import pytest
import robomimic_reference
import torch
from torchvision.models import ResNet18_Weights, resnet18

from lerobot_policy_robomimic.policies.common.vision import ImageEncoder, RandomCrop, Resize, SpatialSoftmax

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
        assert crops.is_contiguous()
        for image, crop in zip(images, crops, strict=True):
            top, left = divmod(int(crop[0, 0, 0] - image[0, 0, 0]), 84)
            torch.testing.assert_close(crop, image[:, top : top + 76, left : left + 76])
            offsets.add((top, left))

    assert {top for top, _ in offsets} == set(range(8))
    assert {left for _, left in offsets} == set(range(8))


def test_random_crop_can_stay_random_at_inference():
    torch.manual_seed(0)
    images = torch.rand(64, 3, 84, 84)
    crop = RandomCrop(CROP, random_at_inference=True).eval()

    assert not torch.equal(crop(images), RandomCrop(CROP).eval()(images))


def test_resize_keeps_strokes_thinner_than_its_step():
    lines = torch.zeros(1, 3, 480, 640)
    lines[..., ::4] = 1.0

    inner = Resize((240, 320))(lines)[..., 1:-1]

    assert (inner > 0).all()
    torch.testing.assert_close(inner.mean(), torch.tensor(0.25))
    assert (torch.nn.functional.interpolate(lines, size=(240, 320), mode="bilinear")[..., 1:-1] == 0).any()


def test_pretrained_backbone_starts_from_imagenet():
    imagenet = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()

    pretrained = ImageEncoder(76, 76, 32, 64, pretrained=True).backbone.state_dict()
    fresh = ImageEncoder(76, 76, 32, 64).backbone.state_dict()

    torch.testing.assert_close(pretrained["0.weight"], imagenet["conv1.weight"])
    torch.testing.assert_close(pretrained["7.1.conv2.weight"], imagenet["layer4.1.conv2.weight"])
    assert not torch.equal(fresh["0.weight"], imagenet["conv1.weight"])


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
