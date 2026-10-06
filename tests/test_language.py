import pytest
import torch
from huggingface_hub import try_to_load_from_cache
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot_policy_robomimic import RobomimicBCConfig, language
from lerobot_policy_robomimic.language import OBS_LANGUAGE_EMBEDDING, CLIPTaskEmbeddingStep
from lerobot_policy_robomimic.processors import make_robomimic_pre_post_processors

CLIP = "openai/clip-vit-large-patch14"


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[len(text)]])}


class FakeCLIP:
    """Embeds a task as its length repeated four times, counting its calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids):
        self.calls += 1
        return {"text_embeds": input_ids.float().expand(1, 4)}


@pytest.fixture
def fake_clip(monkeypatch):
    model = FakeCLIP()
    monkeypatch.setattr(language, "load_clip", lambda name: (FakeTokenizer(), model))
    return model


def make_config(**kwargs):
    return RobomimicBCConfig(
        device="cpu",
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(9,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        language_embedding_dim=4,
        **kwargs,
    )


@pytest.mark.parametrize("conditioning", ["concat", "film"])
def test_preprocessor_embeds_each_task(fake_clip, conditioning):
    preprocessor, _ = make_robomimic_pre_post_processors(make_config(language_conditioning=conditioning))

    batch = preprocessor({OBS_STATE: torch.zeros(3, 9), "task": ["lift", "lift", "stack it"]})

    torch.testing.assert_close(batch[OBS_LANGUAGE_EMBEDDING], torch.tensor([[4.0] * 4, [4.0] * 4, [8.0] * 4]))
    assert fake_clip.calls == 2


def test_preprocessor_batches_a_single_task(fake_clip):
    preprocessor, _ = make_robomimic_pre_post_processors(make_config(language_conditioning="concat"))

    batch = preprocessor({OBS_STATE: torch.zeros(9), "task": "lift"})

    torch.testing.assert_close(batch[OBS_LANGUAGE_EMBEDDING], torch.full((1, 4), 4.0))


def test_preprocessor_without_language_has_no_clip_step():
    preprocessor, _ = make_robomimic_pre_post_processors(make_config())

    assert not any(isinstance(step, CLIPTaskEmbeddingStep) for step in preprocessor.steps)


def test_embedding_size_must_match(fake_clip):
    with pytest.raises(ValueError, match="language_embedding_dim is 5"):
        CLIPTaskEmbeddingStep(embedding_dim=5).embed("lift")


def test_task_is_required(fake_clip):
    preprocessor, _ = make_robomimic_pre_post_processors(make_config(language_conditioning="concat"))

    with pytest.raises(ValueError, match="'task'"):
        preprocessor({OBS_STATE: torch.zeros(9)})


def test_rejects_unknown_conditioning():
    with pytest.raises(ValueError, match="language_conditioning"):
        make_config(language_conditioning="cross_attention")


def test_saved_preprocessor_keeps_the_clip_step(tmp_path, fake_clip):
    preprocessor, _ = make_robomimic_pre_post_processors(
        make_config(language_conditioning="film", clip_model_name="openai/clip-vit-base-patch32")
    )
    preprocessor.save_pretrained(tmp_path)

    loaded = PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename="policy_preprocessor.json")

    steps = [step for step in loaded.steps if isinstance(step, CLIPTaskEmbeddingStep)]
    assert [(step.model_name, step.embedding_dim) for step in steps] == [("openai/clip-vit-base-patch32", 4)]


@pytest.mark.skipif(
    not isinstance(try_to_load_from_cache(CLIP, "model.safetensors"), str),
    reason=f"{CLIP} is not in the Hugging Face cache",
)
def test_embedding_matches_robomimic(monkeypatch):
    lang_utils = pytest.importorskip("robomimic.utils.lang_utils")
    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    monkeypatch.setattr(
        lang_utils, "lang_emb_model", CLIPTextModelWithProjection.from_pretrained(CLIP).eval()
    )
    monkeypatch.setattr(lang_utils, "tz", AutoTokenizer.from_pretrained(CLIP))
    step = CLIPTaskEmbeddingStep(model_name=CLIP)

    for task in ("lift the cube", "pick up the can and place it in the bin"):
        torch.testing.assert_close(step.embed(task), lang_utils.get_lang_emb(task))
