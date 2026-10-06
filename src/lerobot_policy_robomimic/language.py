"""CLIP embeddings of task strings, robomimic's language conditioning input."""

import functools
from dataclasses import dataclass, field
from typing import Any

import torch
from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotObservation, TransitionKey
from lerobot.processor import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import OBS_LANGUAGE
from torch import Tensor

OBS_LANGUAGE_EMBEDDING = f"{OBS_LANGUAGE}.embedding"


@functools.cache
def load_clip(model_name: str) -> tuple[Any, Any]:
    """Loads a CLIP tokenizer and text model with projection, once per model name."""
    try:
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
    except ImportError as err:
        raise ImportError(
            "Language conditioning needs transformers; install it with `uv sync --extra language`."
        ) from err
    return AutoTokenizer.from_pretrained(model_name), CLIPTextModelWithProjection.from_pretrained(
        model_name
    ).eval()


@ProcessorStepRegistry.register(name="robomimic_clip_task_embedding")
@dataclass
class CLIPTaskEmbeddingStep(ObservationProcessorStep):
    """Adds the CLIP text embedding of each sample's task to the observation.

    Each task is tokenized alone, padded to `max_length` tokens, and embedded as CLIP's projected
    text embedding, as robomimic's `get_lang_emb` does. Embeddings are cached per task, and the
    CLIP model loads when the first task arrives.

    Args:
        model_name: CLIP model on the Hugging Face Hub.
        embedding_dim: Size of the projected text embedding, checked against the model's output.
        max_length: Number of tokens each task is padded to.
        task_key: Key of the task string, or of a list with one per sample, in the
            transition's complementary data.
    """

    model_name: str = "openai/clip-vit-large-patch14"
    embedding_dim: int = 768
    max_length: int = 25
    task_key: str = "task"
    _cache: dict[str, Tensor] = field(default_factory=dict, init=False, repr=False)

    def embed(self, task: str) -> Tensor:
        """Returns the (embedding_dim,) embedding of one task."""
        if task not in self._cache:
            tokenizer, model = load_clip(self.model_name)
            tokens = tokenizer(
                text=task,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                embedding = model(**tokens)["text_embeds"][0]
            if embedding.shape != (self.embedding_dim,):
                raise ValueError(
                    f"{self.model_name} embeds tasks in {embedding.shape[0]} dimensions, "
                    f"but language_embedding_dim is {self.embedding_dim}."
                )
            self._cache[task] = embedding
        return self._cache[task]

    def observation(self, observation: RobotObservation) -> RobotObservation:
        complementary_data = self.transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        if self.task_key not in complementary_data:
            raise ValueError(f"Language conditioning needs a {self.task_key!r} string with each sample.")
        tasks = complementary_data[self.task_key]
        tasks = [tasks] if isinstance(tasks, str) else list(tasks)
        return {**observation, OBS_LANGUAGE_EMBEDDING: torch.stack([self.embed(task) for task in tasks])}

    def get_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "max_length": self.max_length,
            "task_key": self.task_key,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_EMBEDDING] = PolicyFeature(
            type=FeatureType.LANGUAGE, shape=(self.embedding_dim,)
        )
        return features
