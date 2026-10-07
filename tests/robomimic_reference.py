"""Builds robomimic's own networks with the settings of its paper experiments."""

import pytest

pytest.importorskip("robomimic", reason="third_party/robomimic submodule is not checked out")

import json  # noqa: E402
from collections import OrderedDict  # noqa: E402
from pathlib import Path  # noqa: E402

import robomimic.utils.obs_utils as obs_utils  # noqa: E402
import robomimic.utils.torch_utils as torch_utils  # noqa: E402
import torch  # noqa: E402
from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE  # noqa: E402
from robomimic.algo import algo_factory  # noqa: E402
from robomimic.algo.bc import BC  # noqa: E402
from robomimic.config import config_factory  # noqa: E402
from robomimic.models.base_nets import SpatialSoftmax  # noqa: E402
from robomimic.models.obs_core import CropRandomizer, VisualCore, VisualCoreLanguageConditioned  # noqa: E402
from robomimic.models.obs_nets import ObservationEncoder, obs_encoder_factory  # noqa: E402
from robomimic.scripts.generate_paper_configs import (  # noqa: E402
    modify_bc_config_for_dataset,
    modify_bc_rnn_config_for_dataset,
    modify_config_for_default_image_exp,
)
from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict  # noqa: E402

from lerobot_policy_robomimic.policies.common.language import OBS_LANGUAGE_EMBEDDING  # noqa: E402

STATE_KEYS = {"robot0_eef_pos": 3, "robot0_eef_quat": 4, "robot0_gripper_qpos": 2}
LANG_EMB = "lang_emb"
CAMERAS = ("agentview", "robot0_eye_in_hand")
IMAGE_SHAPE = (3, 84, 84)


def image_experiment_features() -> dict[str, PolicyFeature]:
    """Returns the LeRobot input features of an image experiment dataset."""
    features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(sum(STATE_KEYS.values()),))}
    for camera in CAMERAS:
        features[f"{OBS_IMAGES}.{camera}"] = PolicyFeature(type=FeatureType.VISUAL, shape=IMAGE_SHAPE)
    return features


def image_experiment_obs_shapes() -> dict[str, list[int]]:
    """Returns the robomimic observation shapes of an image experiment."""
    shapes = {key: [dim] for key, dim in STATE_KEYS.items()}
    return shapes | {f"{camera}_image": list(IMAGE_SHAPE) for camera in CAMERAS}


def random_image_observations(*leading: int) -> tuple[dict, dict]:
    """Returns the same random observations, with leading dimensions such as (B,) or (B, T).

    The first dict is keyed for robomimic, the second for LeRobot.
    """
    obs = {key: torch.randn(*leading, dim) for key, dim in STATE_KEYS.items()}
    obs |= {f"{camera}_image": torch.rand(*leading, *IMAGE_SHAPE) for camera in CAMERAS}
    batch = {OBS_STATE: torch.cat([obs[key] for key in STATE_KEYS], dim=-1)}
    batch |= {f"{OBS_IMAGES}.{camera}": obs[f"{camera}_image"] for camera in CAMERAS}
    return obs, batch


def lerobot_feature_order(features: torch.Tensor, language_dim: int = 0) -> torch.Tensor:
    """Reorders the last dimension of an image experiment from robomimic's sorted keys to LeRobot's order.

    It applies to encoder outputs and to the input columns of the first MLP layer alike. With
    `language_dim`, the features include robomimic's `lang_emb`, which comes last in LeRobot's order.
    """
    feature_dim = rgb_encoder_kwargs()["core_kwargs"]["feature_dimension"]
    sizes = {key: STATE_KEYS.get(key, feature_dim) for key in image_experiment_obs_shapes()}
    if language_dim:
        sizes[LANG_EMB] = language_dim
    sizes = dict(sorted(sizes.items()))
    by_key = dict(zip(sizes, features.split(list(sizes.values()), dim=-1), strict=True))
    ordered = [by_key[key] for key in STATE_KEYS] + [by_key[f"{camera}_image"] for camera in CAMERAS]
    if language_dim:
        ordered.append(by_key[LANG_EMB])
    return torch.cat(ordered, dim=-1)


def random_language(batch_size: int, *steps: int, dim: int = 768) -> tuple[dict, dict]:
    """Returns one random task embedding per sample, keyed for robomimic and for LeRobot.

    robomimic's copy is repeated over `steps`, as its dataset repeats a demo's embedding.
    """
    embedding = torch.randn(batch_size, dim)
    repeated = embedding.reshape(batch_size, *([1] * len(steps)), dim).expand(batch_size, *steps, dim)
    return {LANG_EMB: repeated}, {OBS_LANGUAGE_EMBEDDING: embedding}


def with_language(config, film: bool = False):
    """Adds robomimic's `lang_emb` observation to `config`, and with `film`, FiLM to its cameras.

    FiLM swaps the camera core and backbone as `docs/tutorials/language_conditioning.md` shows,
    keeping the experiment's crop and spatial softmax.
    """
    with config.values_unlocked():
        config.observation.modalities.obs.low_dim = [*config.observation.modalities.obs.low_dim, LANG_EMB]
        if film:
            rgb = config.observation.encoder.rgb
            rgb.core_class = "VisualCoreLanguageConditioned"
            rgb.core_kwargs.backbone_class = "ResNet18ConvFiLM"
    return config


def image_experiment_config():
    """Returns robomimic's BC config for the image experiments."""
    return modify_config_for_default_image_exp(config_factory("bc"))


def rgb_encoder_kwargs() -> dict:
    """Returns the camera encoder settings of the image experiments."""
    return obs_utils.obs_encoder_kwargs_from_config(image_experiment_config().observation.encoder)["rgb"]


def crop_randomizer(image_shape: list[int]) -> CropRandomizer:
    """Returns the camera crop of the image experiments."""
    return CropRandomizer(input_shape=image_shape, **rgb_encoder_kwargs()["obs_randomizer_kwargs"])


def spatial_softmax(feature_shape: list[int]) -> SpatialSoftmax:
    """Returns the spatial softmax of the image experiments."""
    pool_kwargs = rgb_encoder_kwargs()["core_kwargs"]["pool_kwargs"]
    return SpatialSoftmax(input_shape=feature_shape, **pool_kwargs)


def visual_core(image_shape: list[int]) -> VisualCore:
    """Returns the camera encoder of the image experiments."""
    core_kwargs = extract_class_init_kwargs_from_dict(
        VisualCore, dict(rgb_encoder_kwargs()["core_kwargs"]), copy=True
    )
    return VisualCore(input_shape=image_shape, **core_kwargs)


def film_visual_core(image_shape: list[int]) -> VisualCoreLanguageConditioned:
    """Returns the camera encoder of the image experiments with robomimic's FiLM ResNet-18."""
    core_kwargs = extract_class_init_kwargs_from_dict(
        VisualCoreLanguageConditioned, dict(rgb_encoder_kwargs()["core_kwargs"]), copy=True
    )
    core_kwargs["backbone_class"] = "ResNet18ConvFiLM"
    return VisualCoreLanguageConditioned(input_shape=image_shape, **core_kwargs)


def observation_encoder(obs_shapes: dict[str, list[int]]) -> ObservationEncoder:
    """Returns the observation encoder of the image experiments, with keys in sorted order.

    robomimic sorts observation keys when it reads the dataset (`robomimic/utils/file_utils.py:162`).
    """
    config = image_experiment_config()
    obs_utils.initialize_obs_utils_with_config(config)
    return obs_encoder_factory(
        obs_shapes=OrderedDict(sorted(obs_shapes.items())),
        encoder_kwargs=obs_utils.obs_encoder_kwargs_from_config(config.observation.encoder),
    )


def bc_config(gmm: bool = True, **loss_weights: float):
    """Returns robomimic's config for BC in the image experiments on proficient-human Lift data.

    Those experiments use a GMM head (`robomimic/scripts/generate_paper_configs.py:368`);
    `gmm=False` turns it off, which gives robomimic's plain `BC`.
    """
    config = modify_bc_config_for_dataset(image_experiment_config(), "lift", "ph", "image")
    with config.algo.values_unlocked():
        config.algo.gmm.enabled = gmm
        for name, weight in loss_weights.items():
            config.algo.loss[name] = weight
    return config


def bc_gaussian_config(**gaussian):
    """Returns `bc_config` with robomimic's Gaussian head, `BC_Gaussian`, and `gaussian` overrides.

    The paper uses no Gaussian head, so its settings keep robomimic's defaults
    (`robomimic/config/bc_config.py:44-49`) unless overridden.
    """
    config = bc_config(gmm=False)
    with config.algo.values_unlocked():
        config.algo.gaussian.enabled = True
        for name, value in gaussian.items():
            config.algo.gaussian[name] = value
    return config


def bc_vae_config(**vae):
    """Returns `bc_config` with robomimic's VAE policy, `BC_VAE`, and `vae` overrides.

    The paper trains no BC-VAE, so its settings keep robomimic's defaults
    (`robomimic/config/bc_config.py:59-83`) unless overridden; nested settings are given as
    `{"prior": {"learn": True}}`.
    """
    config = bc_config(gmm=False)
    with config.algo.values_unlocked():
        config.algo.vae.enabled = True
        for name, value in vae.items():
            if isinstance(value, dict):
                for key, nested in value.items():
                    config.algo.vae[name][key] = nested
            else:
                config.algo.vae[name] = value
    return config


def bc_rnn_config(hdf5_type: str = "image"):
    """Returns robomimic's config for BC-RNN in the experiments on proficient-human Lift data.

    These experiments use a GMM head (`robomimic/scripts/generate_paper_configs.py:415`).
    """
    return modify_bc_rnn_config_for_dataset(image_experiment_config(), "lift", "ph", hdf5_type)


def robosuite_env(env_args: dict, camera_names: list[str]):
    """Returns robomimic's own `EnvRobosuite`, built from a dataset's env args as its rollouts are.

    robomimic imports `egl_probe` to pick an EGL GPU for offscreen rendering; macOS has none, so a
    stand-in that finds no device takes its place when it is missing.
    """
    import sys
    import types

    from robomimic.utils.env_utils import create_env_from_metadata

    if "egl_probe" not in sys.modules:
        try:
            import egl_probe  # noqa: F401
        except ImportError:
            sys.modules["egl_probe"] = types.SimpleNamespace(get_available_devices=lambda: [])
    obs_utils.initialize_obs_utils_with_obs_specs(
        {
            "obs": {
                "low_dim": list(STATE_KEYS) + ["object"],
                "rgb": [f"{camera}_image" for camera in camera_names],
            }
        }
    )
    env_meta = dict(env_args, env_kwargs=dict(env_args["env_kwargs"], camera_names=camera_names))
    return create_env_from_metadata(
        env_meta=env_meta, render_offscreen=True, use_image_obs=bool(camera_names)
    )


def bc_transformer_template_config():
    """Returns robomimic's tuned BC-Transformer config, loaded as `robomimic/scripts/train.py:475-479` does.

    `docs/tutorials/training_transformers.md` points to this template for transformer policies.
    """
    path = Path(torch_utils.__file__).parents[1] / "config/default_templates/bc_transformer.json"
    config = config_factory("bc")
    with config.values_unlocked():
        config.update(json.loads(path.read_text()))
    return config


def linear_lr_scheduler(optimizer, decay_epochs: int, decay_factor: float):
    """Returns robomimic's "linear" learning rate scheduler, which robomimic steps once per epoch."""
    optim_params = {
        "learning_rate": {
            "initial": optimizer.param_groups[0]["lr"],
            "decay_factor": decay_factor,
            "epoch_schedule": [decay_epochs],
            "scheduler_type": "linear",
        },
        "num_train_batches": None,
        "num_epochs": None,
    }
    return torch_utils.lr_scheduler_from_optim_params(optim_params, net=None, optimizer=optimizer)


def bc_rnn_algo(obs_shapes: dict[str, list[int]], action_dim: int, gmm: bool) -> BC:
    """Returns robomimic's BC-RNN algorithm built from `bc_rnn_config`, with keys in sorted order."""
    config = bc_rnn_config()
    with config.algo.values_unlocked():
        config.algo.gmm.enabled = gmm
    return make_algo(config, obs_shapes, action_dim)


def bc_transformer_algo(obs_shapes: dict[str, list[int]], action_dim: int, gmm: bool) -> BC:
    """Returns robomimic's BC-Transformer from its template, reading the image experiments' observations."""
    config = bc_transformer_template_config()
    image = image_experiment_config()
    with config.values_unlocked():
        config.observation.modalities.obs.low_dim = list(image.observation.modalities.obs.low_dim)
        config.observation.modalities.obs.rgb = list(image.observation.modalities.obs.rgb)
        config.algo.gmm.enabled = gmm
    return make_algo(config, obs_shapes, action_dim)


def bc_algo(obs_shapes: dict[str, list[int]], action_dim: int, gmm: bool, **loss_weights: float) -> BC:
    """Returns robomimic's BC algorithm built from `bc_config`, with keys in sorted order."""
    return make_algo(bc_config(gmm, **loss_weights), obs_shapes, action_dim)


def make_algo(config, obs_shapes: dict[str, list[int]], action_dim: int) -> BC:
    """Builds the robomimic algorithm of `config` on the CPU, with keys in sorted order."""
    obs_utils.initialize_obs_utils_with_config(config)
    return algo_factory(
        algo_name="bc",
        config=config,
        obs_key_shapes=OrderedDict(sorted(obs_shapes.items())),
        ac_dim=action_dim,
        device=torch.device("cpu"),
    )


def load_policy(policy, reference, language_dim: int = 0) -> None:
    """Copies the weights of BC's, BC-RNN's, or BC-Transformer's actor network into our policy.

    The input columns of the first layer after the encoder, the LSTM's, the transformer's
    input projection, or the MLP's, are reordered from robomimic's sorted keys, which include
    `lang_emb` when `language_dim` is set.
    """
    for camera in CAMERAS:
        load_visual_core(
            policy.encoder.cameras[camera].encoder,
            reference.nets["encoder"].nets["obs"].obs_nets[f"{camera}_image"],
        )
    reordered = False
    if "transformer" in reference.nets:
        load_transformer(policy.transformer, reference)
        policy.transformer.input_projection.weight.data.copy_(
            lerobot_feature_order(reference.nets["embed_encoder"].weight, language_dim)
        )
        reordered = True
    if "rnn" in reference.nets:
        lstm = dict(reference.nets["rnn"].nets.state_dict())
        lstm["weight_ih_l0"] = lerobot_feature_order(lstm["weight_ih_l0"], language_dim)
        policy.lstm.load_state_dict(lstm)
        reordered = True
    reference_mlp = reference.nets["mlp"]._model if "mlp" in reference.nets else []
    reference_linears = [layer for layer in reference_mlp if isinstance(layer, torch.nn.Linear)]
    ours_mlp = policy.mlp.layers if hasattr(policy, "mlp") else []
    ours_linears = [layer for layer in ours_mlp if isinstance(layer, torch.nn.Linear)]
    for ours, theirs in zip(ours_linears, reference_linears, strict=True):
        weight = theirs.weight if reordered else lerobot_feature_order(theirs.weight, language_dim)
        ours.load_state_dict({"weight": weight, "bias": theirs.bias})
        reordered = True
    for name, layer in reference.nets["decoder"].nets.items():
        ours = getattr(policy.action_head, "linear" if name == "action" else name)
        if ours is not None:
            ours.load_state_dict(layer.state_dict())


def load_conditioned_mlp(ours, reference, group: str, input_dim: int) -> None:
    """Copies a robomimic `MIMO_MLP` into a `ConditionedMLP`.

    `group` names the MIMO_MLP's observation group, and `input_dim` is the size of the vector
    that precedes the observations; the observation columns of the first layer are reordered
    from robomimic's sorted keys.
    """
    observations = reference.nets["encoder"].nets[group].obs_nets
    for camera in CAMERAS:
        load_visual_core(ours.encoder.cameras[camera].encoder, observations[f"{camera}_image"])
    reference_linears = [
        layer for layer in reference.nets["mlp"]._model if isinstance(layer, torch.nn.Linear)
    ]
    ours_linears = [layer for layer in ours.mlp.layers if isinstance(layer, torch.nn.Linear)]
    for index, (layer, theirs) in enumerate(zip(ours_linears, reference_linears, strict=True)):
        weight = theirs.weight
        if index == 0:
            weight = torch.cat([weight[:, :input_dim], lerobot_feature_order(weight[:, input_dim:])], dim=-1)
        layer.load_state_dict({"weight": weight, "bias": theirs.bias})
    for name, layer in ours.outputs.items():
        layer.load_state_dict(reference.nets["decoder"].nets[name].state_dict())


def load_vae(vae, reference) -> None:
    """Copies the weights of robomimic's `VAE` into an `ActionVAE`."""
    load_conditioned_mlp(
        vae.posterior, reference.nets["encoder"], "condition", reference.input_shapes["action"][0]
    )
    load_conditioned_mlp(vae.decoder, reference.nets["decoder"], "condition", reference.latent_dim)
    prior = reference.nets["prior"]
    if not prior.learnable:
        return
    if prior.prior_module is not None:
        load_conditioned_mlp(vae.prior.network, prior.prior_module, "obs", 0)
    for name, parameter in prior.prior_params.items():
        vae.prior.params[name].data.copy_(parameter)


def load_transformer(transformer, reference) -> None:
    """Copies the embedding and GPT weights of robomimic's `MIMO_Transformer` into a `Transformer`."""
    transformer.input_projection.load_state_dict(reference.nets["embed_encoder"].state_dict())
    transformer.position_embedding.data.copy_(reference.params["embed_timestep"])
    transformer.input_norm.load_state_dict(reference.nets["embed_ln"].state_dict())
    gpt = reference.nets["transformer"].nets
    for ours, theirs in zip(transformer.blocks, gpt["transformer"], strict=True):
        ours.attention_norm.load_state_dict(theirs.nets["ln1"].state_dict())
        ours.attention.qkv.load_state_dict(theirs.nets["attention"].nets["qkv"].state_dict())
        ours.attention.output.load_state_dict(theirs.nets["attention"].nets["output"].state_dict())
        ours.mlp_norm.load_state_dict(theirs.nets["ln2"].state_dict())
        ours.mlp.load_state_dict(theirs.nets["mlp"].state_dict())
    transformer.output_norm.load_state_dict(gpt["output_ln"].state_dict())


def load_visual_core(image_encoder, reference: VisualCore) -> None:
    """Copies the weights of robomimic's `VisualCore` or `VisualCoreLanguageConditioned` into an `ImageEncoder`."""
    backbone = reference.backbone
    if hasattr(backbone, "_film_layers"):
        image_encoder.backbone[:4].load_state_dict(backbone._base_block.state_dict())
        for ours, theirs in zip(image_encoder.residual_blocks(), backbone._conv_blocks, strict=True):
            ours.load_state_dict(theirs.state_dict())
        for ours, theirs in zip(image_encoder.film, backbone._film_layers, strict=True):
            ours.projection.load_state_dict(theirs.lang_proj.state_dict())
    else:
        image_encoder.backbone.load_state_dict(backbone.nets.state_dict())
    image_encoder.pool.keypoints.load_state_dict(reference.pool.nets.state_dict())
    image_encoder.projection.load_state_dict(reference.nets[-1].state_dict())
