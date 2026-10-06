# lerobot-policy-robomimic

English | [繁體中文](README.zh-TW.md)

An out-of-tree [LeRobot](https://github.com/huggingface/lerobot) policy plugin
that ports the imitation learning policies of
[robomimic](https://robomimic.github.io/) (Mandlekar et al., 2021), such as BC
and BC-RNN, to LeRobot v0.6.1 and registers them as LeRobot policy types. The
ported policies can be trained with `lerobot-train` and evaluated with
`lerobot-eval`, while following the network architectures and training
recipes of the original implementation as closely as possible.

## Design

Each robomimic policy is rebuilt here from scratch, one component at a time.
Each section below describes one component and where its behavior comes from.

Sources are cited as follows:

- **Paper**: Mandlekar et al.,
  [What Matters in Learning from Offline Human Demonstrations for Robot Manipulation](https://arxiv.org/abs/2108.03298),
  CoRL 2021.
- **Repo**: [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic)
  at `d309eae`, vendored as the `third_party/robomimic` submodule. Paths are
  relative to that directory.
- **LeRobot**: LeRobot v0.6.1, either the
  [Adding a Policy](https://huggingface.co/docs/lerobot/v0.6.1/en/bring_your_own_policies)
  guide or its source. Paths are relative to the installed `lerobot` package.
- **This port**: a choice made here, not taken from any source.

robomimic is released under the MIT License. This port reimplements its
networks instead of importing robomimic at runtime.

### 1. Policy types and package layout

| Decision                       | Choice                                                                                      | Source                                                                                                                                                                                        |
| ------------------------------ | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Policy type                    | `robomimic_bc`                                                                              | This port: the `robomimic_` prefix avoids clashing with LeRobot's built-in types, such as `diffusion` (`policies/diffusion/`), which robomimic also implements (Repo `robomimic/algo/diffusion_policy.py`) |
| Class and function names       | `RobomimicBCConfig`, `RobomimicBCPolicy`, `make_robomimic_bc_pre_post_processors`           | LeRobot: the policy class name is the config class name with `Config` replaced by `Policy` (`policies/factory.py:409-415`), and the processor factory is `make_<type>_pre_post_processors` (`policies/factory.py:458`) |
| Module names                   | `configuration_robomimic_bc.py`, `modeling_robomimic_bc.py`, `processor_robomimic_bc.py`    | LeRobot: the modeling and processor modules are found by replacing `configuration_` in the config's module path (`policies/factory.py:416`, `:459`); the names follow the guide's template      |
| Distribution name              | `lerobot_policy_robomimic`                                                                  | LeRobot: an installed distribution whose name starts with `lerobot_policy_` is imported by that name (`utils/import_utils.py:231-255`), which runs `@PreTrainedConfig.register_subclass`        |
| Several policies, one package  | Each robomimic algorithm gets its own policy type and its own three modules                 | This port: the guide shows one policy per package, but the factory only needs those three modules per type, so robomimic's algorithms can share one package and its networks              |

### 2. Configuration

`RobomimicBCConfig` registers the policy type `robomimic_bc`. Defaults follow
robomimic's image experiments on the proficient-human (PH) datasets.

| Setting                    | Default                                                       | Config field                                     | Source                                                                                                                                                                       |
| -------------------------- | ------------------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Observations               | Proprioception and cameras; no object state                   | `use_env_state` (False)                          | Repo `robomimic/scripts/generate_paper_configs.py:138-146` (image); low-dim experiments add `object` (`:67-72`)                                                              |
| Observation history        | None (single step)                                            | `n_obs_steps`, `observation_delta_indices`       | Repo `robomimic/config/base_config.py:193`, `:195`; `robomimic/algo/bc.py:111` reads step 0 only                                                                             |
| Action prediction          | One action per observation                                    | `action_delta_indices`                           | Repo `robomimic/algo/bc.py:113`                                                                                                                                              |
| MLP hidden sizes           | (1024, 1024)                                                  | `actor_layer_dims`                               | Repo `robomimic/config/bc_config.py:41`, `robomimic/scripts/generate_paper_configs.py:367`                                                                                   |
| Camera crop                | Random 76×76 in training, center 76×76 at inference           | `crop_shape`                                     | Repo `robomimic/scripts/generate_paper_configs.py:163-168`                                                                                                                  |
| Spatial softmax keypoints  | 32                                                            | `spatial_softmax_num_kp`                         | Repo `robomimic/scripts/generate_paper_configs.py:157`                                                                                                                      |
| Feature size per camera    | 64                                                            | `image_feature_dim`                              | Repo `robomimic/scripts/generate_paper_configs.py:152`                                                                                                                      |
| Loss                       | Mean squared error only                                       | `l2_weight` (1), `l1_weight` (0), `cos_weight` (0) | Repo `robomimic/config/bc_config.py:36-38`                                                                                                                                 |
| Normalization              | None; images are only scaled to `[0, 1]`                      | `normalization_mapping` (`IDENTITY`)             | Repo `robomimic/config/base_config.py:181` (observations), `:225` (actions), `robomimic/utils/obs_utils.py:921` (images); LeRobot already scales images (`datasets/io_utils.py:255-263`) |
| Optimizer                  | Adam                                                          | `get_optimizer_preset()`                         | Repo `robomimic/config/bc_config.py:27`                                                                                                                                      |
| Learning rate              | 1e-4, constant                                                | `optimizer_lr`                                   | Repo `robomimic/config/bc_config.py:28`, `:30` (no decay epochs)                                                                                                             |
| Weight decay               | 0                                                             | `optimizer_weight_decay`                         | Repo `robomimic/config/bc_config.py:33`                                                                                                                                      |
| Gradient clipping          | None                                                          | `optimizer_grad_clip_norm` (0)                   | Repo `robomimic/config/base_config.py:239`; LeRobot skips clipping at 0 (`scripts/lerobot_train.py:178`)                                                                     |

LeRobot feeds every non-action dataset feature to the policy
(`policies/factory.py:305-306`), while robomimic lists the observations each
experiment uses. `use_env_state` keeps that choice in the policy, so one
dataset serves both kinds of experiment.

Batch size and training length belong to `lerobot-train`, not the policy:
robomimic uses batch size 16 for 600 epochs of 500 steps (300K steps) on image
data (`robomimic/scripts/generate_paper_configs.py:113`, `:131-132`), and
batch size 100 for 2000 epochs of 100 steps (200K steps) on low-dim data
(`:43`, `:61-62`).

robomimic's paper runs BC with a GMM action head on human datasets
(`robomimic/scripts/generate_paper_configs.py:368`). This policy starts with
the deterministic head, robomimic's `BC` class.
