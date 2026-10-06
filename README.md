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

### 3. Camera encoder

`lerobot_policy_robomimic/vision.py` holds the camera encoder used by
robomimic's image experiments, robomimic's `VisualCore`
(Repo `robomimic/scripts/generate_paper_configs.py:151-160`). Tests compare
each part with robomimic's own module on the same weights and inputs
(`tests/test_vision.py`).

| Part                 | Behavior                                                                                   | This port      | Source                                                                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------ | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Crop                 | Random crop in training, center crop otherwise                                             | `RandomCrop`   | Repo `robomimic/models/obs_core.py:489` (`CropRandomizer`), `:579` (center crop at evaluation)                                              |
| Random crop offsets  | `floor(rand × (size − crop))`, so 0 to 7 for 76 out of 84; the last offset is never drawn | `RandomCrop`   | Repo `robomimic/utils/obs_utils.py:730`                                                                                                     |
| Center crop offset   | `floor((size − crop) / 2)`, so 4 for 76 out of 84                                          | `RandomCrop`   | Repo `robomimic/utils/obs_utils.py:278`                                                                                                     |
| Backbone             | ResNet-18 without its average pool and classifier, trained from scratch                    | `ImageEncoder` | Repo `robomimic/models/base_nets.py:536` (`ResNet18Conv`); `robomimic/scripts/generate_paper_configs.py:154` (`pretrained = False`)        |
| Spatial softmax      | 1×1 convolution to 32 keypoints, softmax over positions, expected `(x, y)` on a `[-1, 1]` grid | `SpatialSoftmax` | Repo `robomimic/models/base_nets.py:1143`, `:1162`, `:1216`; temperature fixed at 1 without noise (`robomimic/scripts/generate_paper_configs.py:158-160`) |
| Projection           | Linear layer from 32 × 2 keypoint coordinates to 64 features                               | `ImageEncoder` | Repo `robomimic/models/obs_core.py:138`                                                                                                     |

robomimic averages the features of several crops (`robomimic/models/obs_core.py:589-598`);
with one crop per image, as in the paper experiments
(`robomimic/scripts/generate_paper_configs.py:168`), the average is the
feature itself, so `RandomCrop` takes exactly one.

robomimic stores the spatial softmax grid and temperature in the checkpoint
(`robomimic/models/base_nets.py:1159`, `:1167`). Here they are fixed values
rebuilt from the feature map size, so they are left out of the state dict.

Options that robomimic supports but its paper experiments do not use are not
ported: ImageNet-pretrained weights, coordinate convolution, crop position
encoding, and a learnable or noisy spatial softmax.

### 4. Observation encoder

`lerobot_policy_robomimic/observation_encoder.py` turns every observation the
policy reads into one feature vector, like robomimic's `ObservationEncoder`
(Repo `robomimic/models/obs_nets.py:119`). A test loads the same camera
weights into robomimic's encoder and compares the outputs
(`tests/test_observation_encoder.py`).

| Part                 | Behavior                                                         | Source                                                                                                                                                                                  |
| -------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Vector observations  | Concatenated as they are                                         | Repo `robomimic/config/base_config.py:284` (no encoder for low-dim observations), `robomimic/models/obs_nets.py:303-307`                                                                |
| Each camera          | Its own crop, `ImageEncoder`, and ReLU                           | Repo `robomimic/models/obs_nets.py:287-289` (randomizer per key), `:295` (encoder), `:297` (activation); the ReLU is the default `feature_activation` (`:440`, `:33`), which `MIMO_MLP` keeps (`:608-611`) |
| Feature order        | The order of the policy's input features                         | This port; robomimic sorts observation keys instead (Repo `robomimic/utils/file_utils.py:162`, passed on by `robomimic/scripts/train.py:242` and `robomimic/algo/algo.py:154-156`)        |

robomimic's sorted order interleaves cameras and vector observations; for its
image experiments it is `agentview_image`, `robot0_eef_pos`,
`robot0_eef_quat`, `robot0_eye_in_hand_image`, `robot0_gripper_qpos`. The
order only matters when loading robomimic weights, where it decides the
columns of the first MLP layer. Keeping LeRobot's order leaves robomimic's
key names out of the encoder; the test reorders robomimic's output before
comparing.

### 5. Actor network

`MLPActor` in `lerobot_policy_robomimic/actor.py` is robomimic's
`ActorNetwork` (Repo `robomimic/models/policy_nets.py:26`), which BC builds
with `actor_layer_dims` (`robomimic/algo/bc.py:87-92`). It takes the
observation encoder as an argument, so the recurrent and transformer actors of
later policies can reuse the encoder. A test loads robomimic's weights,
reordering the input columns of the first layer as in §4, and compares the
actions (`tests/test_actor.py`).

| Part        | Behavior                                                      | Source                                                                                                                                         |
| ----------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| MLP         | A linear layer and a ReLU for each size in `actor_layer_dims` | Repo `robomimic/models/obs_nets.py:617-623`; the last size is the MLP output and also gets a ReLU (`:623`, `robomimic/models/base_nets.py:256-258`) |
| Action head | Linear layer to the action size                               | Repo `robomimic/models/obs_nets.py:627`, `:392` (`ObservationDecoder`)                                                                         |
| Output      | `tanh`, so actions lie in `[-1, 1]`                           | Repo `robomimic/models/policy_nets.py:107`                                                                                                    |

robomimic splits the MLP into hidden layers and an output layer
(`robomimic/models/obs_nets.py:619-620`), but since both end in a ReLU, the
result is one linear layer and one ReLU per size, which is how `MLPActor`
builds it.
