# lerobot-policy-robomimic

English | [繁體中文](README.zh-TW.md)

An out-of-tree [LeRobot](https://github.com/huggingface/lerobot) policy plugin
that ports the imitation learning policies of
[robomimic](https://robomimic.github.io/) (Mandlekar et al., 2021), such as BC
and BC-RNN, to LeRobot v0.6.1 and registers them as LeRobot policy types. The
ported policies can be trained with `lerobot-train` and evaluated with
`lerobot-eval`, while following the network architectures and training
recipes of the original implementation as closely as possible.

## Usage

Install the plugin with the dependencies of `lerobot-train`:

```bash
uv sync --extra training
```

Convert a robomimic hdf5 file that contains observations into a
LeRobotDataset. Image files are generated from robomimic's raw files with its
`dataset_states_to_obs.py` (Repo `robomimic/scripts/extract_obs_from_raw_datasets.sh:59-61`).

```bash
uv run python -m lerobot_policy_robomimic.convert_dataset \
    --hdf5 data/robomimic/lift/ph/image_v15.hdf5 \
    --repo-id <user>/robomimic_lift_ph_image \
    --task "lift the cube"
```

Train BC on the demos of robomimic's `train` mask, with the batch size and
length of robomimic's image experiments (§2):

```bash
ROOT=~/.cache/huggingface/lerobot/<user>/robomimic_lift_ph_image
EPISODES="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["masks"]["train"])' $ROOT/meta/robomimic_masks.json)"

uv run lerobot-train \
    --policy.type=robomimic_bc \
    --dataset.repo_id=<user>/robomimic_lift_ph_image \
    --dataset.episodes="$EPISODES" \
    --batch_size=16 \
    --steps=300000 \
    --policy.push_to_hub=false
```

For a low-dim experiment, convert the low-dim file instead, add
`--policy.use_env_state=true`, and use `--batch_size=100 --steps=200000`.

For BC-RNN, use `--policy.type=robomimic_bc_rnn` with the same batch size and
length, which robomimic applies to every algorithm in its image and low-dim
experiments (Repo `robomimic/scripts/generate_paper_configs.py:61-62`,
`:131-132`, chosen by data type alone at `:739-746`). Its low-dim experiments
also need `--policy.rnn_hidden_dim=400` (§8).

For BC-Transformer, use `--policy.type=robomimic_bc_transformer`. Its tuned
template trains on low-dim data with `--batch_size=100 --steps=200000` (§13);
the policy sets AdamW and the `robomimic_linear` learning rate schedule
itself.

To run a BC-RNN checkpoint from robomimic's model zoo (Repo
`docs/model_zoo/robomimic_v0.1.md`), download it and convert it into a
LeRobot policy directory (§12):

```bash
curl -L --create-dirs -o data/robomimic/model_zoo/lift_ph_image_epoch_500_succ_100.pth \
    http://downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo/lift/bc_rnn/lift_ph_image_epoch_500_succ_100.pth
uv run python -m lerobot_policy_robomimic.convert_checkpoint \
    --checkpoint data/robomimic/model_zoo/lift_ph_image_epoch_500_succ_100.pth \
    --output-dir outputs/checkpoints/lift_ph_image_bc_rnn
```

Evaluate a policy in simulation with `lerobot-eval` and the `robomimic` env,
which rebuilds the robosuite env of a converted dataset (§ Simulation); this
needs the `sim` extra:

```bash
uv sync --extra training --extra sim
uv run lerobot-eval \
    --policy.path=outputs/checkpoints/lift_ph_image_bc_rnn \
    --env.type=robomimic \
    --env.env_args_path=$ROOT/meta/robomimic_env_args.json \
    --eval.n_episodes=50 \
    --eval.batch_size=1
```

## Datasets

`lerobot_policy_robomimic/convert_dataset.py` turns a robomimic hdf5 file that
contains observations into a LeRobotDataset with one episode per demo. Tests
check the converted values, pixels, episodes, and metadata against a small
hdf5 file (`tests/test_convert_dataset.py`). Sources are cited as described
under Design.

| Decision              | Choice                                                                                                          | Source                                                                                                                                                 |
| --------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Proprioception        | `observation.state`: `robot0_eef_pos`, `robot0_eef_quat`, and `robot0_gripper_qpos`, concatenated in that order | Repo `robomimic/scripts/generate_paper_configs.py:138-142`, `:67-70` (the keys of every paper experiment); LeRobot policies read `observation.state`   |
| Object state          | `observation.environment_state`, from `object`                                                                  | Repo `robomimic/scripts/generate_paper_configs.py:71` (low-dim experiments only); LeRobot gives this key the `ENV` type (`utils/feature_utils.py:168`) |
| Cameras               | `observation.images.<camera>`, from `<camera>_image`                                                            | This port; LeRobot's RoboCasa env also keeps robosuite's camera names (`envs/robocasa.py:44`)                                                          |
| Image storage         | Lossless PNG, or LeRobot's MP4 videos above 2 GiB of raw pixels (`--video auto`)                                | This port: small datasets keep robomimic's exact pixels; large ones stay manageable                                                                    |
| Frame rate            | The `control_freq` of the demos, 20 Hz for Lift                                                                 | The hdf5's `env_args`                                                                                                                                  |
| Episode order         | Numeric demo order, so `demo_2` comes before `demo_10`                                                          | This port                                                                                                                                              |
| Train and valid split | `meta/robomimic_masks.json` maps each mask to episode indices, for `--dataset.episodes`                         | Repo `robomimic/scripts/generate_paper_configs.py:244-245` (the paper trains on `train` and validates on `valid`)                                      |
| Simulator settings    | `meta/robomimic_env_args.json` keeps the hdf5's `env_args` as they are                                          | Repo `robomimic/scripts/train.py:89`, `:150` (robomimic builds its rollout env from them)                                                              |
| Values                | float64 becomes float32                                                                                         | This port                                                                                                                                              |

robomimic's v1.5 image datasets are not released; they are generated from the
raw files with `robomimic/scripts/dataset_states_to_obs.py`, using the cameras
and size of `robomimic/scripts/extract_obs_from_raw_datasets.sh:61`. Generating
the low-dim file the same way reproduces the released one, up to rounding
in one float32 observation.

The simulator settings matter because robosuite 1.5.1's own defaults differ
from those the demos were recorded with: its Panda controller reads actions
in the robot base frame instead of the world frame (`input_ref_frame`), and
the demos turn off `lite_physics`.

## Simulation

`RobomimicEnvConfig` in `lerobot_policy_robomimic/env_config.py` registers the
LeRobot env type `robomimic`, and `RobomimicEnv` in
`lerobot_policy_robomimic/robosuite_env.py` wraps robosuite the way robomimic's
rollouts do. The env is rebuilt from the `meta/robomimic_env_args.json` of a
converted dataset (§ Datasets), and robosuite is imported only when an env is
built, so it stays in the `sim` extra. Sources are cited as described under
Design.

| Part            | Behavior                                                                                                  | Source                                                                                                                          |
| --------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Construction    | `robosuite.make` with the env args, rendering only the requested cameras and always the object state      | Repo `robomimic/envs/env_robosuite.py:87`, `:116`                                                                               |
| Images          | Flipped upright, as `pixels/<camera>`                                                                     | Repo `robomimic/envs/env_robosuite.py:255`                                                                                      |
| Proprioception  | The dataset's state keys, concatenated as `agent_pos`                                                     | § Datasets                                                                                                                      |
| Object state    | robosuite's `object-state`, as `environment_state`                                                        | Repo `robomimic/envs/env_robosuite.py:267`                                                                                      |
| Success and end | `_check_success()`; an episode ends on success                                                            | Repo `robomimic/envs/env_robosuite.py:407`; `robomimic/scripts/generate_paper_configs.py:53`, `:123` (`terminate_on_success`)   |
| Episode length  | 400 steps for Lift, Can, and Square; 700 for Transport and Tool Hang                                      | Repo `robomimic/__init__.py:63`                                                                                                 |
| Initial states  | robosuite's random reset; a seed sets NumPy's global generator, which robosuite 1.5.1 places objects with | robosuite 1.5.1 `robosuite/utils/placement_samplers.py:167`                                                                     |
| Batched envs    | Each env stops simulating once its episode ends, with `NEXT_STEP` autoreset                               | LeRobot `envs/utils.py:241` (`freeze_after_episode_end`), `envs/libero.py:526`                                                  |
| Features        | Read from a robosuite env built without cameras when the config is created                                | This port: LeRobot replaces a loaded policy's action feature with the env's (`policies/factory.py:304`), so sizes must be exact |

Tests build robomimic's own `EnvRobosuite` from the same env args, reset both
with the same seed, and compare five steps of images, proprioception, object
state, reward, and success (`tests/test_env.py`). The comparison starts from
seeded resets rather than a copied simulator state, because robosuite's
controllers and its visual markers keep state that a copied simulator state
leaves behind.

## Design

Each robomimic policy is rebuilt here from scratch, one component at a time.
Each section below describes one component and where its behavior comes from.

Sources are cited as follows:

- **Paper**: Mandlekar et al.,
  [What Matters in Learning from Offline Human Demonstrations for Robot Manipulation](https://arxiv.org/abs/2108.03298),
  CoRL 2021.
- **Repo**: [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic)
  at `d309eae`, vendored as the `third_party/robomimic` submodule. Paths are
  relative to that directory; a `v0.1.0:` prefix marks a path at that tag,
  readable with `git -C third_party/robomimic show v0.1.0:<path>`.
- **LeRobot**: LeRobot v0.6.1, either the
  [Adding a Policy](https://huggingface.co/docs/lerobot/v0.6.1/en/bring_your_own_policies)
  guide or its source. Paths are relative to the installed `lerobot` package.
- **This port**: a choice made here, not taken from any source.

robomimic is released under the MIT License. This port reimplements its
networks instead of importing robomimic at runtime.

### 1. Policy types and package layout

| Decision                      | Choice                                                                                                                                            | Source                                                                                                                                                                                                                 |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Policy types                  | `robomimic_bc` (BC), `robomimic_bc_rnn` (BC-RNN), `robomimic_bc_transformer` (BC-Transformer)                                                     | This port: the `robomimic_` prefix avoids clashing with LeRobot's built-in types, such as `diffusion` (`policies/diffusion/`), which robomimic also implements (Repo `robomimic/algo/diffusion_policy.py`)             |
| Class and function names      | `RobomimicBCConfig`, `RobomimicBCPolicy`, `make_robomimic_bc_pre_post_processors`; likewise for `robomimic_bc_rnn` and `robomimic_bc_transformer` | LeRobot: the policy class name is the config class name with `Config` replaced by `Policy` (`policies/factory.py:409-415`), and the processor factory is `make_<type>_pre_post_processors` (`policies/factory.py:458`) |
| Module names                  | `configuration_<type>.py`, `modeling_<type>.py`, `processor_<type>.py`                                                                            | LeRobot: the modeling and processor modules are found by replacing `configuration_` in the config's module path (`policies/factory.py:416`, `:459`); the names follow the guide's template                             |
| Distribution name             | `lerobot_policy_robomimic`                                                                                                                        | LeRobot: an installed distribution whose name starts with `lerobot_policy_` is imported by that name (`utils/import_utils.py:231-255`), which runs `@PreTrainedConfig.register_subclass`                               |
| Several policies, one package | Each robomimic algorithm gets its own policy type and its own three modules                                                                       | This port: the guide shows one policy per package, but the factory only needs those three modules per type, so robomimic's algorithms can share one package and its networks                                           |

## BC (`robomimic_bc`)

### 2. Configuration

`RobomimicBCConfig` registers the policy type `robomimic_bc`. Defaults follow
robomimic's image experiments on the proficient-human (PH) datasets. Every
setting below except `actor_layer_dims` comes from `RobomimicPolicyConfig` in
`lerobot_policy_robomimic/base_config.py`, which all robomimic policies here
share; it is not a policy type itself.

| Setting                        | Default                                             | Config field                                       | Source                                                                                                                                                                                   |
| ------------------------------ | --------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Observations                   | Proprioception and cameras; no object state         | `use_env_state` (False)                            | Repo `robomimic/scripts/generate_paper_configs.py:138-146` (image); low-dim experiments add `object` (`:67-72`)                                                                          |
| Observation history            | None (single step)                                  | `n_obs_steps`, `observation_delta_indices`         | Repo `robomimic/config/base_config.py:193`, `:195`; `robomimic/algo/bc.py:111` reads step 0 only                                                                                         |
| Action prediction              | One action per observation                          | `action_delta_indices`                             | Repo `robomimic/algo/bc.py:113`                                                                                                                                                          |
| MLP hidden sizes               | (1024, 1024)                                        | `actor_layer_dims`                                 | Repo `robomimic/config/bc_config.py:41`, `robomimic/scripts/generate_paper_configs.py:367`                                                                                               |
| Camera crop                    | Random 76×76 in training, center 76×76 at inference | `crop_shape`                                       | Repo `robomimic/scripts/generate_paper_configs.py:163-168`                                                                                                                               |
| Crop outside training          | Center                                              | `random_crop_at_inference` (False)                 | Repo `robomimic/models/obs_core.py:579`; robomimic v0.1 crops at random instead (`v0.1.0:robomimic/models/base_nets.py:1066`)                                                            |
| Spatial softmax keypoints      | 32                                                  | `spatial_softmax_num_kp`                           | Repo `robomimic/scripts/generate_paper_configs.py:157`                                                                                                                                   |
| Feature size per camera        | 64                                                  | `image_feature_dim`                                | Repo `robomimic/scripts/generate_paper_configs.py:152`                                                                                                                                   |
| Action head                    | Mixture of 5 Gaussians                              | `use_gmm` (True), `gmm_num_modes` (5)              | Repo `robomimic/scripts/generate_paper_configs.py:368` (human datasets), `robomimic/config/bc_config.py:53`                                                                              |
| GMM minimum standard deviation | 1e-4                                                | `gmm_min_std`                                      | Repo `robomimic/config/bc_config.py:54`                                                                                                                                                  |
| GMM noise outside training     | Every standard deviation set to 1e-4                | `gmm_low_noise_eval` (True)                        | Repo `robomimic/config/bc_config.py:56`                                                                                                                                                  |
| Loss without GMM               | Mean squared error only                             | `l2_weight` (1), `l1_weight` (0), `cos_weight` (0) | Repo `robomimic/config/bc_config.py:36-38`                                                                                                                                               |
| Normalization                  | None; images are only scaled to `[0, 1]`            | `normalization_mapping` (`IDENTITY`)               | Repo `robomimic/config/base_config.py:181` (observations), `:225` (actions), `robomimic/utils/obs_utils.py:921` (images); LeRobot already scales images (`datasets/io_utils.py:255-263`) |
| Optimizer                      | Adam                                                | `get_optimizer_preset()`                           | Repo `robomimic/config/bc_config.py:27`                                                                                                                                                  |
| Learning rate                  | 1e-4, constant                                      | `optimizer_lr`                                     | Repo `robomimic/config/bc_config.py:28`, `:30` (no decay epochs)                                                                                                                         |
| Weight decay                   | 0                                                   | `optimizer_weight_decay`                           | Repo `robomimic/config/bc_config.py:33`                                                                                                                                                  |
| Gradient clipping              | None                                                | `optimizer_grad_clip_norm` (0)                     | Repo `robomimic/config/base_config.py:239`; LeRobot skips clipping at 0 (`scripts/lerobot_train.py:178`)                                                                                 |

LeRobot feeds every non-action dataset feature to the policy
(`policies/factory.py:305-306`), while robomimic lists the observations each
experiment uses. `use_env_state` keeps that choice in the policy, so one
dataset serves both kinds of experiment.

Batch size and training length belong to `lerobot-train`, not the policy:
robomimic uses batch size 16 for 600 epochs of 500 steps (300K steps) on image
data (`robomimic/scripts/generate_paper_configs.py:113`, `:131-132`), and
batch size 100 for 2000 epochs of 100 steps (200K steps) on low-dim data
(`:43`, `:61-62`).

`tests/test_paper_defaults.py` checks these defaults against robomimic's own
config for BC in the image experiments on PH Lift. `use_gmm=false` gives
robomimic's plain `BC` class instead, which the paper uses only for
machine-generated datasets (`robomimic/scripts/generate_paper_configs.py:370-372`).

### 3. Camera encoder

`lerobot_policy_robomimic/vision.py` holds the camera encoder used by
robomimic's image experiments, robomimic's `VisualCore`
(Repo `robomimic/scripts/generate_paper_configs.py:151-160`). Tests compare
each part with robomimic's own module on the same weights and inputs
(`tests/test_vision.py`).

| Part                | Behavior                                                                                       | This port        | Source                                                                                                                                                    |
| ------------------- | ---------------------------------------------------------------------------------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Crop                | Random crop in training, center crop otherwise                                                 | `RandomCrop`     | Repo `robomimic/models/obs_core.py:489` (`CropRandomizer`), `:579` (center crop at evaluation)                                                            |
| Random crop offsets | `floor(rand × (size − crop))`, so 0 to 7 for 76 out of 84; the last offset is never drawn      | `RandomCrop`     | Repo `robomimic/utils/obs_utils.py:730`                                                                                                                   |
| Center crop offset  | `floor((size − crop) / 2)`, so 4 for 76 out of 84                                              | `RandomCrop`     | Repo `robomimic/utils/obs_utils.py:278`                                                                                                                   |
| Backbone            | ResNet-18 without its average pool and classifier, trained from scratch                        | `ImageEncoder`   | Repo `robomimic/models/base_nets.py:536` (`ResNet18Conv`); `robomimic/scripts/generate_paper_configs.py:154` (`pretrained = False`)                       |
| Spatial softmax     | 1×1 convolution to 32 keypoints, softmax over positions, expected `(x, y)` on a `[-1, 1]` grid | `SpatialSoftmax` | Repo `robomimic/models/base_nets.py:1143`, `:1162`, `:1216`; temperature fixed at 1 without noise (`robomimic/scripts/generate_paper_configs.py:158-160`) |
| Projection          | Linear layer from 32 × 2 keypoint coordinates to 64 features                                   | `ImageEncoder`   | Repo `robomimic/models/obs_core.py:138`                                                                                                                   |

robomimic averages the features of several crops (`robomimic/models/obs_core.py:589-598`);
with one crop per image, as in the paper experiments
(`robomimic/scripts/generate_paper_configs.py:168`), the average is the
feature itself, so `RandomCrop` takes exactly one.

robomimic v0.1, which trained the model zoo's checkpoints, had no separate
evaluation path: `CropRandomizer.forward_in` always took a random crop
(Repo `v0.1.0:robomimic/models/base_nets.py:1066`, called at
`v0.1.0:robomimic/models/obs_nets.py:271`), and the center crop arrived in
v0.3 (commit `40e427a`). The model zoo's success rates were therefore measured
with random crops at inference. `random_crop_at_inference=true` restores that
behavior for checkpoints converted from the model zoo.

`RandomCrop` stacks one slice per image instead of gathering all crops with
advanced indexing. Gathered crops are not contiguous, and on Apple's MPS
backend the backward pass through the ResNet then fails with `view size is
not compatible with input tensor's size and stride` (this port; found by the
`lerobot-train` run in §7).

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

| Part                | Behavior                                                                                                                         | Source                                                                                                                                                                                                     |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Vector observations | Concatenated as they are                                                                                                         | Repo `robomimic/config/base_config.py:284` (no encoder for low-dim observations), `robomimic/models/obs_nets.py:303-307`                                                                                   |
| Each camera         | Its own crop, `ImageEncoder`, and ReLU                                                                                           | Repo `robomimic/models/obs_nets.py:287-289` (randomizer per key), `:295` (encoder), `:297` (activation); the ReLU is the default `feature_activation` (`:440`, `:33`), which `MIMO_MLP` keeps (`:608-611`) |
| Feature order       | The order of the policy's input features                                                                                         | This port; robomimic sorts observation keys instead (Repo `robomimic/utils/file_utils.py:162`, passed on by `robomimic/scripts/train.py:242` and `robomimic/algo/algo.py:154-156`)                         |
| Leading dimensions  | Any, such as `(B,)` for BC or `(B, T)` for BC-RNN sequences; camera frames are encoded one by one, each with its own random crop | Repo `robomimic/models/obs_nets.py:858` (`time_distributed` over the encoder)                                                                                                                              |

robomimic's sorted order interleaves cameras and vector observations; for its
image experiments it is `agentview_image`, `robot0_eef_pos`,
`robot0_eef_quat`, `robot0_eye_in_hand_image`, `robot0_gripper_qpos`. The
order only matters when loading robomimic weights, where it decides the
columns of the first MLP layer. Keeping LeRobot's order leaves robomimic's
key names out of the encoder; the test reorders robomimic's output before
comparing.

### 5. MLP and action heads

`MLP` in `lerobot_policy_robomimic/mlp.py` maps the encoded observations to
features, and an action head in `lerobot_policy_robomimic/action_heads.py`
turns them into actions. Together they are robomimic's `ActorNetwork`
(Repo `robomimic/models/policy_nets.py:26`) or `GMMActorNetwork` (`:397`),
which BC builds with `actor_layer_dims` (`robomimic/algo/bc.py:87-92`).

| Part                | Behavior                                                                                           | Source                                                                                                                                              |
| ------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| MLP                 | A linear layer and a ReLU for each size in `actor_layer_dims`                                      | Repo `robomimic/models/obs_nets.py:617-623`; the last size is the MLP output and also gets a ReLU (`:623`, `robomimic/models/base_nets.py:256-258`) |
| `DeterministicHead` | Linear layer to the action size, then `tanh`                                                       | Repo `robomimic/models/obs_nets.py:627`, `:392` (`ObservationDecoder`), `robomimic/models/policy_nets.py:107`                                       |
| Its loss            | `l2_weight` × mean squared error + `l1_weight` × smooth L1 + `cos_weight` × cosine loss            | Repo `robomimic/algo/bc.py:182-192`                                                                                                                 |
| Its cosine loss     | Mean of 1 − cosine similarity over the first three action dimensions, the end-effector translation | Repo `robomimic/algo/bc.py:185`, `robomimic/utils/loss_utils.py:22-23`                                                                              |
| `GMMHead` outputs   | Linear layers to the means and scales of each mode's Gaussian and to the mode logits               | Repo `robomimic/models/policy_nets.py:489-491`                                                                                                      |
| Its means           | `tanh` of the output                                                                               | Repo `robomimic/models/policy_nets.py:514`                                                                                                          |
| Its scales          | `softplus` of the output plus `gmm_min_std`; 1e-4 outside training with `gmm_low_noise_eval`       | Repo `robomimic/models/policy_nets.py:519`, `:522`                                                                                                  |
| Its distribution    | A categorical choice of mode, then a diagonal Gaussian over the action dimensions                  | Repo `robomimic/models/policy_nets.py:526-535`                                                                                                      |
| Its loss            | Negative mean log-likelihood of the demonstrated actions                                           | Repo `robomimic/algo/bc.py:300`, `:322` (`BC_Gaussian`, which `BC_GMM` inherits at `:347`)                                                          |
| Its action          | A sample from the distribution                                                                     | Repo `robomimic/models/policy_nets.py:555`                                                                                                          |

robomimic splits the MLP into hidden layers and an output layer
(`robomimic/models/obs_nets.py:619-620`), but since both end in a ReLU, the
result is one linear layer and one ReLU per size, which is how `MLP` builds
it.

Even at inference, `GMMHead` samples the mode from the mixture weights instead
of taking the most likely one; low noise only shrinks the Gaussian around
that mode's mean, as in robomimic.

Each action head carries its own loss and action choice (this port). robomimic
instead pairs each network with an algorithm subclass, chosen from the config
flags (`robomimic/algo/bc.py:46-73`). Keeping the head separate from the MLP
lets the recurrent and transformer policies reuse both heads.

### 6. Policy

`RobomimicBCPolicy` chains the observation encoder, the MLP, and the action
head in the methods LeRobot's training and evaluation loops call, following
robomimic's `BC` and `BC_GMM` classes (Repo `robomimic/algo/bc.py:78`,
`:347`). Tests load the same weights into robomimic's algorithms, reordering
the input columns of the first MLP layer as in §4, and compare the losses and
actions of both heads (`tests/test_policy.py`).

| Method                 | Behavior                                                                | Source                                                                                                                                                                                                                                |
| ---------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `forward`              | The action head's loss and the values it logs                           | Repo `robomimic/algo/bc.py:182-192` (`BC`), `:300`, `:322` (`BC_GMM`)                                                                                                                                                                 |
| `select_action`        | The action head's action for the current observation, without gradients | Repo `robomimic/algo/bc.py:239-251`                                                                                                                                                                                                   |
| `predict_action_chunk` | The same action as a chunk of length 1                                  | This port: LeRobot expects `(B, chunk_size, action_dim)` (Adding a Policy guide); BC predicts one action                                                                                                                              |
| `__init__`             | Takes `**kwargs` and ignores them                                       | LeRobot: `make_policy` also passes `dataset_stats` and `dataset_meta` (`policies/factory.py:323`, `:326`); built-in policies take `**kwargs` (`policies/diffusion/modeling_diffusion.py:65-69`)                                       |
| `reset`                | Nothing to reset                                                        | Repo `robomimic/algo/algo.py:365` (BC keeps the base class's empty `reset`)                                                                                                                                                           |
| `get_optim_params`     | All parameters, returned as `self.parameters()`                         | Repo `robomimic/algo/algo.py:169-193` (one optimizer for the policy network); LeRobot passes the result straight to the optimizer (`optim/factory.py:37-40`), as Diffusion expects (`policies/diffusion/modeling_diffusion.py:89-90`) |

`__init__` and `get_optim_params` differ from the guide's template, whose
`__init__` takes only `dataset_stats` and whose `get_optim_params` returns
`{"params": ...}`. With LeRobot v0.6.1, the first raises `TypeError` for
`dataset_meta` and the second makes the optimizer iterate over the string
`"params"`; both surfaced in the `lerobot-train` run in §7.

Importing robomimic's `BC` also imports every other algorithm
(Repo `robomimic/algo/__init__.py`), so the test needs `diffusers` and
`imageio`. They are dev dependencies, with `diffusers` bounded as in
LeRobot's `diffusion` extra.

### 7. Processor

`make_robomimic_bc_pre_post_processors` returns LeRobot's default pipelines
(`processor/factory.py:116-128`, `:173-174`). Before the policy they rename
nothing, add the batch dimension, move tensors to the policy's device, and
normalize; after it they unnormalize and move the action to the CPU. With the
`IDENTITY` mapping of §2, normalization leaves every value as it is, even when
dataset statistics are given (`tests/test_processor.py`).

A 20-step `lerobot-train` run on the converted Lift image dataset trains,
saves a checkpoint, and loads it back with its processors for
`select_action`. `lerobot-train` needs `lerobot[training]`, available here as
the `training` extra, as in `lerobot_policy_openvla_oft`.

## BC-RNN (`robomimic_bc_rnn`)

BC-RNN is robomimic's main policy in its paper. It reuses the camera encoder
(§3), the observation encoder (§4), and the action heads (§5) of BC.

### 8. Configuration

`RobomimicBCRNNConfig` registers the policy type `robomimic_bc_rnn`. It keeps
the shared settings of §2, which robomimic's BC-RNN experiments also use, and
adds the settings below. Defaults follow the image experiments on PH data;
`tests/test_paper_defaults.py` checks them against robomimic's own config.

| Setting                  | Default                                   | Config field                                                       | Source                                                                           |
| ------------------------ | ----------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| Recurrent network        | LSTM, one direction                       | (fixed)                                                            | Repo `robomimic/config/bc_config.py:89`, `:92`                                   |
| LSTM layers              | 2                                         | `rnn_num_layers`                                                   | Repo `robomimic/config/bc_config.py:90`                                          |
| LSTM hidden size         | 1000 for image data; 400 for low-dim data | `rnn_hidden_dim` (1000)                                            | Repo `robomimic/scripts/generate_paper_configs.py:428` (image), `:416` (low-dim) |
| MLP after the LSTM       | None                                      | `actor_layer_dims` (())                                            | Repo `robomimic/scripts/generate_paper_configs.py:414`                           |
| Sequence length          | 10 steps of observations and actions      | `rnn_horizon`, `observation_delta_indices`, `action_delta_indices` | Repo `robomimic/scripts/generate_paper_configs.py:405`, `:409-410`               |
| State reset at inference | Every 10 steps                            | `rnn_horizon`                                                      | Repo `robomimic/scripts/generate_paper_configs.py:410`                           |

robomimic sets the sequence length (`train.seq_length`) and the reset period
(`algo.rnn.horizon`) separately, but its paper sets both to 10 and its config
notes that they should usually match (`robomimic/config/bc_config.py:87`).
This port uses one field, `rnn_horizon`, for both.

Every frame of a demo starts a training sequence, and a sequence that runs
past the end of its demo repeats the last frame (Repo
`robomimic/utils/dataset.py:225`, `:557-576`). LeRobot builds the same
sequences from the delta indices: it also starts one at every frame and clamps
indices past the end to the last frame (LeRobot
`datasets/dataset_reader.py:223`).

robomimic's open-loop mode (`robomimic/config/bc_config.py:91`), GRU, and
bidirectional LSTMs are not used by the paper experiments and are not ported.

### 9. Network

`RobomimicBCRNNPolicy` encodes each step of a sequence with the observation
encoder (§4), runs a PyTorch `nn.LSTM` over the steps, and applies the MLP and
action head of §5 at every step. Together they are robomimic's
`RNNActorNetwork` and `RNNGMMActorNetwork` (Repo
`robomimic/models/policy_nets.py:563`, `:728`).

| Part     | Behavior                                                                | Source                                                                                       |
| -------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Encoder  | The observation encoder of §4, applied to every step                    | Repo `robomimic/models/obs_nets.py:858`                                                      |
| LSTM     | Batch-first `nn.LSTM`, starting from zeros when no state is given       | Repo `robomimic/models/obs_nets.py:788`, `robomimic/models/base_nets.py:347`, `:372`, `:424` |
| Per step | The MLP (empty by default) and the action head, applied to every output | Repo `robomimic/models/obs_nets.py:765`, `:785`, `robomimic/models/base_nets.py:426`         |

robomimic wraps the LSTM in `RNN_Base` (Repo `robomimic/models/base_nets.py:306`),
which only adds a zero initial state; `nn.LSTM` already starts from zeros, so
this port uses it directly. A robomimic LSTM's state dict therefore loads into
`lstm` as it is, except that the input columns of `weight_ih_l0` follow
robomimic's sorted keys (§4).

### 10. Policy

`RobomimicBCRNNPolicy` follows robomimic's `BC_RNN` and `BC_RNN_GMM` classes
(Repo `robomimic/algo/bc.py:483`, `:578`). Tests load the same weights into
robomimic's algorithms and compare the sequence losses of both heads and the
actions of 25 consecutive steps across two state resets
(`tests/test_policy_bc_rnn.py`).

| Method                 | Behavior                                                                                       | Source                                                                                                                     |
| ---------------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `forward`              | The action head's loss over every step of every sequence, padded steps included                | Repo `robomimic/algo/bc.py:630`, `:652` (`BC_RNN_GMM`); `BC_RNN` uses `BC`'s loss (`:182-192`)                             |
| `select_action`        | Runs one LSTM step from the kept state; the state is cleared before steps 0, 10, 20, and so on | Repo `robomimic/algo/bc.py:551`, `:565`                                                                                    |
| `reset`                | Clears the LSTM state and the step count                                                       | Repo `robomimic/algo/bc.py:570-575`                                                                                        |
| `predict_action_chunk` | Not supported                                                                                  | This port: every action advances the LSTM state, and the guide allows `NotImplementedError` for policies that do not chunk |

The LSTM state does not last a whole episode: robomimic clears it every
`rnn_horizon` steps at inference, matching the 10-step sequences it was
trained on.

The policy builds its encoder and action head with the same
`ObservationEncoder.from_config` and `make_action_head` as BC.

### 11. Processor

`make_robomimic_bc_rnn_pre_post_processors` returns the same default pipelines
as BC (§7). They see one step at a time at inference, and whole sequences in
training, where normalization would apply to every step alike; with the
`IDENTITY` mapping it leaves them unchanged (`tests/test_processor.py`).

A 20-step `lerobot-train` run on the `train` mask of the converted Lift image
dataset trains the 35M-parameter policy, saves a checkpoint, and runs 12 steps
of `select_action` across a state reset after loading it back with its
processors.

### 12. Model zoo checkpoints

`lerobot_policy_robomimic/convert_checkpoint.py` turns a BC-RNN checkpoint of
robomimic's model zoo into a `robomimic_bc_rnn` policy directory with its
config, weights, and processors. The model zoo was trained with robomimic
v0.1 (Repo `docs/model_zoo/robomimic_v0.1.md`), so the converter reads that
version's format; robomimic v0.5 itself cannot load these image checkpoints,
because it upgrades only the config (`robomimic/utils/file_utils.py:247`) and
loads the weights by exact name (`robomimic/algo/algo.py:353`).

The checkpoint is read with `torch.load(weights_only=True)`, so no pickled
code runs. Its config is checked first, and settings this port does not
implement, such as a GRU or an open-loop RNN, stop the conversion with an
error.

| Part of robomimic v0.1                                              | Same as this port?                                           | Source                                                                                             |
| ------------------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| Camera encoder: ResNet-18, spatial softmax, linear projection, ReLU | Yes; the code differs only in default arguments              | Repo `v0.1.0:robomimic/models/base_nets.py:840-873`, `v0.1.0:robomimic/models/obs_nets.py:271-276` |
| Image orientation                                                   | Yes, flipped upright by the env wrapper                      | Repo `v0.1.0:robomimic/envs/env_robosuite.py:185`                                                  |
| Feature order                                                       | Yes, sorted keys                                             | The checkpoint's `shape_metadata["all_shapes"]`                                                    |
| LSTM state reset every `rnn_horizon` steps                          | Yes                                                          | Repo `v0.1.0:robomimic/algo/bc.py:521-535`                                                         |
| Crop at inference                                                   | No, random; converted configs set `random_crop_at_inference` | §3                                                                                                 |
| Weight names                                                        | No; renamed below                                            | The checkpoint's keys                                                                              |

| robomimic v0.1 weight                                                                                   | This port                                                         |
| ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `policy.nets.encoder.nets.obs.obs_nets.<camera>_image.vis_core.nets.*`                                  | `encoder.cameras.<camera>.encoder.backbone.*`                     |
| `…<camera>_image.pool_net.nets.*`                                                                       | `encoder.cameras.<camera>.encoder.pool.keypoints.*`               |
| `…<camera>_image.nets.3.*`                                                                              | `encoder.cameras.<camera>.encoder.projection.*`                   |
| `policy.nets.rnn.nets.*`                                                                                | `lstm.*`, with the input columns of `weight_ih_l0` reordered (§4) |
| `policy.nets.decoder.nets.{mean,scale,logits}.*`                                                        | `action_head.{mean,scale,logits}.*`                               |
| `…<camera>_image.nets.{0,1}.*`, `…pool_net.{temperature,pos_x,pos_y}`, `policy.nets.rnn.per_step_net.*` | Dropped: duplicates of the weights above, or fixed values (§3)    |

The weights then load with `strict=True`, so every weight of the policy must
come from the checkpoint.

Tests write a robomimic v0.5 BC-RNN network in the v0.1 format, convert it,
and compare 12 steps of actions with robomimic's own `get_action`; they also
convert the Lift checkpoint when it is downloaded
(`tests/test_convert_checkpoint.py`).

On the 1026 frames of the 20 `valid` demos of the converted Lift image
dataset, the converted Lift PH image checkpoint
(`lift_ph_image_epoch_500_succ_100.pth`, SHA-256 `37b94a11…bc47fb`) predicts the
demonstrated actions with a mean absolute error of 0.071 per dimension, or
0.073 with center crops. Two controls show that this error is meaningful:
always predicting zeros gives 0.263, and skipping the column reorder of
`weight_ih_l0` gives 0.305. The dataset was regenerated with robosuite 1.5.1,
while the checkpoint was trained on data from robosuite's `offline_study`
branch, so part of the error may come from that difference.

In closed loop, `lerobot-eval` in the `robomimic` env of the Lift image
dataset (50 episodes, seed 1000, about 1 s per episode on an Apple M5 Max)
gives the converted checkpoint a success rate of 100%, matching the model
zoo's reported ~100% (Repo `docs/model_zoo/robomimic_v0.1.md:35`). With
center crops at inference instead of v0.1's random crops (§3), it succeeds in
98% of the same episodes.

## BC-Transformer (`robomimic_bc_transformer`)

BC-Transformer is not part of robomimic's paper; robomimic added it in v0.3
(Repo commit `40e427a`). It reuses the camera encoder (§3), the observation
encoder (§4), and the action heads (§5) of BC.

### 13. Configuration

`RobomimicBCTransformerConfig` registers the policy type
`robomimic_bc_transformer` and keeps the shared settings of §2. The paper has
no BC-Transformer, so the transformer, optimizer, and schedule defaults follow
the template that robomimic's transformer tutorial calls tuned (Repo
`docs/tutorials/training_transformers.md`,
`robomimic/config/default_templates/bc_transformer.json`, cited below as
"template"). `tests/test_paper_defaults.py` checks them against that template,
loaded as `robomimic/scripts/train.py:475-479` loads a config.

| Setting                 | Default                                            | Config field                                                                              | Source                                                                                                                    |
| ----------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Observations seen       | The current step and the 9 before it               | `transformer_context_length` (10), `observation_delta_indices`                            | Template `:47` (`seq_length` 1), `:49` (`frame_stack` 10); Repo `robomimic/config/bc_config.py:97`                        |
| Steps before an episode | Copies of its first frame                          | (fixed)                                                                                   | Repo `robomimic/utils/dataset.py:557-576`; LeRobot clamps indices to the episode start (`datasets/dataset_reader.py:223`) |
| Supervised action       | The current step's only                            | `action_delta_indices` (None)                                                             | Repo `robomimic/algo/bc.py:734`; `supervise_all_steps` is off (`robomimic/config/bc_config.py:106`)                       |
| Transformer size        | Width 512, 6 blocks, 8 heads                       | `transformer_embed_dim`, `transformer_num_layers`, `transformer_num_heads`                | Repo `robomimic/config/bc_config.py:98-100`                                                                               |
| Dropout                 | 0.1 on embeddings, attention, and block outputs    | `transformer_emb_dropout`, `transformer_attn_dropout`, `transformer_block_output_dropout` | Repo `robomimic/config/bc_config.py:101-103`                                                                              |
| Optimizer               | AdamW, learning rate 1e-4, weight decay 0.01       | `get_optimizer_preset()`, `optimizer_lr`, `optimizer_weight_decay` (0.01)                 | Template `:63`, `:65`, `:71`                                                                                              |
| Learning rate schedule  | Falls linearly to 0.1× over 100 epochs, then stays | `scheduler_decay_epochs` (100), `scheduler_decay_factor` (0.1)                            | Template `:66-68`; Repo `robomimic/utils/torch_utils.py:146-157`                                                          |
| Epoch length            | 100 optimizer steps                                | `scheduler_steps_per_epoch` (100)                                                         | Template `:21` (`epoch_every_n_steps`)                                                                                    |

robomimic updates its learning rate schedule once at the end of every epoch
(Repo `robomimic/algo/algo.py:313-315`, called from
`robomimic/scripts/train.py:311`), while LeRobot steps a scheduler after every
optimizer step (LeRobot `scripts/lerobot_train.py:193`) and has no epochs.
`RobomimicLinearSchedulerConfig` in `lerobot_policy_robomimic/schedulers.py`,
registered as the scheduler type `robomimic_linear`, therefore counts steps
in epochs of `scheduler_steps_per_epoch` and keeps the learning rate fixed
within each one. `tests/test_schedulers.py` compares it step by step with
robomimic's own scheduler stepped once per epoch.

The template trains on low-dim data with batch size 100 for 2000 epochs of
100 steps (200K steps; template `:56-57`, `:21`); pass these to
`lerobot-train`.

robomimic's options to supervise every step (`supervise_all_steps`), predict
future actions (`pred_future_acs`), use sinusoidal or `nn.Embedding` position
embeddings, or use GEGLU are off in the template and are not ported.

### 14. Transformer

`Transformer` in `lerobot_policy_robomimic/transformer.py` maps a sequence of
encoded observations to one feature per step. It covers the embedding part of
robomimic's `MIMO_Transformer` (Repo `robomimic/models/obs_nets.py:997-1022`,
`:1088-1102`) and its `GPT_Backbone` (Repo
`robomimic/models/transformers.py`). A test loads robomimic's weights and
compares the outputs (`tests/test_transformer.py`).

| Part            | Behavior                                                                                                                                            | Source                                                                                       |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Input embedding | Linear projection to the width, plus a learned position per step, then LayerNorm and dropout                                                        | Repo `robomimic/models/obs_nets.py:997`, `:1009`, `:1016`, `:1019`, `:1102`                  |
| Attention       | Causal multi-head self-attention, with a bias-free query-key-value projection                                                                       | Repo `robomimic/models/transformers.py:133`, `:143`, `:179-184`                              |
| Block           | Pre-norm: attention and a 4× GELU MLP, each added to its input                                                                                      | Repo `robomimic/models/transformers.py:276`, `:291-292`                                      |
| Output          | LayerNorm after the last block                                                                                                                      | Repo `robomimic/models/transformers.py:392`                                                  |
| Initialization  | Blocks and output norm start from normal(0, 0.02) weights and zero biases; the input projection keeps PyTorch's default and positions start at zero | Repo `robomimic/models/transformers.py:362`, `:394-399`; `robomimic/models/obs_nets.py:1009` |

Attention uses PyTorch's `scaled_dot_product_attention` with `is_causal=True`
(this port), which computes the same masked softmax as robomimic's explicit
mask (`robomimic/models/transformers.py:143`, `:184`) within floating-point
tolerance, without storing the mask.

### 15. Policy

`RobomimicBCTransformerPolicy` encodes the observations of its context, runs
the transformer of §14, and applies the action head of §5 to the last step,
following robomimic's `BC_Transformer` and `BC_Transformer_GMM` classes (Repo
`robomimic/algo/bc.py:677`, `:794`). Tests load the same weights into
robomimic's algorithms and compare the losses of both heads, the GMM at the
last step, and 13 steps of actions on padded observation windows
(`tests/test_policy_bc_transformer.py`).

| Part                   | Behavior                                                                                        | Source                                                                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Camera features        | No ReLU after each camera's encoder, unlike BC and BC-RNN                                       | Repo `robomimic/models/obs_nets.py:991` (`feature_activation=None`)                                                      |
| `forward`              | The action head's loss for the action of the last step of each context                          | Repo `robomimic/algo/bc.py:734`, `:766`                                                                                  |
| `select_action`        | Keeps the last `transformer_context_length` observations; an episode's first one fills them all | Repo `robomimic/envs/wrappers.py:97`, `:130-131` (`FrameStackWrapper`, set up at `robomimic/utils/env_utils.py:342-343`) |
| Action                 | The action head's action at the last step                                                       | Repo `robomimic/algo/bc.py:769`, `:788`                                                                                  |
| `reset`                | Forgets the stored observations                                                                 | Repo `robomimic/envs/wrappers.py:152-165`, which rebuilds the history with `:119-131`                                    |
| `predict_action_chunk` | Not supported                                                                                   | This port: like BC-RNN (§10), each action depends on the observations kept by `select_action`                            |

`ObservationEncoder.from_config` takes `camera_activation`, which only this
policy turns off, so the three policies share one encoder.

robomimic computes BC-Transformer's GMM loss with training scales even outside
training (Repo `robomimic/algo/bc.py:836`), while BC and BC-RNN use the
low-noise scales there; `GMMHead` follows the latter for all three. The two
agree on every training step and differ only for losses computed in
evaluation mode.

robomimic samples an action for every step of the context and keeps the last
(`robomimic/algo/bc.py:788`); this port samples only the last step, which has
the same distribution but draws different random numbers, so tests compare the
distribution rather than samples.

### 16. Processor

`make_robomimic_bc_transformer_pre_post_processors` returns the same default
pipelines as BC (§7), which leave every feature unnormalized
(`tests/test_processor.py`).

A 20-step `lerobot-train` run on the `train` mask of the converted Lift image
dataset trains the 41M-parameter policy with AdamW and the
`robomimic_linear` schedule, saves a checkpoint whose training config keeps
that scheduler type, and runs 12 steps of `select_action` after loading it
back with its processors. With the epoch shortened to 2 steps and the decay
to 5 epochs, the logged learning rate, LeRobot's average over each 5-step
logging window, reads 7.8e-5, 3.2e-5, and then 1.0e-5, as the schedule
predicts.
