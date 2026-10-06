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

| Decision                      | Choice                                                                                                                                      | Source                                                                                                                                                                                                                 |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Policy types                  | `robomimic_bc` (BC), `robomimic_bc_rnn` (BC-RNN)                                                                                            | This port: the `robomimic_` prefix avoids clashing with LeRobot's built-in types, such as `diffusion` (`policies/diffusion/`), which robomimic also implements (Repo `robomimic/algo/diffusion_policy.py`)             |
| Class and function names      | `RobomimicBCConfig`, `RobomimicBCPolicy`, `make_robomimic_bc_pre_post_processors`; likewise `RobomimicBCRNN…` and `make_robomimic_bc_rnn_…` | LeRobot: the policy class name is the config class name with `Config` replaced by `Policy` (`policies/factory.py:409-415`), and the processor factory is `make_<type>_pre_post_processors` (`policies/factory.py:458`) |
| Module names                  | `configuration_<type>.py`, `modeling_<type>.py`, `processor_<type>.py`                                                                      | LeRobot: the modeling and processor modules are found by replacing `configuration_` in the config's module path (`policies/factory.py:416`, `:459`); the names follow the guide's template                             |
| Distribution name             | `lerobot_policy_robomimic`                                                                                                                  | LeRobot: an installed distribution whose name starts with `lerobot_policy_` is imported by that name (`utils/import_utils.py:231-255`), which runs `@PreTrainedConfig.register_subclass`                               |
| Several policies, one package | Each robomimic algorithm gets its own policy type and its own three modules                                                                 | This port: the guide shows one policy per package, but the factory only needs those three modules per type, so robomimic's algorithms can share one package and its networks                                           |

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
