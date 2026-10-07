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

For BC-VAE, use `--policy.type=robomimic_bc_vae`. The paper has no BC-VAE
experiment; robomimic's paper configs pick the batch size and length by data
type alone, so the values above for BC apply (§17).

To condition any of these policies on the task's CLIP embedding, install the
`language` extra and add `--policy.language_conditioning=film`, or `concat`
(§21-23). The task is the `--task` string given to the converter; datasets
converted before the converter kept it in the env args need a `"lang"` entry
added to `meta/robomimic_env_args.json` for `lerobot-eval`.

```bash
uv sync --extra training --extra language
uv run lerobot-train \
    --policy.type=robomimic_bc_rnn \
    --policy.language_conditioning=film \
    --dataset.repo_id=<user>/robomimic_lift_ph_image \
    --dataset.episodes="$EPISODES" \
    --batch_size=16 \
    --steps=300000 \
    --policy.push_to_hub=false
```

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
| Language instruction  | The `--task` string, with every frame and under `lang` in `meta/robomimic_env_args.json`                        | Repo `robomimic/utils/env_utils.py:231` (robomimic reads an env's instruction from `lang` in its env args)                                             |
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

| Part             | Behavior                                                                                                   | Source                                                                                                                          |
| ---------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Construction     | `robosuite.make` with the env args, rendering only the requested cameras and always the object state       | Repo `robomimic/envs/env_robosuite.py:87`, `:116`                                                                               |
| Images           | Flipped upright, as `pixels/<camera>`                                                                      | Repo `robomimic/envs/env_robosuite.py:255`                                                                                      |
| Proprioception   | The dataset's state keys, concatenated as `agent_pos`                                                      | § Datasets                                                                                                                      |
| Object state     | robosuite's `object-state`, as `environment_state`                                                         | Repo `robomimic/envs/env_robosuite.py:267`                                                                                      |
| Success and end  | `_check_success()`; an episode ends on success                                                             | Repo `robomimic/envs/env_robosuite.py:407`; `robomimic/scripts/generate_paper_configs.py:53`, `:123` (`terminate_on_success`)   |
| Episode length   | 400 steps for Lift, Can, and Square; 700 for Transport and Tool Hang                                       | Repo `robomimic/__init__.py:63`                                                                                                 |
| Initial states   | robosuite's random reset; a seed sets NumPy's global generator, which robosuite 1.5.1 places objects with  | robosuite 1.5.1 `robosuite/utils/placement_samplers.py:167`                                                                     |
| Task description | The env args' `lang`, or else the env name; `lerobot-eval` passes it to the preprocessor as the task (§21) | Repo `robomimic/utils/env_utils.py:231`; LeRobot `scripts/lerobot_eval.py:281`                                                  |
| Batched envs     | Each env stops simulating once its episode ends, with `NEXT_STEP` autoreset                                | LeRobot `envs/utils.py:241` (`freeze_after_episode_end`), `envs/libero.py:526`                                                  |
| Features         | Read from a robosuite env built without cameras when the config is created                                 | This port: LeRobot replaces a loaded policy's action feature with the env's (`policies/factory.py:304`), so sizes must be exact |

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

| Decision                      | Choice                                                                                                                                                                 | Source                                                                                                                                                                                                                 |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Policy types                  | `robomimic_bc` (BC), `robomimic_bc_rnn` (BC-RNN), `robomimic_bc_transformer` (BC-Transformer), `robomimic_bc_vae` (BC-VAE)                                             | This port: the `robomimic_` prefix avoids clashing with LeRobot's built-in types, such as `diffusion` (`policies/diffusion/`), which robomimic also implements (Repo `robomimic/algo/diffusion_policy.py`)             |
| Class and function names      | `RobomimicBCConfig`, `RobomimicBCPolicy`, `make_robomimic_bc_pre_post_processors`; likewise for `robomimic_bc_rnn`, `robomimic_bc_transformer`, and `robomimic_bc_vae` | LeRobot: the policy class name is the config class name with `Config` replaced by `Policy` (`policies/factory.py:409-415`), and the processor factory is `make_<type>_pre_post_processors` (`policies/factory.py:458`) |
| Module names                  | `configuration_<type>.py`, `modeling_<type>.py`, `processor_<type>.py`                                                                                                 | LeRobot: the modeling and processor modules are found by replacing `configuration_` in the config's module path (`policies/factory.py:416`, `:459`); the names follow the guide's template                             |
| Distribution name             | `lerobot_policy_robomimic`                                                                                                                                             | LeRobot: an installed distribution whose name starts with `lerobot_policy_` is imported by that name (`utils/import_utils.py:231-255`), which runs `@PreTrainedConfig.register_subclass`                               |
| Several policies, one package | Each robomimic algorithm gets its own policy type and its own three modules                                                                                            | This port: the guide shows one policy per package, but the factory only needs those three modules per type, so robomimic's algorithms can share one package and its networks                                           |

## BC (`robomimic_bc`)

### 2. Configuration

`RobomimicBCConfig` registers the policy type `robomimic_bc`. Defaults follow
robomimic's image experiments on the proficient-human (PH) datasets. Every
setting below except `actor_layer_dims` comes from
`lerobot_policy_robomimic/base_config.py`, which holds two classes that are
not policy types themselves: `RobomimicPolicyConfig`, with the observation,
encoder, and optimizer settings that all robomimic policies here share, and
its subclass `RobomimicActorConfig`, which adds the action head settings that
BC, BC-RNN, and BC-Transformer share.

| Setting                          | Default                                               | Config field                                                                       | Source                                                                                                                                                                                   |
| -------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Observations                     | Proprioception and cameras; no object state           | `use_env_state` (False)                                                            | Repo `robomimic/scripts/generate_paper_configs.py:138-146` (image); low-dim experiments add `object` (`:67-72`)                                                                          |
| Observation history              | None (single step)                                    | `n_obs_steps`, `observation_delta_indices`                                         | Repo `robomimic/config/base_config.py:193`, `:195`; `robomimic/algo/bc.py:111` reads step 0 only                                                                                         |
| Action prediction                | One action per observation                            | `action_delta_indices`                                                             | Repo `robomimic/algo/bc.py:113`                                                                                                                                                          |
| MLP hidden sizes                 | (1024, 1024)                                          | `actor_layer_dims`                                                                 | Repo `robomimic/config/bc_config.py:41`, `robomimic/scripts/generate_paper_configs.py:367`                                                                                               |
| Camera crop                      | Random 76×76 in training, center 76×76 at inference   | `crop_shape`                                                                       | Repo `robomimic/scripts/generate_paper_configs.py:163-168`                                                                                                                               |
| Crop outside training            | Center                                                | `random_crop_at_inference` (False)                                                 | Repo `robomimic/models/obs_core.py:579`; robomimic v0.1 crops at random instead (`v0.1.0:robomimic/models/base_nets.py:1066`)                                                            |
| Spatial softmax keypoints        | 32                                                    | `spatial_softmax_num_kp`                                                           | Repo `robomimic/scripts/generate_paper_configs.py:157`                                                                                                                                   |
| Feature size per camera          | 64                                                    | `image_feature_dim`                                                                | Repo `robomimic/scripts/generate_paper_configs.py:152`                                                                                                                                   |
| Action head                      | Mixture of 5 Gaussians                                | `use_gmm` (True), `gmm_num_modes` (5)                                              | Repo `robomimic/scripts/generate_paper_configs.py:368` (human datasets), `robomimic/config/bc_config.py:53`                                                                              |
| GMM minimum standard deviation   | 1e-4                                                  | `gmm_min_std`                                                                      | Repo `robomimic/config/bc_config.py:54`                                                                                                                                                  |
| GMM noise outside training       | Every standard deviation set to 1e-4                  | `gmm_low_noise_eval` (True)                                                        | Repo `robomimic/config/bc_config.py:56`                                                                                                                                                  |
| Gaussian head                    | Off; when on, it takes precedence over the GMM head   | `use_gaussian` (False)                                                             | Repo `robomimic/config/bc_config.py:44`, `robomimic/algo/bc.py:45-52`                                                                                                                    |
| Gaussian standard deviation      | Learned, 0.1 for a zero network output, at least 0.01 | `gaussian_fixed_std` (False), `gaussian_init_std` (0.1), `gaussian_min_std` (0.01) | Repo `robomimic/config/bc_config.py:45-47`                                                                                                                                               |
| Gaussian action outside training | The mean                                              | `gaussian_low_noise_eval` (True)                                                   | Repo `robomimic/config/bc_config.py:49`                                                                                                                                                  |
| Loss of the deterministic head   | Mean squared error only                               | `l2_weight` (1), `l1_weight` (0), `cos_weight` (0)                                 | Repo `robomimic/config/bc_config.py:36-38`                                                                                                                                               |
| Normalization                    | None; images are only scaled to `[0, 1]`              | `normalization_mapping` (`IDENTITY`)                                               | Repo `robomimic/config/base_config.py:181` (observations), `:225` (actions), `robomimic/utils/obs_utils.py:921` (images); LeRobot already scales images (`datasets/io_utils.py:255-263`) |
| Optimizer                        | Adam                                                  | `get_optimizer_preset()`                                                           | Repo `robomimic/config/bc_config.py:27`                                                                                                                                                  |
| Learning rate                    | 1e-4, constant                                        | `optimizer_lr`                                                                     | Repo `robomimic/config/bc_config.py:28`, `:30` (no decay epochs)                                                                                                                         |
| Weight decay                     | 0                                                     | `optimizer_weight_decay`                                                           | Repo `robomimic/config/bc_config.py:33`                                                                                                                                                  |
| Gradient clipping                | None                                                  | `optimizer_grad_clip_norm` (0)                                                     | Repo `robomimic/config/base_config.py:239`; LeRobot skips clipping at 0 (`scripts/lerobot_train.py:178`)                                                                                 |

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
`use_gaussian=true` gives robomimic's `BC_Gaussian`, which the paper does not
use; its settings keep robomimic's defaults, which the same test checks.

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
(Repo `robomimic/models/policy_nets.py:26`), `GaussianActorNetwork` (`:193`), or
`GMMActorNetwork` (`:397`),
which BC builds with `actor_layer_dims` (`robomimic/algo/bc.py:87-92`).

| Part                   | Behavior                                                                                                                                                                                                                     | Source                                                                                                                                                           |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MLP                    | A linear layer and a ReLU for each size in `actor_layer_dims`                                                                                                                                                                | Repo `robomimic/models/obs_nets.py:617-623`; the last size is the MLP output and also gets a ReLU (`:623`, `robomimic/models/base_nets.py:256-258`)              |
| `DeterministicHead`    | Linear layer to the action size, then `tanh`                                                                                                                                                                                 | Repo `robomimic/models/obs_nets.py:627`, `:392` (`ObservationDecoder`), `robomimic/models/policy_nets.py:107`                                                    |
| Its loss               | `l2_weight` × mean squared error + `l1_weight` × smooth L1 + `cos_weight` × cosine loss                                                                                                                                      | Repo `robomimic/algo/bc.py:182-192`                                                                                                                              |
| Its cosine loss        | Mean of 1 − cosine similarity over the first three action dimensions, the end-effector translation                                                                                                                           | Repo `robomimic/algo/bc.py:185`, `robomimic/utils/loss_utils.py:22-23`                                                                                           |
| `GaussianHead` outputs | Linear layers to the mean and the scale of one diagonal Gaussian; no scale layer with `gaussian_fixed_std`                                                                                                                   | Repo `robomimic/models/policy_nets.py:318-319`, `:338`                                                                                                           |
| Its means              | Clamped to [−9, 9], then `tanh`                                                                                                                                                                                              | Repo `robomimic/models/policy_nets.py:341`, `:345`; `BC_Gaussian` keeps the default limits and no tanh-Gaussian (`:207`, `:210`, `robomimic/algo/bc.py:263-274`) |
| Its scales             | `softplus` of the output, scaled so that zero gives `gaussian_init_std`, or `gaussian_init_std` itself with `gaussian_fixed_std`; clamped to [`gaussian_min_std`, 7.5]; 1e-4 outside training with `gaussian_low_noise_eval` | Repo `robomimic/models/policy_nets.py:280-282`, `:338`, `:350`, `:355`; `robomimic/algo/bc.py:272`                                                               |
| Its loss               | Negative mean log-likelihood of the demonstrated actions                                                                                                                                                                     | Repo `robomimic/algo/bc.py:300`, `:322`                                                                                                                          |
| Its action             | The mean outside training with `gaussian_low_noise_eval`, a sample otherwise                                                                                                                                                 | Repo `robomimic/models/policy_nets.py:386-388`                                                                                                                   |
| `GMMHead` outputs      | Linear layers to the means and scales of each mode's Gaussian and to the mode logits                                                                                                                                         | Repo `robomimic/models/policy_nets.py:489-491`                                                                                                                   |
| Its means              | `tanh` of the output                                                                                                                                                                                                         | Repo `robomimic/models/policy_nets.py:514`                                                                                                                       |
| Its scales             | `softplus` of the output plus `gmm_min_std`; 1e-4 outside training with `gmm_low_noise_eval`                                                                                                                                 | Repo `robomimic/models/policy_nets.py:519`, `:522`                                                                                                               |
| Its distribution       | A categorical choice of mode, then a diagonal Gaussian over the action dimensions                                                                                                                                            | Repo `robomimic/models/policy_nets.py:526-535`                                                                                                                   |
| Its loss               | Negative mean log-likelihood of the demonstrated actions                                                                                                                                                                     | Repo `robomimic/algo/bc.py:300`, `:322` (`BC_Gaussian`, which `BC_GMM` inherits at `:347`)                                                                       |
| Its action             | A sample from the distribution                                                                                                                                                                                               | Repo `robomimic/models/policy_nets.py:555`                                                                                                                       |

robomimic splits the MLP into hidden layers and an output layer
(`robomimic/models/obs_nets.py:619-620`), but since both end in a ReLU, the
result is one linear layer and one ReLU per size, which is how `MLP` builds
it.

Even at inference, `GMMHead` samples the mode from the mixture weights instead
of taking the most likely one; low noise only shrinks the Gaussian around
that mode's mean, as in robomimic.

robomimic still creates the scale layer of a Gaussian with a fixed standard
deviation and leaves it unused (`robomimic/models/policy_nets.py:318-319`,
`:338`); `GaussianHead` leaves it out (this port).

Each action head carries its own loss and action choice (this port). robomimic
instead pairs each network with an algorithm subclass, chosen from the config
flags (`robomimic/algo/bc.py:46-73`). Keeping the head separate from the MLP
lets the recurrent and transformer policies reuse the deterministic and GMM
heads. robomimic has no Gaussian BC-RNN or BC-Transformer
(`robomimic/algo/bc.py:46-49`), so their configs reject `use_gaussian`.

robomimic can also output the standard deviations of both distributions with
`exp` instead of `softplus` (`robomimic/config/bc_config.py:48`, `:55`); its
defaults and paper experiments use `softplus`, and `exp` is not ported.

### 6. Policy

`RobomimicBCPolicy` chains the observation encoder, the MLP, and the action
head in the methods LeRobot's training and evaluation loops call, following
robomimic's `BC`, `BC_Gaussian`, and `BC_GMM` classes (Repo
`robomimic/algo/bc.py:78`, `:254`, `:347`). Tests load the same weights into
robomimic's algorithms, reordering the input columns of the first MLP layer as
in §4, and compare the losses and actions of all three heads
(`tests/test_policy.py`).

| Method                 | Behavior                                                                | Source                                                                                                                                                                                                                                |
| ---------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `forward`              | The action head's loss and the values it logs                           | Repo `robomimic/algo/bc.py:182-192` (`BC`), `:300`, `:322` (`BC_Gaussian`, `BC_GMM`)                                                                                                                                                  |
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

On CUDA, cuDNN keeps the LSTM's weights as views of one buffer after
`flatten_parameters` (PyTorch 2.11 `torch/nn/modules/rnn.py:237`). LeRobot saves a
policy with safetensors' `save_model` (`policies/pretrained.py:160`), which
refuses tensors that share a buffer none of them covers, so a BC-RNN
checkpoint failed to save on a cloud GPU (this port; found by the benchmark
run). `RobomimicBCRNNPolicy._save_pretrained` saves copies of the weights
instead. CPU and MPS keep separate weights, so a test makes the LSTM's
weights share one buffer to reproduce the failure (`tests/test_policy_bc_rnn.py`).

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

## BC-VAE (`robomimic_bc_vae`)

BC-VAE is a variant of BC that robomimic has had since v0.1 (Repo
`robomimic/algo/bc.py:373`, `v0.1.0:robomimic/algo/bc.py:345`;
`docs/introduction/implemented_algorithms.md`). The paper does not train it as
a policy; it only uses a VAE as the action sampler of BCQ
(`robomimic/scripts/generate_paper_configs.py:463-470`). It reuses the camera
encoder (§3) and the observation encoder (§4) of BC, but no action head: a
conditional VAE models the action instead.

### 17. Configuration

`RobomimicBCVAEConfig` registers the policy type `robomimic_bc_vae`. It
inherits only `RobomimicPolicyConfig`, the observation, encoder, and optimizer
settings of §2, since BC-VAE has no action head. BC-VAE uses the same
optimizer as BC (Repo `robomimic/config/bc_config.py:27-33`). The VAE settings
below keep robomimic's defaults; `tests/test_paper_defaults.py` checks them
against robomimic's config for BC-VAE in the image experiments.

| Setting                  | Default                                      | Config field                                                                                         | Source                                                                                                                                  |
| ------------------------ | -------------------------------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Latent size              | 14, twice the size of a 7-dimensional action | `vae_latent_dim`                                                                                     | Repo `robomimic/config/bc_config.py:60`                                                                                                 |
| Latent bound             | None                                         | `vae_latent_clip`                                                                                    | Repo `robomimic/config/bc_config.py:61`                                                                                                 |
| KL weight                | 1                                            | `vae_kl_weight`                                                                                      | Repo `robomimic/config/bc_config.py:62`                                                                                                 |
| Encoder and decoder MLPs | (300, 400) each                              | `vae_encoder_layer_dims`, `vae_decoder_layer_dims`                                                   | Repo `robomimic/config/bc_config.py:81-82`                                                                                              |
| Decoder input            | Latent and observation                       | (fixed)                                                                                              | Repo `robomimic/config/bc_config.py:65`; robomimic conditions the decoder whatever this says (`robomimic/models/vae_nets.py:1049-1050`) |
| Reconstruction loss      | Mean of the squared errors                   | `vae_reconstruction_sum_across_elements` (False)                                                     | Repo `robomimic/config/bc_config.py:66`, `robomimic/models/vae_nets.py:1277-1284`                                                       |
| Prior                    | Fixed N(0, 1)                                | `vae_prior_learn` (False)                                                                            | Repo `robomimic/config/bc_config.py:69`                                                                                                 |
| Learned prior            | One Gaussian, independent of the observation | `vae_prior_is_conditioned` (False), `vae_prior_use_gmm` (False), `vae_prior_layer_dims` ((300, 400)) | Repo `robomimic/config/bc_config.py:70-71`, `:83`                                                                                       |
| Mixture prior            | 10 modes with uniform weights                | `vae_prior_gmm_num_modes` (10), `vae_prior_gmm_learn_weights` (False)                                | Repo `robomimic/config/bc_config.py:72-73`                                                                                              |

The config raises `ValueError` for the combinations robomimic rejects with
assertions: a conditioned or mixture prior that is not learned (Repo
`robomimic/models/vae_nets.py:943`, `:983`).

robomimic's `decoder.is_conditioned` only feeds an assertion that the decoder
or the prior is conditioned (`robomimic/models/vae_nets.py:940`); the decoder
gets the observation group whenever the VAE has observations (`:1049-1050`,
likewise at `v0.1.0:robomimic/models/vae_nets.py:1109`). Setting it to false
in robomimic changes nothing, so it has no field here, and a test sets it to
false in robomimic and still matches (`tests/test_vae.py`).

robomimic's categorical prior (`robomimic/config/bc_config.py:74-79`) is not
ported. It lowers the Gumbel-softmax temperature by a fixed step every epoch
(`robomimic/algo/bc.py:393-400`), and LeRobot gives the policy no epochs.

### 18. Network

`ActionVAE` in `lerobot_policy_robomimic/vae.py` is robomimic's `VAE` as
`VAEActor` builds it for actions (Repo `robomimic/models/policy_nets.py:1336`,
`robomimic/models/vae_nets.py:747`). Its three parts are each a
`ConditionedMLP`, robomimic's `MIMO_MLP` (`robomimic/models/obs_nets.py:541`):
an observation encoder of §4, an MLP, and one linear layer per output.
`tests/test_vae.py` loads the same weights into robomimic's `VAE`, reordering
the observation columns of each first MLP layer as in §4, and compares the
losses and the sampled actions of eight prior and decoder settings.

| Part            | Behavior                                                                                                                                       | Source                                                                                                                       |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Posterior       | The action, then the encoded observation, through the MLP to a latent mean and log variance                                                    | Repo `robomimic/models/vae_nets.py:1014-1016`, `:1026-1029`                                                                  |
| Decoder         | The latent, then the encoded observation, through the MLP to the action, bounded by `tanh`                                                     | Repo `robomimic/models/vae_nets.py:1048-1050`, `:1192`; `robomimic/models/policy_nets.py:1398-1399`                          |
| Prior           | N(0, 1); or learned parameters, initialized from N(0, 1) divided by the square root of their size; or an MLP over its own encoded observation  | Repo `robomimic/models/vae_nets.py:419`, `:128`, `:1068-1069`                                                                |
| Mixture weights | Uniform, or learned and normalized by `log_softmax`                                                                                            | Repo `robomimic/models/vae_nets.py:403`, `:499`                                                                              |
| Latent sample   | Mean plus standard deviation times Gaussian noise, with the log standard deviation clamped to [−4, 15]                                         | Repo `robomimic/utils/torch_utils.py:77`, `:83`                                                                              |
| Prior sample    | A mode drawn from the mixture weights, then a latent from that mode; clamped to `vae_latent_clip` if set                                       | Repo `robomimic/models/vae_nets.py:406-412`, `:422`                                                                          |
| KL loss         | Closed form against N(0, 1) or a learned Gaussian; against a mixture, estimated at the posterior sample with log variances clamped to [−8, 30] | Repo `robomimic/models/vae_nets.py:452`, `:462`, `:470-474`; `robomimic/utils/loss_utils.py:39`, `:56-60`, `:78`, `:104-119` |
| Reconstruction  | Squared error of the decoded posterior sample, averaged over everything or summed per sample                                                   | Repo `robomimic/models/vae_nets.py:1277-1284`                                                                                |

Each part has its own observation encoder, as in robomimic, where every
`MIMO_MLP` builds one (`robomimic/models/obs_nets.py:608`): with two
cameras, the posterior and the decoder hold four ResNet-18s between them, and
a conditioned prior adds two more.

### 19. Policy

`RobomimicBCVAEPolicy` wraps `ActionVAE` in the methods LeRobot's training and
evaluation loops call, following robomimic's `BC_VAE` class (Repo
`robomimic/algo/bc.py:373`). Tests load the same weights into robomimic's
algorithm with a learned, observation-conditioned mixture prior and a KL
weight of 0.5, and compare the loss, the logged values, and the actions
(`tests/test_policy_bc_vae.py`).

| Method                 | Behavior                                                                                      | Source                                                                           |
| ---------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `forward`              | Reconstruction loss plus `vae_kl_weight` × KL loss; logs both and the mean posterior variance | Repo `robomimic/algo/bc.py:403-433`, `:452`, `:477`                              |
| `select_action`        | An action decoded from a latent drawn from the prior, one per observation, without gradients  | Repo `robomimic/algo/bc.py:239-251`; `robomimic/models/policy_nets.py:1565-1570` |
| `predict_action_chunk` | The same action as a chunk of length 1                                                        | This port, as for BC (§6)                                                        |
| `reset`                | Nothing to reset                                                                              | Repo `robomimic/algo/algo.py:365`                                                |
| `get_optim_params`     | All parameters, returned as `self.parameters()`                                               | Repo `robomimic/algo/algo.py:169-193`; LeRobot as for BC (§6)                    |

`BC_VAE` also reads a `freeze_encoder` flag from the batch, which stops the
gradient into the posterior (`robomimic/algo/bc.py:419`). Only BCQ sets it,
for its own action sampler (`robomimic/algo/bcq.py:242-243`), so it is not
ported.

### 20. Processor

`make_robomimic_bc_vae_pre_post_processors` returns the same default
pipelines as BC (§7), which leave every feature unnormalized
(`tests/test_processor.py`).

A 20-step `lerobot-train` run on the `train` mask of the converted Lift image
dataset trains BC-VAE with a learned, observation-conditioned mixture prior,
saves a checkpoint, and runs 12 steps of `select_action` after loading it
back with its processors. The policy has 68M parameters, mostly the six
ResNet-18s of three observation encoders with two cameras each (§18). The
logged KL loss can be negative with a mixture prior, since robomimic
estimates it from one posterior sample (§18).

## Language conditioning

robomimic added language conditioning in v0.5 (Repo commit `ae5799f`,
`docs/tutorials/language_conditioning.md`). It is not part of the paper and
works with every policy here, since it lives in the observation encoder. The
paper's tasks each have one fixed instruction, so robomimic gives every demo
of a dataset the same one.

### 21. Task embedding

`CLIPTaskEmbeddingStep` in `lerobot_policy_robomimic/language.py` embeds the
task string that LeRobot keeps with every frame, and that `lerobot-eval` takes
from the env's `task_description` (LeRobot `scripts/lerobot_eval.py:281`). It
adds the embedding to the observation as `observation.language.embedding`.
`make_robomimic_pre_post_processors` in `lerobot_policy_robomimic/processors.py`
puts it after the batch dimension is added and before tensors move to the
policy's device; every policy's processor factory calls this one function.
`tests/test_language.py` checks the step and, once the CLIP model is in the
Hugging Face cache, compares its embedding with robomimic's `get_lang_emb`.

| Setting or part  | Behavior                                                                  | Config field                                      | Source                                                                                                                            |
| ---------------- | ------------------------------------------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Conditioning     | Off; `concat` or `film` turns it on                                       | `language_conditioning` (None)                    | Repo `docs/tutorials/language_conditioning.md` (section 2)                                                                        |
| CLIP model       | `openai/clip-vit-large-patch14`, its projected text embedding of size 768 | `clip_model_name`, `language_embedding_dim` (768) | Repo `robomimic/utils/lang_utils.py:4`, `:21`, `:43`; `robomimic/models/base_nets.py:666`                                         |
| Tokens           | The task alone, with special tokens, padded to 25 tokens                  | (fixed)                                           | Repo `robomimic/utils/lang_utils.py:35-42`                                                                                        |
| Embedding source | The task string of each frame                                             | (fixed)                                           | This port; robomimic embeds one `lang` per dataset and copies it to every step (`robomimic/utils/dataset.py:113-114`, `:530-532`) |
| Model loading    | On the first task, once per model; embeddings cached per task             | (fixed)                                           | Repo `robomimic/utils/lang_utils.py:8-27` (lazy loading)                                                                          |

Taking the embedding from LeRobot's task string instead of a per-dataset
setting gives the same input for robomimic's single-task datasets, and lets
`lerobot-eval` feed the same instruction to the policy without the env
computing embeddings itself.

Language conditioning needs `transformers`, available as the `language`
extra with LeRobot's own bounds (`lerobot[transformers-dep]`). The CLIP model
is about 1.7 GB and is downloaded to the Hugging Face cache on first use.

### 22. Concatenation

With `language_conditioning=concat`, the policy reads the embedding as one
more vector observation: `observation_features` lists it last, and the
observation encoder (§4) concatenates it unchanged, as robomimic does with a
`lang_emb` key among its low-dim observations (Repo
`docs/tutorials/language_conditioning.md`, "Feature input to action head";
`robomimic/models/obs_nets.py:282-284`, `:303-307`). `tests/test_language_conditioning.py`
compares the encoder with robomimic's, reordering features as in §4 since
robomimic sorts `lang_emb` among its keys, and BC-RNN's loss with robomimic's
on the same weights.

| Part          | Behavior                                                                                 | Source                                                                                            |
| ------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Feature order | Last, after the dataset's features                                                       | This port; robomimic sorts it among the other keys (Repo `robomimic/utils/file_utils.py:162-167`) |
| Sequences     | One embedding per sample, repeated over the steps of BC-RNN and BC-Transformer sequences | Repo `robomimic/utils/dataset.py:530-532` (repeated over the sequence)                            |
| Other inputs  | At least one observation besides the task                                                | This port                                                                                         |

LeRobot's dataset gives one task string per sample even when it returns a
sequence of frames, so the embedding arrives without the time dimension and
the encoder repeats it. Within a demo the task does not change, so this
equals robomimic's repeated embedding.

### 23. FiLM

With `language_conditioning=film`, every camera's ResNet-18 is modulated by
the embedding, robomimic's `ResNet18ConvFiLM` inside
`VisualCoreLanguageConditioned` (Repo `robomimic/models/base_nets.py:657`,
`robomimic/models/obs_core.py:189`). `ImageEncoder` gains a `FiLM` layer after
each residual block, and `CameraEncoder` in
`lerobot_policy_robomimic/observation_encoder.py` passes the embedding to it.
`tests/test_language_conditioning.py` compares the camera encoder with
robomimic's, and BC's loss and actions and BC-RNN's loss with robomimic's
algorithms, on the same weights.

| Part             | Behavior                                                                                         | Source                                                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| FiLM layer       | A linear layer from the embedding to β and γ for each channel, then `(1 + γ) × x + β` and a ReLU | Repo `robomimic/models/base_nets.py:630`, `:649-654`                                                                         |
| Placement        | After each of the 8 residual blocks; the stem before them is unchanged                           | Repo `robomimic/models/base_nets.py:699-713`, `:735-740`                                                                     |
| Embedding size   | `language_embedding_dim`, 768 for the default CLIP model                                         | Repo `robomimic/models/base_nets.py:666`                                                                                     |
| Features         | The embedding modulates the cameras and is left out of the concatenated features                 | Repo `robomimic/models/obs_nets.py:248-251`, `:282-284`                                                                      |
| Crop and pooling | Kept from the image experiments: random 76×76 crops and the spatial softmax                      | This port; `VisualCoreLanguageConditioned` pools with a spatial softmax by default (Repo `robomimic/models/obs_core.py:197`) |

robomimic's tutorial shows FiLM with no pooling and no crop
(`docs/tutorials/language_conditioning.md:88`, `:91`). That is one example
config, not a default, so this port changes only the backbone and keeps every
other camera setting of §2; the tests build robomimic's FiLM encoder the same
way.

`CameraEncoder` replaces the `nn.Sequential` each camera used before, which
cannot pass a second input to its encoder; its parameters keep the same
names, so earlier checkpoints still load.

### 24. Runs

With `openai/clip-vit-large-patch14` in the Hugging Face cache, its embedding
of two tasks equals robomimic's `get_lang_emb` (`tests/test_language.py`).
A 20-step `lerobot-train` run of BC-RNN with `language_conditioning=film` on
the `train` mask of the converted Lift image dataset trains the 41M-parameter
policy and saves a preprocessor that includes the CLIP step. `lerobot-eval`
then loads it with the `robomimic` env, whose task description is the
dataset's `lift the cube`, and runs two 30-step episodes. No success rate is
reported here: robomimic publishes none for language conditioning, and the
paper's tasks each have a single instruction.

## Not ported

Every robomimic option below is left out. None of them is used by the paper's
BC and BC-RNN experiments (Repo `robomimic/scripts/generate_paper_configs.py`)
or by the BC-Transformer template, so leaving them out does not change what
this port reproduces. Options that LeRobot already covers, such as batch
size, epochs, seed, device, data workers, and rollout settings, are not
listed; they are `lerobot-train` and `lerobot-eval` arguments.

| Option                                                                                                                | robomimic default, and the paper's setting                   | Source                                                                                                        | Details                                                   |
| --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Depth and scan observations (`ScanCore`)                                                                              | None; the paper uses low-dim and RGB only                    | Repo `robomimic/config/base_config.py:271-272`, `:313`; `robomimic/models/obs_core.py:240`                    | This list                                                 |
| Goal-conditioned observations (`modalities.goal`, `train.goal_mode`)                                                  | None; the paper sets the goal lists to empty                 | Repo `robomimic/config/base_config.py:230`, `:273-276`; `robomimic/scripts/generate_paper_configs.py:147-148` | This list                                                 |
| Encoders and randomizers for low-dim observations (`encoder.low_dim`)                                                 | None                                                         | Repo `robomimic/config/base_config.py:284-291`                                                                | This list                                                 |
| Other camera backbones: `ResNet50Conv`, `R3MConv`, `MVPConv`, `ShallowConv`                                           | `ResNet18Conv`, which the paper uses                         | Repo `robomimic/models/base_nets.py:561`, `:749`, `:828`, `:993`                                              | This list                                                 |
| `SpatialMeanPool` or no pooling                                                                                       | The paper uses `SpatialSoftmax`                              | Repo `robomimic/scripts/generate_paper_configs.py:156`; `robomimic/models/base_nets.py:1248`                  | This list                                                 |
| `ColorRandomizer` and `GaussianNoiseRandomizer`                                                                       | The paper uses `CropRandomizer` only                         | Repo `robomimic/scripts/generate_paper_configs.py:163`; `robomimic/models/obs_core.py:629`, `:786`            | This list                                                 |
| ImageNet weights, coordinate convolution, crop position encoding, several crops, a learnable or noisy spatial softmax | Off in the paper                                             | §3                                                                                                            | §3                                                        |
| Stacked observation history for BC and BC-RNN (`train.frame_stack > 1`)                                               | 1, which the paper keeps                                     | Repo `robomimic/config/base_config.py:195`                                                                    | This list; `n_obs_steps` other than 1 raises `ValueError` |
| `exp` standard deviations for the Gaussian and GMM heads                                                              | `softplus`                                                   | §5                                                                                                            | §5                                                        |
| GRU, bidirectional LSTM, and open-loop BC-RNN                                                                         | One-way LSTM, closed loop                                    | §8                                                                                                            | §8                                                        |
| BC-Transformer's all-step supervision, future actions, other position embeddings, GEGLU                               | Off in the template                                          | §13                                                                                                           | §13                                                       |
| BC-VAE's categorical prior and `freeze_encoder`                                                                       | Off; `freeze_encoder` is only set by BCQ                     | §17, §19                                                                                                      | §17, §19                                                  |
| Learning rate schedules for BC, BC-RNN, and BC-VAE (`multistep`, `linear`, `cosine`)                                  | No decay epochs, so the rate stays constant, as in the paper | Repo `robomimic/config/bc_config.py:28-31`; `robomimic/utils/torch_utils.py:141-180`                          | This list                                                 |
| A choice of optimizer (`optimizer_type`)                                                                              | Adam, or AdamW in the BC-Transformer template                | Repo `robomimic/config/bc_config.py:27`; `robomimic/utils/torch_utils.py:106-115`                             | §2, §13                                                   |
| Action normalization and rotation conversion (`train.action_config`)                                                  | No normalization                                             | Repo `robomimic/config/base_config.py:223-227`                                                                | This list                                                 |
| Observation normalization (`train.hdf5_normalize_obs`)                                                                | Off                                                          | Repo `robomimic/config/base_config.py:181`                                                                    | §2 (normalization)                                        |
| Weighting several datasets by size (`train.normalize_weights_by_ds_size`)                                             | Off; the paper trains on one dataset                         | Repo `robomimic/config/base_config.py:161`                                                                    | This list                                                 |

Each policy has its optimizer fixed rather than chosen by a setting: Adam for
BC, BC-RNN, and BC-VAE, and AdamW for BC-Transformer, which are the only
optimizers robomimic's configs and templates use for them.

## Citation

This plugin is an independent port of robomimic, which was developed by Ajay
Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun
Kulkarni, Li Fei-Fei, Silvio Savarese, Yuke Zhu, and Roberto Martín-Martín. If
you use it, please cite their [paper](https://arxiv.org/abs/2108.03298):

```bibtex
@inproceedings{robomimic2021,
  title={What Matters in Learning from Offline Human Demonstrations for Robot Manipulation},
  author={Ajay Mandlekar and Danfei Xu and Josiah Wong and Soroush Nasiriany and Chen Wang and Rohun Kulkarni and Li Fei-Fei and Silvio Savarese and Yuke Zhu and Roberto Mart\'{i}n-Mart\'{i}n},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2021}
}
```
