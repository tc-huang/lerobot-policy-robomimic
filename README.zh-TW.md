# lerobot-policy-robomimic

[English](README.md) | 繁體中文

本專案是 [LeRobot](https://github.com/huggingface/lerobot) 的 out-of-tree
policy 插件，將 [robomimic](https://robomimic.github.io/)（Mandlekar 等人，2021）
中的 imitation learning policy（例如 BC 與 BC-RNN）移植到 LeRobot v0.6.1，並註冊
為 LeRobot 的 policy type。移植後的 policy 可以用 `lerobot-train` 訓練、以
`lerobot-eval` 評估，網路架構與訓練設定則盡可能貼近原始實作。

## 使用方式

安裝插件與 `lerobot-train` 需要的依賴：

```bash
uv sync --extra training
```

把含有 observation 的 robomimic hdf5 檔轉成 LeRobotDataset。影像檔是用 robomimic 的
`dataset_states_to_obs.py` 從 raw 檔產生的（原 repo
`robomimic/scripts/extract_obs_from_raw_datasets.sh:59-61`）。

```bash
uv run python -m lerobot_policy_robomimic.convert_dataset \
    --hdf5 data/robomimic/lift/ph/image_v15.hdf5 \
    --repo-id <user>/robomimic_lift_ph_image \
    --task "lift the cube"
```

用 robomimic `train` mask 中的 demo 訓練 BC，batch size 與訓練長度沿用 robomimic 的
image 實驗（§2）：

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

若要做 low-dim 實驗，改轉換 low-dim 檔，加上 `--policy.use_env_state=true`，並使用
`--batch_size=100 --steps=200000`。

若要訓練 BC-RNN，使用 `--policy.type=robomimic_bc_rnn`，batch size 與訓練長度不變；
robomimic 在 image 與 low-dim 實驗中對所有演算法都使用相同設定（原 repo
`robomimic/scripts/generate_paper_configs.py:61-62`、`:131-132`；`:739-746` 只依資料
類型選擇設定）。它的 low-dim 實驗還需要 `--policy.rnn_hidden_dim=400`（§8）。

若要訓練 BC-Transformer，使用 `--policy.type=robomimic_bc_transformer`。它調好參數的
template 是在 low-dim 資料上以 `--batch_size=100 --steps=200000` 訓練（§13）；AdamW 與
`robomimic_linear` learning rate schedule 由 policy 自行設定。

若要訓練 BC-VAE，使用 `--policy.type=robomimic_bc_vae`。論文沒有 BC-VAE 的實驗；robomimic
的論文 config 只依資料類型決定 batch size 與訓練長度，因此沿用上面 BC 的數值（§17）。

若要讓這些 policy 依 task 的 CLIP embedding 條件化，安裝 `language` extra，並加上
`--policy.language_conditioning=film` 或 `concat`（§21-23）。task 就是轉換時給的 `--task`
字串；在 converter 開始把它存進 env args 之前轉換的 dataset，需要在
`meta/robomimic_env_args.json` 加上 `"lang"`，`lerobot-eval` 才拿得到它。

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

若要使用 robomimic model zoo 的 BC-RNN checkpoint（原 repo `docs/model_zoo/robomimic_v0.1.md`），
先下載，再轉換成 LeRobot 的 policy 目錄（§12）：

```bash
curl -L --create-dirs -o data/robomimic/model_zoo/lift_ph_image_epoch_500_succ_100.pth \
    http://downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo/lift/bc_rnn/lift_ph_image_epoch_500_succ_100.pth
uv run python -m lerobot_policy_robomimic.convert_checkpoint \
    --checkpoint data/robomimic/model_zoo/lift_ph_image_epoch_500_succ_100.pth \
    --output-dir outputs/checkpoints/lift_ph_image_bc_rnn
```

以 `lerobot-eval` 與 `robomimic` env 在模擬環境中評估 policy；這個 env 會重建轉換後 dataset
的 robosuite env（見 Simulation 一節），需要 `sim` extra：

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

`lerobot_policy_robomimic/convert_dataset.py` 把含有 observation 的 robomimic hdf5 檔轉成
LeRobotDataset，每個 demo 一個 episode。測試會用一個小型 hdf5 檔比對轉換後的數值、像素、
episode 與 metadata（`tests/test_convert_dataset.py`）。來源標示方式見「設計」一節。

| 決策                | 選擇                                                                                       | 來源                                                                                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 本體感知            | `observation.state`：依序串接 `robot0_eef_pos`、`robot0_eef_quat`、`robot0_gripper_qpos`   | 原 repo `robomimic/scripts/generate_paper_configs.py:138-142`、`:67-70`（所有論文實驗都用這些 key）；LeRobot 的 policy 讀取 `observation.state` |
| 物體狀態            | `observation.environment_state`，來自 `object`                                             | 原 repo `robomimic/scripts/generate_paper_configs.py:71`（僅 low-dim 實驗）；LeRobot 把這個 key 定為 `ENV` 類型（`utils/feature_utils.py:168`） |
| 相機                | `observation.images.<camera>`，來自 `<camera>_image`                                       | 本專案；LeRobot 的 RoboCasa env 也沿用 robosuite 的相機名稱（`envs/robocasa.py:44`）                                                            |
| 影像儲存            | 無損 PNG；原始像素超過 2 GiB 時改用 LeRobot 的 MP4 影片（`--video auto`）                  | 本專案：小資料集保留 robomimic 的精確像素，大資料集維持可處理的大小                                                                             |
| Frame rate          | demo 的 `control_freq`，Lift 為 20 Hz                                                      | hdf5 的 `env_args`                                                                                                                              |
| Episode 順序        | 依 demo 編號的數值順序，因此 `demo_2` 排在 `demo_10` 之前                                  | 本專案                                                                                                                                          |
| Train 與 valid 切分 | `meta/robomimic_masks.json` 把每個 mask 對應到 episode index，供 `--dataset.episodes` 使用 | 原 repo `robomimic/scripts/generate_paper_configs.py:244-245`（論文以 `train` 訓練、以 `valid` 驗證）                                           |
| 模擬器設定          | `meta/robomimic_env_args.json` 原樣保留 hdf5 的 `env_args`                                 | 原 repo `robomimic/scripts/train.py:89`、`:150`（robomimic 以它建立 rollout 用的 env）                                                          |
| 語言指令            | `--task` 字串，放在每個 frame，也以 `lang` 存在 `meta/robomimic_env_args.json`             | 原 repo `robomimic/utils/env_utils.py:231`（robomimic 從 env args 的 `lang` 讀取 env 的指令）                                                   |
| 數值                | float64 轉為 float32                                                                       | 本專案                                                                                                                                          |

robomimic 沒有釋出 v1.5 的 image dataset；它是用 `robomimic/scripts/dataset_states_to_obs.py`
從 raw 檔產生的，相機與尺寸依照 `robomimic/scripts/extract_obs_from_raw_datasets.sh:61`。用同樣
方式產生 low-dim 檔，可以重現官方釋出的檔案，只有一個 float32 observation 有捨入差異。

模擬器設定之所以重要，是因為 robosuite 1.5.1 自己的預設值和錄製 demo 時不同：它的 Panda
控制器以機器人底座座標系、而非世界座標系解讀 action（`input_ref_frame`），而 demo 關閉了
`lite_physics`。

## Simulation

`lerobot_policy_robomimic/env_config.py` 的 `RobomimicEnvConfig` 註冊 LeRobot env type
`robomimic`，`lerobot_policy_robomimic/robosuite_env.py` 的 `RobomimicEnv` 則以 robomimic rollout
的方式包裝 robosuite。env 是依轉換後 dataset 的 `meta/robomimic_env_args.json` 重建的（見
Datasets 一節）；robosuite 只在建立 env 時才 import，因此它留在 `sim` extra 中。來源標示方式見
「設計」一節。

| 部分         | 行為                                                                                         | 來源                                                                                                                              |
| ------------ | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 建立         | 以 env args 呼叫 `robosuite.make`，只算繪需要的相機，並一律提供物體狀態                      | 原 repo `robomimic/envs/env_robosuite.py:87`、`:116`                                                                              |
| 影像         | 上下翻正，作為 `pixels/<camera>`                                                             | 原 repo `robomimic/envs/env_robosuite.py:255`                                                                                     |
| 本體感知     | dataset 的 state key 串接後作為 `agent_pos`                                                  | 見 Datasets 一節                                                                                                                  |
| 物體狀態     | robosuite 的 `object-state`，作為 `environment_state`                                        | 原 repo `robomimic/envs/env_robosuite.py:267`                                                                                     |
| 成功與結束   | `_check_success()`；成功時結束 episode                                                       | 原 repo `robomimic/envs/env_robosuite.py:407`；`robomimic/scripts/generate_paper_configs.py:53`、`:123`（`terminate_on_success`） |
| Episode 長度 | Lift、Can、Square 為 400 步；Transport、Tool Hang 為 700 步                                  | 原 repo `robomimic/__init__.py:63`                                                                                                |
| 初始狀態     | robosuite 的隨機 reset；seed 設定的是 NumPy 的全域 generator，robosuite 1.5.1 以它擺放物體   | robosuite 1.5.1 `robosuite/utils/placement_samplers.py:167`                                                                       |
| Task 描述    | env args 的 `lang`，沒有時用 env 名稱；`lerobot-eval` 把它當作 task 交給 preprocessor（§21） | 原 repo `robomimic/utils/env_utils.py:231`；LeRobot `scripts/lerobot_eval.py:281`                                                 |
| 批次 env     | 每個 env 在 episode 結束後停止模擬，並使用 `NEXT_STEP` autoreset                             | LeRobot `envs/utils.py:241`（`freeze_after_episode_end`）、`envs/libero.py:526`                                                   |
| Features     | 建立 config 時，從一個不算繪相機的 robosuite env 讀出                                        | 本專案：LeRobot 會以 env 的 action feature 取代載入 policy 的設定（`policies/factory.py:304`），因此大小必須正確                  |

測試會以相同的 env args 建立 robomimic 自己的 `EnvRobosuite`，兩者以相同 seed reset 後，比對
5 步的影像、本體感知、物體狀態、reward 與成功與否（`tests/test_env.py`）。比對是從各自以 seed
reset 的狀態開始，而不是複製模擬器狀態，因為 robosuite 的控制器與視覺化標記都有複製模擬器狀態
時帶不過去的狀態。

## 設計

本專案從零開始重建每個 robomimic policy，一次建構一個元件。以下每一節說明一個元件，
以及它的行為從何而來。

來源標示方式如下：

- **論文**：Mandlekar 等人，
  [What Matters in Learning from Offline Human Demonstrations for Robot Manipulation](https://arxiv.org/abs/2108.03298)，
  CoRL 2021。
- **原 repo**：[ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic)
  的 `d309eae`，以 submodule 放在 `third_party/robomimic`。路徑皆相對於該目錄；前綴
  `v0.1.0:` 表示該 tag 的路徑，可用 `git -C third_party/robomimic show v0.1.0:<path>`
  讀取。
- **LeRobot**：LeRobot v0.6.1 的
  [Adding a Policy](https://huggingface.co/docs/lerobot/v0.6.1/en/bring_your_own_policies)
  指南或原始碼。路徑皆相對於安裝後的 `lerobot` 套件。
- **本專案**：本專案自行決定，並非取自上述任一來源。

robomimic 以 MIT License 釋出。本專案重新實作其網路，執行時不 import robomimic。

### 1. Policy type 與套件結構

| 決策                  | 選擇                                                                                                                                                             | 來源                                                                                                                                                                                                |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Policy type           | `robomimic_bc`（BC）、`robomimic_bc_rnn`（BC-RNN）、`robomimic_bc_transformer`（BC-Transformer）、`robomimic_bc_vae`（BC-VAE）                                   | 本專案：加上 `robomimic_` 前綴，避免和 LeRobot 內建的 type 衝突，例如 `diffusion`（`policies/diffusion/`），而 robomimic 也有實作它（原 repo `robomimic/algo/diffusion_policy.py`）                 |
| Class 與函式名稱      | `RobomimicBCConfig`、`RobomimicBCPolicy`、`make_robomimic_bc_pre_post_processors`；`robomimic_bc_rnn`、`robomimic_bc_transformer` 與 `robomimic_bc_vae` 依此類推 | LeRobot：policy class 名稱是把 config class 名稱的 `Config` 換成 `Policy`（`policies/factory.py:409-415`），processor factory 名稱是 `make_<type>_pre_post_processors`（`policies/factory.py:458`） |
| 模組名稱              | `configuration_<type>.py`、`modeling_<type>.py`、`processor_<type>.py`                                                                                           | LeRobot：modeling 與 processor 模組是把 config 模組路徑中的 `configuration_` 替換後找到的（`policies/factory.py:416`、`:459`）；命名沿用指南的 template                                             |
| Distribution 名稱     | `lerobot_policy_robomimic`                                                                                                                                       | LeRobot：名稱以 `lerobot_policy_` 開頭的已安裝 distribution 會以該名稱被 import（`utils/import_utils.py:231-255`），進而執行 `@PreTrainedConfig.register_subclass`                                  |
| 一個套件、多個 policy | 每個 robomimic 演算法各自有 policy type 與三個模組                                                                                                               | 本專案：指南示範一個套件一個 policy，但 factory 只需要每個 type 各有這三個模組，因此 robomimic 的演算法可以共用一個套件與其網路                                                                     |

## BC（`robomimic_bc`）

### 2. Configuration

`RobomimicBCConfig` 註冊 policy type `robomimic_bc`。預設值沿用 robomimic 在
proficient-human（PH）資料集上的 image 實驗設定。下表中除了 `actor_layer_dims` 以外的設定，
都來自 `lerobot_policy_robomimic/base_config.py`，裡面有兩個本身不是 policy type 的 class：
`RobomimicPolicyConfig` 放本專案所有 robomimic policy 共用的 observation、encoder 與
optimizer 設定；它的子類別 `RobomimicActorConfig` 再加上 BC、BC-RNN 與 BC-Transformer
共用的 action head 設定。

| 設定                         | 預設值                                 | Config 欄位                                                                           | 來源                                                                                                                                                                             |
| ---------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Observation                  | 本體感知與相機；不含物體狀態           | `use_env_state`（False）                                                              | 原 repo `robomimic/scripts/generate_paper_configs.py:138-146`（image）；low-dim 實驗另加 `object`（`:67-72`）                                                                    |
| 觀測歷史                     | 無（只用當下這一步）                   | `n_obs_steps`、`observation_delta_indices`                                            | 原 repo `robomimic/config/base_config.py:193`、`:195`；`robomimic/algo/bc.py:111` 只讀第 0 步                                                                                    |
| Action 預測                  | 每個 observation 預測一個 action       | `action_delta_indices`                                                                | 原 repo `robomimic/algo/bc.py:113`                                                                                                                                               |
| MLP 隱藏層大小               | (1024, 1024)                           | `actor_layer_dims`                                                                    | 原 repo `robomimic/config/bc_config.py:41`、`robomimic/scripts/generate_paper_configs.py:367`                                                                                    |
| 相機裁切                     | 訓練時隨機裁 76×76，推論時中央裁 76×76 | `crop_shape`                                                                          | 原 repo `robomimic/scripts/generate_paper_configs.py:163-168`                                                                                                                    |
| 訓練以外的裁切               | 中央                                   | `random_crop_at_inference`（False）                                                   | 原 repo `robomimic/models/obs_core.py:579`；robomimic v0.1 則是隨機裁切（`v0.1.0:robomimic/models/base_nets.py:1066`）                                                           |
| Spatial softmax keypoint     | 32                                     | `spatial_softmax_num_kp`                                                              | 原 repo `robomimic/scripts/generate_paper_configs.py:157`                                                                                                                        |
| 每支相機的特徵大小           | 64                                     | `image_feature_dim`                                                                   | 原 repo `robomimic/scripts/generate_paper_configs.py:152`                                                                                                                        |
| Action head                  | 5 個高斯分佈的混合                     | `use_gmm`（True）、`gmm_num_modes`（5）                                               | 原 repo `robomimic/scripts/generate_paper_configs.py:368`（人類示範資料集）、`robomimic/config/bc_config.py:53`                                                                  |
| GMM 最小標準差               | 1e-4                                   | `gmm_min_std`                                                                         | 原 repo `robomimic/config/bc_config.py:54`                                                                                                                                       |
| GMM 在訓練以外的雜訊         | 所有標準差設為 1e-4                    | `gmm_low_noise_eval`（True）                                                          | 原 repo `robomimic/config/bc_config.py:56`                                                                                                                                       |
| Gaussian head                | 關閉；開啟時優先於 GMM head            | `use_gaussian`（False）                                                               | 原 repo `robomimic/config/bc_config.py:44`、`robomimic/algo/bc.py:45-52`                                                                                                         |
| Gaussian 標準差              | 可學習，網路輸出 0 時為 0.1，至少 0.01 | `gaussian_fixed_std`（False）、`gaussian_init_std`（0.1）、`gaussian_min_std`（0.01） | 原 repo `robomimic/config/bc_config.py:45-47`                                                                                                                                    |
| Gaussian 在訓練以外的 action | 平均值                                 | `gaussian_low_noise_eval`（True）                                                     | 原 repo `robomimic/config/bc_config.py:49`                                                                                                                                       |
| Deterministic head 的 loss   | 只用均方誤差                           | `l2_weight`（1）、`l1_weight`（0）、`cos_weight`（0）                                 | 原 repo `robomimic/config/bc_config.py:36-38`                                                                                                                                    |
| 正規化                       | 無；影像只縮放到 `[0, 1]`              | `normalization_mapping`（`IDENTITY`）                                                 | 原 repo `robomimic/config/base_config.py:181`（observation）、`:225`（action）、`robomimic/utils/obs_utils.py:921`（影像）；LeRobot 已縮放影像（`datasets/io_utils.py:255-263`） |
| Optimizer                    | Adam                                   | `get_optimizer_preset()`                                                              | 原 repo `robomimic/config/bc_config.py:27`                                                                                                                                       |
| Learning rate                | 1e-4，固定不變                         | `optimizer_lr`                                                                        | 原 repo `robomimic/config/bc_config.py:28`、`:30`（沒有衰減的 epoch）                                                                                                            |
| Weight decay                 | 0                                      | `optimizer_weight_decay`                                                              | 原 repo `robomimic/config/bc_config.py:33`                                                                                                                                       |
| Gradient clipping            | 無                                     | `optimizer_grad_clip_norm`（0）                                                       | 原 repo `robomimic/config/base_config.py:239`；LeRobot 在 0 時不做 clipping（`scripts/lerobot_train.py:178`）                                                                    |

LeRobot 會把 dataset 裡所有非 action 的 feature 都交給 policy
（`policies/factory.py:305-306`），而 robomimic 是由每個實驗列出要用的
observation。`use_env_state` 讓這個選擇留在 policy 裡，因此同一份 dataset 可以同時
支援兩種實驗。

Batch size 與訓練長度屬於 `lerobot-train`，不屬於 policy：robomimic 在 image 資料上
用 batch size 16，訓練 600 個 epoch、每個 epoch 500 步（共 30 萬步）
（`robomimic/scripts/generate_paper_configs.py:113`、`:131-132`）；在 low-dim 資料上
用 batch size 100，訓練 2000 個 epoch、每個 epoch 100 步（共 20 萬步）
（`:43`、`:61-62`）。

`tests/test_paper_defaults.py` 會把這些預設值和 robomimic 自己在 PH Lift image 實驗中的
BC config 比對。設定 `use_gmm=false` 則得到 robomimic 原本的 `BC` class，論文只在
機器產生的資料集上這樣用（`robomimic/scripts/generate_paper_configs.py:370-372`）。
設定 `use_gaussian=true` 則得到 robomimic 的 `BC_Gaussian`，論文沒有用它；它的設定沿用
robomimic 的預設值，由同一個測試檢查。

### 3. 相機 encoder

`lerobot_policy_robomimic/vision.py` 是 robomimic image 實驗所用的相機 encoder，對應
robomimic 的 `VisualCore`（原 repo `robomimic/scripts/generate_paper_configs.py:151-160`）。
測試會用相同的權重和輸入，把每個部分和 robomimic 自己的模組比對
（`tests/test_vision.py`）。

| 部分            | 行為                                                                                | 本專案           | 來源                                                                                                                                                     |
| --------------- | ----------------------------------------------------------------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 裁切            | 訓練時隨機裁切，其他時候中央裁切                                                    | `RandomCrop`     | 原 repo `robomimic/models/obs_core.py:489`（`CropRandomizer`）、`:579`（評估時中央裁切）                                                                 |
| 隨機裁切的位移  | `floor(rand × (size − crop))`，84 裁 76 時為 0 到 7；最後一個位移永遠抽不到         | `RandomCrop`     | 原 repo `robomimic/utils/obs_utils.py:730`                                                                                                               |
| 中央裁切的位移  | `floor((size − crop) / 2)`，84 裁 76 時為 4                                         | `RandomCrop`     | 原 repo `robomimic/utils/obs_utils.py:278`                                                                                                               |
| Backbone        | 去掉 average pool 與分類層的 ResNet-18，從頭訓練                                    | `ImageEncoder`   | 原 repo `robomimic/models/base_nets.py:536`（`ResNet18Conv`）；`robomimic/scripts/generate_paper_configs.py:154`（`pretrained = False`）                 |
| Spatial softmax | 1×1 卷積產生 32 個 keypoint，在位置上做 softmax，取 `[-1, 1]` 網格上的期望 `(x, y)` | `SpatialSoftmax` | 原 repo `robomimic/models/base_nets.py:1143`、`:1162`、`:1216`；temperature 固定為 1 且不加噪聲（`robomimic/scripts/generate_paper_configs.py:158-160`） |
| 投影            | Linear 層，把 32 × 2 個 keypoint 座標投影成 64 維特徵                               | `ImageEncoder`   | 原 repo `robomimic/models/obs_core.py:138`                                                                                                               |

robomimic 會平均多個裁切的特徵（`robomimic/models/obs_core.py:589-598`）；論文實驗每張
影像只裁一次（`robomimic/scripts/generate_paper_configs.py:168`），平均後就是該特徵本身，
因此 `RandomCrop` 只取一個裁切。

訓練出 model zoo checkpoint 的 robomimic v0.1 沒有另外的評估路徑：`CropRandomizer.forward_in`
永遠隨機裁切（原 repo `v0.1.0:robomimic/models/base_nets.py:1066`，於
`v0.1.0:robomimic/models/obs_nets.py:271` 呼叫），中央裁切是 v0.3 才加入的（commit
`40e427a`）。因此 model zoo 的成功率是在推論時隨機裁切的情況下量測的。
`random_crop_at_inference=true` 可以讓從 model zoo 轉換來的 checkpoint 恢復這個行為。

`RandomCrop` 對每張影像各切一塊再疊起來，而不是用 advanced indexing 一次取出所有裁切。
用 indexing 取出的裁切不是連續的 tensor，在 Apple 的 MPS backend 上，經過 ResNet 的
backward 會失敗，錯誤訊息是 `view size is not compatible with input tensor's size
and stride`（本專案；由 §7 的 `lerobot-train` 執行發現）。

robomimic 把 spatial softmax 的網格與 temperature 存在 checkpoint 裡
（`robomimic/models/base_nets.py:1159`、`:1167`）。本專案中它們是依特徵圖大小重建的
固定值，因此不放進 state dict。

robomimic 支援、但論文實驗沒有用到的選項不移植：ImageNet 預訓練權重、coordinate
convolution、裁切的位置編碼，以及可學習或加噪聲的 spatial softmax。

### 4. Observation encoder

`lerobot_policy_robomimic/observation_encoder.py` 把 policy 讀取的所有 observation
轉成一個特徵向量，對應 robomimic 的 `ObservationEncoder`
（原 repo `robomimic/models/obs_nets.py:119`）。測試會把相同的相機權重載入 robomimic
的 encoder，並比對輸出（`tests/test_observation_encoder.py`）。

| 部分             | 行為                                                                                              | 來源                                                                                                                                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 向量 observation | 原樣串接                                                                                          | 原 repo `robomimic/config/base_config.py:284`（low-dim observation 沒有 encoder）、`robomimic/models/obs_nets.py:303-307`                                                                                     |
| 每支相機         | 各自的裁切、`ImageEncoder` 與 ReLU                                                                | 原 repo `robomimic/models/obs_nets.py:287-289`（每個 key 一個 randomizer）、`:295`（encoder）、`:297`（activation）；ReLU 是預設的 `feature_activation`（`:440`、`:33`），`MIMO_MLP` 沿用預設值（`:608-611`） |
| 特徵順序         | 依 policy input feature 的順序                                                                    | 本專案；robomimic 則是把 observation key 排序（原 repo `robomimic/utils/file_utils.py:162`，經 `robomimic/scripts/train.py:242` 與 `robomimic/algo/algo.py:154-156` 傳入）                                    |
| 前置維度         | 任意，例如 BC 的 `(B,)` 或 BC-RNN 序列的 `(B, T)`；相機影像逐 frame 編碼，每個 frame 各自隨機裁切 | 原 repo `robomimic/models/obs_nets.py:858`（以 `time_distributed` 套用 encoder）                                                                                                                              |

robomimic 排序後，相機與向量 observation 會交錯排列；以 image 實驗為例，順序是
`agentview_image`、`robot0_eef_pos`、`robot0_eef_quat`、`robot0_eye_in_hand_image`、
`robot0_gripper_qpos`。這個順序只在載入 robomimic 權重時有影響，它決定 MLP 第一層的
輸入欄位。沿用 LeRobot 的順序，encoder 就不需要知道 robomimic 的 key 名稱；測試則在
比對前先重新排列 robomimic 的輸出。

### 5. MLP 與 action head

`lerobot_policy_robomimic/mlp.py` 的 `MLP` 把編碼後的 observation 轉成特徵，
`lerobot_policy_robomimic/action_heads.py` 的 action head 再把特徵轉成 action。兩者合起來
對應 robomimic 的 `ActorNetwork`（原 repo `robomimic/models/policy_nets.py:26`）、
`GaussianActorNetwork`（`:193`）或 `GMMActorNetwork`（`:397`），BC 用 `actor_layer_dims` 建立它們（`robomimic/algo/bc.py:87-92`）。

| 部分                  | 行為                                                                                                                                                                                                    | 來源                                                                                                                                                        |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MLP                   | `actor_layer_dims` 的每個大小各一層 Linear 加 ReLU                                                                                                                                                      | 原 repo `robomimic/models/obs_nets.py:617-623`；最後一個大小是 MLP 的輸出，同樣接 ReLU（`:623`、`robomimic/models/base_nets.py:256-258`）                   |
| `DeterministicHead`   | Linear 層輸出 action 維度，再接 `tanh`                                                                                                                                                                  | 原 repo `robomimic/models/obs_nets.py:627`、`:392`（`ObservationDecoder`）、`robomimic/models/policy_nets.py:107`                                           |
| 它的 loss             | `l2_weight` × 均方誤差 + `l1_weight` × smooth L1 + `cos_weight` × cosine loss                                                                                                                           | 原 repo `robomimic/algo/bc.py:182-192`                                                                                                                      |
| 它的 cosine loss      | 前三個 action 維度（末端執行器的位移）上 1 − cosine similarity 的平均                                                                                                                                   | 原 repo `robomimic/algo/bc.py:185`、`robomimic/utils/loss_utils.py:22-23`                                                                                   |
| `GaussianHead` 的輸出 | Linear 層分別輸出單一對角高斯分佈的平均值與尺度；開啟 `gaussian_fixed_std` 時沒有尺度層                                                                                                                 | 原 repo `robomimic/models/policy_nets.py:318-319`、`:338`                                                                                                   |
| 它的平均值            | 先限制在 [−9, 9]，再取 `tanh`                                                                                                                                                                           | 原 repo `robomimic/models/policy_nets.py:341`、`:345`；`BC_Gaussian` 沿用預設的範圍，且不用 tanh-Gaussian（`:207`、`:210`、`robomimic/algo/bc.py:263-274`） |
| 它的尺度              | 輸出取 `softplus` 並縮放成輸出 0 時為 `gaussian_init_std`，開啟 `gaussian_fixed_std` 時就是 `gaussian_init_std`；再限制在 [`gaussian_min_std`, 7.5]；開啟 `gaussian_low_noise_eval` 時，訓練以外為 1e-4 | 原 repo `robomimic/models/policy_nets.py:280-282`、`:338`、`:350`、`:355`；`robomimic/algo/bc.py:272`                                                       |
| 它的 loss             | 示範 action 的平均對數似然取負                                                                                                                                                                          | 原 repo `robomimic/algo/bc.py:300`、`:322`                                                                                                                  |
| 它的 action           | 開啟 `gaussian_low_noise_eval` 時，訓練以外取平均值，其他時候抽樣                                                                                                                                       | 原 repo `robomimic/models/policy_nets.py:386-388`                                                                                                           |
| `GMMHead` 的輸出      | Linear 層分別輸出每個 mode 的高斯平均值、尺度，以及 mode 的 logits                                                                                                                                      | 原 repo `robomimic/models/policy_nets.py:489-491`                                                                                                           |
| 它的平均值            | 輸出取 `tanh`                                                                                                                                                                                           | 原 repo `robomimic/models/policy_nets.py:514`                                                                                                               |
| 它的尺度              | 輸出取 `softplus` 再加 `gmm_min_std`；開啟 `gmm_low_noise_eval` 時，訓練以外為 1e-4                                                                                                                     | 原 repo `robomimic/models/policy_nets.py:519`、`:522`                                                                                                       |
| 它的分佈              | 先以 categorical 選 mode，再在各 action 維度上用對角高斯分佈                                                                                                                                            | 原 repo `robomimic/models/policy_nets.py:526-535`                                                                                                           |
| 它的 loss             | 示範 action 的平均對數似然取負                                                                                                                                                                          | 原 repo `robomimic/algo/bc.py:300`、`:322`（`BC_Gaussian`，`BC_GMM` 在 `:347` 繼承它）                                                                      |
| 它的 action           | 從分佈中抽樣                                                                                                                                                                                            | 原 repo `robomimic/models/policy_nets.py:555`                                                                                                               |

robomimic 把 MLP 拆成隱藏層與輸出層（`robomimic/models/obs_nets.py:619-620`），但兩者
最後都接 ReLU，結果就是每個大小各一層 Linear 加 ReLU，`MLP` 也就這樣建立。

即使在推論時，`GMMHead` 也是依混合權重抽樣出 mode，而不是選機率最高的 mode；低雜訊只會
把高斯分佈縮小到該 mode 的平均值附近，和 robomimic 相同。

標準差固定時，robomimic 仍會建立 Gaussian 的尺度層，只是不使用它
（`robomimic/models/policy_nets.py:318-319`、`:338`）；`GaussianHead` 則不建立（本專案）。

每個 action head 各自帶有 loss 與 action 的選擇方式（本專案）。robomimic 則是依 config
旗標，為每種網路搭配一個演算法子類別（`robomimic/algo/bc.py:46-73`）。把 head 和 MLP
分開，之後的 recurrent 與 transformer policy 就能重用 deterministic 與 GMM head。
robomimic 沒有 Gaussian 的 BC-RNN 或 BC-Transformer（`robomimic/algo/bc.py:46-49`），
因此它們的 config 不接受 `use_gaussian`。

robomimic 也可以用 `exp` 取代 `softplus` 來輸出兩種分佈的標準差
（`robomimic/config/bc_config.py:48`、`:55`）；它的預設值與論文實驗都用 `softplus`，
`exp` 不移植。

### 6. Policy

`RobomimicBCPolicy` 在 LeRobot 訓練與評估流程會呼叫的 method 中，依序串接
observation encoder、MLP 與 action head，行為依照 robomimic 的 `BC`、`BC_Gaussian` 與
`BC_GMM` class（原 repo `robomimic/algo/bc.py:78`、`:254`、`:347`）。測試會把相同權重載入
robomimic 的演算法，依 §4 的方式重新排列 MLP 第一層的輸入欄位，並比對三種 head 的 loss
與 action
（`tests/test_policy.py`）。

| Method                 | 行為                                                     | 來源                                                                                                                                                                                                                  |
| ---------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `forward`              | action head 的 loss 與它記錄的數值                       | 原 repo `robomimic/algo/bc.py:182-192`（`BC`）、`:300`、`:322`（`BC_Gaussian`、`BC_GMM`）                                                                                                                             |
| `select_action`        | action head 對當下 observation 給出的 action，不計算梯度 | 原 repo `robomimic/algo/bc.py:239-251`                                                                                                                                                                                |
| `predict_action_chunk` | 同一個 action，作為長度 1 的 chunk                       | 本專案：LeRobot 預期 `(B, chunk_size, action_dim)`（Adding a Policy 指南）；BC 只預測一個 action                                                                                                                      |
| `__init__`             | 接收 `**kwargs` 並忽略                                   | LeRobot：`make_policy` 還會傳入 `dataset_stats` 與 `dataset_meta`（`policies/factory.py:323`、`:326`）；內建 policy 都接收 `**kwargs`（`policies/diffusion/modeling_diffusion.py:65-69`）                             |
| `reset`                | 沒有需要重設的狀態                                       | 原 repo `robomimic/algo/algo.py:365`（BC 沿用基底類別空的 `reset`）                                                                                                                                                   |
| `get_optim_params`     | 所有參數，以 `self.parameters()` 回傳                    | 原 repo `robomimic/algo/algo.py:169-193`（policy 網路只有一個 optimizer）；LeRobot 會把結果直接交給 optimizer（`optim/factory.py:37-40`），Diffusion 也是這樣回傳（`policies/diffusion/modeling_diffusion.py:89-90`） |

`__init__` 與 `get_optim_params` 和指南的 template 不同：template 的 `__init__` 只接收
`dataset_stats`，`get_optim_params` 回傳 `{"params": ...}`。在 LeRobot v0.6.1 中，前者
會因 `dataset_meta` 拋出 `TypeError`，後者會讓 optimizer 迭代到字串 `"params"`；兩者都是
在 §7 的 `lerobot-train` 執行中發現的。

import robomimic 的 `BC` 時會一併 import 所有其他演算法（原 repo
`robomimic/algo/__init__.py`），因此測試需要 `diffusers` 與 `imageio`。它們是 dev
依賴，`diffusers` 的版本範圍沿用 LeRobot 的 `diffusion` extra。

### 7. Processor

`make_robomimic_bc_pre_post_processors` 回傳 LeRobot 的預設 pipeline
（`processor/factory.py:116-128`、`:173-174`）。policy 之前的 pipeline 不重新命名任何
key、加上 batch 維度、把 tensor 移到 policy 的裝置並做正規化；policy 之後的 pipeline
做反正規化並把 action 移回 CPU。在 §2 的 `IDENTITY` 對應下，即使給了 dataset 統計值，
正規化也不會改變任何數值（`tests/test_processor.py`）。

在轉換好的 Lift image dataset 上跑 20 步的 `lerobot-train`，可以完成訓練、存下
checkpoint，並連同 processor 讀回後執行 `select_action`。`lerobot-train` 需要
`lerobot[training]`，本專案和 `lerobot_policy_openvla_oft` 一樣以 `training` extra
提供。

## BC-RNN（`robomimic_bc_rnn`）

BC-RNN 是 robomimic 論文的主力 policy。它重用 BC 的相機 encoder（§3）、observation
encoder（§4）與 action head（§5）。

### 8. Configuration

`RobomimicBCRNNConfig` 註冊 policy type `robomimic_bc_rnn`。它沿用 §2 的共用設定（robomimic
的 BC-RNN 實驗也使用相同設定），並新增下列設定。預設值沿用 PH 資料上的 image 實驗；
`tests/test_paper_defaults.py` 會和 robomimic 自己的 config 比對。

| 設定             | 預設值                                | Config 欄位                                                        | 來源                                                                                  |
| ---------------- | ------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| Recurrent 網路   | 單向 LSTM                             | （固定）                                                           | 原 repo `robomimic/config/bc_config.py:89`、`:92`                                     |
| LSTM 層數        | 2                                     | `rnn_num_layers`                                                   | 原 repo `robomimic/config/bc_config.py:90`                                            |
| LSTM hidden size | image 資料為 1000；low-dim 資料為 400 | `rnn_hidden_dim`（1000）                                           | 原 repo `robomimic/scripts/generate_paper_configs.py:428`（image）、`:416`（low-dim） |
| LSTM 之後的 MLP  | 無                                    | `actor_layer_dims`（()）                                           | 原 repo `robomimic/scripts/generate_paper_configs.py:414`                             |
| 序列長度         | 10 步的 observation 與 action         | `rnn_horizon`、`observation_delta_indices`、`action_delta_indices` | 原 repo `robomimic/scripts/generate_paper_configs.py:405`、`:409-410`                 |
| 推論時重設狀態   | 每 10 步                              | `rnn_horizon`                                                      | 原 repo `robomimic/scripts/generate_paper_configs.py:410`                             |

robomimic 把序列長度（`train.seq_length`）與重設週期（`algo.rnn.horizon`）分開設定，但論文
把兩者都設為 10，其 config 也註明兩者通常應該相同（`robomimic/config/bc_config.py:87`）。
本專案用同一個欄位 `rnn_horizon` 表示兩者。

demo 中每個 frame 都是一段訓練序列的起點；序列超出 demo 結尾時會重複最後一個 frame
（原 repo `robomimic/utils/dataset.py:225`、`:557-576`）。LeRobot 依 delta index 建立相同的
序列：它同樣以每個 frame 為起點，並把超出結尾的 index 夾到最後一個 frame
（LeRobot `datasets/dataset_reader.py:223`）。

robomimic 的 open-loop 模式（`robomimic/config/bc_config.py:91`）、GRU 與雙向 LSTM，
論文實驗都沒有使用，因此不移植。

### 9. 網路

`RobomimicBCRNNPolicy` 以 observation encoder（§4）編碼序列的每一步，用 PyTorch 的
`nn.LSTM` 依序處理各步，再在每一步套用 §5 的 MLP 與 action head。三者合起來對應 robomimic
的 `RNNActorNetwork` 與 `RNNGMMActorNetwork`（原 repo `robomimic/models/policy_nets.py:563`、
`:728`）。

| 部分    | 行為                                              | 來源                                                                                            |
| ------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Encoder | §4 的 observation encoder，套用到每一步           | 原 repo `robomimic/models/obs_nets.py:858`                                                      |
| LSTM    | batch-first 的 `nn.LSTM`，沒有給狀態時從零開始    | 原 repo `robomimic/models/obs_nets.py:788`、`robomimic/models/base_nets.py:347`、`:372`、`:424` |
| 每一步  | MLP（預設為空）與 action head，套用到每一步的輸出 | 原 repo `robomimic/models/obs_nets.py:765`、`:785`、`robomimic/models/base_nets.py:426`         |

robomimic 用 `RNN_Base` 包裝 LSTM（原 repo `robomimic/models/base_nets.py:306`），它只多了
補零的初始狀態；`nn.LSTM` 本來就從零開始，因此本專案直接使用它。robomimic LSTM 的 state
dict 因此可以原樣載入 `lstm`，只有 `weight_ih_l0` 的輸入欄位是依 robomimic 排序後的 key
排列（§4）。

### 10. Policy

`RobomimicBCRNNPolicy` 依照 robomimic 的 `BC_RNN` 與 `BC_RNN_GMM` class（原 repo
`robomimic/algo/bc.py:483`、`:578`）。測試會把相同權重載入 robomimic 的演算法，比對兩種
head 的序列 loss，以及連續 25 步、跨越兩次狀態重設的 action（`tests/test_policy_bc_rnn.py`）。

| Method                 | 行為                                                     | 來源                                                                                                 |
| ---------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `forward`              | 每段序列每一步的 action head loss，包含補值的步          | 原 repo `robomimic/algo/bc.py:630`、`:652`（`BC_RNN_GMM`）；`BC_RNN` 沿用 `BC` 的 loss（`:182-192`） |
| `select_action`        | 從保留的狀態跑一步 LSTM；在第 0、10、20 步等之前清空狀態 | 原 repo `robomimic/algo/bc.py:551`、`:565`                                                           |
| `reset`                | 清空 LSTM 狀態與步數                                     | 原 repo `robomimic/algo/bc.py:570-575`                                                               |
| `predict_action_chunk` | 不支援                                                   | 本專案：每個 action 都會推進 LSTM 狀態，而指南允許不做 chunk 的 policy 拋出 `NotImplementedError`    |

LSTM 狀態不會延續整個 episode：robomimic 在推論時每 `rnn_horizon` 步清空一次，和訓練時
10 步的序列長度一致。

這個 policy 和 BC 一樣，用 `ObservationEncoder.from_config` 與 `make_action_head` 建立
encoder 與 action head。

在 CUDA 上，cuDNN 執行 `flatten_parameters` 後，會把 LSTM 的權重都放成同一塊記憶體的 view
（PyTorch 2.11 `torch/nn/modules/rnn.py:237`）。LeRobot 用 safetensors 的 `save_model` 儲存 policy
（`policies/pretrained.py:160`），而它拒絕儲存「共用同一塊記憶體、卻沒有任何一個完整涵蓋它」
的 tensor，因此 BC-RNN 的 checkpoint 在雲端 GPU 上存檔失敗（本專案；由 benchmark 執行發現）。
`RobomimicBCRNNPolicy._save_pretrained` 改為儲存權重的副本。CPU 與 MPS 上的權重是各自獨立的，
因此測試會讓 LSTM 的權重共用同一塊記憶體來重現這個錯誤（`tests/test_policy_bc_rnn.py`）。

### 11. Processor

`make_robomimic_bc_rnn_pre_post_processors` 回傳和 BC 相同的預設 pipeline（§7）。推論時
它們一次處理一步，訓練時則處理整段序列，正規化會同樣套用到每一步；在 `IDENTITY` 對應下
數值不會改變（`tests/test_processor.py`）。

在轉換好的 Lift image dataset 上，用 `train` mask 跑 20 步的 `lerobot-train`，可以訓練這個
35M 參數的 policy、存下 checkpoint，並連同 processor 讀回後執行 12 步 `select_action`，
期間跨過一次狀態重設。

### 12. Model zoo checkpoint

`lerobot_policy_robomimic/convert_checkpoint.py` 把 robomimic model zoo 的 BC-RNN
checkpoint 轉成 `robomimic_bc_rnn` 的 policy 目錄，包含 config、權重與 processor。model zoo
是用 robomimic v0.1 訓練的（原 repo `docs/model_zoo/robomimic_v0.1.md`），因此轉換程式讀取
該版本的格式；robomimic v0.5 本身無法載入這些 image checkpoint，因為它只升級 config
（`robomimic/utils/file_utils.py:247`），載入權重時則要求名稱完全相同
（`robomimic/algo/algo.py:353`）。

checkpoint 以 `torch.load(weights_only=True)` 讀取，不會執行 pickle 中的程式碼。轉換前會先
檢查 config；遇到本專案沒有實作的設定，例如 GRU 或 open-loop RNN，會報錯並停止轉換。

| robomimic v0.1 的部分                                    | 和本專案相同嗎？                                                    | 來源                                                                                                  |
| -------------------------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| 相機 encoder：ResNet-18、spatial softmax、線性投影、ReLU | 相同；程式碼只差在預設參數                                          | 原 repo `v0.1.0:robomimic/models/base_nets.py:840-873`、`v0.1.0:robomimic/models/obs_nets.py:271-276` |
| 影像方向                                                 | 相同，由 env wrapper 翻正                                           | 原 repo `v0.1.0:robomimic/envs/env_robosuite.py:185`                                                  |
| 特徵順序                                                 | 相同，依排序後的 key                                                | checkpoint 的 `shape_metadata["all_shapes"]`                                                          |
| 每 `rnn_horizon` 步重設 LSTM 狀態                        | 相同                                                                | 原 repo `v0.1.0:robomimic/algo/bc.py:521-535`                                                         |
| 推論時的裁切                                             | 不同，為隨機裁切；轉換後的 config 會設定 `random_crop_at_inference` | §3                                                                                                    |
| 權重名稱                                                 | 不同；依下表改名                                                    | checkpoint 的 key                                                                                     |

| robomimic v0.1 的權重                                                                                   | 本專案                                                   |
| ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `policy.nets.encoder.nets.obs.obs_nets.<camera>_image.vis_core.nets.*`                                  | `encoder.cameras.<camera>.encoder.backbone.*`            |
| `…<camera>_image.pool_net.nets.*`                                                                       | `encoder.cameras.<camera>.encoder.pool.keypoints.*`      |
| `…<camera>_image.nets.3.*`                                                                              | `encoder.cameras.<camera>.encoder.projection.*`          |
| `policy.nets.rnn.nets.*`                                                                                | `lstm.*`，其中 `weight_ih_l0` 的輸入欄位會重新排列（§4） |
| `policy.nets.decoder.nets.{mean,scale,logits}.*`                                                        | `action_head.{mean,scale,logits}.*`                      |
| `…<camera>_image.nets.{0,1}.*`、`…pool_net.{temperature,pos_x,pos_y}`、`policy.nets.rnn.per_step_net.*` | 捨棄：它們是上述權重的重複，或是固定值（§3）             |

之後以 `strict=True` 載入權重，因此 policy 的每個權重都必須來自 checkpoint。

測試會把 robomimic v0.5 的 BC-RNN 網路寫成 v0.1 格式、轉換後，和 robomimic 自己的
`get_action` 比對 12 步的 action；若已下載 Lift checkpoint，也會轉換它
（`tests/test_convert_checkpoint.py`）。

在轉換好的 Lift image dataset 中，20 個 `valid` demo 共 1026 個 frame 上，轉換後的 Lift PH
image checkpoint（`lift_ph_image_epoch_500_succ_100.pth`，SHA-256 `37b94a11…bc47fb`）
預測示範 action 的平均絕對誤差為每維 0.071，改用中央裁切時為 0.073。兩個對照組說明這個誤差
是有意義的：一律預測 0 時為 0.263，略過 `weight_ih_l0` 的欄位重新排列時為 0.305。這份
dataset 是用 robosuite 1.5.1 重新產生的，而 checkpoint 是用 robosuite `offline_study`
branch 的資料訓練，因此部分誤差可能來自這個差異。

在閉環評估中，以 `lerobot-eval` 在 Lift image dataset 的 `robomimic` env 上評估（50 個
episode、seed 1000，在 Apple M5 Max 上每個 episode 約 1 秒），轉換後的 checkpoint 成功率為
100%，和 model zoo 回報的約 100% 一致（原 repo `docs/model_zoo/robomimic_v0.1.md:35`）。若推論時
改用中央裁切、而不是 v0.1 的隨機裁切（§3），在相同的 episode 上成功率為 98%。

## BC-Transformer（`robomimic_bc_transformer`）

BC-Transformer 不在 robomimic 的論文中，是 robomimic 在 v0.3 加入的（原 repo commit
`40e427a`）。它重用 BC 的相機 encoder（§3）、observation encoder（§4）與 action head（§5）。

### 13. Configuration

`RobomimicBCTransformerConfig` 註冊 policy type `robomimic_bc_transformer`，並沿用 §2 的
共用設定。論文沒有 BC-Transformer，因此 transformer、optimizer 與 schedule 的預設值沿用
robomimic transformer 教學稱為「調好參數」的 template（原 repo
`docs/tutorials/training_transformers.md`、
`robomimic/config/default_templates/bc_transformer.json`，下表稱為「template」）。
`tests/test_paper_defaults.py` 會以 `robomimic/scripts/train.py:475-479` 載入 config 的方式
讀取該 template 並比對。

| 設定                   | 預設值                                       | Config 欄位                                                                               | 來源                                                                                                                 |
| ---------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| 看到的 observation     | 當下這一步與之前的 9 步                      | `transformer_context_length`（10）、`observation_delta_indices`                           | template `:47`（`seq_length` 1）、`:49`（`frame_stack` 10）；原 repo `robomimic/config/bc_config.py:97`              |
| episode 開始前的步     | 以第一個 frame 補齊                          | （固定）                                                                                  | 原 repo `robomimic/utils/dataset.py:557-576`；LeRobot 把 index 夾到 episode 開頭（`datasets/dataset_reader.py:223`） |
| 監督的 action          | 只有當下這一步                               | `action_delta_indices`（None）                                                            | 原 repo `robomimic/algo/bc.py:734`；`supervise_all_steps` 為關閉（`robomimic/config/bc_config.py:106`）              |
| Transformer 大小       | 寬度 512、6 個 block、8 個 head              | `transformer_embed_dim`、`transformer_num_layers`、`transformer_num_heads`                | 原 repo `robomimic/config/bc_config.py:98-100`                                                                       |
| Dropout                | embedding、attention 與 block 輸出都是 0.1   | `transformer_emb_dropout`、`transformer_attn_dropout`、`transformer_block_output_dropout` | 原 repo `robomimic/config/bc_config.py:101-103`                                                                      |
| Optimizer              | AdamW，learning rate 1e-4，weight decay 0.01 | `get_optimizer_preset()`、`optimizer_lr`、`optimizer_weight_decay`（0.01）                | template `:63`、`:65`、`:71`                                                                                         |
| Learning rate schedule | 100 個 epoch 內線性降到 0.1 倍，之後維持     | `scheduler_decay_epochs`（100）、`scheduler_decay_factor`（0.1）                          | template `:66-68`；原 repo `robomimic/utils/torch_utils.py:146-157`                                                  |
| Epoch 長度             | 100 次 optimizer 更新                        | `scheduler_steps_per_epoch`（100）                                                        | template `:21`（`epoch_every_n_steps`）                                                                              |

robomimic 在每個 epoch 結束時才更新一次 learning rate schedule（原 repo
`robomimic/algo/algo.py:313-315`，由 `robomimic/scripts/train.py:311` 呼叫），而 LeRobot 在
每次 optimizer 更新後都會呼叫 scheduler（LeRobot `scripts/lerobot_train.py:193`），也沒有
epoch 的概念。因此 `lerobot_policy_robomimic/schedulers.py` 中的
`RobomimicLinearSchedulerConfig`（註冊為 scheduler type `robomimic_linear`）以
`scheduler_steps_per_epoch` 步為一個 epoch 計數，並在每個 epoch 內固定 learning rate。
`tests/test_schedulers.py` 會和每個 epoch 呼叫一次的 robomimic scheduler 逐步比對。

template 是 low-dim 資料的設定：batch size 100，訓練 2000 個 epoch、每個 epoch 100 步
（共 20 萬步；template `:56-57`、`:21`），這些要傳給 `lerobot-train`。

robomimic 中監督每一步（`supervise_all_steps`）、預測未來 action（`pred_future_acs`）、
sinusoidal 或 `nn.Embedding` 位置編碼，以及 GEGLU 的選項，在 template 中都是關閉的，因此
不移植。

### 14. Transformer

`lerobot_policy_robomimic/transformer.py` 的 `Transformer` 把一段編碼後的 observation 序列
轉成每一步各一個特徵。它涵蓋 robomimic `MIMO_Transformer` 的 embedding 部分（原 repo
`robomimic/models/obs_nets.py:997-1022`、`:1088-1102`）與其 `GPT_Backbone`（原 repo
`robomimic/models/transformers.py`）。測試會載入 robomimic 的權重並比對輸出
（`tests/test_transformer.py`）。

| 部分           | 行為                                                                                                     | 來源                                                                                            |
| -------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 輸入 embedding | 線性投影到寬度，加上每一步可學習的位置向量，再做 LayerNorm 與 dropout                                    | 原 repo `robomimic/models/obs_nets.py:997`、`:1009`、`:1016`、`:1019`、`:1102`                  |
| Attention      | causal multi-head self-attention，query-key-value 投影沒有 bias                                          | 原 repo `robomimic/models/transformers.py:133`、`:143`、`:179-184`                              |
| Block          | pre-norm：attention 與 4 倍寬的 GELU MLP，各自加回輸入                                                   | 原 repo `robomimic/models/transformers.py:276`、`:291-292`                                      |
| 輸出           | 最後一個 block 之後的 LayerNorm                                                                          | 原 repo `robomimic/models/transformers.py:392`                                                  |
| 初始化         | block 與輸出 norm 的權重從 normal(0, 0.02) 開始、bias 為 0；輸入投影沿用 PyTorch 預設，位置向量從 0 開始 | 原 repo `robomimic/models/transformers.py:362`、`:394-399`；`robomimic/models/obs_nets.py:1009` |

attention 使用 PyTorch 的 `scaled_dot_product_attention` 並設定 `is_causal=True`（本專案），
它和 robomimic 明確寫出的遮罩（`robomimic/models/transformers.py:143`、`:184`）計算相同的
masked softmax，差異在浮點誤差範圍內，而且不需要儲存遮罩。

### 15. Policy

`RobomimicBCTransformerPolicy` 編碼 context 中的 observation，執行 §14 的 transformer，並在
最後一步套用 §5 的 action head，行為依照 robomimic 的 `BC_Transformer` 與
`BC_Transformer_GMM` class（原 repo `robomimic/algo/bc.py:677`、`:794`）。測試會把相同權重
載入 robomimic 的演算法，比對兩種 head 的 loss、最後一步的 GMM，以及在補齊的
observation 視窗上連續 13 步的 action（`tests/test_policy_bc_transformer.py`）。

| 部分                   | 行為                                                                                          | 來源                                                                                                                       |
| ---------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 相機特徵               | 每支相機的 encoder 之後不接 ReLU，和 BC、BC-RNN 不同                                          | 原 repo `robomimic/models/obs_nets.py:991`（`feature_activation=None`）                                                    |
| `forward`              | 每段 context 最後一步 action 的 action head loss                                              | 原 repo `robomimic/algo/bc.py:734`、`:766`                                                                                 |
| `select_action`        | 保留最近 `transformer_context_length` 個 observation；episode 的第一個 observation 會填滿全部 | 原 repo `robomimic/envs/wrappers.py:97`、`:130-131`（`FrameStackWrapper`，於 `robomimic/utils/env_utils.py:342-343` 套用） |
| Action                 | action head 在最後一步給出的 action                                                           | 原 repo `robomimic/algo/bc.py:769`、`:788`                                                                                 |
| `reset`                | 清空保存的 observation                                                                        | 原 repo `robomimic/envs/wrappers.py:152-165`，以 `:119-131` 重建記錄                                                       |
| `predict_action_chunk` | 不支援                                                                                        | 本專案：和 BC-RNN（§10）一樣，每個 action 都取決於 `select_action` 保存的 observation                                      |

`ObservationEncoder.from_config` 接受 `camera_activation` 參數，只有這個 policy 會關閉它，
因此三個 policy 共用同一個 encoder。

robomimic 在計算 BC-Transformer 的 GMM loss 時，即使不在訓練模式也使用訓練時的尺度（原 repo
`robomimic/algo/bc.py:836`），而 BC 與 BC-RNN 在此情況下使用低雜訊的尺度；`GMMHead` 對三者
都採用後者。兩者在每個訓練步都相同，只有在評估模式下計算 loss 時才有差異。

robomimic 會為 context 中每一步都抽樣 action，再取最後一步（`robomimic/algo/bc.py:788`）；
本專案只抽樣最後一步，分佈相同但使用的亂數不同，因此測試比對的是分佈而不是抽樣結果。

### 16. Processor

`make_robomimic_bc_transformer_pre_post_processors` 回傳和 BC 相同的預設 pipeline（§7），
不會正規化任何 feature（`tests/test_processor.py`）。

在轉換好的 Lift image dataset 上，用 `train` mask 跑 20 步的 `lerobot-train`，可以用 AdamW 與
`robomimic_linear` schedule 訓練這個 41M 參數的 policy、存下 checkpoint（其訓練設定保留了這個
scheduler type），並連同 processor 讀回後執行 12 步 `select_action`。把 epoch 縮短為 2 步、
衰減期設為 5 個 epoch 時，記錄到的 learning rate（LeRobot 對每 5 步記錄區間取的平均）依序為
7.8e-5、3.2e-5，之後為 1.0e-5，和 schedule 的預期相同。

## BC-VAE（`robomimic_bc_vae`）

BC-VAE 是 robomimic 從 v0.1 就有的 BC 變體（原 repo `robomimic/algo/bc.py:373`、
`v0.1.0:robomimic/algo/bc.py:345`；`docs/introduction/implemented_algorithms.md`）。論文沒有把它
當作 policy 訓練，只把 VAE 用作 BCQ 的 action sampler
（`robomimic/scripts/generate_paper_configs.py:463-470`）。它重用 BC 的相機 encoder（§3）與
observation encoder（§4），但沒有 action head，而是以條件式 VAE 建模 action。

### 17. Configuration

`RobomimicBCVAEConfig` 註冊 policy type `robomimic_bc_vae`。BC-VAE 沒有 action head，因此它
只繼承 `RobomimicPolicyConfig`，也就是 §2 的 observation、encoder 與 optimizer 設定；
BC-VAE 的 optimizer 和 BC 相同（原 repo `robomimic/config/bc_config.py:27-33`）。下表的
VAE 設定沿用 robomimic 的預設值；`tests/test_paper_defaults.py` 會把它們和 robomimic 在
image 實驗中的 BC-VAE config 比對。

| 設定                   | 預設值                              | Config 欄位                                                                                             | 來源                                                                                                                                            |
| ---------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Latent 大小            | 14，即 7 維 action 的兩倍           | `vae_latent_dim`                                                                                        | 原 repo `robomimic/config/bc_config.py:60`                                                                                                      |
| Latent 範圍限制        | 無                                  | `vae_latent_clip`                                                                                       | 原 repo `robomimic/config/bc_config.py:61`                                                                                                      |
| KL 權重                | 1                                   | `vae_kl_weight`                                                                                         | 原 repo `robomimic/config/bc_config.py:62`                                                                                                      |
| Encoder 與 decoder MLP | 各為 (300, 400)                     | `vae_encoder_layer_dims`、`vae_decoder_layer_dims`                                                      | 原 repo `robomimic/config/bc_config.py:81-82`                                                                                                   |
| Decoder 輸入           | Latent 與 observation               | （固定）                                                                                                | 原 repo `robomimic/config/bc_config.py:65`；不論這個設定為何，robomimic 都讓 decoder 讀 observation（`robomimic/models/vae_nets.py:1049-1050`） |
| 重建 loss              | 平方誤差的平均                      | `vae_reconstruction_sum_across_elements`（False）                                                       | 原 repo `robomimic/config/bc_config.py:66`、`robomimic/models/vae_nets.py:1277-1284`                                                            |
| Prior                  | 固定的 N(0, 1)                      | `vae_prior_learn`（False）                                                                              | 原 repo `robomimic/config/bc_config.py:69`                                                                                                      |
| 可學習的 prior         | 單一高斯分佈，不依 observation 而變 | `vae_prior_is_conditioned`（False）、`vae_prior_use_gmm`（False）、`vae_prior_layer_dims`（(300, 400)） | 原 repo `robomimic/config/bc_config.py:70-71`、`:83`                                                                                            |
| 混合 prior             | 10 個 mode，權重均等                | `vae_prior_gmm_num_modes`（10）、`vae_prior_gmm_learn_weights`（False）                                 | 原 repo `robomimic/config/bc_config.py:72-73`                                                                                                   |

robomimic 以 assertion 拒絕的組合，config 會丟出 `ValueError`：prior 依 observation
條件化或使用混合分佈、卻沒有設為可學習（原 repo `robomimic/models/vae_nets.py:943`、`:983`）。

robomimic 的 `decoder.is_conditioned` 只用在一個 assertion，檢查 decoder 或 prior 至少有一個
依 observation 條件化（`robomimic/models/vae_nets.py:940`）；只要 VAE 有 observation，decoder
就會拿到 observation 這一組輸入（`:1049-1050`，`v0.1.0:robomimic/models/vae_nets.py:1109`
也一樣）。在 robomimic 中把它設為 false 沒有任何效果，因此本專案沒有這個欄位；測試在
robomimic 端把它設為 false，結果仍然一致（`tests/test_vae.py`）。

robomimic 的 categorical prior（`robomimic/config/bc_config.py:74-79`）不移植。它每個 epoch
把 Gumbel-softmax 的溫度降低固定的量（`robomimic/algo/bc.py:393-400`），而 LeRobot 不會把
epoch 交給 policy。

### 18. 網路

`lerobot_policy_robomimic/vae.py` 的 `ActionVAE` 是 `VAEActor` 為 action 建立的 robomimic
`VAE`（原 repo `robomimic/models/policy_nets.py:1336`、`robomimic/models/vae_nets.py:747`）。
它的三個部分各是一個 `ConditionedMLP`，對應 robomimic 的 `MIMO_MLP`
（`robomimic/models/obs_nets.py:541`）：一個 §4 的 observation encoder、一個 MLP，以及每個
輸出各一層 Linear。`tests/test_vae.py` 把相同權重載入 robomimic 的 `VAE`，依 §4 的方式重新
排列每個 MLP 第一層的 observation 欄位，並在八種 prior 與 decoder 設定下比對 loss 與抽樣出的
action。

| 部分        | 行為                                                                                                    | 來源                                                                                                                            |
| ----------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Posterior   | Action 接上編碼後的 observation，經 MLP 輸出 latent 的平均值與對數變異數                                | 原 repo `robomimic/models/vae_nets.py:1014-1016`、`:1026-1029`                                                                  |
| Decoder     | Latent 接上編碼後的 observation，經 MLP 輸出 action，再以 `tanh` 限制範圍                               | 原 repo `robomimic/models/vae_nets.py:1048-1050`、`:1192`；`robomimic/models/policy_nets.py:1398-1399`                          |
| Prior       | N(0, 1)；或是可學習的參數，以 N(0, 1) 除以參數大小的平方根初始化；或是讀取自己編碼的 observation 的 MLP | 原 repo `robomimic/models/vae_nets.py:419`、`:128`、`:1068-1069`                                                                |
| 混合權重    | 均等，或是可學習並經 `log_softmax` 正規化                                                               | 原 repo `robomimic/models/vae_nets.py:403`、`:499`                                                                              |
| Latent 抽樣 | 平均值加上標準差乘以高斯雜訊，對數標準差限制在 [−4, 15]                                                 | 原 repo `robomimic/utils/torch_utils.py:77`、`:83`                                                                              |
| Prior 抽樣  | 依混合權重抽出 mode，再從該 mode 抽 latent；有設定 `vae_latent_clip` 時再限制範圍                       | 原 repo `robomimic/models/vae_nets.py:406-412`、`:422`                                                                          |
| KL loss     | 對 N(0, 1) 或可學習的高斯分佈用解析解；對混合分佈則在 posterior 抽樣點上估計，對數變異數限制在 [−8, 30] | 原 repo `robomimic/models/vae_nets.py:452`、`:462`、`:470-474`；`robomimic/utils/loss_utils.py:39`、`:56-60`、`:78`、`:104-119` |
| 重建 loss   | 解碼 posterior 抽樣後的平方誤差，對所有元素取平均，或對每個樣本加總                                     | 原 repo `robomimic/models/vae_nets.py:1277-1284`                                                                                |

和 robomimic 一樣，每個部分都有自己的 observation encoder，因為 robomimic 的每個 `MIMO_MLP`
都會建立一個（`robomimic/models/obs_nets.py:608`）：兩支相機時，posterior 與 decoder
共有四個 ResNet-18，條件化的 prior 再多兩個。

### 19. Policy

`RobomimicBCVAEPolicy` 在 LeRobot 訓練與評估流程會呼叫的 method 中包裝 `ActionVAE`，行為依照
robomimic 的 `BC_VAE` class（原 repo `robomimic/algo/bc.py:373`）。測試會把相同權重載入
robomimic 的演算法，使用可學習、依 observation 條件化的混合 prior 與 0.5 的 KL 權重，比對
loss、記錄的數值與 action（`tests/test_policy_bc_vae.py`）。

| Method                 | 行為                                                                          | 來源                                                                                |
| ---------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `forward`              | 重建 loss 加上 `vae_kl_weight` × KL loss；記錄這兩者與 posterior 變異數的平均 | 原 repo `robomimic/algo/bc.py:403-433`、`:452`、`:477`                              |
| `select_action`        | 從 prior 抽 latent 再解碼出的 action，每個 observation 一個，不計算梯度       | 原 repo `robomimic/algo/bc.py:239-251`；`robomimic/models/policy_nets.py:1565-1570` |
| `predict_action_chunk` | 同一個 action，作為長度 1 的 chunk                                            | 本專案，與 BC 相同（§6）                                                            |
| `reset`                | 沒有需要重設的狀態                                                            | 原 repo `robomimic/algo/algo.py:365`                                                |
| `get_optim_params`     | 所有參數，以 `self.parameters()` 回傳                                         | 原 repo `robomimic/algo/algo.py:169-193`；LeRobot 部分與 BC 相同（§6）              |

`BC_VAE` 也會從 batch 讀取 `freeze_encoder` 旗標，用來擋住流向 posterior 的梯度
（`robomimic/algo/bc.py:419`）。只有 BCQ 會為自己的 action sampler 設定它
（`robomimic/algo/bcq.py:242-243`），因此不移植。

### 20. Processor

`make_robomimic_bc_vae_pre_post_processors` 回傳和 BC 相同的預設 pipeline（§7），不會正規化
任何 feature（`tests/test_processor.py`）。

在轉換後的 Lift image dataset 的 `train` mask 上執行 20 步 `lerobot-train`，以可學習、依
observation 條件化的混合 prior 訓練 BC-VAE，儲存 checkpoint，並連同 processor 載入後執行 12 步
`select_action`。這個 policy 有 6800 萬個參數，大多來自三個 observation encoder、每個各兩支
相機的六個 ResNet-18（§18）。使用混合 prior 時，記錄的 KL loss 可能是負的，因為 robomimic 只用
一個 posterior 樣本估計它（§18）。

## 語言條件化

robomimic 在 v0.5 加入語言條件化（原 repo commit `ae5799f`、
`docs/tutorials/language_conditioning.md`）。它不在論文中；因為它位於 observation encoder，
本專案的每個 policy 都能使用。論文的每個任務只有一句固定的指令，因此 robomimic 讓同一個
dataset 的所有示範共用同一句。

### 21. 任務 embedding

`lerobot_policy_robomimic/language.py` 的 `CLIPTaskEmbeddingStep` 把 LeRobot 每個 frame 都有的
task 字串轉成 embedding；`lerobot-eval` 則從 env 的 `task_description` 取得這個字串（LeRobot
`scripts/lerobot_eval.py:281`）。它把 embedding 以 `observation.language.embedding` 加進
observation。`lerobot_policy_robomimic/processors.py` 的 `make_robomimic_pre_post_processors`
把這個 step 放在加上 batch 維度之後、tensor 移到 policy 裝置之前；每個 policy 的 processor
factory 都呼叫這同一個函式。`tests/test_language.py` 會檢查這個 step，並在 Hugging Face
cache 中有 CLIP 模型時，把它的 embedding 和 robomimic 的 `get_lang_emb` 比對。

| 設定或部分     | 行為                                                               | Config 欄位                                        | 來源                                                                                                                   |
| -------------- | ------------------------------------------------------------------ | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| 條件化方式     | 關閉；設為 `concat` 或 `film` 時開啟                               | `language_conditioning`（None）                    | 原 repo `docs/tutorials/language_conditioning.md`（第 2 節）                                                           |
| CLIP 模型      | `openai/clip-vit-large-patch14` 投影後的文字 embedding，768 維     | `clip_model_name`、`language_embedding_dim`（768） | 原 repo `robomimic/utils/lang_utils.py:4`、`:21`、`:43`；`robomimic/models/base_nets.py:666`                           |
| Token          | 單獨處理每個 task，加上特殊 token，補齊到 25 個 token              | （固定）                                           | 原 repo `robomimic/utils/lang_utils.py:35-42`                                                                          |
| Embedding 來源 | 每個 frame 的 task 字串                                            | （固定）                                           | 本專案；robomimic 則是每個 dataset 嵌入一句 `lang`，再複製到每一步（`robomimic/utils/dataset.py:113-114`、`:530-532`） |
| 模型載入       | 第一個 task 出現時載入，每個模型只載入一次；embedding 依 task 快取 | （固定）                                           | 原 repo `robomimic/utils/lang_utils.py:8-27`（延遲載入）                                                               |

改用 LeRobot 的 task 字串、而不是每個 dataset 的設定，對 robomimic 的單一任務 dataset 會得到
相同的輸入，也讓 `lerobot-eval` 能把同一句指令交給 policy，env 不必自己計算 embedding。

語言條件化需要 `transformers`，以 `language` extra 提供，版本範圍沿用 LeRobot 自己的
`lerobot[transformers-dep]`。CLIP 模型約 1.7 GB，第一次使用時會下載到 Hugging Face cache。

### 22. 串接

設定 `language_conditioning=concat` 時，policy 把 embedding 當成多一個向量 observation：
`observation_features` 把它列在最後，§4 的 observation encoder 原樣串接它，和 robomimic 把
`lang_emb` 放在 low-dim observation 中的做法相同（原 repo `docs/tutorials/language_conditioning.md`
的「Feature input to action head」；`robomimic/models/obs_nets.py:282-284`、`:303-307`）。
`tests/test_language_conditioning.py` 會把 encoder 與 robomimic 的比對（robomimic 會把
`lang_emb` 和其他 key 一起排序，因此依 §4 的方式重新排列特徵），也會在相同權重下比對 BC-RNN
的 loss。

| 部分     | 行為                                                                 | 來源                                                                                           |
| -------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| 特徵順序 | 放在最後，接在 dataset 的 feature 之後                               | 本專案；robomimic 則把它和其他 key 一起排序（原 repo `robomimic/utils/file_utils.py:162-167`） |
| 序列     | 每個樣本一個 embedding，在 BC-RNN 與 BC-Transformer 的序列中逐步重複 | 原 repo `robomimic/utils/dataset.py:530-532`（在序列中重複）                                   |
| 其他輸入 | 除了 task 以外至少要有一個 observation                               | 本專案                                                                                         |

即使回傳的是一段 frame 序列，LeRobot 的 dataset 對每個樣本也只給一個 task 字串，因此
embedding 沒有時間維度，由 encoder 重複它。同一個示範中的 task 不會改變，所以結果和
robomimic 重複的 embedding 相同。

### 23. FiLM

設定 `language_conditioning=film` 時，每支相機的 ResNet-18 都由 embedding 調製，對應
robomimic 在 `VisualCoreLanguageConditioned` 中使用的 `ResNet18ConvFiLM`（原 repo
`robomimic/models/base_nets.py:657`、`robomimic/models/obs_core.py:189`）。`ImageEncoder` 在每個
residual block 之後加上一層 `FiLM`，並由 `lerobot_policy_robomimic/observation_encoder.py` 的
`CameraEncoder` 把 embedding 傳給它。`tests/test_language_conditioning.py` 會在相同權重下，把
相機 encoder、BC 的 loss 與 action、BC-RNN 的 loss 和 robomimic 比對。

| 部分           | 行為                                                                                    | 來源                                                                                                                 |
| -------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| FiLM 層        | 一層 Linear 從 embedding 算出每個 channel 的 β 與 γ，再計算 `(1 + γ) × x + β` 並接 ReLU | 原 repo `robomimic/models/base_nets.py:630`、`:649-654`                                                              |
| 位置           | 8 個 residual block 之後各一層；之前的 stem 不變                                        | 原 repo `robomimic/models/base_nets.py:699-713`、`:735-740`                                                          |
| Embedding 大小 | `language_embedding_dim`，預設 CLIP 模型為 768                                          | 原 repo `robomimic/models/base_nets.py:666`                                                                          |
| 特徵           | Embedding 用來調製相機，不放進串接的特徵                                                | 原 repo `robomimic/models/obs_nets.py:248-251`、`:282-284`                                                           |
| 裁切與 pooling | 沿用 image 實驗：隨機裁 76×76 與 spatial softmax                                        | 本專案；`VisualCoreLanguageConditioned` 預設以 spatial softmax pooling（原 repo `robomimic/models/obs_core.py:197`） |

robomimic 的教學示範的 FiLM 沒有 pooling 也沒有裁切
（`docs/tutorials/language_conditioning.md:88`、`:91`）。那只是一個範例 config，不是預設值，
因此本專案只換掉 backbone，§2 的其他相機設定都保留；測試也以相同方式建立 robomimic 的 FiLM
encoder。

`CameraEncoder` 取代了原本每支相機使用的 `nn.Sequential`，因為後者無法把第二個輸入傳給
encoder；它的參數名稱不變，因此先前的 checkpoint 仍可載入。

### 24. 執行紀錄

Hugging Face cache 中有 `openai/clip-vit-large-patch14` 時，它對兩個 task 的 embedding 和
robomimic 的 `get_lang_emb` 相同（`tests/test_language.py`）。在轉換後的 Lift image dataset 的
`train` mask 上，以 `language_conditioning=film` 執行 20 步 BC-RNN 的 `lerobot-train`，會訓練這個
4100 萬參數的 policy，並儲存包含 CLIP step 的 preprocessor。接著 `lerobot-eval` 以 `robomimic`
env 載入它，env 的 task 描述是 dataset 的 `lift the cube`，並執行兩個 30 步的 episode。這裡不
報告成功率：robomimic 沒有公布語言條件化的成功率，而論文的每個任務也只有一句指令。

## 未移植的設定

下列 robomimic 選項都沒有移植。論文的 BC 與 BC-RNN 實驗（原 repo
`robomimic/scripts/generate_paper_configs.py`）以及 BC-Transformer 的 template 都沒有用到它們，
因此不影響本專案要重現的結果。LeRobot 已經涵蓋的選項，例如 batch size、epoch、seed、裝置、
data worker 與 rollout 設定，屬於 `lerobot-train` 與 `lerobot-eval` 的參數，不列在這裡。

| 選項                                                                                            | robomimic 預設值與論文設定                       | 來源                                                                                                             | 說明位置                                       |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Depth 與 scan 觀測（`ScanCore`）                                                                | 無；論文只用 low-dim 與 RGB                      | 原 repo `robomimic/config/base_config.py:271-272`、`:313`；`robomimic/models/obs_core.py:240`                    | 本表                                           |
| 依 goal 條件化的觀測（`modalities.goal`、`train.goal_mode`）                                    | 無；論文把 goal 清單設為空                       | 原 repo `robomimic/config/base_config.py:230`、`:273-276`；`robomimic/scripts/generate_paper_configs.py:147-148` | 本表                                           |
| Low-dim 觀測的 encoder 與 randomizer（`encoder.low_dim`）                                       | 無                                               | 原 repo `robomimic/config/base_config.py:284-291`                                                                | 本表                                           |
| 其他相機 backbone：`ResNet50Conv`、`R3MConv`、`MVPConv`、`ShallowConv`                          | `ResNet18Conv`，也就是論文使用的                 | 原 repo `robomimic/models/base_nets.py:561`、`:749`、`:828`、`:993`                                              | 本表                                           |
| `SpatialMeanPool` 或不做 pooling                                                                | 論文用 `SpatialSoftmax`                          | 原 repo `robomimic/scripts/generate_paper_configs.py:156`；`robomimic/models/base_nets.py:1248`                  | 本表                                           |
| `ColorRandomizer` 與 `GaussianNoiseRandomizer`                                                  | 論文只用 `CropRandomizer`                        | 原 repo `robomimic/scripts/generate_paper_configs.py:163`；`robomimic/models/obs_core.py:629`、`:786`            | 本表                                           |
| ImageNet 權重、coordinate convolution、裁切位置編碼、多個裁切、可學習或加噪聲的 spatial softmax | 論文都關閉                                       | §3                                                                                                               | §3                                             |
| BC 與 BC-RNN 的多步觀測（`train.frame_stack > 1`）                                              | 1，論文維持不變                                  | 原 repo `robomimic/config/base_config.py:195`                                                                    | 本表；`n_obs_steps` 不是 1 時丟出 `ValueError` |
| Gaussian 與 GMM head 用 `exp` 輸出標準差                                                        | `softplus`                                       | §5                                                                                                               | §5                                             |
| GRU、雙向 LSTM 與 open-loop 的 BC-RNN                                                           | 單向 LSTM、closed loop                           | §8                                                                                                               | §8                                             |
| BC-Transformer 監督所有步、預測未來 action、其他位置編碼、GEGLU                                 | template 中都關閉                                | §13                                                                                                              | §13                                            |
| BC-VAE 的 categorical prior 與 `freeze_encoder`                                                 | 關閉；只有 BCQ 會設定 `freeze_encoder`           | §17、§19                                                                                                         | §17、§19                                       |
| BC、BC-RNN 與 BC-VAE 的 learning rate schedule（`multistep`、`linear`、`cosine`）               | 沒有衰減的 epoch，learning rate 固定，和論文相同 | 原 repo `robomimic/config/bc_config.py:28-31`；`robomimic/utils/torch_utils.py:141-180`                          | 本表                                           |
| 選擇 optimizer（`optimizer_type`）                                                              | Adam；BC-Transformer 的 template 用 AdamW        | 原 repo `robomimic/config/bc_config.py:27`；`robomimic/utils/torch_utils.py:106-115`                             | §2、§13                                        |
| Action 正規化與旋轉格式轉換（`train.action_config`）                                            | 不正規化                                         | 原 repo `robomimic/config/base_config.py:223-227`                                                                | 本表                                           |
| Observation 正規化（`train.hdf5_normalize_obs`）                                                | 關閉                                             | 原 repo `robomimic/config/base_config.py:181`                                                                    | §2（正規化）                                   |
| 多個 dataset 依大小加權（`train.normalize_weights_by_ds_size`）                                 | 關閉；論文只用一個 dataset 訓練                  | 原 repo `robomimic/config/base_config.py:161`                                                                    | 本表                                           |

每個 policy 的 optimizer 是固定的，不能用設定切換：BC、BC-RNN 與 BC-VAE 用 Adam，BC-Transformer
用 AdamW，這也是 robomimic 的 config 與 template 對它們用過的唯一 optimizer。
