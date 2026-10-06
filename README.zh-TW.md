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

## 設計

本專案從零開始重建每個 robomimic policy，一次建構一個元件。以下每一節說明一個元件，
以及它的行為從何而來。

來源標示方式如下：

- **論文**：Mandlekar 等人，
  [What Matters in Learning from Offline Human Demonstrations for Robot Manipulation](https://arxiv.org/abs/2108.03298)，
  CoRL 2021。
- **原 repo**：[ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic)
  的 `d309eae`，以 submodule 放在 `third_party/robomimic`。路徑皆相對於該目錄。
- **LeRobot**：LeRobot v0.6.1 的
  [Adding a Policy](https://huggingface.co/docs/lerobot/v0.6.1/en/bring_your_own_policies)
  指南或原始碼。路徑皆相對於安裝後的 `lerobot` 套件。
- **本專案**：本專案自行決定，並非取自上述任一來源。

robomimic 以 MIT License 釋出。本專案重新實作其網路，執行時不 import robomimic。

### 1. Policy type 與套件結構

| 決策                  | 選擇                                                                                                                                       | 來源                                                                                                                                                                                                |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Policy type           | `robomimic_bc`（BC）、`robomimic_bc_rnn`（BC-RNN）                                                                                         | 本專案：加上 `robomimic_` 前綴，避免和 LeRobot 內建的 type 衝突，例如 `diffusion`（`policies/diffusion/`），而 robomimic 也有實作它（原 repo `robomimic/algo/diffusion_policy.py`）                 |
| Class 與函式名稱      | `RobomimicBCConfig`、`RobomimicBCPolicy`、`make_robomimic_bc_pre_post_processors`；`RobomimicBCRNN…` 與 `make_robomimic_bc_rnn_…` 依此類推 | LeRobot：policy class 名稱是把 config class 名稱的 `Config` 換成 `Policy`（`policies/factory.py:409-415`），processor factory 名稱是 `make_<type>_pre_post_processors`（`policies/factory.py:458`） |
| 模組名稱              | `configuration_<type>.py`、`modeling_<type>.py`、`processor_<type>.py`                                                                     | LeRobot：modeling 與 processor 模組是把 config 模組路徑中的 `configuration_` 替換後找到的（`policies/factory.py:416`、`:459`）；命名沿用指南的 template                                             |
| Distribution 名稱     | `lerobot_policy_robomimic`                                                                                                                 | LeRobot：名稱以 `lerobot_policy_` 開頭的已安裝 distribution 會以該名稱被 import（`utils/import_utils.py:231-255`），進而執行 `@PreTrainedConfig.register_subclass`                                  |
| 一個套件、多個 policy | 每個 robomimic 演算法各自有 policy type 與三個模組                                                                                         | 本專案：指南示範一個套件一個 policy，但 factory 只需要每個 type 各有這三個模組，因此 robomimic 的演算法可以共用一個套件與其網路                                                                     |

## BC（`robomimic_bc`）

### 2. Configuration

`RobomimicBCConfig` 註冊 policy type `robomimic_bc`。預設值沿用 robomimic 在
proficient-human（PH）資料集上的 image 實驗設定。下表中除了 `actor_layer_dims` 以外的設定，
都來自 `lerobot_policy_robomimic/base_config.py` 的 `RobomimicPolicyConfig`，由本專案所有
robomimic policy 共用；它本身不是 policy type。

| 設定                     | 預設值                                 | Config 欄位                                           | 來源                                                                                                                                                                             |
| ------------------------ | -------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Observation              | 本體感知與相機；不含物體狀態           | `use_env_state`（False）                              | 原 repo `robomimic/scripts/generate_paper_configs.py:138-146`（image）；low-dim 實驗另加 `object`（`:67-72`）                                                                    |
| 觀測歷史                 | 無（只用當下這一步）                   | `n_obs_steps`、`observation_delta_indices`            | 原 repo `robomimic/config/base_config.py:193`、`:195`；`robomimic/algo/bc.py:111` 只讀第 0 步                                                                                    |
| Action 預測              | 每個 observation 預測一個 action       | `action_delta_indices`                                | 原 repo `robomimic/algo/bc.py:113`                                                                                                                                               |
| MLP 隱藏層大小           | (1024, 1024)                           | `actor_layer_dims`                                    | 原 repo `robomimic/config/bc_config.py:41`、`robomimic/scripts/generate_paper_configs.py:367`                                                                                    |
| 相機裁切                 | 訓練時隨機裁 76×76，推論時中央裁 76×76 | `crop_shape`                                          | 原 repo `robomimic/scripts/generate_paper_configs.py:163-168`                                                                                                                    |
| Spatial softmax keypoint | 32                                     | `spatial_softmax_num_kp`                              | 原 repo `robomimic/scripts/generate_paper_configs.py:157`                                                                                                                        |
| 每支相機的特徵大小       | 64                                     | `image_feature_dim`                                   | 原 repo `robomimic/scripts/generate_paper_configs.py:152`                                                                                                                        |
| Action head              | 5 個高斯分佈的混合                     | `use_gmm`（True）、`gmm_num_modes`（5）               | 原 repo `robomimic/scripts/generate_paper_configs.py:368`（人類示範資料集）、`robomimic/config/bc_config.py:53`                                                                  |
| GMM 最小標準差           | 1e-4                                   | `gmm_min_std`                                         | 原 repo `robomimic/config/bc_config.py:54`                                                                                                                                       |
| GMM 在訓練以外的雜訊     | 所有標準差設為 1e-4                    | `gmm_low_noise_eval`（True）                          | 原 repo `robomimic/config/bc_config.py:56`                                                                                                                                       |
| 不用 GMM 時的 loss       | 只用均方誤差                           | `l2_weight`（1）、`l1_weight`（0）、`cos_weight`（0） | 原 repo `robomimic/config/bc_config.py:36-38`                                                                                                                                    |
| 正規化                   | 無；影像只縮放到 `[0, 1]`              | `normalization_mapping`（`IDENTITY`）                 | 原 repo `robomimic/config/base_config.py:181`（observation）、`:225`（action）、`robomimic/utils/obs_utils.py:921`（影像）；LeRobot 已縮放影像（`datasets/io_utils.py:255-263`） |
| Optimizer                | Adam                                   | `get_optimizer_preset()`                              | 原 repo `robomimic/config/bc_config.py:27`                                                                                                                                       |
| Learning rate            | 1e-4，固定不變                         | `optimizer_lr`                                        | 原 repo `robomimic/config/bc_config.py:28`、`:30`（沒有衰減的 epoch）                                                                                                            |
| Weight decay             | 0                                      | `optimizer_weight_decay`                              | 原 repo `robomimic/config/bc_config.py:33`                                                                                                                                       |
| Gradient clipping        | 無                                     | `optimizer_grad_clip_norm`（0）                       | 原 repo `robomimic/config/base_config.py:239`；LeRobot 在 0 時不做 clipping（`scripts/lerobot_train.py:178`）                                                                    |

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
對應 robomimic 的 `ActorNetwork`（原 repo `robomimic/models/policy_nets.py:26`）或
`GMMActorNetwork`（`:397`），BC 用 `actor_layer_dims` 建立它們（`robomimic/algo/bc.py:87-92`）。

| 部分                | 行為                                                                                | 來源                                                                                                                                      |
| ------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| MLP                 | `actor_layer_dims` 的每個大小各一層 Linear 加 ReLU                                  | 原 repo `robomimic/models/obs_nets.py:617-623`；最後一個大小是 MLP 的輸出，同樣接 ReLU（`:623`、`robomimic/models/base_nets.py:256-258`） |
| `DeterministicHead` | Linear 層輸出 action 維度，再接 `tanh`                                              | 原 repo `robomimic/models/obs_nets.py:627`、`:392`（`ObservationDecoder`）、`robomimic/models/policy_nets.py:107`                         |
| 它的 loss           | `l2_weight` × 均方誤差 + `l1_weight` × smooth L1 + `cos_weight` × cosine loss       | 原 repo `robomimic/algo/bc.py:182-192`                                                                                                    |
| 它的 cosine loss    | 前三個 action 維度（末端執行器的位移）上 1 − cosine similarity 的平均               | 原 repo `robomimic/algo/bc.py:185`、`robomimic/utils/loss_utils.py:22-23`                                                                 |
| `GMMHead` 的輸出    | Linear 層分別輸出每個 mode 的高斯平均值、尺度，以及 mode 的 logits                  | 原 repo `robomimic/models/policy_nets.py:489-491`                                                                                         |
| 它的平均值          | 輸出取 `tanh`                                                                       | 原 repo `robomimic/models/policy_nets.py:514`                                                                                             |
| 它的尺度            | 輸出取 `softplus` 再加 `gmm_min_std`；開啟 `gmm_low_noise_eval` 時，訓練以外為 1e-4 | 原 repo `robomimic/models/policy_nets.py:519`、`:522`                                                                                     |
| 它的分佈            | 先以 categorical 選 mode，再在各 action 維度上用對角高斯分佈                        | 原 repo `robomimic/models/policy_nets.py:526-535`                                                                                         |
| 它的 loss           | 示範 action 的平均對數似然取負                                                      | 原 repo `robomimic/algo/bc.py:300`、`:322`（`BC_Gaussian`，`BC_GMM` 在 `:347` 繼承它）                                                    |
| 它的 action         | 從分佈中抽樣                                                                        | 原 repo `robomimic/models/policy_nets.py:555`                                                                                             |

robomimic 把 MLP 拆成隱藏層與輸出層（`robomimic/models/obs_nets.py:619-620`），但兩者
最後都接 ReLU，結果就是每個大小各一層 Linear 加 ReLU，`MLP` 也就這樣建立。

即使在推論時，`GMMHead` 也是依混合權重抽樣出 mode，而不是選機率最高的 mode；低雜訊只會
把高斯分佈縮小到該 mode 的平均值附近，和 robomimic 相同。

每個 action head 各自帶有 loss 與 action 的選擇方式（本專案）。robomimic 則是依 config
旗標，為每種網路搭配一個演算法子類別（`robomimic/algo/bc.py:46-73`）。把 head 和 MLP
分開，之後的 recurrent 與 transformer policy 就能重用這兩種 head。

### 6. Policy

`RobomimicBCPolicy` 在 LeRobot 訓練與評估流程會呼叫的 method 中，依序串接
observation encoder、MLP 與 action head，行為依照 robomimic 的 `BC` 與 `BC_GMM` class
（原 repo `robomimic/algo/bc.py:78`、`:347`）。測試會把相同權重載入 robomimic 的演算法，
依 §4 的方式重新排列 MLP 第一層的輸入欄位，並比對兩種 head 的 loss 與 action
（`tests/test_policy.py`）。

| Method                 | 行為                                                     | 來源                                                                                                                                                                                                                  |
| ---------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `forward`              | action head 的 loss 與它記錄的數值                       | 原 repo `robomimic/algo/bc.py:182-192`（`BC`）、`:300`、`:322`（`BC_GMM`）                                                                                                                                            |
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

### 11. Processor

`make_robomimic_bc_rnn_pre_post_processors` 回傳和 BC 相同的預設 pipeline（§7）。推論時
它們一次處理一步，訓練時則處理整段序列，正規化會同樣套用到每一步；在 `IDENTITY` 對應下
數值不會改變（`tests/test_processor.py`）。

在轉換好的 Lift image dataset 上，用 `train` mask 跑 20 步的 `lerobot-train`，可以訓練這個
35M 參數的 policy、存下 checkpoint，並連同 processor 讀回後執行 12 步 `select_action`，
期間跨過一次狀態重設。
