# lerobot-policy-robomimic

[English](README.md) | 繁體中文

本專案是 [LeRobot](https://github.com/huggingface/lerobot) 的 out-of-tree
policy 插件，將 [robomimic](https://robomimic.github.io/)（Mandlekar 等人，2021）
中的 imitation learning policy（例如 BC 與 BC-RNN）移植到 LeRobot v0.6.1，並註冊
為 LeRobot 的 policy type。移植後的 policy 可以用 `lerobot-train` 訓練、以
`lerobot-eval` 評估，網路架構與訓練設定則盡可能貼近原始實作。

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

| 決策                     | 選擇                                                                                        | 來源                                                                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Policy type              | `robomimic_bc`                                                                              | 本專案：加上 `robomimic_` 前綴，避免和 LeRobot 內建的 type 衝突，例如 `diffusion`（`policies/diffusion/`），而 robomimic 也有實作它（原 repo `robomimic/algo/diffusion_policy.py`） |
| Class 與函式名稱         | `RobomimicBCConfig`、`RobomimicBCPolicy`、`make_robomimic_bc_pre_post_processors`           | LeRobot：policy class 名稱是把 config class 名稱的 `Config` 換成 `Policy`（`policies/factory.py:409-415`），processor factory 名稱是 `make_<type>_pre_post_processors`（`policies/factory.py:458`） |
| 模組名稱                 | `configuration_robomimic_bc.py`、`modeling_robomimic_bc.py`、`processor_robomimic_bc.py`    | LeRobot：modeling 與 processor 模組是把 config 模組路徑中的 `configuration_` 替換後找到的（`policies/factory.py:416`、`:459`）；命名沿用指南的 template           |
| Distribution 名稱        | `lerobot_policy_robomimic`                                                                  | LeRobot：名稱以 `lerobot_policy_` 開頭的已安裝 distribution 會以該名稱被 import（`utils/import_utils.py:231-255`），進而執行 `@PreTrainedConfig.register_subclass` |
| 一個套件、多個 policy    | 每個 robomimic 演算法各自有 policy type 與三個模組                                          | 本專案：指南示範一個套件一個 policy，但 factory 只需要每個 type 各有這三個模組，因此 robomimic 的演算法可以共用一個套件與其網路                                     |

### 2. Configuration

`RobomimicBCConfig` 註冊 policy type `robomimic_bc`。預設值沿用 robomimic 在
proficient-human（PH）資料集上的 image 實驗設定。

| 設定                   | 預設值                                             | Config 欄位                                      | 來源                                                                                                                                                         |
| ---------------------- | -------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Observation            | 本體感知與相機；不含物體狀態                       | `use_env_state`（False）                         | 原 repo `robomimic/scripts/generate_paper_configs.py:138-146`（image）；low-dim 實驗另加 `object`（`:67-72`）                                                 |
| 觀測歷史               | 無（只用當下這一步）                               | `n_obs_steps`、`observation_delta_indices`       | 原 repo `robomimic/config/base_config.py:193`、`:195`；`robomimic/algo/bc.py:111` 只讀第 0 步                                                                 |
| Action 預測            | 每個 observation 預測一個 action                   | `action_delta_indices`                           | 原 repo `robomimic/algo/bc.py:113`                                                                                                                            |
| MLP 隱藏層大小         | (1024, 1024)                                       | `actor_layer_dims`                               | 原 repo `robomimic/config/bc_config.py:41`、`robomimic/scripts/generate_paper_configs.py:367`                                                                  |
| 相機裁切               | 訓練時隨機裁 76×76，推論時中央裁 76×76             | `crop_shape`                                     | 原 repo `robomimic/scripts/generate_paper_configs.py:163-168`                                                                                                 |
| Spatial softmax keypoint | 32                                               | `spatial_softmax_num_kp`                         | 原 repo `robomimic/scripts/generate_paper_configs.py:157`                                                                                                     |
| 每支相機的特徵大小     | 64                                                 | `image_feature_dim`                              | 原 repo `robomimic/scripts/generate_paper_configs.py:152`                                                                                                     |
| Loss                   | 只用均方誤差                                       | `l2_weight`（1）、`l1_weight`（0）、`cos_weight`（0） | 原 repo `robomimic/config/bc_config.py:36-38`                                                                                                           |
| 正規化                 | 無；影像只縮放到 `[0, 1]`                          | `normalization_mapping`（`IDENTITY`）            | 原 repo `robomimic/config/base_config.py:181`（observation）、`:225`（action）、`robomimic/utils/obs_utils.py:921`（影像）；LeRobot 已縮放影像（`datasets/io_utils.py:255-263`） |
| Optimizer              | Adam                                               | `get_optimizer_preset()`                         | 原 repo `robomimic/config/bc_config.py:27`                                                                                                                    |
| Learning rate          | 1e-4，固定不變                                     | `optimizer_lr`                                   | 原 repo `robomimic/config/bc_config.py:28`、`:30`（沒有衰減的 epoch）                                                                                         |
| Weight decay           | 0                                                  | `optimizer_weight_decay`                         | 原 repo `robomimic/config/bc_config.py:33`                                                                                                                    |
| Gradient clipping      | 無                                                 | `optimizer_grad_clip_norm`（0）                  | 原 repo `robomimic/config/base_config.py:239`；LeRobot 在 0 時不做 clipping（`scripts/lerobot_train.py:178`）                                                 |

LeRobot 會把 dataset 裡所有非 action 的 feature 都交給 policy
（`policies/factory.py:305-306`），而 robomimic 是由每個實驗列出要用的
observation。`use_env_state` 讓這個選擇留在 policy 裡，因此同一份 dataset 可以同時
支援兩種實驗。

Batch size 與訓練長度屬於 `lerobot-train`，不屬於 policy：robomimic 在 image 資料上
用 batch size 16，訓練 600 個 epoch、每個 epoch 500 步（共 30 萬步）
（`robomimic/scripts/generate_paper_configs.py:113`、`:131-132`）；在 low-dim 資料上
用 batch size 100，訓練 2000 個 epoch、每個 epoch 100 步（共 20 萬步）
（`:43`、`:61-62`）。

robomimic 論文在人類示範資料集上跑 BC 時使用 GMM action head
（`robomimic/scripts/generate_paper_configs.py:368`）。本 policy 先從 deterministic
head 開始，也就是 robomimic 的 `BC` class。

### 3. 相機 encoder

`lerobot_policy_robomimic/vision.py` 是 robomimic image 實驗所用的相機 encoder，對應
robomimic 的 `VisualCore`（原 repo `robomimic/scripts/generate_paper_configs.py:151-160`）。
測試會用相同的權重和輸入，把每個部分和 robomimic 自己的模組比對
（`tests/test_vision.py`）。

| 部分                 | 行為                                                                     | 本專案           | 來源                                                                                                                                         |
| -------------------- | ------------------------------------------------------------------------ | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 裁切                 | 訓練時隨機裁切，其他時候中央裁切                                         | `RandomCrop`     | 原 repo `robomimic/models/obs_core.py:489`（`CropRandomizer`）、`:579`（評估時中央裁切）                                                     |
| 隨機裁切的位移       | `floor(rand × (size − crop))`，84 裁 76 時為 0 到 7；最後一個位移永遠抽不到 | `RandomCrop`   | 原 repo `robomimic/utils/obs_utils.py:730`                                                                                                   |
| 中央裁切的位移       | `floor((size − crop) / 2)`，84 裁 76 時為 4                              | `RandomCrop`     | 原 repo `robomimic/utils/obs_utils.py:278`                                                                                                   |
| Backbone             | 去掉 average pool 與分類層的 ResNet-18，從頭訓練                         | `ImageEncoder`   | 原 repo `robomimic/models/base_nets.py:536`（`ResNet18Conv`）；`robomimic/scripts/generate_paper_configs.py:154`（`pretrained = False`）     |
| Spatial softmax      | 1×1 卷積產生 32 個 keypoint，在位置上做 softmax，取 `[-1, 1]` 網格上的期望 `(x, y)` | `SpatialSoftmax` | 原 repo `robomimic/models/base_nets.py:1143`、`:1162`、`:1216`；temperature 固定為 1 且不加噪聲（`robomimic/scripts/generate_paper_configs.py:158-160`） |
| 投影                 | Linear 層，把 32 × 2 個 keypoint 座標投影成 64 維特徵                    | `ImageEncoder`   | 原 repo `robomimic/models/obs_core.py:138`                                                                                                   |

robomimic 會平均多個裁切的特徵（`robomimic/models/obs_core.py:589-598`）；論文實驗每張
影像只裁一次（`robomimic/scripts/generate_paper_configs.py:168`），平均後就是該特徵本身，
因此 `RandomCrop` 只取一個裁切。

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

| 部分                 | 行為                                                     | 來源                                                                                                                                                                                    |
| -------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 向量 observation     | 原樣串接                                                 | 原 repo `robomimic/config/base_config.py:284`（low-dim observation 沒有 encoder）、`robomimic/models/obs_nets.py:303-307`                                                               |
| 每支相機             | 各自的裁切、`ImageEncoder` 與 ReLU                       | 原 repo `robomimic/models/obs_nets.py:287-289`（每個 key 一個 randomizer）、`:295`（encoder）、`:297`（activation）；ReLU 是預設的 `feature_activation`（`:440`、`:33`），`MIMO_MLP` 沿用預設值（`:608-611`） |
| 特徵順序             | 依 policy input feature 的順序                           | 本專案；robomimic 則是把 observation key 排序（原 repo `robomimic/utils/file_utils.py:162`，經 `robomimic/scripts/train.py:242` 與 `robomimic/algo/algo.py:154-156` 傳入）              |

robomimic 排序後，相機與向量 observation 會交錯排列；以 image 實驗為例，順序是
`agentview_image`、`robot0_eef_pos`、`robot0_eef_quat`、`robot0_eye_in_hand_image`、
`robot0_gripper_qpos`。這個順序只在載入 robomimic 權重時有影響，它決定 MLP 第一層的
輸入欄位。沿用 LeRobot 的順序，encoder 就不需要知道 robomimic 的 key 名稱；測試則在
比對前先重新排列 robomimic 的輸出。
