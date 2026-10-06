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
