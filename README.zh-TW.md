# lerobot-policy-robomimic

[English](README.md) | 繁體中文

本專案是 [LeRobot](https://github.com/huggingface/lerobot) 的 out-of-tree
policy 插件，將 [robomimic](https://robomimic.github.io/)（Mandlekar 等人，2021）
中的 imitation learning policy（例如 BC 與 BC-RNN）移植到 LeRobot v0.6.1，並註冊
為 LeRobot 的 policy type。移植後的 policy 可以用 `lerobot-train` 訓練、以
`lerobot-eval` 評估，網路架構與訓練設定則盡可能貼近原始實作。
