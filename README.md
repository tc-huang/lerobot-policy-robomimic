# lerobot-policy-robomimic

English | [繁體中文](README.zh-TW.md)

An out-of-tree [LeRobot](https://github.com/huggingface/lerobot) policy plugin
that ports the imitation learning policies of
[robomimic](https://robomimic.github.io/) (Mandlekar et al., 2021), such as BC
and BC-RNN, to LeRobot v0.6.1 and registers them as LeRobot policy types. The
ported policies can be trained with `lerobot-train` and evaluated with
`lerobot-eval`, while following the network architectures and training
recipes of the original implementation as closely as possible.
