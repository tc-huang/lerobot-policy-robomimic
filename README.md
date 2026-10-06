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
