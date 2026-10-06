import robomimic_reference

from lerobot_policy_robomimic import RobomimicBCConfig


def test_bc_defaults_match_robomimic_image_experiment():
    ours = RobomimicBCConfig(device="cpu")
    paper = robomimic_reference.bc_config()
    rgb = paper.observation.encoder.rgb

    assert ours.actor_layer_dims == tuple(paper.algo.actor_layer_dims)
    assert ours.optimizer_lr == paper.algo.optim_params.policy.learning_rate.initial
    assert list(paper.algo.optim_params.policy.learning_rate.epoch_schedule) == []
    assert ours.optimizer_weight_decay == paper.algo.optim_params.policy.regularization.L2
    assert paper.train.max_grad_norm is None and ours.optimizer_grad_clip_norm == 0
    assert ours.use_gmm == paper.algo.gmm.enabled
    assert ours.gmm_num_modes == paper.algo.gmm.num_modes
    assert ours.gmm_min_std == paper.algo.gmm.min_std
    assert ours.gmm_low_noise_eval == paper.algo.gmm.low_noise_eval
    assert paper.algo.gmm.std_activation == "softplus"
    assert (ours.l2_weight, ours.l1_weight, ours.cos_weight) == (
        paper.algo.loss.l2_weight,
        paper.algo.loss.l1_weight,
        paper.algo.loss.cos_weight,
    )
    assert ours.crop_shape == (rgb.obs_randomizer_kwargs.crop_height, rgb.obs_randomizer_kwargs.crop_width)
    assert ours.spatial_softmax_num_kp == rgb.core_kwargs.pool_kwargs.num_kp
    assert ours.image_feature_dim == rgb.core_kwargs.feature_dimension
