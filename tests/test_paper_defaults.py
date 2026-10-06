import robomimic_reference

from lerobot_policy_robomimic import RobomimicBCConfig, RobomimicBCRNNConfig


def assert_shared_defaults_match(ours, paper):
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


def test_bc_defaults_match_robomimic_image_experiment():
    assert_shared_defaults_match(RobomimicBCConfig(device="cpu"), robomimic_reference.bc_config())


def test_bc_rnn_defaults_match_robomimic_image_experiment():
    ours = RobomimicBCRNNConfig(device="cpu")
    paper = robomimic_reference.bc_rnn_config()

    assert_shared_defaults_match(ours, paper)
    assert paper.algo.rnn.enabled and paper.algo.rnn.rnn_type == "LSTM"
    assert not paper.algo.rnn.kwargs.bidirectional and not paper.algo.rnn.open_loop
    assert ours.rnn_hidden_dim == paper.algo.rnn.hidden_dim
    assert ours.rnn_num_layers == paper.algo.rnn.num_layers
    assert ours.rnn_horizon == paper.algo.rnn.horizon == paper.train.seq_length


def test_bc_rnn_low_dim_experiment_uses_a_smaller_lstm():
    assert robomimic_reference.bc_rnn_config("low_dim").algo.rnn.hidden_dim == 400
