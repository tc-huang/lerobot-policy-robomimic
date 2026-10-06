import robomimic_reference

from lerobot_policy_robomimic import (
    RobomimicBCConfig,
    RobomimicBCRNNConfig,
    RobomimicBCTransformerConfig,
    RobomimicBCVAEConfig,
)


def assert_policy_defaults_match(ours, paper):
    rgb = paper.observation.encoder.rgb
    assert ours.optimizer_lr == paper.algo.optim_params.policy.learning_rate.initial
    assert ours.optimizer_weight_decay == paper.algo.optim_params.policy.regularization.L2
    assert paper.train.max_grad_norm is None and ours.optimizer_grad_clip_norm == 0
    assert ours.crop_shape == (rgb.obs_randomizer_kwargs.crop_height, rgb.obs_randomizer_kwargs.crop_width)
    assert ours.spatial_softmax_num_kp == rgb.core_kwargs.pool_kwargs.num_kp
    assert ours.image_feature_dim == rgb.core_kwargs.feature_dimension


def assert_shared_defaults_match(ours, paper):
    assert_policy_defaults_match(ours, paper)
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


def test_bc_defaults_match_robomimic_image_experiment():
    ours = RobomimicBCConfig(device="cpu")
    paper = robomimic_reference.bc_config()

    assert_shared_defaults_match(ours, paper)
    assert ours.actor_layer_dims == tuple(paper.algo.actor_layer_dims)
    assert list(paper.algo.optim_params.policy.learning_rate.epoch_schedule) == []


def test_bc_gaussian_defaults_match_robomimic():
    ours = RobomimicBCConfig(device="cpu")
    gaussian = robomimic_reference.bc_config().algo.gaussian

    assert ours.use_gaussian == gaussian.enabled
    assert ours.gaussian_fixed_std == gaussian.fixed_std
    assert ours.gaussian_init_std == gaussian.init_std
    assert ours.gaussian_min_std == gaussian.min_std
    assert ours.gaussian_low_noise_eval == gaussian.low_noise_eval
    assert gaussian.std_activation == "softplus"


def test_bc_rnn_defaults_match_robomimic_image_experiment():
    ours = RobomimicBCRNNConfig(device="cpu")
    paper = robomimic_reference.bc_rnn_config()

    assert_shared_defaults_match(ours, paper)
    assert ours.actor_layer_dims == tuple(paper.algo.actor_layer_dims)
    assert list(paper.algo.optim_params.policy.learning_rate.epoch_schedule) == []
    assert paper.algo.rnn.enabled and paper.algo.rnn.rnn_type == "LSTM"
    assert not paper.algo.rnn.kwargs.bidirectional and not paper.algo.rnn.open_loop
    assert ours.rnn_hidden_dim == paper.algo.rnn.hidden_dim
    assert ours.rnn_num_layers == paper.algo.rnn.num_layers
    assert ours.rnn_horizon == paper.algo.rnn.horizon == paper.train.seq_length


def test_bc_rnn_low_dim_experiment_uses_a_smaller_lstm():
    assert robomimic_reference.bc_rnn_config("low_dim").algo.rnn.hidden_dim == 400


def test_bc_transformer_defaults_match_robomimic_template():
    ours = RobomimicBCTransformerConfig(device="cpu")
    template = robomimic_reference.bc_transformer_template_config()
    transformer = template.algo.transformer
    learning_rate = template.algo.optim_params.policy.learning_rate

    assert_shared_defaults_match(ours, template)
    assert transformer.enabled and not template.algo.rnn.enabled
    assert ours.transformer_context_length == transformer.context_length == template.train.frame_stack
    assert template.train.seq_length == 1
    assert ours.transformer_embed_dim == transformer.embed_dim
    assert ours.transformer_num_layers == transformer.num_layers
    assert ours.transformer_num_heads == transformer.num_heads
    assert ours.transformer_emb_dropout == transformer.emb_dropout
    assert ours.transformer_attn_dropout == transformer.attn_dropout
    assert ours.transformer_block_output_dropout == transformer.block_output_dropout
    assert transformer.activation == "gelu" and not transformer.sinusoidal_embedding
    assert transformer.nn_parameter_for_timesteps
    assert not transformer.supervise_all_steps and not transformer.pred_future_acs
    assert template.algo.optim_params.policy.optimizer_type == "adamw"
    assert learning_rate.scheduler_type == "linear"
    assert [ours.scheduler_decay_epochs] == list(learning_rate.epoch_schedule)
    assert ours.scheduler_decay_factor == learning_rate.decay_factor
    assert ours.scheduler_steps_per_epoch == template.experiment.epoch_every_n_steps


def test_bc_vae_defaults_match_robomimic():
    ours = RobomimicBCVAEConfig(device="cpu")
    reference = robomimic_reference.bc_vae_config()
    vae = reference.algo.vae

    assert_policy_defaults_match(ours, reference)
    assert list(reference.algo.optim_params.policy.learning_rate.epoch_schedule) == []
    assert ours.vae_latent_dim == vae.latent_dim
    assert ours.vae_latent_clip == vae.latent_clip
    assert ours.vae_kl_weight == vae.kl_weight
    assert ours.vae_encoder_layer_dims == tuple(vae.encoder_layer_dims)
    assert ours.vae_decoder_layer_dims == tuple(vae.decoder_layer_dims)
    assert ours.vae_decoder_is_conditioned == vae.decoder.is_conditioned
    assert ours.vae_reconstruction_sum_across_elements == vae.decoder.reconstruction_sum_across_elements
    assert ours.vae_prior_learn == vae.prior.learn
    assert ours.vae_prior_is_conditioned == vae.prior.is_conditioned
    assert ours.vae_prior_layer_dims == tuple(vae.prior_layer_dims)
    assert ours.vae_prior_use_gmm == vae.prior.use_gmm
    assert ours.vae_prior_gmm_num_modes == vae.prior.gmm_num_modes
    assert ours.vae_prior_gmm_learn_weights == vae.prior.gmm_learn_weights
    assert not vae.prior.use_categorical
