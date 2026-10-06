import pytest
import robomimic_reference
import torch

from lerobot_policy_robomimic.transformer import Transformer

INPUT_DIM = 9 + 2 * 64


def make_transformer() -> Transformer:
    return Transformer(
        input_dim=INPUT_DIM,
        context_length=10,
        embed_dim=512,
        num_layers=6,
        num_heads=8,
        emb_dropout=0.1,
        attn_dropout=0.1,
        block_output_dropout=0.1,
    )


def test_matches_robomimic_mimo_transformer():
    torch.manual_seed(0)
    reference = (
        robomimic_reference.bc_transformer_algo(
            robomimic_reference.image_experiment_obs_shapes(), action_dim=7, gmm=True
        )
        .nets["policy"]
        .eval()
    )
    ours = make_transformer().eval()
    robomimic_reference.load_transformer(ours, reference)
    features = torch.randn(4, 10, INPUT_DIM)

    expected = reference.nets["transformer"](reference.input_embedding(features))

    torch.testing.assert_close(ours(features), expected)


def test_steps_only_attend_to_the_past():
    torch.manual_seed(0)
    transformer = make_transformer().eval()
    features = torch.randn(1, 10, INPUT_DIM)
    changed = features.clone()
    changed[:, 6:] = torch.randn(1, 4, INPUT_DIM)

    torch.testing.assert_close(transformer(changed)[:, :6], transformer(features)[:, :6])
    assert not torch.allclose(transformer(changed)[:, 6:], transformer(features)[:, 6:])


def test_gpt_blocks_start_from_gpt2_initialization():
    torch.manual_seed(0)
    transformer = make_transformer()
    linears = [module for module in transformer.blocks.modules() if isinstance(module, torch.nn.Linear)]
    weights = torch.cat([linear.weight.flatten() for linear in linears])

    assert weights.std().item() == pytest.approx(0.02, rel=0.05)
    assert torch.count_nonzero(transformer.position_embedding) == 0
