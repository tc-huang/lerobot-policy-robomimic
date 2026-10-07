"""The causal transformer of robomimic's BC-Transformer."""

import torch
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention in which each step attends to itself and earlier steps."""

    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float, output_dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.output = nn.Linear(embed_dim, embed_dim)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Maps (B, T, embed_dim) inputs to (B, T, embed_dim) outputs."""
        q, k, v = (
            t.unflatten(-1, (self.num_heads, -1)).transpose(1, 2) for t in self.qkv(x).chunk(3, dim=-1)
        )
        y = scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0, is_causal=True
        )
        return self.output_dropout(self.output(y.transpose(1, 2).flatten(start_dim=2)))


class TransformerBlock(nn.Module):
    """A pre-norm GPT block: causal self-attention, then a GELU MLP, each with a residual."""

    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float, output_dropout: float):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.attention = CausalSelfAttention(embed_dim, num_heads, attn_dropout, output_dropout)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(output_dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class Transformer(nn.Module):
    """Embeds a sequence of features with learned step positions and runs GPT blocks over it.

    The GPT blocks and the output norm start from GPT-2's initialization; the input projection
    keeps PyTorch's default and the step positions start at zero, as in robomimic.
    """

    def __init__(
        self,
        input_dim: int,
        context_length: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        emb_dropout: float,
        attn_dropout: float,
        block_output_dropout: float,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, context_length, embed_dim))
        self.input_norm = nn.LayerNorm(embed_dim)
        self.input_dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.Sequential(
            *(
                TransformerBlock(embed_dim, num_heads, attn_dropout, block_output_dropout)
                for _ in range(num_layers)
            )
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.blocks.apply(self._init_weights)
        self.output_norm.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, features: Tensor) -> Tensor:
        """Maps (B, context_length, input_dim) features to (B, context_length, embed_dim) outputs."""
        x = self.input_projection(features) + self.position_embedding
        x = self.input_dropout(self.input_norm(x))
        return self.output_norm(self.blocks(x))
