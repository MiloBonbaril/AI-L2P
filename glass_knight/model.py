"""Pre-norm transformer encoder over the 68-token board sequence.

Policy head reads the [MOVE] token and predicts a UCI move index. Value head
predicts a 64-bin win-probability bucket. Every block can optionally capture
its attention weights, residual stream, and pooled MLP activity — the
producer side of the visualization contract (concept doc section 3.3 / 8).
Capture is off by default (zero overhead path uses fused SDPA); turning it
on switches attention to a manual softmax so weights are observable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from glass_knight.move_vocab import NUM_MOVES
from glass_knight.tokenizer import SEQ_LEN, VOCAB_SIZE

MOVE_TOKEN_POS = SEQ_LEN - 1  # index 67, always the [MOVE] readout token
VALUE_BINS = 64
BRAIN_GROUPS = 64  # channel-groups the MLP hidden state is pooled into


@dataclass
class GlassKnightConfig:
    layers: int
    heads: int
    d_model: int
    d_ff: int
    dropout: float = 0.0
    seq_len: int = SEQ_LEN
    vocab_size: int = VOCAB_SIZE
    num_moves: int = NUM_MOVES
    value_bins: int = VALUE_BINS


PRESETS = {
    "tiny": GlassKnightConfig(layers=2, heads=4, d_model=128, d_ff=512),
    "small": GlassKnightConfig(layers=8, heads=8, d_model=512, d_ff=2048),
    "medium": GlassKnightConfig(layers=12, heads=12, d_model=768, d_ff=3072),
}


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        hidden = F.silu(gate) * up  # (..., d_ff), returned for activation capture
        return self.down(hidden), hidden


class Attention(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, capture: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t_.view(b, t, self.heads, self.head_dim).transpose(1, 2) for t_ in (q, k, v))

        if capture:
            # Manual attention so we can return the weights for the heatmap.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            weights = torch.softmax(scores, dim=-1)  # (b, heads, t, t)
            attn_out = weights @ v
        else:
            weights = None
            attn_out = F.scaled_dot_product_attention(q, k, v)  # fused, no weight materialization

        attn_out = attn_out.transpose(1, 2).reshape(b, t, self.heads * self.head_dim)
        return self.out(attn_out), weights


class Block(nn.Module):
    def __init__(self, cfg: GlassKnightConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg.d_model, cfg.heads)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x: torch.Tensor, capture: bool):
        attn_out, attn_weights = self.attn(self.norm1(x), capture)
        x = x + attn_out
        mlp_out, mlp_hidden = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x, attn_weights, (mlp_hidden if capture else None)


class BoardEmbedding(nn.Module):
    """Token identity + per-square learned position + rank/file factorized position."""

    def __init__(self, cfg: GlassKnightConfig):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.square_emb = nn.Embedding(64, cfg.d_model)
        self.file_emb = nn.Embedding(8, cfg.d_model)
        self.rank_emb = nn.Embedding(8, cfg.d_model)
        self.extra_pos_emb = nn.Embedding(4, cfg.d_model)  # stm, castling, ep, [MOVE]
        self.register_buffer("file_idx", torch.arange(64) % 8, persistent=False)
        self.register_buffer("rank_idx", torch.arange(64) // 8, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(tokens)
        x[:, :64] = x[:, :64] + self.square_emb.weight + self.file_emb(self.file_idx) + self.rank_emb(self.rank_idx)
        x[:, 64:68] = x[:, 64:68] + self.extra_pos_emb.weight
        return x


class GlassKnight(nn.Module):
    def __init__(self, cfg: GlassKnightConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = BoardEmbedding(cfg)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.final_norm = RMSNorm(cfg.d_model)
        self.policy_head = nn.Linear(cfg.d_model, cfg.num_moves, bias=False)
        self.value_head = nn.Linear(cfg.d_model, cfg.value_bins, bias=False)

    def forward(self, tokens: torch.Tensor, capture: bool = False):
        """tokens: (batch, 68) long.

        Returns (policy_logits, value_logits, trace) where trace is None
        unless capture=True. trace shapes (per layer, batch included):
          attn        : (batch, heads, 68, 68)
          residual    : (batch, 68, d_model)   -- block output, post-residual
          brain       : (batch, 64)            -- pooled |MLP hidden| per group
          logit_lens  : (batch, num_moves)     -- policy head applied early
        """
        x = self.embed(tokens)
        trace = {"attn": [], "residual": [], "brain": [], "logit_lens": []} if capture else None

        for block in self.blocks:
            x, attn_weights, mlp_hidden = block(x, capture)
            if capture:
                trace["attn"].append(attn_weights)
                trace["residual"].append(x)
                trace["brain"].append(_pool_brain(mlp_hidden, BRAIN_GROUPS))
                trace["logit_lens"].append(self.policy_head(self.final_norm(x)[:, MOVE_TOKEN_POS]))

        x = self.final_norm(x)
        move_repr = x[:, MOVE_TOKEN_POS]
        policy_logits = self.policy_head(move_repr)
        value_logits = self.value_head(move_repr)
        return policy_logits, value_logits, trace


def _pool_brain(mlp_hidden: torch.Tensor, groups: int) -> torch.Tensor:
    # ponytail: mean-pool over sequence then split channels into `groups`
    # even chunks. Good enough for "is this layer lit up"; if uneven-size
    # chunks ever visibly skew the brain panel, switch to a fixed reshape.
    per_channel = mlp_hidden.abs().mean(dim=1)  # (batch, d_ff)
    return torch.stack([chunk.mean(dim=-1) for chunk in per_channel.chunk(groups, dim=-1)], dim=-1)


def _demo() -> None:
    torch.manual_seed(0)
    cfg = PRESETS["tiny"]
    model = GlassKnight(cfg)
    n = num_params(model)
    print(f"tiny params: {n:,}")

    batch = 3
    tokens = torch.randint(0, cfg.vocab_size, (batch, cfg.seq_len))

    policy_logits, value_logits, trace = model(tokens, capture=False)
    assert policy_logits.shape == (batch, cfg.num_moves)
    assert value_logits.shape == (batch, cfg.value_bins)
    assert trace is None

    policy_logits2, value_logits2, trace = model(tokens, capture=True)
    assert torch.allclose(policy_logits, policy_logits2, atol=1e-4), "capture path must match fused path"
    assert len(trace["attn"]) == cfg.layers
    assert trace["attn"][0].shape == (batch, cfg.heads, cfg.seq_len, cfg.seq_len)
    assert trace["residual"][0].shape == (batch, cfg.seq_len, cfg.d_model)
    assert trace["brain"][0].shape == (batch, BRAIN_GROUPS)
    assert trace["logit_lens"][0].shape == (batch, cfg.num_moves)
    # attention rows are probability distributions
    assert torch.allclose(trace["attn"][0].sum(dim=-1), torch.ones(batch, cfg.heads, cfg.seq_len), atol=1e-4)

    for name, cfg in PRESETS.items():
        m = GlassKnight(cfg)
        print(f"{name}: {num_params(m):,} params")

    print("ok: forward shapes, capture-path parity, and hook contract check out")


if __name__ == "__main__":
    _demo()
