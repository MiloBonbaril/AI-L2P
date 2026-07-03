"""Convert a capture-mode forward pass into the exhibition wire format
(concept doc section 8): one dict per model move, quantized tensors ready
to ship over WebSocket and pace client-side into the replay animation.

Ponytail scope cuts vs the concept doc's exact contract:
  - Only the [MOVE] token's attention to the 64 board squares is sent
    (heads x 64), not the full 68x68 matrix -- nothing in the three M2
    views reads square-to-square attention, only "what is the network
    looking at when it's about to move." Revisit if a future view needs it.
  - Brain and value-bucket tensors are min-max normalized per-move (the
    "deviation from baseline" the concept doc asks for in section 7.3),
    not against a running session baseline -- revisit once that matters.
  - Tensors are base64-encoded uint8 inside JSON, not msgpack -- avoids a
    JS dependency; still a ~4x size win over a plain JSON int list.
"""
from __future__ import annotations

import base64

import torch

from glass_knight.model import MOVE_TOKEN_POS
from glass_knight.move_vocab import index_to_move, legal_move_indices
from glass_knight.tokenizer import tokenize

TOP_K = 3


def _b64_u8(arr: torch.Tensor) -> str:
    return base64.b64encode(arr.to(torch.uint8).cpu().numpy().tobytes()).decode("ascii")


def _quantize01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(0, 1) * 255).round()


def _normalize_quantize(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.min(), x.max()
    if (hi - lo).item() < 1e-8:
        return torch.zeros_like(x)
    return ((x - lo) / (hi - lo) * 255).round()


def _topk_moves(logits: torch.Tensor, legal_idx: list[int], k: int = TOP_K) -> list[dict]:
    mask = torch.full_like(logits, float("-inf"))
    mask[legal_idx] = 0.0
    probs = torch.softmax(logits + mask, dim=-1)
    top_probs, top_idx = probs.topk(min(k, len(legal_idx)))
    return [{"move": index_to_move(i.item()), "prob": round(p.item(), 4)} for p, i in zip(top_probs, top_idx)]


def build_trace_message(board, model, device: str = "cpu") -> dict:
    """Runs one capture=True forward pass for `board`'s side to move and
    returns the wire-format dict for that move (concept doc section 8)."""
    legal_idx = legal_move_indices(board)
    tokens = torch.from_numpy(tokenize(board).astype("int64")).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        policy_logits, value_logits, trace = model(tokens, capture=True)

    mask = torch.full_like(policy_logits[0], float("-inf"))
    mask[legal_idx] = 0.0
    chosen_idx = (policy_logits[0] + mask).argmax().item()
    chosen_move = index_to_move(chosen_idx)

    layers = []
    for attn, brain, logit_lens in zip(trace["attn"], trace["brain"], trace["logit_lens"]):
        move_token_attn = attn[0, :, MOVE_TOKEN_POS, :64]  # (heads, 64 squares)
        layers.append(
            {
                "attn": _b64_u8(_quantize01(move_token_attn)),
                "attn_heads": move_token_attn.shape[0],
                "brain": _b64_u8(_normalize_quantize(brain[0])),
                "logits_topk": _topk_moves(logit_lens[0], legal_idx),
            }
        )

    value_probs = torch.softmax(value_logits[0], dim=-1)
    bucket_centers = (torch.arange(value_probs.shape[0], device=value_probs.device) + 0.5) / value_probs.shape[0]
    win_prob = (value_probs * bucket_centers).sum().item()

    return {
        "fen": board.fen(),
        "chosen_move": chosen_move,
        "legal_moves": [index_to_move(i) for i in legal_idx],
        "per_layer": layers,
        # value_bucket_dist is min-max normalized for a legible sparkline shape;
        # value_winprob is the real (unnormalized) expected win probability.
        "value_bucket_dist": _b64_u8(_normalize_quantize(value_probs)),
        "value_winprob": round(win_prob, 4),
    }


def _demo() -> None:
    import chess

    from glass_knight.model import PRESETS, GlassKnight

    torch.manual_seed(0)
    cfg = PRESETS["tiny"]
    model = GlassKnight(cfg)
    board = chess.Board()

    msg = build_trace_message(board, model)
    assert msg["fen"] == board.fen()
    assert chess.Move.from_uci(msg["chosen_move"]) in board.legal_moves
    assert set(msg["legal_moves"]) == {m.uci() for m in board.legal_moves}
    assert len(msg["per_layer"]) == cfg.layers

    layer0 = msg["per_layer"][0]
    attn_bytes = base64.b64decode(layer0["attn"])
    assert len(attn_bytes) == cfg.heads * 64
    assert layer0["attn_heads"] == cfg.heads
    brain_bytes = base64.b64decode(layer0["brain"])
    assert len(brain_bytes) == 64
    assert len(layer0["logits_topk"]) == TOP_K
    assert all(chess.Move.from_uci(m["move"]) in board.legal_moves for m in layer0["logits_topk"])

    value_bytes = base64.b64decode(msg["value_bucket_dist"])
    assert len(value_bytes) == cfg.value_bins
    assert 0.0 <= msg["value_winprob"] <= 1.0

    print("ok: trace message matches the wire contract shapes and decodes cleanly")


if __name__ == "__main__":
    _demo()
