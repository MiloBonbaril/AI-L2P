"""Training loop + the M0 overfit check.

Overfit check: train the `tiny` preset on a small slice of a shard with no
regularization until it memorizes (near-zero loss, near-100% move accuracy).
That's the standard "does the pipeline have a bug" sanity test — if a tiny
model can't memorize a few thousand positions, tokenizer/loss/model wiring
is broken, no point scaling up.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from glass_knight.data import ShardDataset
from glass_knight.model import PRESETS, GlassKnight, num_params

VALUE_LOSS_WEIGHT = 0.3  # lambda from concept doc section 3.2


def loss_fn(policy_logits, value_logits, move_idx, value_bucket):
    policy_loss = F.cross_entropy(policy_logits, move_idx)
    value_loss = F.cross_entropy(value_logits, value_bucket)
    return policy_loss + VALUE_LOSS_WEIGHT * value_loss, policy_loss, value_loss


def train(
    shard_path: Path,
    preset: str = "tiny",
    steps: int = 300,
    batch_size: int = 256,
    subset_size: int | None = 2000,
    lr: float = 3e-4,
    device: str | None = None,
    log_every: int = 25,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = ShardDataset(shard_path)
    if subset_size is not None:
        ds = Subset(ds, range(min(subset_size, len(ds))))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    cfg = PRESETS[preset]
    model = GlassKnight(cfg).to(device)
    print(f"{preset}: {num_params(model):,} params on {device}, {len(ds):,} positions")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    step = 0
    last_acc = 0.0
    last_loss = float("inf")
    model.train()
    while step < steps:
        for tokens, move_idx, value_bucket in loader:
            if step >= steps:
                break
            tokens, move_idx, value_bucket = (t.to(device) for t in (tokens, move_idx, value_bucket))

            policy_logits, value_logits, _ = model(tokens, capture=False)
            loss, policy_loss, value_loss = loss_fn(policy_logits, value_logits, move_idx, value_bucket)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            acc = (policy_logits.argmax(-1) == move_idx).float().mean().item()
            last_acc, last_loss = acc, loss.item()
            if step % log_every == 0 or step == steps - 1:
                print(f"step {step:4d}  loss {loss.item():.4f}  policy_acc {acc:.3f}")
            step += 1

    return model, last_loss, last_acc


def _demo() -> None:
    import tempfile

    from glass_knight.data import generate_shard

    with tempfile.TemporaryDirectory() as tmp:
        shard = Path(tmp) / "overfit.bin"
        generate_shard(shard, num_positions=1000, seed=1, max_plies=60)

        _, final_loss, final_acc = train(
            shard, preset="tiny", steps=2000, batch_size=128, subset_size=1000, lr=1e-3, log_every=1000
        )
        assert final_acc > 0.9, f"tiny model failed to memorize 1000 positions (acc={final_acc:.3f})"
        assert final_loss < 0.5, f"loss did not collapse on overfit check (loss={final_loss:.3f})"

    print(f"ok: overfit check passed (final loss {final_loss:.4f}, policy acc {final_acc:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, default=Path("data/shard_000.bin"))
    parser.add_argument("--preset", choices=list(PRESETS), default="tiny")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--subset-size", type=int, default=2000)
    parser.add_argument("--full", action="store_true", help="train on the whole shard, not a subset")
    parser.add_argument("--demo", action="store_true", help="run the M0 overfit self-check")
    args = parser.parse_args()

    if args.demo:
        _demo()
    else:
        train(
            args.shard,
            preset=args.preset,
            steps=args.steps,
            batch_size=args.batch_size,
            subset_size=None if args.full else args.subset_size,
        )
