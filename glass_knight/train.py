"""Training loop, checkpointing, and the M0 overfit check.

Overfit check: train the `tiny` preset on a small slice of a shard with no
regularization until it memorizes (near-zero loss, near-100% move accuracy).
That's the standard "does the pipeline have a bug" sanity test — if a tiny
model can't memorize a few thousand positions, tokenizer/loss/model wiring
is broken, no point scaling up.

Real runs (M1+) point --shard-glob at Lichess shards and use --preset small;
checkpoints land in ckpt/step_NNNNNN/ with a ckpt/latest symlink, which is
what glass_knight.arena consumes.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from glass_knight.data import load_shards
from glass_knight.model import PRESETS, GlassKnight, GlassKnightConfig, num_params

VALUE_LOSS_WEIGHT = 0.3  # lambda from concept doc section 3.2


def loss_fn(policy_logits, value_logits, move_idx, value_bucket):
    policy_loss = F.cross_entropy(policy_logits, move_idx)
    value_loss = F.cross_entropy(value_logits, value_bucket)
    return policy_loss + VALUE_LOSS_WEIGHT * value_loss, policy_loss, value_loss


def warmup_cosine(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def save_checkpoint(model: GlassKnight, cfg: GlassKnightConfig, step: int, ckpt_dir: Path) -> Path:
    step_dir = ckpt_dir / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), step_dir / "model.pt")
    (step_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg)))

    latest = ckpt_dir / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(step_dir.name)
    return step_dir


def load_checkpoint(step_dir: Path, device: str = "cpu") -> tuple[GlassKnight, GlassKnightConfig]:
    cfg = GlassKnightConfig(**json.loads((step_dir / "config.json").read_text()))
    model = GlassKnight(cfg)
    model.load_state_dict(torch.load(step_dir / "model.pt", map_location=device))
    model.to(device)
    return model, cfg


def train(
    shard_path: str | Path,
    preset: str = "tiny",
    steps: int = 300,
    batch_size: int = 256,
    subset_size: int | None = 2000,
    lr: float = 3e-4,
    warmup_steps: int = 0,
    device: str | None = None,
    log_every: int = 25,
    ckpt_dir: Path | None = None,
    ckpt_every: int = 0,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_shards(str(shard_path))
    if subset_size is not None:
        ds = Subset(ds, range(min(subset_size, len(ds))))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

    cfg = PRESETS[preset]
    model = GlassKnight(cfg).to(device)
    print(f"{preset}: {num_params(model):,} params on {device}, {len(ds):,} positions")

    use_amp = device == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=use_amp)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, warmup_steps, steps))

    step = 0
    last_acc = 0.0
    last_loss = float("inf")
    model.train()
    while step < steps:
        for tokens, move_idx, value_bucket in loader:
            if step >= steps:
                break
            tokens, move_idx, value_bucket = (t.to(device) for t in (tokens, move_idx, value_bucket))

            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                policy_logits, value_logits, _ = model(tokens, capture=False)
                loss, policy_loss, value_loss = loss_fn(policy_logits, value_logits, move_idx, value_bucket)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            acc = (policy_logits.argmax(-1) == move_idx).float().mean().item()
            last_acc, last_loss = acc, loss.item()
            if step % log_every == 0 or step == steps - 1:
                lr_now = sched.get_last_lr()[0]
                print(f"step {step:6d}  loss {loss.item():.4f}  policy_acc {acc:.3f}  lr {lr_now:.2e}")
            if ckpt_dir is not None and ckpt_every > 0 and (step + 1) % ckpt_every == 0:
                save_checkpoint(model, cfg, step + 1, ckpt_dir)
            step += 1

    if ckpt_dir is not None:
        save_checkpoint(model, cfg, step, ckpt_dir)
        print(f"saved checkpoint at step {step} -> {ckpt_dir}/latest")

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

        # checkpoint save/load round-trips to an identical model
        ckpt_dir = Path(tmp) / "ckpt"
        model, _, _ = train(shard, preset="tiny", steps=1, batch_size=128, subset_size=1000, ckpt_dir=ckpt_dir)
        loaded, cfg = load_checkpoint(ckpt_dir / "latest")
        assert cfg == PRESETS["tiny"]
        for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
            assert torch.allclose(a.cpu(), b.cpu())

    print(f"ok: overfit check passed (final loss {final_loss:.4f}, policy acc {final_acc:.3f}); checkpoint round-trip ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=str, default="data/shard_000.bin", help="path or glob, e.g. data/lichess/shard_*.bin")
    parser.add_argument("--preset", choices=list(PRESETS), default="tiny")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--subset-size", type=int, default=2000)
    parser.add_argument("--full", action="store_true", help="train on the whole shard, not a subset")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--ckpt-every", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--demo", action="store_true", help="run the overfit self-check")
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
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            log_every=args.log_every,
            ckpt_dir=args.ckpt_dir,
            ckpt_every=args.ckpt_every,
        )
