"""Position generation and the mmap shard format (concept doc section 4.3).

M0 has no Lichess dump yet, so positions come from random legal-move
self-play games. That's fine for a pipeline proof: M0 only needs to show the
tokenize -> shard -> mmap-dataset -> train loop is correct and the tiny
model can memorize it. Swapping in real PGN data later only touches
`play_random_game` / a new `generate_shard_from_pgn`, not the shard format
or the dataset reader.
"""
from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path

import chess
import numpy as np
import torch

from glass_knight.move_vocab import move_to_index
from glass_knight.tokenizer import SEQ_LEN, STM_OFFSET, tokenize

RECORD_DTYPE = np.dtype(
    [("tokens", np.uint8, (SEQ_LEN,)), ("move_idx", np.uint16), ("value_bucket", np.uint8)],
    align=False,
)
assert RECORD_DTYPE.itemsize == SEQ_LEN + 2 + 1  # 71 bytes/position, per spec

NUM_VALUE_BINS = 64


def play_random_game(rng: random.Random, max_plies: int = 200):
    board = chess.Board()
    records = []
    for _ in range(max_plies):
        if board.is_game_over():
            break
        legal = list(board.legal_moves)
        move = rng.choice(legal)
        records.append((tokenize(board), move_to_index(move.uci())))
        board.push(move)
    result = board.result(claim_draw=True) if board.is_game_over() else "1/2-1/2"
    return records, result


def prob_to_bucket(win_prob: float, num_bins: int = NUM_VALUE_BINS) -> int:
    return min(max(int(win_prob * num_bins), 0), num_bins - 1)


def result_to_prob(result: str, stm_is_white: bool) -> float:
    if result == "1/2-1/2":
        return 0.5
    return 1.0 if ((result == "1-0") == stm_is_white) else 0.0


def value_bucket(result: str, stm_is_white: bool, num_bins: int = NUM_VALUE_BINS) -> int:
    return prob_to_bucket(result_to_prob(result, stm_is_white), num_bins)


def generate_shard(path: Path, num_positions: int, seed: int = 0, max_plies: int = 200) -> int:
    rng = random.Random(seed)
    buf = np.zeros(num_positions, dtype=RECORD_DTYPE)
    count = 0
    while count < num_positions:
        records, result = play_random_game(rng, max_plies)
        for tokens, move_idx in records:
            if count >= num_positions:
                break
            stm_is_white = bool(tokens[64] == STM_OFFSET)
            buf[count] = (tokens, move_idx, value_bucket(result, stm_is_white))
            count += 1
    buf.tofile(path)
    return count


class ShardDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path):
        self.data = np.memmap(path, dtype=RECORD_DTYPE, mode="r")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        rec = self.data[idx]
        tokens = torch.from_numpy(rec["tokens"].astype(np.int64))
        return tokens, int(rec["move_idx"]), int(rec["value_bucket"])


def load_shards(pattern: str) -> torch.utils.data.Dataset:
    """`pattern` is a glob (absolute or relative), e.g.
    "data/lichess/shard_*.bin". Multiple shards concatenate into one
    dataset; a plain path with no wildcard works too."""
    is_glob = any(c in pattern for c in "*?[")
    paths = sorted(Path(p) for p in glob.glob(pattern)) if is_glob else [Path(pattern)]
    if not paths:
        raise FileNotFoundError(f"no shards match {pattern}")
    datasets = [ShardDataset(p) for p in paths]
    return datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)


def _demo() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shard.bin"
        n = generate_shard(path, num_positions=2000, seed=0, max_plies=80)
        assert n == 2000
        assert path.stat().st_size == 2000 * RECORD_DTYPE.itemsize

        ds = ShardDataset(path)
        assert len(ds) == 2000
        tokens, move_idx, bucket = ds[0]
        assert tokens.shape == (SEQ_LEN,)
        assert tokens.dtype == torch.int64
        assert 0 <= move_idx < 1968
        assert 0 <= bucket < NUM_VALUE_BINS

        # every record's token ids and buckets are in-range
        all_tokens = ds.data["tokens"]
        assert all_tokens.max() < 41
        assert ds.data["value_bucket"].max() < NUM_VALUE_BINS

        # load_shards concatenates multiple shard files matched by a glob
        generate_shard(Path(tmp) / "shard_001.bin", num_positions=500, seed=1, max_plies=80)
        (Path(tmp) / "shard_000.bin").write_bytes(path.read_bytes())
        multi = load_shards(str(Path(tmp) / "shard_*.bin"))
        assert len(multi) == 2500

    print("ok: shard round-trips through mmap with correct dtype and ranges")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/shard_000.bin"))
    parser.add_argument("--num-positions", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        _demo()
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        n = generate_shard(args.out, args.num_positions, args.seed, args.max_plies)
        size_mb = args.out.stat().st_size / 1e6
        print(f"wrote {n:,} positions to {args.out} ({size_mb:.1f} MB)")
