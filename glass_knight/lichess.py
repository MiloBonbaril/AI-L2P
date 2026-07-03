"""Stream-filter a Lichess monthly PGN dump straight into shards.

Reads PGN text from stdin one game at a time (never buffers the whole dump)
so it composes with a streaming download+decompress pipe:

    curl -s https://database.lichess.org/standard/lichess_db_standard_rated_2026-06.pgn.zst \
        | zstd -dc \
        | uv run python -m glass_knight.lichess --out data/lichess/shard --num-positions 8000000

Filters: rated games, both players >=1800, drop bullet (base time <3min),
drop games under ~10 full moves (section 4.2). Stage-A value target is the
game outcome, blended 50/50 toward the position's Lichess %eval annotation
when the PGN carries one (ponytail: fixed 50/50 blend rather than the
full discounting scheme the concept doc leaves unspecified -- revisit once
Stage-B distillation needs a sharper value signal). Opening positions (first
~10 full moves) are capped per unique board FEN so repeated openings across
millions of games don't dominate the diet.
"""
from __future__ import annotations

import argparse
import io
import math
import re
import sys
from collections import Counter
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from glass_knight.data import RECORD_DTYPE, prob_to_bucket, result_to_prob
from glass_knight.move_vocab import move_to_index
from glass_knight.tokenizer import STM_OFFSET, tokenize

MIN_ELO = 1800
MIN_BASE_SECONDS = 180  # drop bullet/ultrabullet, keep blitz..classical
MIN_PLIES = 20  # ~10 full moves
OPENING_PLY_CUTOFF = 20
OPENING_FEN_CAP = 200
SHARD_SIZE = 10_000_000

EVAL_RE = re.compile(r"\[%eval\s+(#?-?\d+(?:\.\d+)?)\]")


def eval_to_white_winprob(raw: str) -> float:
    """Lichess %eval is always from White's perspective: a pawn-unit
    centipawn score, or "#N" / "#-N" for mate in N."""
    if raw.startswith("#"):
        return 1.0 if int(raw[1:]) > 0 else 0.0
    cp = float(raw) * 100
    return 1.0 / (1.0 + math.exp(-cp / 400.0))


def game_qualifies(headers: chess.pgn.Headers) -> bool:
    if "Rated" not in headers.get("Event", ""):
        return False
    tc = headers.get("TimeControl", "-")
    if tc == "-":
        return False
    try:
        base = int(tc.split("+")[0])
        white_elo = int(headers.get("WhiteElo", "0"))
        black_elo = int(headers.get("BlackElo", "0"))
    except ValueError:
        return False
    if base < MIN_BASE_SECONDS:
        return False
    return white_elo >= MIN_ELO and black_elo >= MIN_ELO


class ShardWriter:
    def __init__(self, out_prefix: Path, shard_size: int = SHARD_SIZE):
        self.out_prefix = out_prefix
        self.shard_size = shard_size
        self.shard_idx = 0
        self.buf = np.zeros(shard_size, dtype=RECORD_DTYPE)
        self.count_in_shard = 0
        self.total = 0

    def add(self, tokens, move_idx, bucket) -> None:
        self.buf[self.count_in_shard] = (tokens, move_idx, bucket)
        self.count_in_shard += 1
        self.total += 1
        if self.count_in_shard == self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if self.count_in_shard == 0:
            return
        path = self.out_prefix.with_name(f"{self.out_prefix.name}_{self.shard_idx:03d}.bin")
        self.buf[: self.count_in_shard].tofile(path)
        self.shard_idx += 1
        self.count_in_shard = 0

    def close(self) -> None:
        self._flush()


def _extract_eval(comment: str) -> str | None:
    m = EVAL_RE.search(comment)
    return m.group(1) if m else None


def stream_filter(handle, writer: ShardWriter, target_positions: int, log_every_games: int = 5000) -> None:
    opening_counts: Counter[str] = Counter()
    games_seen = games_kept = 0

    while writer.total < target_positions:
        game = chess.pgn.read_game(handle)
        if game is None:
            break
        games_seen += 1
        if not game_qualifies(game.headers):
            continue
        result = game.headers.get("Result", "*")
        if result not in ("1-0", "0-1", "1/2-1/2"):
            continue

        board = game.board()
        node = game
        ply = 0
        game_records = []
        while node.variations:
            next_node = node.variations[0]
            move = next_node.move
            eval_raw = _extract_eval(next_node.comment)

            tokens = tokenize(board)
            stm_is_white = bool(tokens[64] == STM_OFFSET)
            outcome_prob = result_to_prob(result, stm_is_white)
            if eval_raw is not None:
                white_wp = eval_to_white_winprob(eval_raw)
                eval_prob = white_wp if stm_is_white else 1.0 - white_wp
                win_prob = 0.5 * outcome_prob + 0.5 * eval_prob
            else:
                win_prob = outcome_prob

            keep = True
            if ply < OPENING_PLY_CUTOFF:
                fen_key = board.board_fen()
                opening_counts[fen_key] += 1
                keep = opening_counts[fen_key] <= OPENING_FEN_CAP
            if keep:
                game_records.append((tokens, move_to_index(move.uci()), prob_to_bucket(win_prob)))

            board.push(move)
            node = next_node
            ply += 1

        if ply >= MIN_PLIES:
            games_kept += 1
            for tokens, move_idx, bucket in game_records:
                if writer.total >= target_positions:
                    break
                writer.add(tokens, move_idx, bucket)

        if games_seen % log_every_games == 0:
            print(f"games_seen={games_seen} games_kept={games_kept} positions={writer.total}", file=sys.stderr, flush=True)

    writer.close()
    print(f"done: games_seen={games_seen} games_kept={games_kept} positions={writer.total}", file=sys.stderr)


_SAMPLE_PGN = """[Event "Rated Blitz game"]
[Result "1-0"]
[WhiteElo "2100"]
[BlackElo "1950"]
[TimeControl "300+3"]

1. e4 { [%eval 0.3] } e5 2. Nf3 { [%eval 0.35] } Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 O-O 8. c3 d6 9. h3 Nb8 10. d4 Nbd7 1-0

[Event "Rated Bullet game"]
[Result "0-1"]
[WhiteElo "2200"]
[BlackElo "2100"]
[TimeControl "60+0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 O-O 0-1

[Event "Rated Blitz game"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "300+3"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 O-O 1-0

[Event "Casual Blitz game"]
[Result "1-0"]
[WhiteElo "2100"]
[BlackElo "1950"]
[TimeControl "300+3"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 O-O 1-0

[Event "Rated Blitz game"]
[Result "1/2-1/2"]
[WhiteElo "2000"]
[BlackElo "2000"]
[TimeControl "300+3"]

1. e4 e5 2. Nf3 1/2-1/2
"""


def _demo() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "shard"
        writer = ShardWriter(out, shard_size=1000)
        handle = io.StringIO(_SAMPLE_PGN)
        stream_filter(handle, writer, target_positions=1000, log_every_games=10_000)

        # Only the first game qualifies: rated, both >=1800 Elo, blitz, 20 plies.
        # (bullet, low-elo, casual, and the 3-ply draw are all filtered out.)
        assert writer.total == 20, f"expected 20 kept positions, got {writer.total}"

        shard_path = out.with_name("shard_000.bin")
        data = np.fromfile(shard_path, dtype=RECORD_DTYPE)
        assert len(data) == 20
        # first position (starting board, White to move) had %eval 0.3 blended with a White win
        assert data[0]["value_bucket"] > prob_to_bucket(0.5)

    assert not game_qualifies({"Event": "Rated Bullet game", "TimeControl": "60+0", "WhiteElo": "2000", "BlackElo": "2000"})
    assert eval_to_white_winprob("#-3") == 0.0
    assert eval_to_white_winprob("#2") == 1.0
    assert abs(eval_to_white_winprob("0") - 0.5) < 1e-9

    print("ok: lichess filter keeps qualifying games/positions and rejects the rest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/lichess/shard"))
    parser.add_argument("--num-positions", type=int, default=8_000_000)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        _demo()
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        handle = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
        stream_filter(handle, ShardWriter(args.out), args.num_positions)
