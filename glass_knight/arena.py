"""Stockfish gauntlet + Elo fit (concept doc section 6).

Plays the current model (one forward pass per move, no search) against
Stockfish at fixed UCI_Elo levels from a small opening book, alternating
colors for variance reduction. Results go to SQLite; `fit_elo` turns a
checkpoint's game results into a single Elo number anchored to Stockfish's
own calibrated UCI_Elo scale.

Scope cut for M1: anchored 1D Elo fit (checkpoint vs known-strength
Stockfish levels), not the full Ordo/BayesElo multi-entity matrix from the
concept doc -- that matters once checkpoints also play each other. Revisit
when the ladder has enough checkpoint-vs-checkpoint games to need it.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import time
from pathlib import Path

import chess
import chess.engine
import torch

from glass_knight.move_vocab import index_to_move, legal_move_indices
from glass_knight.tokenizer import tokenize
from glass_knight.train import load_checkpoint

OPENING_BOOK = {
    "start": [],
    "italian": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
    "sicilian": ["e2e4", "c7c5"],
    "french": ["e2e4", "e7e6"],
    "qgd": ["d2d4", "d7d5", "c2c4", "e7e6"],
    "kings_indian": ["d2d4", "g8f6", "c2c4", "g7g6"],
}

MAX_PLIES = 200
STOCKFISH_MOVETIME_S = 0.1

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint TEXT NOT NULL,
    opponent_elo INTEGER NOT NULL,
    model_color TEXT NOT NULL,
    opening TEXT NOT NULL,
    result TEXT NOT NULL,
    model_score REAL NOT NULL,
    plies INTEGER NOT NULL,
    timestamp REAL NOT NULL
);
"""


class ModelPlayer:
    """Greedy: highest-probability legal move, one forward pass, no search."""

    def __init__(self, model, device: str = "cpu"):
        self.model = model.eval()
        self.device = device

    @torch.no_grad()
    def choose_move(self, board: chess.Board) -> chess.Move:
        tokens = torch.from_numpy(tokenize(board).astype("int64")).unsqueeze(0).to(self.device)
        policy_logits, _, _ = self.model(tokens, capture=False)
        mask = torch.full_like(policy_logits, float("-inf"))
        mask[0, legal_move_indices(board)] = 0.0
        best_idx = (policy_logits + mask)[0].argmax().item()
        return chess.Move.from_uci(index_to_move(best_idx))


def play_game(
    model_player: ModelPlayer,
    engine: "chess.engine.SimpleEngine",
    model_is_white: bool,
    opening_moves: list[str],
) -> tuple[str, float, int]:
    board = chess.Board()
    for uci in opening_moves:
        board.push_uci(uci)

    limit = chess.engine.Limit(time=STOCKFISH_MOVETIME_S)
    plies = len(opening_moves)
    while not board.is_game_over(claim_draw=True) and plies < MAX_PLIES:
        model_turn = (board.turn == chess.WHITE) == model_is_white
        move = model_player.choose_move(board) if model_turn else engine.play(board, limit).move
        board.push(move)
        plies += 1

    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
    model_score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
    if not model_is_white:
        model_score = 1.0 - model_score
    return result, model_score, plies


def run_gauntlet(
    checkpoint_dir: Path,
    stockfish_path: str,
    levels: list[int],
    db_path: Path,
    device: str | None = None,
) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(checkpoint_dir, device=device)
    player = ModelPlayer(model, device=device)
    checkpoint_name = str(checkpoint_dir.resolve())

    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)

    for elo in levels:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        for opening_name, moves in OPENING_BOOK.items():
            for model_is_white in (True, False):
                result, score, plies = play_game(player, engine, model_is_white, moves)
                conn.execute(
                    "INSERT INTO games (checkpoint, opponent_elo, model_color, opening, result, model_score, plies, timestamp)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (
                        checkpoint_name,
                        elo,
                        "white" if model_is_white else "black",
                        opening_name,
                        result,
                        score,
                        plies,
                        time.time(),
                    ),
                )
                conn.commit()
                color = "W" if model_is_white else "B"
                print(f"vs SF{elo:>4} [{opening_name:>12}] model={color}  {result}  ({plies} plies)")
        engine.quit()

    conn.close()
    return checkpoint_name


def fit_elo(results: list[tuple[float, float]], lo: float = 0.0, hi: float = 3200.0, iters: int = 60) -> float:
    """results: (opponent_elo, model_score) pairs. Bisects for the rating
    whose total expected score (logistic Elo formula) matches the total
    actual score -- a standard performance rating against known opponents."""
    total_score = sum(s for _, s in results)

    def expected_score(rating: float) -> float:
        return sum(1.0 / (1.0 + 10 ** ((opp - rating) / 400)) for opp, _ in results)

    for _ in range(iters):
        mid = (lo + hi) / 2
        if expected_score(mid) < total_score:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def elo_for_checkpoint(db_path: Path, checkpoint_name: str) -> float:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT opponent_elo, model_score FROM games WHERE checkpoint = ?", (checkpoint_name,)).fetchall()
    conn.close()
    if not rows:
        raise ValueError(f"no games recorded for checkpoint {checkpoint_name}")
    return fit_elo(rows)


def elo_ladder(db_path: Path) -> list[tuple[str, float, int]]:
    """The hero graph's data (concept doc section 6): fitted Elo per
    checkpoint, in training-step order. Step number is parsed from the
    checkpoint dir name (step_NNNNNN); checkpoints without that pattern sort last."""
    conn = sqlite3.connect(db_path)
    names = [row[0] for row in conn.execute("SELECT DISTINCT checkpoint FROM games")]
    conn.close()

    def step_of(name: str) -> int:
        stem = Path(name).name
        return int(stem.removeprefix("step_")) if stem.startswith("step_") else 1 << 62

    ladder = [(name, elo_for_checkpoint(db_path, name), step_of(name)) for name in names]
    return sorted(ladder, key=lambda row: row[2])


def _demo() -> None:
    # fit_elo is pure math -- verify it recovers a known rating from
    # synthetic games against opponents of known strength, no Stockfish needed.
    true_rating = 1450.0
    synthetic = [(opp, 1.0 / (1.0 + 10 ** ((opp - true_rating) / 400))) for opp in (1200, 1320, 1500, 1700, 1900)]
    fitted = fit_elo(synthetic)
    assert abs(fitted - true_rating) < 2.0, f"elo fit off: {fitted} vs {true_rating}"
    print(f"ok: fit_elo recovers a known rating ({fitted:.1f} vs {true_rating})")

    sf_path = shutil.which("stockfish")
    if sf_path is None:
        print("skip: stockfish not installed, not running the live gauntlet check")
        return

    import tempfile

    from glass_knight.data import generate_shard
    from glass_knight.train import train

    with tempfile.TemporaryDirectory() as tmp:
        shard = Path(tmp) / "shard.bin"
        generate_shard(shard, num_positions=500, seed=0, max_plies=40)
        ckpt_dir = Path(tmp) / "ckpt"
        train(shard, preset="tiny", steps=5, batch_size=64, subset_size=500, ckpt_dir=ckpt_dir)

        db_path = Path(tmp) / "arena.db"
        name = run_gauntlet(ckpt_dir / "latest", sf_path, levels=[1320], db_path=db_path)
        rating = elo_for_checkpoint(db_path, name)
        print(f"ok: live gauntlet ran, fitted rating = {rating:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("ckpt/latest"))
    parser.add_argument("--stockfish-path", type=str, default=None)
    parser.add_argument("--levels", type=int, nargs="+", default=[1320, 1500, 1700])
    parser.add_argument("--db", type=Path, default=Path("data/arena.db"))
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--report", action="store_true", help="print the Elo ladder (all checkpoints in --db) instead of running a gauntlet")
    args = parser.parse_args()

    if args.demo:
        _demo()
    elif args.report:
        for name, elo, step in elo_ladder(args.db):
            label = f"step {step:,}" if step < (1 << 62) else Path(name).name
            print(f"{label:>14}  Elo {elo:6.0f}   {name}")
    else:
        sf_path = args.stockfish_path or shutil.which("stockfish")
        if sf_path is None:
            raise SystemExit("stockfish not found on PATH; pass --stockfish-path")
        args.db.parent.mkdir(parents=True, exist_ok=True)
        name = run_gauntlet(args.ckpt, sf_path, args.levels, args.db)
        rating = elo_for_checkpoint(args.db, name)
        print(f"\n{args.ckpt}: fitted Elo = {rating:.0f}")
