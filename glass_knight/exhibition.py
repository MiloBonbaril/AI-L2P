"""Exhibition server (concept doc section 8): loads a checkpoint, plays a
game one ply at a time on request, and streams each of the model's moves
as a full interpretability trace over WebSocket. The client owns replay
pacing (concept doc section 7.1) -- this server just answers "what's the
next ply" instantly.

Ponytail scope cuts vs the concept doc, see also trace.py:
  - No NATS bus. One producer (this server), one consumer (the UI) --
    add NATS when the trainer/arena also need to publish to the same
    stream, not before.
  - One game per WebSocket connection, held in memory. No multi-client
    shared-game registry -- add one if "watch the same game from two
    browsers" becomes a real requirement.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import chess
import chess.engine
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from glass_knight.arena import STOCKFISH_MOVETIME_S
from glass_knight.trace import build_trace_message
from glass_knight.train import load_checkpoint

EVAL_LIMIT = chess.engine.Limit(time=0.1)

app = FastAPI()
_state: dict = {}  # model / device / stockfish_path, populated before uvicorn.run


class ExhibitionGame:
    def __init__(self, model, device: str, stockfish_path: str, opponent: str, model_color: str, elo: int):
        self.model = model
        self.device = device
        self.board = chess.Board()
        self.opponent = opponent
        self.model_is_white = model_color == "white"

        self.eval_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.opponent_engine = None
        if opponent == "stockfish":
            self.opponent_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            self.opponent_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})

    def sf_eval(self) -> float | None:
        try:
            info = self.eval_engine.analyse(self.board, EVAL_LIMIT)
            return info["score"].white().score(mate_score=100_000) / 100.0
        except Exception:
            return None

    def next_ply(self) -> dict:
        if self.board.is_game_over(claim_draw=True):
            return {"type": "game_over", "result": self.board.result(claim_draw=True)}

        model_turn = (self.board.turn == chess.WHITE) == self.model_is_white
        color = "white" if self.board.turn == chess.WHITE else "black"

        if model_turn:
            msg = build_trace_message(self.board, self.model, self.device)
            move = chess.Move.from_uci(msg["chosen_move"])
            self.board.push(move)
            payload = {"type": "move", "color": color, "uci": move.uci(), "fen": self.board.fen(), "trace": msg}
        elif self.opponent_engine is not None:
            move = self.opponent_engine.play(self.board, chess.engine.Limit(time=STOCKFISH_MOVETIME_S)).move
            self.board.push(move)
            payload = {"type": "move", "color": color, "uci": move.uci(), "fen": self.board.fen(), "trace": None}
        else:
            # opponent == "self": the model plays this side too, but it's
            # not "on stage" so no trace is captured/sent for its move.
            msg = build_trace_message(self.board, self.model, self.device)
            move = chess.Move.from_uci(msg["chosen_move"])
            self.board.push(move)
            payload = {"type": "move", "color": color, "uci": move.uci(), "fen": self.board.fen(), "trace": None}

        payload["sf_eval"] = self.sf_eval()
        return payload

    def close(self) -> None:
        self.eval_engine.quit()
        if self.opponent_engine is not None:
            self.opponent_engine.quit()


@app.websocket("/ws/exhibition")
async def exhibition_ws(ws: WebSocket) -> None:
    await ws.accept()
    params = ws.query_params
    opponent = params.get("opponent", "stockfish")
    model_color = params.get("model_color", "white")
    elo = int(params.get("elo", "1500"))

    game = ExhibitionGame(_state["model"], _state["device"], _state["stockfish_path"], opponent, model_color, elo)
    await ws.send_json({"type": "init", "fen": game.board.fen(), "model_color": model_color, "opponent": opponent})

    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "next":
                await ws.send_json(game.next_ply())
    except WebSocketDisconnect:
        pass
    finally:
        game.close()


def _demo() -> None:
    """Drives the WebSocket protocol end to end with a tiny model and (if
    installed) real Stockfish, asserting message shapes match the contract."""
    from fastapi.testclient import TestClient

    from glass_knight.model import PRESETS, GlassKnight

    sf_path = shutil.which("stockfish")
    if sf_path is None:
        print("skip: stockfish not installed, not running the exhibition server check")
        return

    model = GlassKnight(PRESETS["tiny"])
    _state.update(model=model, device="cpu", stockfish_path=sf_path)

    client = TestClient(app)
    with client.websocket_connect("/ws/exhibition?opponent=stockfish&model_color=white&elo=1320") as ws:
        init = ws.receive_json()
        assert init["type"] == "init"
        assert init["model_color"] == "white"

        ws.send_json({"type": "next"})
        model_move = ws.receive_json()
        assert model_move["type"] == "move" and model_move["color"] == "white"
        assert model_move["trace"] is not None
        assert model_move["trace"]["chosen_move"] == model_move["uci"]
        assert len(model_move["trace"]["per_layer"]) == PRESETS["tiny"].layers

        ws.send_json({"type": "next"})
        opp_move = ws.receive_json()
        assert opp_move["type"] == "move" and opp_move["color"] == "black"
        assert opp_move["trace"] is None  # opponent moves aren't visualized

    print("ok: exhibition WS protocol produces init -> model move (with trace) -> opponent move (no trace)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("ckpt/latest"))
    parser.add_argument("--stockfish-path", type=str, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        _demo()
    else:
        sf_path = args.stockfish_path or shutil.which("stockfish")
        if sf_path is None:
            raise SystemExit("stockfish not found on PATH; pass --stockfish-path")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = load_checkpoint(args.ckpt, device=device)
        _state.update(model=model, device=device, stockfish_path=sf_path)
        uvicorn.run(app, host=args.host, port=args.port)
