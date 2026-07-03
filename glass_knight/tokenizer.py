"""FEN -> fixed 68-token sequence.

64 square tokens (a1..h8, python-chess's native SQUARES order) + side-to-move
+ castling rights + en-passant file + a constant [MOVE] readout token.
"""
from __future__ import annotations

import chess
import numpy as np

PIECE_SYMBOLS = "PNBRQKpnbrqk"  # 1..12, 0 = empty square
PIECE_TO_ID = {s: i + 1 for i, s in enumerate(PIECE_SYMBOLS)}

STM_OFFSET = 13       # 2 values -> ids 13-14
CASTLING_OFFSET = 15  # 16 values (KQkq bitmask) -> ids 15-30
EP_OFFSET = 31        # 9 values (file 0-7, 8 = none) -> ids 31-39
MOVE_TOKEN_ID = 40

VOCAB_SIZE = 41
SEQ_LEN = 68


def tokenize(board: chess.Board) -> np.ndarray:
    tokens = np.zeros(SEQ_LEN, dtype=np.uint8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        tokens[square] = PIECE_TO_ID[piece.symbol()] if piece else 0

    tokens[64] = STM_OFFSET + (0 if board.turn == chess.WHITE else 1)

    castling = (
        (board.has_kingside_castling_rights(chess.WHITE) << 0)
        | (board.has_queenside_castling_rights(chess.WHITE) << 1)
        | (board.has_kingside_castling_rights(chess.BLACK) << 2)
        | (board.has_queenside_castling_rights(chess.BLACK) << 3)
    )
    tokens[65] = CASTLING_OFFSET + castling

    ep_file = chess.square_file(board.ep_square) if board.ep_square is not None else 8
    tokens[66] = EP_OFFSET + ep_file

    tokens[67] = MOVE_TOKEN_ID
    return tokens


def tokenize_fen(fen: str) -> np.ndarray:
    return tokenize(chess.Board(fen))


def _demo() -> None:
    start = tokenize(chess.Board())
    assert start.shape == (SEQ_LEN,)
    assert start.dtype == np.uint8
    assert start[0] == PIECE_TO_ID["R"]  # a1 = white rook
    assert start[4] == PIECE_TO_ID["K"]  # e1 = white king
    assert start[60] == PIECE_TO_ID["k"]  # e8 = black king
    assert start[20] == 0  # e3 empty
    assert start[64] == STM_OFFSET + 0  # white to move
    assert start[65] == CASTLING_OFFSET + 0b1111  # all castling rights
    assert start[66] == EP_OFFSET + 8  # no en-passant
    assert start[67] == MOVE_TOKEN_ID
    assert tokens_max_id(start) < VOCAB_SIZE

    ep_board = chess.Board()
    ep_board.push_uci("e2e4")
    ep_tokens = tokenize(ep_board)
    assert ep_tokens[66] == EP_OFFSET + 4  # e-file

    print("ok: tokenizer shapes and start-position tokens check out")


def tokens_max_id(tokens: np.ndarray) -> int:
    return int(tokens.max())


if __name__ == "__main__":
    _demo()
