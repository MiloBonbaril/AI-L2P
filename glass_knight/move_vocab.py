"""Fixed action vocabulary: every geometrically possible UCI move.

Built once from square geometry (queen-slide + knight patterns + promotions),
independent of any board state. Every move python-chess ever emits as legal
is a member of this set, so policy targets and legality masks are simple
dict lookups.
"""
from __future__ import annotations

FILES = "abcdefgh"
DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
KNIGHT_DELTAS = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]


def _sq(f: int, r: int) -> str:
    return f"{FILES[f]}{r + 1}"


def build_move_vocab() -> list[str]:
    moves: set[str] = set()

    for f in range(8):
        for r in range(8):
            frm = _sq(f, r)
            for df, dr in DIRECTIONS:
                nf, nr = f + df, r + dr
                while 0 <= nf < 8 and 0 <= nr < 8:
                    moves.add(frm + _sq(nf, nr))
                    nf, nr = nf + df, nr + dr
            for df, dr in KNIGHT_DELTAS:
                nf, nr = f + df, r + dr
                if 0 <= nf < 8 and 0 <= nr < 8:
                    moves.add(frm + _sq(nf, nr))

    # Promotions need an explicit piece suffix in UCI; add straight + both
    # diagonal captures for both colors, all four promotion pieces.
    for f in range(8):
        for df in (-1, 0, 1):
            nf = f + df
            if not 0 <= nf < 8:
                continue
            for promo in "qrbn":
                moves.add(f"{FILES[f]}7{FILES[nf]}8{promo}")
                moves.add(f"{FILES[f]}2{FILES[nf]}1{promo}")

    return sorted(moves)


MOVE_VOCAB: list[str] = build_move_vocab()
MOVE_TO_INDEX: dict[str, int] = {m: i for i, m in enumerate(MOVE_VOCAB)}
NUM_MOVES: int = len(MOVE_VOCAB)


def move_to_index(uci: str) -> int:
    return MOVE_TO_INDEX[uci]


def index_to_move(idx: int) -> str:
    return MOVE_VOCAB[idx]


def legal_move_indices(board) -> list[int]:
    """board: a chess.Board. Returns the vocab indices of its legal moves."""
    return [MOVE_TO_INDEX[m.uci()] for m in board.legal_moves]


def _demo() -> None:
    import chess

    assert 1900 <= NUM_MOVES <= 2100, f"unexpected vocab size {NUM_MOVES}"
    assert len(MOVE_TO_INDEX) == NUM_MOVES  # no duplicate indices

    # Every legal move from a handful of positions must be in the vocab.
    boards = [chess.Board()]
    b = chess.Board()
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]:
        b.push_uci(uci)
        boards.append(b.copy())
    # A position with an en-passant capture and a promotion available.
    boards.append(chess.Board("8/P7/8/8/8/8/k1K5/8 w - - 0 1"))

    checked = 0
    for board in boards:
        for move in board.legal_moves:
            assert move.uci() in MOVE_TO_INDEX, f"missing legal move {move.uci()}"
            checked += 1
        assert len(legal_move_indices(board)) == board.legal_moves.count()
    assert checked > 0

    print(f"ok: {NUM_MOVES} moves in vocab, {checked} legal moves all resolved")


if __name__ == "__main__":
    _demo()
