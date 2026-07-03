"""M0 self-check: run every module's demo() end to end.

python -m pytest tests/  (or just: python tests/test_pipeline.py)
"""
from glass_knight import arena, data, lichess, model, move_vocab, tokenizer, train


def test_move_vocab():
    move_vocab._demo()


def test_arena_elo_fit():
    arena._demo()


def test_tokenizer():
    tokenizer._demo()


def test_model_hooks():
    model._demo()


def test_data_shard_roundtrip():
    data._demo()


def test_lichess_filter():
    lichess._demo()


def test_overfit_check():
    train._demo()


if __name__ == "__main__":
    test_move_vocab()
    test_arena_elo_fit()
    test_tokenizer()
    test_model_hooks()
    test_data_shard_roundtrip()
    test_lichess_filter()
    test_overfit_check()
    print("\nall checks passed")
