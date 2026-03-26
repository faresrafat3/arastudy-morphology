"""Dataset utilities placeholders for AraStudy."""

from __future__ import annotations


def build_train_valid_split(corpus_path: str, train_ratio: float = 0.95) -> tuple[str, str]:
    raise NotImplementedError("Implemented in training data pipeline.")
